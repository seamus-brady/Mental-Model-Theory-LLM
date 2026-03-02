"""
Eval: Illusory Inferences from Mental Model Theory.

Tests the agent on classic illusory inference problems from:
- Johnson-Laird & Savary (1999): "Illusory inferences: a novel class of erroneous deductions"
- Johnson-Laird (2006): "How We Reason"

The key prediction of MMT: System 1 (principle of truth) yields systematic errors
on these problems. System 2 (counterexample search) should correct them.

Run with a real ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from mmt._logging import get_logger
from mmt.agent import MentalModelAgent
from mmt.models import Judgment

_logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Problem definitions
# ─────────────────────────────────────────────


@dataclass
class IllusionProblem:
    name: str
    premises: list[str]
    query: str
    # What System 1 (Principle of Truth) incorrectly predicts
    illusory_answer: Judgment
    # What the logically correct answer is
    correct_answer: Judgment
    notes: str = ""


ILLUSORY_INFERENCE_PROBLEMS: list[IllusionProblem] = [
    # ── Classic King-Ace illusion (Johnson-Laird & Savary 1999) ──────────────
    IllusionProblem(
        name="king_ace_illusion_classic",
        premises=[
            "If there is a king then there is an ace, or else if there is not a king then there is an ace.",
        ],
        query="Is there an ace?",
        illusory_answer=Judgment.VALID,
        correct_answer=Judgment.UNDERDETERMINED,
        notes=(
            "Classic illusion: people say 'yes' (ace must exist). "
            "But the exclusive-or reading means both conditionals can't both hold. "
            "One possibility has no ace."
        ),
    ),

    # ── Bidirectional illusion ────────────────────────────────────────────────
    IllusionProblem(
        name="king_ace_bidirectional",
        premises=[
            "If there is a king then there is an ace.",
            "If there is not a king then there is an ace.",
        ],
        query="Is there necessarily an ace?",
        illusory_answer=Judgment.VALID,
        correct_answer=Judgment.VALID,
        notes=(
            "This version IS valid — both conditionals cover all cases. "
            "System 1 gets it right for the wrong reason (principle of truth). "
            "Used as a control problem."
        ),
    ),

    # ── Exclusive disjunction illusion ────────────────────────────────────────
    IllusionProblem(
        name="exclusive_disjunction_illusion",
        premises=[
            "Either there is a circle or there is a triangle, but not both.",
            "Either there is a circle or there is a square, but not both.",
        ],
        query="Is there a triangle and a square?",
        illusory_answer=Judgment.VALID,
        correct_answer=Judgment.UNDERDETERMINED,
        notes=(
            "People conclude triangle and square must coexist. "
            "But the model with a circle satisfies both premises and has neither."
        ),
    ),

    # ── Conditional with negation illusion ────────────────────────────────────
    IllusionProblem(
        name="negated_conditional_illusion",
        premises=[
            "If there is not a queen then there is a seven.",
            "There is a queen.",
        ],
        query="Is there a seven?",
        illusory_answer=Judgment.UNDERDETERMINED,
        correct_answer=Judgment.UNDERDETERMINED,
        notes=(
            "The conditional only fires when queen is absent. "
            "With queen present, we can't determine if seven exists. "
            "Both S1 and S2 should agree here — included as a non-illusion control."
        ),
    ),

    # ── Modus tollens illusion ────────────────────────────────────────────────
    IllusionProblem(
        name="modus_tollens_illusion",
        premises=[
            "If there is a ten then there is a queen.",
            "There is not a queen.",
        ],
        query="Is there a ten?",
        illusory_answer=Judgment.UNDERDETERMINED,
        correct_answer=Judgment.INVALID,
        notes=(
            "Valid modus tollens: no queen → no ten. "
            "Principle of truth may miss this because the conditional's "
            "false-antecedent cases aren't represented in the initial model."
        ),
    ),

    # ── Double illusion (harder) ───────────────────────────────────────────────
    IllusionProblem(
        name="double_negation_illusion",
        premises=[
            "If there is not a king then there is not an ace.",
            "There is a king.",
        ],
        query="Is there an ace?",
        illusory_answer=Judgment.UNDERDETERMINED,
        correct_answer=Judgment.UNDERDETERMINED,
        notes=(
            "Conditional with double negation. With king present the "
            "conditional doesn't fire — ace is indeterminate. "
            "Both systems should return UNDERDETERMINED. Control problem."
        ),
    ),
]


# ─────────────────────────────────────────────
# Result tracking
# ─────────────────────────────────────────────


@dataclass
class IllusionResult:
    problem_name: str
    s1_judgment: Optional[Judgment] = None
    s2_judgment: Optional[Judgment] = None
    correct_answer: Judgment = Judgment.UNDERDETERMINED
    illusory_answer: Judgment = Judgment.UNDERDETERMINED
    s1_correct: bool = False
    s2_correct: bool = False
    s1_shows_illusion: bool = False   # S1 gives the illusory (wrong) answer
    s2_corrects_illusion: bool = False  # S2 fixes what S1 got wrong
    s1_explanation: str = ""
    s2_explanation: str = ""
    error: Optional[str] = None
    latency_s1: float = 0.0
    latency_s2: float = 0.0


# ─────────────────────────────────────────────
# Eval runner
# ─────────────────────────────────────────────


def run_illusory_inference_eval(
    agent: MentalModelAgent,
    problems: list[IllusionProblem] | None = None,
    verbose: bool = True,
) -> list[IllusionResult]:
    """Run the illusory inference evaluation suite."""
    if problems is None:
        problems = ILLUSORY_INFERENCE_PROBLEMS

    results: list[IllusionResult] = []

    for prob in problems:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Problem: {prob.name}")
            print(f"Notes:   {prob.notes}")
            print(f"Premises: {prob.premises}")
            print(f"Query:    {prob.query}")
            print(f"Expected (correct): {prob.correct_answer.value}")
            print(f"Expected (illusion): {prob.illusory_answer.value}")

        result = IllusionResult(
            problem_name=prob.name,
            correct_answer=prob.correct_answer,
            illusory_answer=prob.illusory_answer,
        )

        try:
            # ── System 1 run ──────────────────────────────────────
            t0 = time.perf_counter()
            r1 = agent.reason(
                premises=prob.premises,
                query=prob.query,
                deliberate=False,  # Force System 1
            )
            result.latency_s1 = time.perf_counter() - t0
            result.s1_judgment = r1.judgment
            result.s1_explanation = r1.explanation
            result.s1_correct = (r1.judgment == prob.correct_answer)
            result.s1_shows_illusion = (r1.judgment == prob.illusory_answer
                                        and prob.illusory_answer != prob.correct_answer)

            if verbose:
                print(f"\n  System 1 → {r1.judgment.value} "
                      f"({'✓' if result.s1_correct else '✗'}) "
                      f"[{result.latency_s1:.1f}s]")
                if result.s1_shows_illusion:
                    print(f"  ⚠ System 1 shows the classic illusion!")

            # ── System 2 run ──────────────────────────────────────
            t0 = time.perf_counter()
            r2 = agent.reason(
                premises=prob.premises,
                query=prob.query,
                deliberate=True,   # Force System 2
            )
            result.latency_s2 = time.perf_counter() - t0
            result.s2_judgment = r2.judgment
            result.s2_explanation = r2.explanation
            result.s2_correct = (r2.judgment == prob.correct_answer)
            result.s2_corrects_illusion = (
                result.s1_shows_illusion and result.s2_correct
            )

            if verbose:
                print(f"  System 2 → {r2.judgment.value} "
                      f"({'✓' if result.s2_correct else '✗'}) "
                      f"[{result.latency_s2:.1f}s]")
                if result.s2_corrects_illusion:
                    print(f"  ✓ System 2 corrected the illusion!")
                print(f"\n  S1 explanation: {r1.explanation[:120]}...")
                print(f"  S2 explanation: {r2.explanation[:120]}...")

        except anthropic.AuthenticationError:
            raise
        except Exception as exc:
            result.error = str(exc)
            _logger.exception("eval_illusory_inferences: problem=%s", prob.name)
            if verbose:
                print(f"  ERROR: {exc}")
                print(f"  (full traceback written to mmt_errors.log)")

        results.append(result)

    return results


def print_illusory_summary(results: list[IllusionResult]) -> dict:
    """Print a summary table and return metrics."""
    n = len(results)
    s1_correct = sum(r.s1_correct for r in results if r.error is None)
    s2_correct = sum(r.s2_correct for r in results if r.error is None)
    illusions_shown = sum(r.s1_shows_illusion for r in results if r.error is None)
    illusions_corrected = sum(r.s2_corrects_illusion for r in results if r.error is None)
    errors = sum(r.error is not None for r in results)

    print(f"\n{'='*60}")
    print("ILLUSORY INFERENCE EVAL SUMMARY")
    print(f"{'='*60}")
    print(f"Problems:            {n}")
    print(f"System 1 correct:    {s1_correct}/{n - errors}  "
          f"({100*s1_correct/max(1, n - errors):.0f}%)")
    print(f"System 2 correct:    {s2_correct}/{n - errors}  "
          f"({100*s2_correct/max(1, n - errors):.0f}%)")
    print(f"Illusions shown (S1): {illusions_shown}")
    print(f"Illusions fixed (S2): {illusions_corrected}")
    if errors:
        print(f"Errors:              {errors}")

    print(f"\n{'Problem':<35} {'S1':>6} {'S2':>6} {'Illusion':>8} {'Fixed':>6}")
    print("-" * 65)
    for r in results:
        s1 = r.s1_judgment.value if r.s1_judgment else "ERR"
        s2 = r.s2_judgment.value if r.s2_judgment else "ERR"
        ill = "yes" if r.s1_shows_illusion else "-"
        fix = "yes" if r.s2_corrects_illusion else "-"
        print(f"{r.problem_name:<35} {s1:>6} {s2:>6} {ill:>8} {fix:>6}")

    return {
        "n": n,
        "s1_accuracy": s1_correct / max(1, n - errors),
        "s2_accuracy": s2_correct / max(1, n - errors),
        "illusions_shown": illusions_shown,
        "illusions_corrected": illusions_corrected,
        "errors": errors,
    }


if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ANTHROPIC_API_KEY before running evals.")

    client = anthropic.Anthropic(api_key=api_key)
    agent = MentalModelAgent(client=client)

    results = run_illusory_inference_eval(agent, verbose=True)
    metrics = print_illusory_summary(results)
