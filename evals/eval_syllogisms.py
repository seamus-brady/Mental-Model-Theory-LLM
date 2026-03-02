"""
Eval: Syllogistic Reasoning from Mental Model Theory.

Tests the agent on Aristotelian syllogisms using quantified premises.
MMT predicts difficulty of a syllogism correlates with:
  1. Number of mental models needed (one-model < two-model < three-model)
  2. Presence of negative or particular premises (harder to represent)
  3. Figural effect: the order of terms affects which conclusions feel natural

Problem types:
  - Valid syllogisms: agent should return VALID
  - Invalid syllogisms: agent should return INVALID or UNDERDETERMINED
  - Figural effect: same logical form in different figures

Reference:
  Johnson-Laird & Bara (1984) "Syllogistic inference"
  Johnson-Laird (1983) Ch.6 "Syllogistic Reasoning"
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
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
class SyllogismProblem:
    name: str
    premises: list[str]
    query: str
    expected: Judgment
    figure: str   # Standard syllogistic figure: "AA1", "AE2", etc.
    mood: str     # "one_model" | "two_model" | "three_model"
    notes: str = ""


SYLLOGISM_PROBLEMS: list[SyllogismProblem] = [
    # ══════════════════════════════════════════
    # VALID SYLLOGISMS (one-model)
    # ══════════════════════════════════════════

    # Barbara (AAA-1): All A are B, All B are C → All A are C
    SyllogismProblem(
        name="barbara_AAA1",
        premises=[
            "All artists are beekeepers.",
            "All beekeepers are chemists.",
        ],
        query="Are all artists chemists?",
        expected=Judgment.VALID,
        figure="AAA-1",
        mood="one_model",
        notes="Barbara: the canonical valid universal syllogism.",
    ),

    # Celarent (EAE-1): No A are B, All B are C → No A are C... wait
    # Actually Celarent: No A are B, All C are A → No C are B
    # Let me use standard forms correctly:
    # Barbara (AAA-1): All M are P, All S are M → All S are P
    SyllogismProblem(
        name="celarent_EAE1",
        premises=[
            "No artists are beekeepers.",
            "All chemists are artists.",
        ],
        query="Are no chemists beekeepers?",
        expected=Judgment.VALID,
        figure="EAE-1",
        mood="one_model",
        notes="Celarent: No M are P, All S are M → No S are P.",
    ),

    # Darii (AII-1): All M are P, Some S are M → Some S are P
    SyllogismProblem(
        name="darii_AII1",
        premises=[
            "All artists are beekeepers.",
            "Some chemists are artists.",
        ],
        query="Are some chemists beekeepers?",
        expected=Judgment.VALID,
        figure="AII-1",
        mood="one_model",
        notes="Darii: All M are P, Some S are M → Some S are P.",
    ),

    # Ferio (EIO-1): No M are P, Some S are M → Some S are not P
    SyllogismProblem(
        name="ferio_EIO1",
        premises=[
            "No artists are beekeepers.",
            "Some chemists are artists.",
        ],
        query="Are some chemists not beekeepers?",
        expected=Judgment.VALID,
        figure="EIO-1",
        mood="one_model",
        notes="Ferio: No M are P, Some S are M → Some S are not P.",
    ),

    # ══════════════════════════════════════════
    # VALID SYLLOGISMS (two-model — harder)
    # ══════════════════════════════════════════

    # Camestres (AEE-2): All P are M, No S are M → No S are P
    SyllogismProblem(
        name="camestres_AEE2",
        premises=[
            "All beekeepers are artists.",
            "No chemists are artists.",
        ],
        query="Are no chemists beekeepers?",
        expected=Judgment.VALID,
        figure="AEE-2",
        mood="two_model",
        notes="Camestres (figure 2): All P are M, No S are M → No S are P.",
    ),

    # Baroco (AOO-2): All P are M, Some S are not M → Some S are not P
    SyllogismProblem(
        name="baroco_AOO2",
        premises=[
            "All beekeepers are artists.",
            "Some chemists are not artists.",
        ],
        query="Are some chemists not beekeepers?",
        expected=Judgment.VALID,
        figure="AOO-2",
        mood="two_model",
        notes="Baroco: All P are M, Some S are not M → Some S are not P.",
    ),

    # ══════════════════════════════════════════
    # INVALID SYLLOGISMS (common errors)
    # ══════════════════════════════════════════

    # Affirming the consequent form — invalid
    SyllogismProblem(
        name="invalid_undistributed_middle",
        premises=[
            "All artists are creative.",
            "All beekeepers are creative.",
        ],
        query="Are all artists beekeepers?",
        expected=Judgment.INVALID,
        figure="AA-?",
        mood="two_model",
        notes=(
            "Undistributed middle (classic fallacy). "
            "Being creative doesn't make artists = beekeepers."
        ),
    ),

    # Some A are B, Some B are C → Some A are C (invalid!)
    SyllogismProblem(
        name="invalid_some_some",
        premises=[
            "Some artists are beekeepers.",
            "Some beekeepers are chemists.",
        ],
        query="Are some artists chemists?",
        expected=Judgment.UNDERDETERMINED,
        figure="II-1",
        mood="two_model",
        notes=(
            "Classic invalid: Some A are B, Some B are C → ? A are C. "
            "Not a valid inference — no artists need be chemists."
        ),
    ),

    # Illicit major: All A are B, No C are A → No C are B (INVALID)
    SyllogismProblem(
        name="invalid_illicit_major",
        premises=[
            "All artists are beekeepers.",
            "No chemists are artists.",
        ],
        query="Are no chemists beekeepers?",
        expected=Judgment.UNDERDETERMINED,
        figure="AE-1",
        mood="two_model",
        notes=(
            "Illicit major. Even though no chemist is an artist, "
            "chemists might still be beekeepers through some other path."
        ),
    ),

    # ══════════════════════════════════════════
    # QUANTIFICATIONAL EDGE CASES
    # ══════════════════════════════════════════

    # Existential import: All unicorns are horses → Some unicorns are horses?
    SyllogismProblem(
        name="existential_import_empty",
        premises=[
            "All politicians are honest.",
        ],
        query="Are some politicians honest?",
        expected=Judgment.UNDERDETERMINED,
        figure="A→I",
        mood="one_model",
        notes=(
            "Existential import question: does 'All A are B' imply 'Some A are B'? "
            "Under open-world assumption (MMT): underdetermined, since there may be "
            "no politicians in the model."
        ),
    ),

    # Syllogism with contradiction
    SyllogismProblem(
        name="contradiction_universal_particular",
        premises=[
            "All artists are beekeepers.",
            "No artists are beekeepers.",
        ],
        query="Are some artists beekeepers?",
        expected=Judgment.INCONSISTENT_PREMISES,
        figure="AE-contradiction",
        mood="one_model",
        notes="Contradictory premises: All A are B and No A are B.",
    ),

    # ══════════════════════════════════════════
    # MULTI-STEP QUANTIFICATIONAL CHAINS
    # ══════════════════════════════════════════

    SyllogismProblem(
        name="three_term_chain_valid",
        premises=[
            "All athletes are bold.",
            "All bold people are curious.",
            "All curious people are diligent.",
        ],
        query="Are all athletes diligent?",
        expected=Judgment.VALID,
        figure="AAA-chain",
        mood="one_model",
        notes="Three-step Barbara chain: A→B→C→D, so A→D.",
    ),

    SyllogismProblem(
        name="some_all_chain",
        premises=[
            "Some athletes are bold.",
            "All bold people are curious.",
        ],
        query="Are some athletes curious?",
        expected=Judgment.VALID,
        figure="IA-1",
        mood="one_model",
        notes="Some A are B, All B are C → Some A are C (Darii-like).",
    ),

    SyllogismProblem(
        name="none_some_invalid",
        premises=[
            "No athletes are bold.",
            "Some bold people are curious.",
        ],
        query="Are no athletes curious?",
        expected=Judgment.UNDERDETERMINED,
        figure="EI-1",
        mood="two_model",
        notes=(
            "No A are B, Some B are C. Athletes might still be curious "
            "through non-bold paths. Result is underdetermined."
        ),
    ),
]


# ─────────────────────────────────────────────
# Result tracking
# ─────────────────────────────────────────────


@dataclass
class SyllogismResult:
    problem_name: str
    figure: str
    mood: str
    expected: Judgment
    judgment: Optional[Judgment] = None
    correct: bool = False
    system_used: str = ""
    models_checked: int = 0
    explanation: str = ""
    error: Optional[str] = None
    latency: float = 0.0


# ─────────────────────────────────────────────
# Eval runner
# ─────────────────────────────────────────────


def run_syllogism_eval(
    agent: MentalModelAgent,
    problems: list[SyllogismProblem] | None = None,
    deliberate: bool | None = None,
    verbose: bool = True,
) -> list[SyllogismResult]:
    """Run the syllogism evaluation."""
    if problems is None:
        problems = SYLLOGISM_PROBLEMS

    results: list[SyllogismResult] = []

    for prob in problems:
        if verbose:
            print(f"\n{'─'*60}")
            print(f"[{prob.figure} / {prob.mood}] {prob.name}")
            print(f"  Premises: {prob.premises}")
            print(f"  Query:    {prob.query}")
            print(f"  Expected: {prob.expected.value}")
            if prob.notes:
                print(f"  Notes:    {prob.notes}")

        result = SyllogismResult(
            problem_name=prob.name,
            figure=prob.figure,
            mood=prob.mood,
            expected=prob.expected,
        )

        try:
            t0 = time.perf_counter()
            r = agent.reason(
                premises=prob.premises,
                query=prob.query,
                deliberate=deliberate,
            )
            result.latency = time.perf_counter() - t0
            result.judgment = r.judgment
            result.correct = (r.judgment == prob.expected)
            result.system_used = r.system_used
            result.models_checked = r.models_checked
            result.explanation = r.explanation

            if verbose:
                mark = "✓" if result.correct else "✗"
                print(f"  Result:   {r.judgment.value} {mark} "
                      f"[{result.system_used}, {r.models_checked} models, "
                      f"{result.latency:.1f}s]")
                if not result.correct:
                    print(f"  ⚠ Got {r.judgment.value}, expected {prob.expected.value}")

        except anthropic.AuthenticationError:
            raise
        except Exception as exc:
            result.error = str(exc)
            _logger.exception("eval_syllogisms: problem=%s", prob.name)
            if verbose:
                print(f"  ERROR: {exc}")
                print(f"  (full traceback written to mmt_errors.log)")

        results.append(result)

    return results


def print_syllogism_summary(results: list[SyllogismResult]) -> dict:
    """Print accuracy summary broken down by mood difficulty."""
    from collections import defaultdict
    mood_totals: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        if r.error is None:
            mood_totals[r.mood].append(r.correct)

    n = len(results)
    errors = sum(r.error is not None for r in results)
    overall_correct = sum(r.correct for r in results if r.error is None)

    print(f"\n{'='*60}")
    print("SYLLOGISM EVAL SUMMARY")
    print(f"{'='*60}")
    print(f"Overall: {overall_correct}/{n - errors} "
          f"({100*overall_correct/max(1, n - errors):.0f}%)")
    print()
    for mood in ["one_model", "two_model", "three_model"]:
        if mood in mood_totals:
            outcomes = mood_totals[mood]
            acc = sum(outcomes) / max(1, len(outcomes))
            print(f"  {mood:<15} {sum(outcomes)}/{len(outcomes)} ({100*acc:.0f}%)")

    # Validity breakdown
    valid_correct = [r for r in results if r.expected == Judgment.VALID and r.error is None]
    invalid_correct = [r for r in results if r.expected != Judgment.VALID and r.error is None]
    if valid_correct:
        vc = sum(r.correct for r in valid_correct)
        print(f"\n  Valid syllogisms:   {vc}/{len(valid_correct)} "
              f"({100*vc/len(valid_correct):.0f}%)")
    if invalid_correct:
        ic = sum(r.correct for r in invalid_correct)
        print(f"  Invalid/Und.:      {ic}/{len(invalid_correct)} "
              f"({100*ic/len(invalid_correct):.0f}%)")

    if errors:
        print(f"\nErrors: {errors}")

    print(f"\n{'Problem':<40} {'Figure':>10} {'Expected':>12} {'Got':>12} {'OK':>4}")
    print("-" * 82)
    for r in results:
        got = r.judgment.value if r.judgment else "ERROR"
        ok = "✓" if r.correct else "✗"
        print(f"{r.problem_name:<40} {r.figure:>10} {r.expected.value:>12} {got:>12} {ok:>4}")

    return {
        "n": n,
        "overall_accuracy": overall_correct / max(1, n - errors),
        "by_mood": {
            mood: sum(v)/len(v) for mood, v in mood_totals.items()
        },
        "errors": errors,
    }


if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ANTHROPIC_API_KEY before running evals.")

    client = anthropic.Anthropic(api_key=api_key)
    agent = MentalModelAgent(client=client)

    results = run_syllogism_eval(agent, verbose=True)
    print_syllogism_summary(results)
