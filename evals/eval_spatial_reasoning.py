"""
Eval: Spatial Reasoning from Mental Model Theory.

Tests the agent on spatial linear-ordering problems.
MMT predicts:
  - Valid inferences from a single spatial model are easy and accurate.
  - Problems requiring multiple spatial models (or indeterminate orderings)
    trigger System 2 and show more errors.

Problem categories:
  1. One-model problems (determinate ordering): should be near-perfect.
  2. Transitive-inference problems (emergent from iconic structure): should succeed.
  3. Indeterminate problems (multiple orderings consistent): should return UNDERDETERMINED.
  4. Impossible premises (contradictory): should return INCONSISTENT_PREMISES.

Reference:
  Johnson-Laird (1983), Ch.5 "Spatial Reasoning"
  Byrne & Johnson-Laird (1989) "Spatial Reasoning"
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
class SpatialProblem:
    name: str
    premises: list[str]
    query: str
    expected: Judgment
    category: str  # "one_model" | "transitive" | "indeterminate" | "impossible"
    notes: str = ""


SPATIAL_PROBLEMS: list[SpatialProblem] = [
    # ── One-model problems ───────────────────────────────────────────────────
    SpatialProblem(
        name="simple_left_right_AB",
        premises=["A is to the left of B."],
        query="Is A to the left of B?",
        expected=Judgment.VALID,
        category="one_model",
        notes="Trivial: conclusion restates premise.",
    ),
    SpatialProblem(
        name="simple_left_right_reverse",
        premises=["A is to the left of B."],
        query="Is B to the right of A?",
        expected=Judgment.VALID,
        category="one_model",
        notes="Converse spatial relation — iconic structure makes this emergent.",
    ),
    SpatialProblem(
        name="simple_false_conclusion",
        premises=["A is to the left of B."],
        query="Is B to the left of A?",
        expected=Judgment.INVALID,
        category="one_model",
        notes="Conclusion contradicts the iconic spatial model.",
    ),

    # ── Transitive inference (emergent) ──────────────────────────────────────
    SpatialProblem(
        name="transitive_three_terms",
        premises=[
            "A is to the left of B.",
            "B is to the left of C.",
        ],
        query="Is A to the left of C?",
        expected=Judgment.VALID,
        category="transitive",
        notes="Classic transitive inference. Emerges from iconic ordering [A B C].",
    ),
    SpatialProblem(
        name="transitive_four_terms",
        premises=[
            "A is to the left of B.",
            "B is to the left of C.",
            "C is to the left of D.",
        ],
        query="Is A to the left of D?",
        expected=Judgment.VALID,
        category="transitive",
        notes="Extended chain. Single iconic model [A B C D].",
    ),
    SpatialProblem(
        name="transitive_skip_one",
        premises=[
            "A is to the left of B.",
            "B is to the left of C.",
            "C is to the left of D.",
        ],
        query="Is B to the left of D?",
        expected=Judgment.VALID,
        category="transitive",
        notes="Non-adjacent transitive inference in a 4-term chain.",
    ),
    SpatialProblem(
        name="transitive_right_of_mixed",
        premises=[
            "A is to the left of B.",
            "C is to the right of B.",
        ],
        query="Is A to the left of C?",
        expected=Judgment.VALID,
        category="transitive",
        notes="Mixed left/right premises; ordering [A B C].",
    ),
    SpatialProblem(
        name="transitive_invalid",
        premises=[
            "A is to the left of B.",
            "B is to the left of C.",
        ],
        query="Is C to the left of A?",
        expected=Judgment.INVALID,
        category="transitive",
        notes="Conclusion reverses the direction of the chain.",
    ),

    # ── Indeterminate problems (multiple models) ──────────────────────────────
    SpatialProblem(
        name="indeterminate_two_valid_orders",
        premises=[
            "A is to the left of B.",
            "C is to the right of B.",
        ],
        query="Is A to the left of C?",
        expected=Judgment.VALID,
        category="one_model",  # Actually determinate: A-B-C
        notes="Both premises imply a unique ordering [A B C].",
    ),
    SpatialProblem(
        name="indeterminate_AC_order",
        premises=[
            "A is to the left of B.",
            "C is to the left of B.",
        ],
        query="Is A to the left of C?",
        expected=Judgment.UNDERDETERMINED,
        category="indeterminate",
        notes=(
            "Both A and C are left of B, but their relative order is unknown. "
            "Two models: [A C B] or [C A B]."
        ),
    ),
    SpatialProblem(
        name="indeterminate_three_entities_partial",
        premises=[
            "A is to the left of B.",
            "C is somewhere.",
        ],
        query="Is A to the left of C?",
        expected=Judgment.UNDERDETERMINED,
        category="indeterminate",
        notes="C's position is unconstrained relative to A.",
    ),

    # ── Impossible / contradictory premises ──────────────────────────────────
    SpatialProblem(
        name="contradiction_simple",
        premises=[
            "A is to the left of B.",
            "B is to the left of A.",
        ],
        query="Is A to the left of B?",
        expected=Judgment.INCONSISTENT_PREMISES,
        category="impossible",
        notes="Antisymmetry violation: A < B and B < A simultaneously.",
    ),
    SpatialProblem(
        name="contradiction_circular",
        premises=[
            "A is to the left of B.",
            "B is to the left of C.",
            "C is to the left of A.",
        ],
        query="Is A to the left of C?",
        expected=Judgment.INCONSISTENT_PREMISES,
        category="impossible",
        notes="Circular chain violates strict ordering (transitivity + antisymmetry).",
    ),
]


# ─────────────────────────────────────────────
# Result tracking
# ─────────────────────────────────────────────


@dataclass
class SpatialResult:
    problem_name: str
    category: str
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


def run_spatial_eval(
    agent: MentalModelAgent,
    problems: list[SpatialProblem] | None = None,
    deliberate: bool | None = None,  # None = auto
    verbose: bool = True,
) -> list[SpatialResult]:
    """Run the spatial reasoning evaluation."""
    if problems is None:
        problems = SPATIAL_PROBLEMS

    results: list[SpatialResult] = []

    for prob in problems:
        if verbose:
            print(f"\n{'─'*55}")
            print(f"[{prob.category}] {prob.name}")
            print(f"  Premises: {prob.premises}")
            print(f"  Query:    {prob.query}")
            print(f"  Expected: {prob.expected.value}")

        result = SpatialResult(
            problem_name=prob.name,
            category=prob.category,
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
                    print(f"  ⚠ Expected {prob.expected.value}, got {r.judgment.value}")

        except anthropic.AuthenticationError:
            raise
        except Exception as exc:
            result.error = str(exc)
            _logger.exception("eval_spatial_reasoning: problem=%s", prob.name)
            if verbose:
                print(f"  ERROR: {exc}")
                print(f"  (full traceback written to mmt_errors.log)")

        results.append(result)

    return results


def print_spatial_summary(results: list[SpatialResult]) -> dict:
    """Print category-level accuracy summary."""
    from collections import defaultdict
    cat_totals: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        if r.error is None:
            cat_totals[r.category].append(r.correct)

    n = len(results)
    errors = sum(r.error is not None for r in results)
    overall_correct = sum(r.correct for r in results if r.error is None)

    print(f"\n{'='*60}")
    print("SPATIAL REASONING EVAL SUMMARY")
    print(f"{'='*60}")
    print(f"Overall: {overall_correct}/{n - errors} "
          f"({100*overall_correct/max(1, n - errors):.0f}%)")
    print()
    for cat, outcomes in sorted(cat_totals.items()):
        acc = sum(outcomes) / max(1, len(outcomes))
        print(f"  {cat:<20} {sum(outcomes)}/{len(outcomes)} ({100*acc:.0f}%)")

    if errors:
        print(f"\nErrors: {errors}")

    print(f"\n{'Problem':<45} {'Expected':>12} {'Got':>12} {'OK':>4}")
    print("-" * 77)
    for r in results:
        got = r.judgment.value if r.judgment else "ERROR"
        ok = "✓" if r.correct else "✗"
        print(f"{r.problem_name:<45} {r.expected.value:>12} {got:>12} {ok:>4}")

    return {
        "n": n,
        "overall_accuracy": overall_correct / max(1, n - errors),
        "by_category": {
            cat: sum(v)/len(v) for cat, v in cat_totals.items()
        },
        "errors": errors,
    }


if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ANTHROPIC_API_KEY before running evals.")

    client = anthropic.Anthropic(api_key=api_key)
    agent = MentalModelAgent(client=client)

    results = run_spatial_eval(agent, verbose=True)
    print_spatial_summary(results)
