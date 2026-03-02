"""
Main eval runner for MentalModelAgent.

Usage:
    python -m evals.run_evals                   # all evals
    python -m evals.run_evals --suite spatial   # spatial only
    python -m evals.run_evals --suite illusions
    python -m evals.run_evals --suite syllogisms
    python -m evals.run_evals --output results.json

Requires ANTHROPIC_API_KEY to be set in the environment.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import anthropic

from mmt.agent import MentalModelAgent

from evals.eval_illusory_inferences import (
    run_illusory_inference_eval,
    print_illusory_summary,
)
from evals.eval_spatial_reasoning import (
    run_spatial_eval,
    print_spatial_summary,
)
from evals.eval_syllogisms import (
    run_syllogism_eval,
    print_syllogism_summary,
)


def build_agent() -> MentalModelAgent:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    # Validate the key before running any evals
    print("Validating API key...", end=" ", flush=True)
    try:
        client.models.list(limit=1)
        print("OK")
    except anthropic.AuthenticationError:
        print(
            "\nError: invalid API key.\n"
            "Make sure ANTHROPIC_API_KEY is set to a valid key from "
            "https://console.anthropic.com/settings/keys",
            file=sys.stderr,
        )
        sys.exit(1)

    return MentalModelAgent(client=client)


def result_to_dict(r) -> dict:
    """Convert a result dataclass to a JSON-serialisable dict."""
    d = {}
    for k, v in vars(r).items():
        if hasattr(v, "value"):
            d[k] = v.value  # Enum → string
        else:
            d[k] = v
    return d


def run_all(
    agent: MentalModelAgent,
    suites: list[str],
    verbose: bool,
) -> dict:
    """Run the requested eval suites and return aggregated metrics."""
    all_metrics: dict[str, dict] = {}
    start = time.perf_counter()

    if "illusions" in suites:
        print("\n" + "═"*60)
        print("SUITE: Illusory Inferences")
        print("═"*60)
        results = run_illusory_inference_eval(agent, verbose=verbose)
        metrics = print_illusory_summary(results)
        all_metrics["illusions"] = {
            "metrics": metrics,
            "results": [result_to_dict(r) for r in results],
        }

    if "spatial" in suites:
        print("\n" + "═"*60)
        print("SUITE: Spatial Reasoning")
        print("═"*60)
        results = run_spatial_eval(agent, verbose=verbose)
        metrics = print_spatial_summary(results)
        all_metrics["spatial"] = {
            "metrics": metrics,
            "results": [result_to_dict(r) for r in results],
        }

    if "syllogisms" in suites:
        print("\n" + "═"*60)
        print("SUITE: Syllogistic Reasoning")
        print("═"*60)
        results = run_syllogism_eval(agent, verbose=verbose)
        metrics = print_syllogism_summary(results)
        all_metrics["syllogisms"] = {
            "metrics": metrics,
            "results": [result_to_dict(r) for r in results],
        }

    total_time = time.perf_counter() - start

    # Print overall summary
    print(f"\n{'═'*60}")
    print("OVERALL EVAL SUMMARY")
    print(f"{'═'*60}")
    for suite, data in all_metrics.items():
        m = data["metrics"]
        acc = m.get("overall_accuracy") or m.get("s2_accuracy", 0)
        print(f"  {suite:<20} accuracy={100*acc:.0f}%  "
              f"errors={m.get('errors', 0)}")
    print(f"\nTotal runtime: {total_time:.1f}s")

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "suites": all_metrics,
        "total_time_s": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Run MentalModelAgent evaluations")
    parser.add_argument(
        "--suite",
        choices=["illusions", "spatial", "syllogisms", "all"],
        default="all",
        help="Which eval suite to run (default: all)",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write JSON results to this file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-problem output",
    )
    args = parser.parse_args()

    suites = (
        ["illusions", "spatial", "syllogisms"]
        if args.suite == "all"
        else [args.suite]
    )

    agent = build_agent()
    report = run_all(agent, suites=suites, verbose=not args.quiet)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nResults written to {args.output}")

    # Exit 1 if any suite had >50% error rate
    for suite, data in report["suites"].items():
        m = data["metrics"]
        acc = m.get("overall_accuracy") or m.get("s2_accuracy", 0)
        if acc < 0.5 and m.get("errors", 0) == 0:
            print(f"\n⚠ {suite} accuracy below 50%: {100*acc:.0f}%", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
