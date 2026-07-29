"""
Faithfulness experiment runner.

Scores each extraction arm against gold triples (objective, no judge) and
optionally against an arm-BLIND reconstruction judge. Reports precision/recall/F1
per domain plus a per-problem disagreement log.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m faithfulness.run                 # triple-level scoring only (cheap)
    python -m faithfulness.run --judge         # add arm-blind reconstruction judge
    python -m faithfulness.run --output f.json # save full results

The triple-level score is the headline number and needs NO judge — it is fully
objective. The judge is a secondary cross-check.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from .problems import generate
from .arms import iconic_triples, fol_triples


# ─────────────────────────────────────────────────────────────────────────────
# Objective scoring: precision / recall / F1 against gold triples
# ─────────────────────────────────────────────────────────────────────────────

def prf(pred: set, gold: set) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _reconstruct(triples: set[tuple[str, str, str]]) -> list[str]:
    """Render canonical triples back to neutral NL for the blind judge."""
    verb = {"left_of": "is to the left of", "before": "happens before",
            "causes": "causes", "enables": "enables"}
    return [f"{a} {verb[rel]} {b}." for (rel, a, b) in sorted(triples)]


# ─────────────────────────────────────────────────────────────────────────────
# Arm-blind reconstruction judge
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You compare two sets of statements for logical equivalence. Given ORIGINAL
premises and a RECONSTRUCTION, answer whether the reconstruction encodes exactly
the same relations — no relation added, dropped, or reversed.

Return ONLY JSON: {"equivalent": true or false, "reason": "..."}
"""


def judge_equivalence(client, model, original: list[str], reconstruction: list[str]) -> dict:
    import re
    o = "\n".join(original)
    r = "\n".join(reconstruction) or "(no relations extracted)"
    resp = client.messages.create(
        model=model, max_tokens=512, system=_JUDGE_SYSTEM,
        messages=[{"role": "user",
                   "content": f"ORIGINAL:\n{o}\n\nRECONSTRUCTION:\n{r}\n\nReturn ONLY the JSON."}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"equivalent": None, "reason": "unparseable judge output"}


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run(use_judge: bool = False, seed: int = 0, model: str | None = None):
    import anthropic
    from mmt.compiler import SemanticCompiler

    client = anthropic.Anthropic()
    # Both arms read compiler.model, so overriding it here keeps the comparison
    # fair. mmt/compiler.py defaults to an out-of-date model string; pass a valid
    # one via --model rather than editing the repo.
    compiler = SemanticCompiler(client=client, model=model) if model else SemanticCompiler(client=client)
    problems = generate(seed=seed)
    rng = random.Random(seed)

    agg = defaultdict(lambda: defaultdict(list))   # domain -> arm -> [f1,...]
    judge_agg = defaultdict(lambda: defaultdict(list))
    log = []

    for prob in problems:
        icon = iconic_triples(compiler, prob.premises)
        fol = fol_triples(compiler, prob.premises)

        ip, ir, iff = prf(icon, prob.gold_triples)
        fp, fr, ff = prf(fol, prob.gold_triples)
        agg[prob.domain]["iconic"].append(iff)
        agg[prob.domain]["fol"].append(ff)

        entry = {
            "name": prob.name, "domain": prob.domain, "premises": prob.premises,
            "gold": sorted(prob.gold_triples),
            "iconic": {"triples": sorted(icon), "P": ip, "R": ir, "F1": iff},
            "fol": {"triples": sorted(fol), "P": fp, "R": fr, "F1": ff},
        }

        if use_judge:
            # Blind: judge sees only reconstructions, order randomised, arm hidden.
            arms = [("iconic", icon), ("fol", fol)]
            rng.shuffle(arms)
            for arm_name, triples in arms:
                verdict = judge_equivalence(
                    client, compiler.model, prob.premises, _reconstruct(triples))
                eq = verdict.get("equivalent")
                judge_agg[prob.domain][arm_name].append(1.0 if eq else 0.0)
                entry[arm_name]["judge_equivalent"] = eq

        log.append(entry)
        print(f"{prob.name:28s} iconic F1={iff:.2f}  fol F1={ff:.2f}")

    _summary(agg, judge_agg if use_judge else None)
    return {"log": log, "aggregate": _to_means(agg),
            "judge": _to_means(judge_agg) if use_judge else None}


def _to_means(agg):
    return {d: {arm: (sum(v) / len(v) if v else 0.0) for arm, v in arms.items()}
            for d, arms in agg.items()}


def _summary(agg, judge_agg):
    print("\n" + "=" * 60)
    print("TRIPLE-LEVEL F1 (objective, no judge)")
    print("=" * 60)
    print(f"{'domain':14s} {'iconic':>10s} {'fol':>10s} {'delta':>10s}")
    for d, arms in agg.items():
        i = sum(arms["iconic"]) / len(arms["iconic"])
        f = sum(arms["fol"]) / len(arms["fol"])
        print(f"{d:14s} {i:10.3f} {f:10.3f} {i - f:+10.3f}")
    if judge_agg:
        print("\n" + "=" * 60)
        print("ARM-BLIND RECONSTRUCTION EQUIVALENCE (fraction judged equivalent)")
        print("=" * 60)
        print(f"{'domain':14s} {'iconic':>10s} {'fol':>10s} {'delta':>10s}")
        for d, arms in judge_agg.items():
            i = sum(arms["iconic"]) / len(arms["iconic"])
            f = sum(arms["fol"]) / len(arms["fol"])
            print(f"{d:14s} {i:10.3f} {f:10.3f} {i - f:+10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="add arm-blind reconstruction judge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--model", type=str, default="claude-opus-4-8",
                    help="model for both arms (overrides mmt/compiler.py default)")
    args = ap.parse_args()
    results = run(use_judge=args.judge, seed=args.seed, model=args.model)
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
