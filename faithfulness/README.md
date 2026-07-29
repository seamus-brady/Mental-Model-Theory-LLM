# Extraction-Faithfulness Experiment

Tests one narrow, falsifiable claim:

> For spatial / temporal / causal natural-language premises, extracting to
> **iconic MMT structures** recovers the stated relations more faithfully than
> extracting to **first-order logic** — the translation step the neurosymbolic
> literature repeatedly names as its bottleneck.

This is measured **upstream of the solver**, so the repo's weak downstream
accuracy (syllogisms etc.) does not confound it. We score the *translation*, not
the reasoning.

## What it does

Two arms, same client / model / token budget / temperature:

| Arm | Extractor | Source |
|-----|-----------|--------|
| iconic | `SemanticCompiler.extract` → `ConstraintSet` | repo, unchanged |
| fol | matched-effort FOL parser → ground atoms | `arms.FOL_SYSTEM` (verbatim, auditable) |

Both outputs are normalised to a shared canonical triple vocabulary
(`left_of`, `before`, `causes`, `enables`) so scoring is arm-neutral and converse
phrasings ("B is to the right of A") are held to the same gold standard on both
sides.

## Scoring

1. **Triple-level P/R/F1 vs gold** — fully objective, no judge. Headline number.
2. **Arm-blind reconstruction judge** — sees only NL reconstructions, arm hidden,
   order randomised; cross-check on the objective score. (`--judge`)

## Guards against the obvious referee objections

- *Rigged baseline?* FOL prompt is matched in structure/examples and printed
  verbatim in `arms.py`.
- *Biased judge?* Judge never sees formats, only reconstructions, arm-blind.
- *Small n?* 54 augmented problems across 3 domains and chain lengths 2–4;
  this is a **pilot** reporting effect size, not a p-value.

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m faithfulness.test_smoke          # no API — proves plumbing (already passing)
python -m faithfulness.run                 # objective triple scoring (~108 calls)
python -m faithfulness.run --judge --output faith.json   # + blind judge (~216 calls)
```

## Reading the result

- iconic − fol delta **clearly positive** on triple F1 → the claimed effect is real.
- delta ≈ 0 → IR choice doesn't affect fidelity here (still a clean result).
- delta negative → FOL wins; the hypothesis is killed cleanly.

Every per-problem disagreement is in the `log` of the output JSON — eyeball it;
the qualitative failures usually persuade more than the aggregate.
