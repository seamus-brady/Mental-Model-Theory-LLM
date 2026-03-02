# Mental Model Theory LLM Agent

An LLM agent that reasons using **Johnson-Laird's Mental Model Theory (MMT)** — a cognitive psychology framework that models human reasoning as the construction and inspection of iconic simulations of possibilities.

## Table of Contents

- [Overview](#overview)
- [Theoretical Background](#theoretical-background)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Domains](#supported-domains)
- [Evaluation Results](#evaluation-results)
- [Running Tests](#running-tests)
- [Running Evaluations](#running-evaluations)
- [Design Decisions](#design-decisions)
- [References](#references)
- [License](#license)

## Overview

Mental Model Theory claims that humans reason by:

1. **Constructing** one or more mental models — iconic simulations of the described situation
2. **Searching** for a conclusion that holds in all models
3. **Checking** whether a counterexample exists to falsify the conclusion

This implementation embeds that process in a dual-process LLM agent:

- **System 1** builds a single model and evaluates the query immediately (fast, heuristic)
- **System 2** constructs multiple alternative models and searches for counterexamples (deliberative, thorough)

A metacognitive layer (`should_deliberate`) triggers System 2 when the problem shows patterns that fool System 1: disjunctions, counterfactuals, negated conditionals, or known illusory inference patterns.

## Theoretical Background

### The Principle of Truth

MMT's central claim: people only mentally represent what is *explicitly true*, not what is false or absent. This explains *illusory inferences* — systematic errors on problems that require representing false possibilities.

For example:
```
If there is a king then there is an ace.
If there is not a king then there is an ace.
∴ There is necessarily an ace.
```

System 1 (principle of truth) sees "king→ace" and "¬king→ace" and concludes ace is always present. But on an exclusive-or reading, both conditionals cannot simultaneously hold — one branch has no ace. System 2 (counterexample search) finds this falsifying model.

### Iconicity

For spatial, temporal, and causal domains, models are *iconic* — they structurally resemble what they represent. A spatial ordering `[A B C]` directly encodes `left_of(A,B)`, `left_of(A,C)`, `left_of(B,C)` without any deduction. Predicates are *read off* the structure, not inferred.

### Open-World Assumption

Following MMT's principle that models represent only what is explicitly true: an absent fact is **unknown** (not false). The checker returns `True`, `False`, or `None` — three-valued logic throughout.

## Architecture

```
MentalModelAgent.reason(premises, query, deliberate)
│
├─ Phase 1: SemanticCompiler.extract(premises)
│   └─ LLM → structured JSON → ConstraintSet
│
├─ Phase 2: ModelBuilder.construct(constraints, mode)
│   ├─ Spatial / Temporal / Causal:
│   │   LLM → iconic layout → derive predicates (pure Python)
│   └─ Propositional / Quantificational:
│       LLM → explicit facts + schema rules
│
├─ Phase 3: ConstraintChecker.check_consistency(model)
│   └─ Transitivity, antisymmetry, iconic-symbolic match,
│      conditional triggers, provenance enforcement
│
├─ Phase 4: Evaluate query against each model
│   └─ Iconic layer first, then symbolic facts, then schema rules
│
├─ Phase 5: should_deliberate() → System 2 if needed
│   ├─ UNDERDETERMINED result
│   ├─ None verdicts in any model
│   ├─ Unexamined disjunction branches
│   ├─ Known illusory inference pattern
│   └─ Counterfactuals / negated conditionals
│
└─ Phase 6: _aggregate(verdicts) → Judgment + narrate()
    ├─ All True  → VALID
    ├─ Any False → INVALID
    └─ else      → UNDERDETERMINED
```

### Modules

| Module | Purpose |
|--------|---------|
| `mmt/models.py` | Pydantic data structures: `Model`, `ConstraintSet`, layouts, `Fact`, `Judgment`, `ReasoningResult` |
| `mmt/compiler.py` | `SemanticCompiler`: parses natural language premises into `ConstraintSet` via LLM structured output |
| `mmt/checker.py` | `ConstraintChecker`: pure-Python consistency checking; `ProvenanceEnforcer`: strips unsupported facts |
| `mmt/builder.py` | `ModelBuilder`: iconic-first construction for spatial/temporal/causal; LLM-guided for propositional |
| `mmt/counterexample.py` | `CounterexampleFinder`: three-phase search — existing models, LLM expansion, unexplored branches |
| `mmt/agent.py` | `MentalModelAgent`: orchestrates the full reasoning pipeline |

## Installation

```bash
pip install -e ".[dev]"
```

Set your API key:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quick Start

```python
import anthropic
from mmt.agent import MentalModelAgent

client = anthropic.Anthropic()
agent = MentalModelAgent(client=client)

result = agent.reason(
    premises=[
        "A is to the left of B.",
        "B is to the left of C.",
    ],
    query="Is A to the left of C?",
)

print(result.judgment)      # Judgment.VALID
print(result.system_used)   # "system1"
print(result.explanation)   # natural language explanation
```

### Forcing System 2

```python
result = agent.reason(
    premises=[
        "If there is a king then there is an ace.",
        "If there is not a king then there is an ace.",
    ],
    query="Is there necessarily an ace?",
    deliberate=True,
)
print(result.judgment)        # may differ from System 1
print(result.counterexample)  # the falsifying model, if found
```

### Result Fields

```python
result.judgment          # Judgment.VALID | INVALID | UNDERDETERMINED | INCONSISTENT_PREMISES
result.system_used       # "system1" | "system2"
result.models_checked    # number of models evaluated
result.confidence        # float 0.0–1.0
result.explanation       # natural language narration
result.counterexample    # Model or None
result.verdicts          # list[bool | None] per model
```

## Supported Domains

### Spatial

Premise: `"A is to the left of B."` → builds iconic ordering `[A B C]` → derives `left_of`, `right_of` predicates. Supports transitive inference (emergent from iconic structure), antisymmetry checking, and contradiction detection.

### Temporal

Premise: `"Event E1 happens before event E2."` → builds `TemporalLayout` with intervals → derives Allen's interval relations: `before`, `after`, `overlaps`, `during`, `meets`.

### Causal

Premise: `"A causes B, B enables C."` → builds `CausalLayout` DAG → derives `causes`, `enables`, `causally_influences` (transitive) predicates with topological ordering.

### Propositional

Conditionals, disjunctions, biconditionals. Facts tagged with provenance (`EXPLICIT`/`DERIVED`/`ICONIC`/`ASSUMED`). Conditional triggers evaluated forward (modus ponens). `ProvenanceEnforcer` blocks confabulation.

### Quantificational

Universal/existential quantifiers. Schema rules (`All A are B`) applied to entities. Appropriate for syllogistic reasoning.

## Evaluation Results

Evaluations run against Claude Opus using real API calls. Results from the evaluation suite (33 problems total):

### Spatial Reasoning — 84.6% accuracy (11/13)

| Category | Accuracy | Notes |
|----------|----------|-------|
| One-model | 100% (4/4) | Simple determinate orderings — near-perfect as expected |
| Transitive | 100% (6/6) | Emergent from iconic structure — no deduction needed |
| Indeterminate | 0% (0/2) | System 1 builds one model and misses alternatives |
| Impossible | 100% (2/2) | Cycle detection catches contradictions immediately |

Spatial reasoning is a strong suit. The iconic representation makes transitive inferences trivial — the answer is read directly off the layout. The main weakness is indeterminate problems where System 1's single-model heuristic fails to trigger System 2.

### Illusory Inferences — 50% accuracy (S1 and S2)

| Problem | S1 | S2 | Pattern |
|---------|----|----|---------|
| King-ace classic | Illusory | Illusory | Both systems fall for the illusion |
| King-ace bidirectional | Wrong | Wrong | Fails to recognize exhaustive cases |
| Exclusive disjunction | Correct | Correct | Correctly identifies underdetermination |
| Negated conditional | Correct | Over-corrects | S2 finds spurious counterexample |
| Modus tollens | Illusory | Correct | S2 successfully corrects S1's illusion |
| Double negation | Correct | Correct | Both systems handle correctly |

The classic MMT prediction — that System 1 falls for illusions while System 2 corrects them — is partially supported. System 2 corrects the modus tollens illusion, but both systems fall for the king-ace illusion, suggesting the counterexample search needs refinement for problems requiring exhaustive case analysis.

### Syllogistic Reasoning — 35.7% accuracy (5/14)

| Mood | Accuracy | Notes |
|------|----------|-------|
| One-model (Barbara, Celarent, Darii, Ferio) | 12.5% (1/8) | Unexpectedly poor — fails on basic syllogisms |
| Two-model (Camestres, Baroco, invalid forms) | 66.7% (4/6) | Better on harder problems with counterexamples |

Syllogistic reasoning is the weakest domain. The system correctly identifies invalid syllogisms and underdetermined cases but struggles with valid syllogisms across all moods. The quantificational model builder fails to construct supporting models even when the conclusion follows necessarily. This inverts the expected MMT difficulty gradient (one-model should be easiest).

### Summary

| Domain | Accuracy | Strongest | Weakest |
|--------|----------|-----------|---------|
| Spatial | **84.6%** | Transitive (100%) | Indeterminate (0%) |
| Illusory Inferences | **50.0%** | Double negation, exclusive-or | King-ace illusions |
| Syllogisms | **35.7%** | Invalid/underdetermined detection | Valid syllogism confirmation |

## Running Tests

```bash
pytest tests/ -v
```

Tests are fully mocked — no API calls required:

```
tests/test_models.py    — SpatialLayout, TemporalLayout, CausalLayout, Model
tests/test_checker.py   — ConstraintChecker, ProvenanceEnforcer
tests/test_builder.py   — Predicate derivation (pure Python)
tests/test_compiler.py  — Compiler conversion logic + mocked API
tests/test_agent.py     — Agent logic + full integration with mocked deps
```

## Running Evaluations

Evaluations make real API calls. Set `ANTHROPIC_API_KEY` first.

```bash
# All eval suites
python -m evals.run_evals

# Individual suites
python -m evals.run_evals --suite illusions
python -m evals.run_evals --suite spatial
python -m evals.run_evals --suite syllogisms

# Save results to JSON
python -m evals.run_evals --output results.json

# Quiet mode (summary only)
python -m evals.run_evals --quiet
```

## Design Decisions

### LLM Proposes, Checker Verifies

The LLM never outputs free-form reasoning. Every LLM call returns structured JSON validated against a Pydantic schema. The `ConstraintChecker` independently verifies consistency — the LLM cannot override logical constraints.

### Provenance Tracking

Every `Fact` carries a `Provenance` tag: `EXPLICIT` (stated in premise), `DERIVED` (inferred by LLM), `ICONIC` (read off structure), or `ASSUMED`. The `ProvenanceEnforcer` rejects `ASSUMED` facts that aren't derivable from constraints.

### Iconic-First Construction

Spatial/temporal/causal models are built as layout structures first, then predicates are derived in pure Python. The LLM parses orderings/intervals/edges from premises — not generate logical propositions. This separates *reading off structure* (iconic) from *logical deduction* (propositional).

### Adaptive Thinking

Uses Claude Opus with `thinking: {type: "adaptive"}` — the model dynamically decides how much to think based on problem complexity.

### Structured Output

All LLM outputs use JSON schema validation for guaranteed parseable responses.

## References

- Johnson-Laird, P.N. (1983). *Mental Models*. Harvard University Press.
- Johnson-Laird, P.N. & Byrne, R.M.J. (1991). *Deduction*. Lawrence Erlbaum.
- Johnson-Laird, P.N. & Savary, F. (1999). Illusory inferences: a novel class of erroneous deductions. *Cognition, 71*(3), 191–229.
- Johnson-Laird, P.N. & Bara, B.G. (1984). Syllogistic inference. *Cognition, 16*(1), 1–61.
- Byrne, R.M.J. & Johnson-Laird, P.N. (1989). Spatial reasoning. *Journal of Memory and Language, 28*(5), 564–575.
- Johnson-Laird, P.N. (2006). *How We Reason*. Oxford University Press.
- Allen, J.F. (1983). Maintaining knowledge about temporal intervals. *Communications of the ACM, 26*(11), 832–843.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
