# MMT Experiment — Evaluation Report

**Date:** 2026-02-19
**Eval timestamp:** 2026-02-19T21:40:15
**Total problems:** 33 (6 illusions, 13 spatial, 14 syllogisms)
**Total runtime:** ~16 minutes

---

## 1. Executive Summary

The Mental Model Theory (MMT) agent was evaluated across three cognitive psychology benchmarks. Overall accuracy across suites was approximately **57%**, with two of three suites failing their targets. The central architectural innovation — a dual-process System 1 / System 2 escalation — delivers no measurable benefit: S2 accuracy equals S1 accuracy on the illusions suite. The counterexample finder, which is the mechanism that makes S2 meaningful, fails silently on virtually every call. The system is not viable for commercialisation in its current form.

| Suite | N | Accuracy | Target | Result |
|-------|---|----------|--------|--------|
| Illusory Inferences (S2) | 6 | 50% | 70–90% | **FAIL** |
| Spatial Reasoning | 13 | 84.6% | >85% | Borderline |
| Syllogisms | 14 | 35.7% | — | **FAIL** |
| **Combined** | **33** | **~57%** | | **FAIL** |

---

## 2. Error Log Analysis

Before examining per-suite results, the error log (`mmt_errors.log`) reveals two distinct classes of failure that shaped the outcomes.

### 2.1 Pydantic Schema Crashes (Earlier Run, ~18:54–18:59)

Five illusions problems and one spatial problem failed completely in an earlier run due to hard validation errors. The LLM was returning fields not present in the strict Pydantic schemas:

- `reasoning_notes` — a natural scratchpad field the LLM added unprompted
- `antecedent_polarity` / `consequent_polarity` — boolean fields added to conditional constraints
- `schema_rules` — rules returned inside the propositional model structure

These caused `ValidationError: Extra inputs are not permitted` and crashed before any reasoning occurred. The problems were retried in a later run. The root cause is that the schemas use `model_config = ConfigDict(extra="forbid")` with no mechanism to strip or tolerate extra LLM output. This is a fragile design.

### 2.2 CounterexampleFinder JSON Failures (Successful Run, ~21:22–21:40)

The `_targeted_llm_expansion` method in `counterexample.py` failed with `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` on **every single call** across the syllogisms suite (~20+ occurrences). The LLM was returning an empty string instead of valid JSON.

These errors are caught with `try/except` and logged, but the failure is silent to the eval harness — the top-level report records **0 errors** for all suites. This means the counterexample search was completely non-functional throughout the syllogisms eval, and the results reflect a system running without its primary System 2 deliberation mechanism.

---

## 3. Suite Results

### 3.1 Illusory Inferences

**Result: FAIL**

| Metric | Result | Design Target |
|--------|--------|---------------|
| S1 accuracy | 50% (3/6) | 40–60% |
| S2 accuracy | 50% (3/6) | 70–90% |
| Illusions shown by S1 | 2/6 | — |
| Illusions corrected by S2 | 1/2 (50%) | >70% |

S1 hits its design target — it is expected to be occasionally fooled. The failure is that **S2 provides zero net improvement**: it matches S1's accuracy exactly.

#### Problem-by-problem breakdown

| Problem | Correct Answer | S1 | S2 | S1 ✓ | S2 ✓ |
|---------|---------------|----|----|-------|-------|
| king_ace_illusion_classic | underdetermined | valid | valid | ✗ | ✗ |
| king_ace_bidirectional | valid | underdetermined | invalid | ✗ | ✗ |
| exclusive_disjunction_illusion | underdetermined | underdetermined | underdetermined | ✓ | ✓ |
| negated_conditional_illusion | underdetermined | underdetermined | invalid | ✓ | ✗ |
| modus_tollens_illusion | invalid | underdetermined | invalid | ✗ | ✓ |
| double_negation_illusion | underdetermined | underdetermined | underdetermined | ✓ | ✓ |

**Key failures:**

`king_ace_illusion_classic` is the canonical MMT test case — the problem Johnson-Laird built the theory around. The system fails it at both levels. More seriously, in `king_ace_bidirectional`, S2's own explanation correctly identifies that both conditional branches cover all possibilities, then returns `invalid` anyway. The judgment contradicts the reasoning — the explanation and verdict are incoherent with each other.

`negated_conditional_illusion` is a regression: S1 gets it right (`underdetermined`), S2 gets it wrong (`invalid`) by accepting a counterexample that violates one of the premises. S2 is actively harmful on this problem.

### 3.2 Spatial Reasoning

**Result: Borderline (84.6%, just below the 85% target)**

| Category | N | Accuracy | Notes |
|----------|---|----------|-------|
| one_model | 4 | 100% | Simple direct inference |
| transitive | 5 | 100% | Chained left-of relations |
| indeterminate | 2 | **0%** | Both returned `valid` instead of `underdetermined` |
| impossible | 2 | 100% | Cycle detection works correctly |

Determinate spatial reasoning is the system's strongest area. The iconic layout construction (placing entities on a left-to-right line and deriving relations from positions) works well. Cycle detection for contradictory premises is fast and reliable.

The complete failure on **indeterminate cases** is architecturally expected but damaging. When two premises are consistent with multiple orderings (e.g., `A left-of B, C left-of B` does not determine whether A is left or right of C), System 1 builds the first consistent model and returns `valid` without exploring alternatives. S2 escalation does not trigger for spatial problems unless other flags are set, so the indeterminate case is never examined further.

Note: `indeterminate_two_valid_orders` was miscategorised as `one_model` in the output (the expected answer was `valid`, which it got), which artificially inflates the `one_model` count and masks what should have been a harder test.

### 3.3 Syllogisms

**Result: FAIL — fundamental breakdown**

| Metric | Result |
|--------|--------|
| Overall accuracy | 35.7% (5/14) |
| one_model mood | 12.5% (1/8) |
| two_model mood | 66.7% (4/6) |

The system returned `underdetermined` for nearly every problem regardless of the correct answer. The 66.7% two_model score is misleading: those problems happen to have `underdetermined` or `invalid` as correct answers, so the system lucked into correct responses for the wrong reason.

#### Problem-by-problem breakdown

| Problem | Mood | Expected | Got | ✓ |
|---------|------|----------|-----|---|
| barbara_AAA1 | one_model | valid | underdetermined | ✗ |
| celarent_EAE1 | one_model | valid | underdetermined | ✗ |
| darii_AII1 | one_model | valid | underdetermined | ✗ |
| ferio_EIO1 | one_model | valid | underdetermined | ✗ |
| camestres_AEE2 | two_model | valid | underdetermined | ✗ |
| baroco_AOO2 | two_model | valid | underdetermined | ✗ |
| invalid_undistributed_middle | two_model | invalid | invalid | ✓ |
| invalid_some_some | two_model | underdetermined | underdetermined | ✓ |
| invalid_illicit_major | two_model | underdetermined | underdetermined | ✓ |
| existential_import_empty | one_model | underdetermined | underdetermined | ✓ |
| contradiction_universal_particular | one_model | inconsistent_premises | underdetermined | ✗ |
| three_term_chain_valid | one_model | valid | underdetermined | ✗ |
| some_all_chain | one_model | valid | underdetermined | ✗ |
| none_some_invalid | two_model | underdetermined | underdetermined | ✓ |

The system cannot handle `All A are B` universal reasoning. It fails Barbara (`All A→B, All B→C ∴ All A→C`), which is the simplest and oldest valid syllogism in classical logic. With the counterexample finder broken, System 2 has no mechanism to confirm validity, so it defaults to `underdetermined` as a safe non-answer. The system is stuck in permanent epistemic uncertainty for all quantificational problems.

---

## 4. Cross-Cutting Issues

### 4.1 System 2 Provides No Benefit

The design premise is that S2 deliberation — building multiple models, searching for counterexamples — corrects S1 errors. The results do not support this:

- Illusions: S1 = 50%, S2 = 50%. No improvement.
- Spatial: S2 is rarely triggered; when it is, the counterexample finder crashes.
- Syllogisms: S2 is triggered for all problems but the counterexample finder returns empty, leaving S2 with nothing to work with.

In `negated_conditional_illusion`, S2 actively worsens the result (correct → incorrect). The dual-process architecture is currently providing no empirical value.

### 4.2 The Counterexample Finder Is Broken

`CounterexampleFinder._targeted_llm_expansion` returns empty JSON on every invocation during the syllogisms run. This is the most critical failure in the system — it is the entire basis of System 2 deliberation for propositional and quantificational problems. The failure mode (empty LLM response) suggests the prompt is triggering a refusal or the response format expectation does not match what the model returns. There is no retry logic, no structured output enforcement (`response_format`), and no fallback strategy.

### 4.3 Reliability and Latency

- Average latency per problem: **15–40 seconds**
- Average latency per S1+S2 pair: **~30–60 seconds combined**
- Total eval runtime: **~16 minutes for 33 problems**

This is far too slow for any interactive application. Each reasoning call involves 2–4 LLM round-trips without caching or batching.

### 4.4 Schema Fragility

The Pydantic models use strict `extra="forbid"` validation against LLM output. The LLM naturally adds fields that make sense in context (`reasoning_notes`, polarity fields, schema rules) and these cause hard crashes. The appropriate fix is to use `extra="ignore"` or to enforce structured output via the API's `response_format` / tool-use mechanism, which guarantees schema compliance without fragility.

---

## 5. What Works

- **Determinate spatial reasoning**: The iconic layout approach (left-to-right ordering, derived predicates) works well and produces accurate, well-explained results.
- **Contradiction detection**: Cycle detection in spatial constraints is fast and correct.
- **Unit test suite**: 112 unit tests pass with full mock coverage. The internal module contracts are sound.
- **Theoretical design**: The six-phase pipeline, provenance tracking, and domain-specific iconic layers are intellectually coherent. The grounding in Johnson-Laird's literature is serious.
- **Explanation quality**: Even on wrong answers, the natural language explanations are clear and readable.

---

## 6. Commercial Viability Assessment

**Verdict: Not viable for commercialisation in current form.**

### Against

1. **The core innovation fails.** The dual-process S1→S2 escalation — the entire architectural rationale — delivers no measurable improvement over S1 alone. A well-crafted chain-of-thought prompt to a frontier model would likely outperform this system at a fraction of the complexity and cost.

2. **Quantificational reasoning is non-functional.** 35.7% on syllogisms, with 0% on the easiest one-model problems, means the system cannot handle "all", "some", "none" reasoning. This disqualifies it from most real-world logic tasks: legal inference, policy analysis, medical decision support, contract review.

3. **The counterexample finder is broken and the failure is hidden.** Silent failures that corrupt top-level metrics are a serious reliability problem. Users and downstream systems would have no way to know the System 2 path was non-functional.

4. **Latency is prohibitive.** 15–40 seconds per call rules out interactive use cases. Batch use cases would incur significant API costs for a system performing at ~57% accuracy.

5. **Thin moat over baseline.** The scaffolding adds substantial complexity but the underlying capability is just calling Claude. The value-add must be large to justify this; currently it is negative in two of three domains.

6. **The king-ace illusion fails.** This is the most-cited result in the MMT literature and the primary demonstration case for the theory. A commercial product claiming to implement MMT must get this right.

### For (as research)

1. **Sound theoretical foundation.** The research plan is thorough and the MMT grounding is genuine.
2. **Interesting negative result.** The finding that explicit MMT scaffolding does not improve over baseline LLM reasoning is a publishable contribution.
3. **Promising spatial subsystem.** The iconic spatial reasoning module could be extracted and developed as a standalone spatial inference tool.

---

## 7. Recommended Next Steps (if continued as research)

1. **Fix the counterexample finder first.** Use `response_format` / tool-use structured outputs to guarantee valid JSON. Add retry logic with exponential backoff. Without this, System 2 is inoperative.

2. **Fix schema fragility.** Switch to `extra="ignore"` or use API-level structured output enforcement. Do not rely on brittle field-exact matching against free LLM output.

3. **Rebuild universal/existential model construction.** The syllogism failures stem from the model builder not correctly representing `All A are B` in a way the checker can reason over. This requires a dedicated quantificational model construction path.

4. **Run a proper baseline comparison.** Prompt Claude directly (no MMT scaffolding) on the same 33 problems with chain-of-thought. If baseline Claude outperforms the MMT agent — which is likely — this either falsifies the premise or identifies exactly where the implementation needs to improve.

5. **Address the spatial indeterminate failure.** Add a flag in `ConstraintChecker` or `ModelBuilder` to detect when multiple consistent orderings exist and trigger S2 automatically for spatial problems.

6. **Consider structured output throughout.** The entire LLM interface layer should use the Anthropic API's tool-use or `response_format` to enforce schemas rather than parsing free text and hoping for compliance.
