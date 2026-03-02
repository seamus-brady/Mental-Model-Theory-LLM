"""
Mental Model Agent: The main orchestrator.

Implements Johnson-Laird's dual-process architecture:

System 1 (fast, intuitive):
  - Builds ONE initial model (Principle of Truth)
  - No counterexample search
  - May produce illusory inferences (intentionally)

System 2 (deliberative):
  - Builds MULTIPLE models, fleshes out implicit alternatives
  - Searches for counterexamples
  - Corrects System 1 errors

The should_deliberate() method implements metacognitive triggering:
triggers System 2 when premise patterns are known to cause errors.

Reference: Johnson-Laird (2006), Khemlani et al. (2018)
"""

from __future__ import annotations

import anthropic

from ._logging import get_logger
from .builder import ModelBuilder, SYSTEM1_CONFIG, SYSTEM2_CONFIG
from .checker import ConstraintChecker, ProvenanceEnforcer
from .compiler import SemanticCompiler
from .counterexample import CounterexampleFinder
from .models import (
    ConstraintSet,
    Domain,
    InconsistentConstraintsError,
    Judgment,
    Model,
    ReasoningResult,
    SearchResult,
)

# ─────────────────────────────────────────────
# Response narration prompt
# ─────────────────────────────────────────────

_logger = get_logger(__name__)

_NARRATE_SYSTEM = """\
You are a reasoning assistant explaining results from a Mental Model Theory analysis.

Explain the reasoning clearly and concisely:
- What models were constructed
- Why the conclusion follows (or does not)
- If there is a counterexample, describe it in plain language
- Note whether System 1 or System 2 reasoning was used
"""

_NARRATE_USER = """\
Premises: {premises}
Query: {query}
Judgment: {judgment}
Models checked: {models_checked}
System used: {system_used}
Counterexample: {counterexample}
Supporting model: {supporting_model}

Explain this reasoning result in 2-3 sentences for a non-expert.
"""


class MentalModelAgent:
    """
    LLM agent implementing Johnson-Laird's Mental Model Theory.

    Usage:
        client = anthropic.Anthropic()
        agent = MentalModelAgent(client)
        result = agent.reason(
            premises=["A is to the left of B", "B is to the left of C"],
            query="Is A to the left of C?"
        )
        print(result.judgment)   # Judgment.VALID
        print(result.explanation)
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model_id: str = "claude-opus-4-6",
    ):
        if client is None:
            client = anthropic.Anthropic()
        self.client = client
        self.model_id = model_id

        self.compiler = SemanticCompiler(client, model_id)
        self.builder = ModelBuilder(client, model_id)
        self.checker = ConstraintChecker()
        self.enforcer = ProvenanceEnforcer()
        self.ce_finder = CounterexampleFinder(self.checker, client, model_id)

    # ─────────────────────────────────────────────────────────────────────
    # Main reasoning method
    # ─────────────────────────────────────────────────────────────────────

    def reason(
        self,
        premises: list[str],
        query: str,
        deliberate: bool | None = None,
    ) -> ReasoningResult:
        """
        Reason about premises and evaluate a query.

        Args:
            premises: Natural language premise strings
            query: The conclusion/question to evaluate
            deliberate: Force System 1 (False) or System 2 (True).
                       If None, auto-detect via should_deliberate().

        Returns:
            ReasoningResult with judgment, explanation, and supporting models.
        """
        # ── Phase 1: Compile premises ──────────────────────────────────
        constraints = self.compiler.extract(premises)

        # ── Phase 2: Build initial model (System 1) ───────────────────
        try:
            models = self.builder.construct(constraints, mode="system1")
        except InconsistentConstraintsError as exc:
            return ReasoningResult(
                judgment=Judgment.INCONSISTENT_PREMISES,
                explanation=f"Premises are contradictory: {exc}",
                models_checked=0,
            )

        if not models:
            return ReasoningResult(
                judgment=Judgment.UNDERDETERMINED,
                explanation="Could not construct any model from the premises.",
                models_checked=0,
            )

        # ── Phase 3: Check consistency ─────────────────────────────────
        for model in models:
            consistency = self.checker.check_consistency(model)
            if not consistency.consistent:
                details = "; ".join(v.detail for v in consistency.violations)
                return ReasoningResult(
                    judgment=Judgment.INCONSISTENT_PREMISES,
                    explanation=f"Premises are contradictory: {details}",
                    models_checked=len(models),
                )

        # ── Phase 4: Parse and evaluate query ─────────────────────────
        parsed_query = self.compiler.parse_query(query, constraints.domain)
        pred = parsed_query["predicate"]
        args = parsed_query["args"]
        polarity = parsed_query["polarity"]
        query_type = parsed_query.get("query_type", "atomic")

        verdicts = [
            self._evaluate_model(model, pred, args, polarity, query_type)
            for model in models
        ]
        initial_judgment = self._aggregate(verdicts)

        # ── Phase 5: Decide whether to deliberate ─────────────────────
        should_use_s2 = deliberate
        if should_use_s2 is None:
            should_use_s2 = self.should_deliberate(
                constraints, initial_judgment, verdicts
            )

        system_used = "system1"

        if should_use_s2:
            system_used = "system2"

            # Expand model set
            try:
                models = self.builder.construct(
                    constraints, mode="system2", existing_models=models
                )
            except InconsistentConstraintsError as exc:
                return ReasoningResult(
                    judgment=Judgment.INCONSISTENT_PREMISES,
                    explanation=f"Premises are contradictory: {exc}",
                    models_checked=0,
                )

            # Re-evaluate with fuller model set
            verdicts = [
                self._evaluate_model(model, pred, args, polarity, query_type)
                for model in models
            ]

            # Counterexample search (the critical check)
            ce_result = self.ce_finder.search(
                models, pred, args, polarity, constraints
            )
            if ce_result.found and ce_result.counterexample:
                explanation = self._narrate(
                    premises, query, Judgment.INVALID,
                    len(models), system_used,
                    ce_result.counterexample, None
                )
                return ReasoningResult(
                    judgment=Judgment.INVALID,
                    explanation=explanation,
                    models_checked=len(models),
                    counterexample=ce_result.counterexample,
                    system_used=system_used,
                )

        # ── Phase 6: Final judgment ────────────────────────────────────
        final_judgment = self._aggregate(verdicts)
        supporting = models[0] if final_judgment == Judgment.VALID else None

        explanation = self._narrate(
            premises, query, final_judgment,
            len(models), system_used,
            None, supporting
        )

        return ReasoningResult(
            judgment=final_judgment,
            explanation=explanation,
            models_checked=len(models),
            supporting_model=supporting,
            system_used=system_used,
            confidence=self._compute_confidence(verdicts, final_judgment),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Evaluation helpers
    # ─────────────────────────────────────────────────────────────────────

    def _evaluate_model(
        self,
        model: Model,
        predicate: str,
        args: list[str],
        polarity: bool,
        query_type: str,
    ) -> bool | None:
        """Evaluate a parsed query against a single model."""
        if query_type == "existential":
            subject = args[0] if args else predicate
            return self.checker.evaluate_existential(model, subject)
        if query_type == "universal":
            # Need antecedent_pred too; fall back to simple evaluation
            return self.checker.evaluate(model, predicate, args, polarity)
        # Atomic
        return self.checker.evaluate(model, predicate, args, polarity)

    def _aggregate(self, verdicts: list[bool | None]) -> Judgment:
        """
        Aggregate verdicts across models.

        all True → VALID
        any False → INVALID
        else → UNDERDETERMINED
        """
        if not verdicts:
            return Judgment.UNDERDETERMINED
        if all(v is True for v in verdicts):
            return Judgment.VALID
        if any(v is False for v in verdicts):
            return Judgment.INVALID
        return Judgment.UNDERDETERMINED

    # ─────────────────────────────────────────────────────────────────────
    # Metacognitive trigger (System 2)
    # ─────────────────────────────────────────────────────────────────────

    def should_deliberate(
        self,
        constraints: ConstraintSet,
        initial_judgment: Judgment,
        verdicts: list[bool | None],
    ) -> bool:
        """
        Metacognitive trigger: decide whether to engage System 2.

        Triggers when:
        - Initial judgment is UNDERDETERMINED
        - Open-world uncertainty (None verdicts)
        - Unexamined disjunction branches exist
        - Premises match a known illusion pattern
        - Counterfactuals or negated conditionals present
        """
        triggers = [
            initial_judgment == Judgment.UNDERDETERMINED,
            None in verdicts,
            constraints.has_unexamined_branches([]),
            constraints.matches_known_illusion,
            constraints.has_counterfactuals,
            constraints.has_negated_conditionals,
            constraints.has_disjunctions,
        ]
        return any(triggers)

    # ─────────────────────────────────────────────────────────────────────
    # Confidence score
    # ─────────────────────────────────────────────────────────────────────

    def _compute_confidence(
        self, verdicts: list[bool | None], judgment: Judgment
    ) -> float:
        """Heuristic confidence based on verdict consistency."""
        if not verdicts:
            return 0.0
        n = len(verdicts)
        if judgment == Judgment.VALID:
            return sum(1 for v in verdicts if v is True) / n
        if judgment == Judgment.INVALID:
            return sum(1 for v in verdicts if v is False) / n
        return 0.5

    # ─────────────────────────────────────────────────────────────────────
    # Natural language narration
    # ─────────────────────────────────────────────────────────────────────

    def _narrate(
        self,
        premises: list[str],
        query: str,
        judgment: Judgment,
        models_checked: int,
        system_used: str,
        counterexample: Model | None,
        supporting: Model | None,
    ) -> str:
        """Generate a natural language explanation of the reasoning result."""
        ce_str = "none"
        if counterexample:
            facts = [
                f"{'NOT ' if not f.polarity else ''}{f.predicate}({', '.join(f.args)})"
                for f in counterexample.relations[:5]
            ]
            ce_str = "; ".join(facts) or "empty model"

        sup_str = "none"
        if supporting:
            if supporting.iconic_layer.spatial:
                sup_str = f"spatial ordering: {supporting.iconic_layer.spatial.ordering}"
            elif supporting.iconic_layer.temporal:
                sup_str = f"temporal intervals: {dict(list(supporting.iconic_layer.temporal.intervals.items())[:3])}"
            else:
                facts = [
                    f"{'NOT ' if not f.polarity else ''}{f.predicate}({', '.join(f.args)})"
                    for f in supporting.relations[:3]
                ]
                sup_str = "; ".join(facts) or "empty model"

        try:
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=512,
                system=_NARRATE_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": _NARRATE_USER.format(
                        premises=premises,
                        query=query,
                        judgment=judgment.value,
                        models_checked=models_checked,
                        system_used=system_used,
                        counterexample=ce_str,
                        supporting_model=sup_str,
                    ),
                }],
            )
            return next(b.text for b in response.content if b.type == "text")
        except Exception:
            _logger.exception("MentalModelAgent._narrate failed")
            # Fallback: generate simple explanation
            return (
                f"Judgment: {judgment.value}. "
                f"Checked {models_checked} model(s) using {system_used}. "
                f"{'Counterexample found.' if counterexample else ''}"
            )

    # ─────────────────────────────────────────────────────────────────────
    # Convenience: just compile (useful for inspection/testing)
    # ─────────────────────────────────────────────────────────────────────

    def compile(self, premises: list[str]) -> ConstraintSet:
        """Compile premises into a ConstraintSet (no reasoning)."""
        return self.compiler.extract(premises)

    def build_models(
        self, premises: list[str], mode: str = "system1"
    ) -> list[Model]:
        """Compile premises and build models."""
        constraints = self.compiler.extract(premises)
        return self.builder.construct(constraints, mode=mode)
