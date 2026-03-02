"""
Counterexample Finder: the core of MMT validation.

Strategy (three-phase search):
1. Check EXISTING models — do any falsify the conclusion?
2. Try TARGETED EXPANSION — can we flesh out a model to falsify it?
3. EXPLORE UNEXPLORED BRANCHES — disjunctions not yet modeled

This prevents the LLM from "feeling" validity without actually checking.

Reference: Khemlani & Johnson-Laird (2012), Johnson-Laird (2006)
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import anthropic

from ._logging import get_logger
from .checker import ConstraintChecker
from .models import (
    Constraint,
    ConstraintSet,
    ConstraintType,
    Domain,
    Entity,
    Fact,
    IconicLayer,
    LLMCounterexampleAttempt,
    Model,
    Provenance,
    SearchResult,
)

if TYPE_CHECKING:
    pass

_logger = get_logger(__name__)

_CE_SYSTEM = """\
You are attempting to DISPROVE a reasoning conclusion using Johnson-Laird's Mental Model Theory.

Your goal: construct a model that satisfies ALL premises but makes the conclusion FALSE.
This is "counterexample search" — the key mechanism preventing illusory inferences.

Rules:
1. The model MUST satisfy every premise constraint listed.
2. The conclusion MUST be FALSE in your model.
3. Use MINIMAL additions — don't add unnecessary facts.
4. If it is IMPOSSIBLE to construct such a model (due to the constraints), explain why.

OUTPUT FORMAT: Return ONLY a valid JSON object, no markdown, no code fences.
{
  "possible": true or false,
  "reasoning": "your explanation",
  "blocking_constraint": "why impossible (only include if possible=false)",
  "model": {
    "entities": [{"id": "...", "entity_type": "...", "properties_true": ["..."]}],
    "facts": [{"predicate": "...", "args": ["..."], "polarity": true}],
    "implicit_note": "..."
  }
}
Include "model" only if possible=true. Include "blocking_constraint" only if possible=false.
"""

_CE_USER = """\
Conclusion to disprove: {conclusion}

Premise constraints that MUST be satisfied:
{constraints}

Existing models (all currently support the conclusion):
{models_summary}

Try to construct a valid model where the conclusion is FALSE.
"""


# ─────────────────────────────────────────────
# JSON response parser
# ─────────────────────────────────────────────


def _parse_json_response(response) -> dict:
    """Extract and parse the JSON object from an LLM response."""
    text = next(b.text for b in response.content if b.type == "text")
    # Strip markdown code fences if the model wraps the output
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


class CounterexampleFinder:
    """
    Searches for models that falsify the proposed conclusion.

    Three phases:
    1. Check existing models
    2. Targeted LLM-guided expansion
    3. Unexplored branch exploration
    """

    def __init__(
        self,
        checker: ConstraintChecker,
        client: anthropic.Anthropic,
        model_id: str = "claude-opus-4-6",
        max_search_depth: int = 3,
    ):
        self.checker = checker
        self.client = client
        self.model_id = model_id
        self.max_search_depth = max_search_depth

    def search(
        self,
        models: list[Model],
        conclusion_pred: str,
        conclusion_args: list[str],
        conclusion_polarity: bool,
        constraints: ConstraintSet,
    ) -> SearchResult:
        """
        Search for a model that satisfies all constraints but falsifies conclusion.

        Returns SearchResult with found=True if a counterexample is found.
        """
        # Phase 1: Check existing models
        for model in models:
            if not self.checker.satisfies_constraints(model, constraints):
                continue  # Skip inconsistent models
            result = self.checker.evaluate(
                model, conclusion_pred, conclusion_args, conclusion_polarity
            )
            if result is False:
                return SearchResult(
                    found=True,
                    counterexample=model,
                    method="existing_model",
                    coverage=len(models),
                )

        # Phase 2: LLM-guided targeted expansion
        ce_model = self._targeted_llm_expansion(
            models, conclusion_pred, conclusion_args, conclusion_polarity, constraints
        )
        if ce_model is not None:
            return SearchResult(
                found=True,
                counterexample=ce_model,
                method="llm_expansion",
                coverage=len(models),
            )

        # Phase 3: Explore unexplored disjunction branches
        unexplored = self._get_unexplored_branches(models, constraints)
        for branch_constraint in unexplored[: self.max_search_depth]:
            branch_model = self._construct_branch_model(models[0] if models else None,
                                                         branch_constraint, constraints)
            if branch_model is None:
                continue
            if not self.checker.satisfies_constraints(branch_model, constraints):
                continue
            result = self.checker.evaluate(
                branch_model, conclusion_pred, conclusion_args, conclusion_polarity
            )
            if result is False:
                return SearchResult(
                    found=True,
                    counterexample=branch_model,
                    method="branch_exploration",
                    coverage=len(models),
                )

        return SearchResult(found=False, coverage=len(models))

    def _targeted_llm_expansion(
        self,
        models: list[Model],
        conclusion_pred: str,
        conclusion_args: list[str],
        conclusion_polarity: bool,
        constraints: ConstraintSet,
    ) -> Model | None:
        """Ask the LLM to construct a counterexample model."""
        conclusion_str = (
            f"{'NOT ' if not conclusion_polarity else ''}"
            f"{conclusion_pred}({', '.join(conclusion_args)})"
        )

        cs_str = self._format_constraints(constraints.constraints)
        models_summary = self._summarize_models(models[:3])  # show first 3

        try:
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=2048,
                system=_CE_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": _CE_USER.format(
                        conclusion=conclusion_str,
                        constraints=cs_str,
                        models_summary=models_summary,
                    ),
                }],
            )
            raw = _parse_json_response(response)
            attempt = LLMCounterexampleAttempt.model_validate(raw)

            if not attempt.possible or attempt.model is None:
                return None

            # Build a model from the LLM's counterexample
            return self._build_model_from_llm(attempt.model, constraints)

        except Exception:
            _logger.exception("CounterexampleFinder._targeted_llm_expansion failed")
            return None

    def _get_unexplored_branches(
        self, models: list[Model], constraints: ConstraintSet
    ) -> list[Constraint]:
        """Find disjunction branches not yet represented in models."""
        unexplored = []
        for c in constraints.get_disjunctions():
            if not c.disjuncts:
                continue
            for disjunct in c.disjuncts:
                pred = disjunct.replace(" ", "_").lower()
                # Check if any model has this disjunct as true
                covered = any(
                    self.checker.evaluate(m, pred, [], True) is True
                    for m in models
                )
                if not covered:
                    unexplored.append(Constraint(
                        type=c.type,
                        disjuncts=[disjunct],
                        predicate=pred,
                        args=[],
                        polarity=True,
                    ))
        return unexplored

    def _construct_branch_model(
        self,
        base_model: Model | None,
        branch: Constraint,
        constraints: ConstraintSet,
    ) -> Model | None:
        """Construct a new model with a specific disjunct branch forced true."""
        if base_model is None:
            return None

        model = base_model.copy()
        if branch.predicate and branch.args is not None:
            model.add_fact(Fact(
                predicate=branch.predicate,
                args=branch.args,
                polarity=True,
                provenance=Provenance.DERIVED,
            ))
        return model

    def _build_model_from_llm(
        self, llm_model, constraints: ConstraintSet
    ) -> Model:
        """Build an internal Model from an LLM-proposed counterexample."""
        entities = []
        for e in llm_model.entities:
            props = {p: True for p in e.properties_true}
            entities.append(Entity(
                id=e.id,
                entity_type=e.entity_type,
                properties=props,
            ))

        relations = []
        for f in llm_model.facts:
            relations.append(Fact(
                predicate=f.predicate,
                args=f.args,
                polarity=f.polarity,
                provenance=Provenance.ASSUMED,  # LLM proposed; check provenance
            ))

        return Model(
            domain=constraints.domain,
            iconic_layer=IconicLayer(),
            entities=entities,
            relations=relations,
            is_fleshed_out=True,
            implicit_note=llm_model.implicit_note,
        )

    def _format_constraints(self, constraints: list[Constraint]) -> str:
        lines = []
        for c in constraints:
            lines.append(f"  - {c.type.value}: {c.model_dump(exclude_none=True)}")
        return "\n".join(lines) if lines else "  (none)"

    def _summarize_models(self, models: list[Model]) -> str:
        summaries = []
        for i, m in enumerate(models):
            facts = [
                f"{'NOT ' if not f.polarity else ''}{f.predicate}({', '.join(f.args)})"
                for f in m.relations[:5]
            ]
            summaries.append(f"Model {i+1}: {'; '.join(facts) or '(empty)'}")
        return "\n".join(summaries) if summaries else "(no models yet)"
