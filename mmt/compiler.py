"""
Semantic Compiler: LLM-powered premise parser.

Converts natural language premises into a ConstraintSet following the
Mental Model Theory semantics (Johnson-Laird, 1983).

Principle of Truth: Only represent what is explicitly true.
The compiler identifies:
- Domain (spatial/temporal/causal/propositional/quantificational)
- Logical connectives (conditionals, disjunctions, universals, etc.)
- Entities introduced
- Known illusion patterns (from MMT literature)
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import anthropic

from ._logging import get_logger
from .models import (
    Constraint,
    ConstraintSet,
    ConstraintType,
    Domain,
    LLMConstraintSet,
)

if TYPE_CHECKING:
    pass

_logger = get_logger(__name__)

# ─────────────────────────────────────────────
# System prompt for the compiler
# ─────────────────────────────────────────────

_COMPILER_SYSTEM = """\
You are a semantic compiler for Johnson-Laird's Mental Model Theory (MMT).

Your task: parse natural language premises into a structured constraint set.

KEY PRINCIPLES:
1. PRINCIPLE OF TRUTH: Only represent what is explicitly true. Do not infer.
2. OPEN WORLD: Absence of information means UNKNOWN, not false.
3. DOMAIN DETECTION: Identify the primary reasoning domain.
4. CONNECTIVE TYPES:
   - "If A then B" → conditional (antecedent=A, consequent=B)
   - "A or B (or both)" → disjunction (disjuncts=[A,B], exclusive=false)
   - "Either A or B (but not both)" → exclusive_disjunction (exclusive=true)
   - "All X are Y" → universal (antecedent_pred=X, consequent_pred=Y)
   - "Some X is Y" → existential (properties=[X,Y])
   - "A is to the left of B" → spatial_relation (entity1=A, relation=left_of, entity2=B)
   - "A before B" → temporal_relation (entity1=A, relation=before, entity2=B)
   - "A causes B" → causal_causes (cause=A, effect=B, causal_type=causes)
   - "A enables B" → causal_enables (cause=A, effect=B, causal_type=enables)
   - "If A had happened, B would have" → counterfactual
   - Negation: polarity=false

KNOWN ILLUSION PATTERNS (set matches_known_illusion=true):
- "If P then Q, or else if not-P then Q" type biconditionals
- Nested conditionals with shared consequents
- Exclusive disjunctions combined with conditionals

DOMAIN SELECTION:
- SPATIAL: any left/right/above/below/between relations
- TEMPORAL: any before/after/during/while/overlap relations
- CAUSAL: any causes/enables/prevents/leads_to
- QUANTIFICATIONAL: all/some/none/most quantifiers
- PROPOSITIONAL: if/or/and/not with non-spatial atomic facts
- MIXED: multiple domains present

OUTPUT FORMAT: Return ONLY a valid JSON object — no markdown, no explanation, no code fences.
The JSON must have exactly these fields:
{
  "domain": "propositional" or "spatial" or "temporal" or "causal" or "quantificational" or "mixed",
  "constraints": [
    {
      "type": "conditional" or "counterfactual" or "disjunction" or "exclusive_disjunction" or
               "conjunction" or "causal_causes" or "causal_enables" or "spatial_relation" or
               "temporal_relation" or "universal" or "existential" or "negation" or "atomic",
      "antecedent": "...",
      "consequent": "...",
      "disjuncts": ["..."],
      "exclusive": true or false,
      "entity1": "...",
      "relation": "...",
      "entity2": "...",
      "antecedent_pred": "...",
      "consequent_pred": "...",
      "properties": ["..."],
      "predicate": "...",
      "args": ["..."],
      "polarity": true or false,
      "cause": "...",
      "effect": "...",
      "causal_type": "causes" or "enables"
    }
  ],
  "entities": ["..."],
  "has_disjunctions": true or false,
  "has_counterfactuals": true or false,
  "has_negated_conditionals": true or false,
  "matches_known_illusion": true or false,
  "reasoning_notes": "..."
}
Only include fields that are relevant for each constraint type.
"""

_COMPILER_USER_TEMPLATE = """\
Parse these premises into a constraint set:

PREMISES:
{premises}

Return ONLY the JSON object. No other text.
"""

_QUERY_PARSE_SYSTEM = """\
Parse a reasoning query into a structured predicate.
Extract: what is being asked (predicate), about whom (args), and whether it is
a positive or negative claim (polarity).
query_type: "atomic" for simple facts, "existential" for "some X", "universal" for "all X".

OUTPUT FORMAT: Return ONLY a valid JSON object — no markdown, no explanation.
{
  "predicate": "...",
  "args": ["..."],
  "polarity": true or false,
  "query_type": "atomic" or "existential" or "universal"
}
"""


# ─────────────────────────────────────────────
# JSON response parser
# ─────────────────────────────────────────────


def _parse_json_response(response) -> dict:
    """Extract and parse the JSON object from an LLM response."""
    text = next(block.text for block in response.content if block.type == "text")
    # Strip markdown code fences if the model wraps the output
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


# ─────────────────────────────────────────────
# Compiler implementation
# ─────────────────────────────────────────────


class SemanticCompiler:
    """
    Uses the LLM to parse natural language premises into a ConstraintSet.

    Follows the "LLM proposes, checker verifies" pattern:
    - LLM extracts the logical structure
    - The resulting ConstraintSet is validated by the ConstraintChecker
    """

    def __init__(self, client: anthropic.Anthropic, model: str = "claude-opus-4-6"):
        self.client = client
        self.model = model

    def extract(self, premises: list[str]) -> ConstraintSet:
        """
        Parse a list of premises into a ConstraintSet.

        Args:
            premises: Natural language premise strings

        Returns:
            ConstraintSet with extracted constraints
        """
        numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(premises))
        user_content = _COMPILER_USER_TEMPLATE.format(premises=numbered)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=_COMPILER_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )

        raw = _parse_json_response(response)
        llm_output = LLMConstraintSet.model_validate(raw)

        return _convert_llm_output(llm_output, premises)

    def parse_query(self, query: str, domain: Domain) -> dict:
        """
        Parse a query string into a structured predicate representation.

        Returns a dict with keys: predicate, args, polarity, query_type
        """
        from .models import LLMQueryParse

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_QUERY_PARSE_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Domain: {domain}\nQuery: {query}\n\nParse this query.",
            }],
        )

        raw = _parse_json_response(response)
        return LLMQueryParse.model_validate(raw).model_dump()


# ─────────────────────────────────────────────
# Conversion helpers
# ─────────────────────────────────────────────

_CONSTRAINT_TYPE_MAP: dict[str, ConstraintType] = {
    "conditional": ConstraintType.CONDITIONAL,
    "counterfactual": ConstraintType.COUNTERFACTUAL,
    "disjunction": ConstraintType.DISJUNCTION,
    "exclusive_disjunction": ConstraintType.EXCLUSIVE_DISJUNCTION,
    "conjunction": ConstraintType.CONJUNCTION,
    "causal_causes": ConstraintType.CAUSAL_CAUSES,
    "causal_enables": ConstraintType.CAUSAL_ENABLES,
    "spatial_relation": ConstraintType.SPATIAL_RELATION,
    "temporal_relation": ConstraintType.TEMPORAL_RELATION,
    "universal": ConstraintType.UNIVERSAL,
    "existential": ConstraintType.EXISTENTIAL,
    "negation": ConstraintType.NEGATION,
    "atomic": ConstraintType.ATOMIC,
}


def _convert_llm_output(llm: LLMConstraintSet, premises: list[str]) -> ConstraintSet:
    """Convert LLM structured output into the internal ConstraintSet format."""
    constraints = []
    for llm_c in llm.constraints:
        ctype = _CONSTRAINT_TYPE_MAP.get(llm_c.type, ConstraintType.ATOMIC)
        constraint = Constraint(
            type=ctype,
            antecedent=llm_c.antecedent,
            consequent=llm_c.consequent,
            disjuncts=llm_c.disjuncts,
            exclusive=llm_c.exclusive or False,
            entity1=llm_c.entity1,
            relation=llm_c.relation,
            entity2=llm_c.entity2,
            antecedent_pred=llm_c.antecedent_pred,
            consequent_pred=llm_c.consequent_pred,
            properties=llm_c.properties,
            predicate=llm_c.predicate,
            args=llm_c.args,
            polarity=llm_c.polarity,
            cause=llm_c.cause,
            effect=llm_c.effect,
            causal_type=llm_c.causal_type,
        )
        constraints.append(constraint)

    try:
        domain = Domain(llm.domain)
    except ValueError:
        domain = Domain.MIXED

    return ConstraintSet(
        domain=domain,
        constraints=constraints,
        entities=llm.entities,
        has_disjunctions=llm.has_disjunctions,
        has_counterfactuals=llm.has_counterfactuals,
        has_negated_conditionals=llm.has_negated_conditionals,
        matches_known_illusion=llm.matches_known_illusion,
        raw_premises=premises,
    )
