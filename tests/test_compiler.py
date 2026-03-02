"""
Unit tests for SemanticCompiler with mocked LLM calls.
These tests verify the compiler's conversion logic without API calls.
"""

import pytest
from unittest.mock import MagicMock, patch
from mmt.compiler import SemanticCompiler, _convert_llm_output
from mmt.models import (
    ConstraintType,
    Domain,
    LLMConstraint,
    LLMConstraintSet,
)


# ─────────────────────────────────────────────
# _convert_llm_output tests (pure Python)
# ─────────────────────────────────────────────


class TestConvertLLMOutput:
    def _make_llm_output(self, **kwargs) -> LLMConstraintSet:
        defaults = {
            "domain": "propositional",
            "constraints": [],
            "entities": [],
            "has_disjunctions": False,
            "has_counterfactuals": False,
            "has_negated_conditionals": False,
            "matches_known_illusion": False,
            "reasoning_notes": "test",
        }
        defaults.update(kwargs)
        return LLMConstraintSet(**defaults)

    def test_domain_conversion(self):
        llm = self._make_llm_output(domain="spatial")
        cs = _convert_llm_output(llm, [])
        assert cs.domain == Domain.SPATIAL

    def test_unknown_domain_becomes_mixed(self):
        llm = self._make_llm_output(domain="unknown_domain")
        cs = _convert_llm_output(llm, [])
        assert cs.domain == Domain.MIXED

    def test_conditional_constraint(self):
        llm = self._make_llm_output(
            constraints=[
                LLMConstraint(
                    type="conditional",
                    antecedent="raining",
                    consequent="ground_wet",
                )
            ]
        )
        cs = _convert_llm_output(llm, ["If it rains, the ground is wet"])
        assert len(cs.constraints) == 1
        c = cs.constraints[0]
        assert c.type == ConstraintType.CONDITIONAL
        assert c.antecedent == "raining"
        assert c.consequent == "ground_wet"

    def test_spatial_constraint(self):
        llm = self._make_llm_output(
            domain="spatial",
            constraints=[
                LLMConstraint(
                    type="spatial_relation",
                    entity1="A",
                    relation="left_of",
                    entity2="B",
                )
            ],
            entities=["A", "B"],
        )
        cs = _convert_llm_output(llm, ["A is to the left of B"])
        assert len(cs.constraints) == 1
        c = cs.constraints[0]
        assert c.type == ConstraintType.SPATIAL_RELATION
        assert c.entity1 == "A"
        assert c.relation == "left_of"
        assert c.entity2 == "B"

    def test_disjunction_flags(self):
        llm = self._make_llm_output(
            has_disjunctions=True,
            has_negated_conditionals=True,
        )
        cs = _convert_llm_output(llm, [])
        assert cs.has_disjunctions is True
        assert cs.has_negated_conditionals is True

    def test_premises_preserved(self):
        premises = ["P1", "P2", "P3"]
        llm = self._make_llm_output()
        cs = _convert_llm_output(llm, premises)
        assert cs.raw_premises == premises

    def test_universal_constraint(self):
        llm = self._make_llm_output(
            domain="quantificational",
            constraints=[
                LLMConstraint(
                    type="universal",
                    antecedent_pred="musician",
                    consequent_pred="creative",
                )
            ],
        )
        cs = _convert_llm_output(llm, ["All musicians are creative"])
        c = cs.constraints[0]
        assert c.type == ConstraintType.UNIVERSAL
        assert c.antecedent_pred == "musician"
        assert c.consequent_pred == "creative"

    def test_exclusive_disjunction(self):
        llm = self._make_llm_output(
            constraints=[
                LLMConstraint(
                    type="exclusive_disjunction",
                    disjuncts=["soup", "salad"],
                    exclusive=True,
                )
            ],
            has_disjunctions=True,
        )
        cs = _convert_llm_output(llm, ["Either soup or salad but not both"])
        c = cs.constraints[0]
        assert c.type == ConstraintType.EXCLUSIVE_DISJUNCTION
        assert c.exclusive is True
        assert c.disjuncts == ["soup", "salad"]

    def test_causal_constraint(self):
        llm = self._make_llm_output(
            domain="causal",
            constraints=[
                LLMConstraint(
                    type="causal_causes",
                    cause="fire",
                    effect="smoke",
                    causal_type="causes",
                )
            ],
        )
        cs = _convert_llm_output(llm, ["Fire causes smoke"])
        c = cs.constraints[0]
        assert c.type == ConstraintType.CAUSAL_CAUSES
        assert c.cause == "fire"
        assert c.effect == "smoke"

    def test_known_illusion_flagged(self):
        llm = self._make_llm_output(matches_known_illusion=True)
        cs = _convert_llm_output(llm, [])
        assert cs.matches_known_illusion is True


# ─────────────────────────────────────────────
# SemanticCompiler with mocked client
# ─────────────────────────────────────────────


class TestSemanticCompilerMocked:
    def _make_mock_response(self, llm_dict: dict) -> MagicMock:
        """Create a mock Anthropic response returning the given dict as JSON."""
        import json
        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = json.dumps(llm_dict)
        mock_response.content = [mock_text_block]
        return mock_response

    def _make_compiler(self) -> tuple[SemanticCompiler, MagicMock]:
        mock_client = MagicMock()
        compiler = SemanticCompiler(mock_client)
        return compiler, mock_client

    def test_extract_spatial_premises(self):
        compiler, mock_client = self._make_compiler()
        mock_client.messages.create.return_value = self._make_mock_response({
            "domain": "spatial",
            "constraints": [
                {
                    "type": "spatial_relation",
                    "entity1": "A",
                    "relation": "left_of",
                    "entity2": "B",
                }
            ],
            "entities": ["A", "B"],
            "has_disjunctions": False,
            "has_counterfactuals": False,
            "has_negated_conditionals": False,
            "matches_known_illusion": False,
            "reasoning_notes": "Simple spatial ordering",
        })

        cs = compiler.extract(["A is to the left of B"])
        assert cs.domain == Domain.SPATIAL
        assert len(cs.constraints) == 1
        assert cs.constraints[0].type == ConstraintType.SPATIAL_RELATION
        assert cs.entities == ["A", "B"]

    def test_extract_detects_illusion_pattern(self):
        compiler, mock_client = self._make_compiler()
        mock_client.messages.create.return_value = self._make_mock_response({
            "domain": "propositional",
            "constraints": [
                {
                    "type": "conditional",
                    "antecedent": "king",
                    "consequent": "ace",
                },
                {
                    "type": "conditional",
                    "antecedent": "not_king",
                    "consequent": "ace",
                },
            ],
            "entities": [],
            "has_disjunctions": False,
            "has_counterfactuals": False,
            "has_negated_conditionals": False,
            "matches_known_illusion": True,
            "reasoning_notes": "Classic illusory inference pattern",
        })

        cs = compiler.extract([
            "If there is a king then there is an ace",
            "If there is not a king then there is an ace",
        ])
        assert cs.matches_known_illusion is True

    def test_extract_preserves_raw_premises(self):
        compiler, mock_client = self._make_compiler()
        mock_client.messages.create.return_value = self._make_mock_response({
            "domain": "propositional",
            "constraints": [],
            "entities": [],
            "has_disjunctions": False,
            "has_counterfactuals": False,
            "has_negated_conditionals": False,
            "matches_known_illusion": False,
            "reasoning_notes": "",
        })

        premises = ["P1", "P2"]
        cs = compiler.extract(premises)
        assert cs.raw_premises == premises

    def test_parse_query(self):
        compiler, mock_client = self._make_compiler()
        mock_client.messages.create.return_value = self._make_mock_response({
            "predicate": "left_of",
            "args": ["A", "C"],
            "polarity": True,
            "query_type": "atomic",
        })

        result = compiler.parse_query("Is A to the left of C?", Domain.SPATIAL)
        assert result["predicate"] == "left_of"
        assert result["args"] == ["A", "C"]
        assert result["polarity"] is True
        assert result["query_type"] == "atomic"
