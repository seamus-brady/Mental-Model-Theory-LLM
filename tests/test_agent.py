"""
Unit tests for MentalModelAgent with mocked dependencies.
Tests the dual-process logic, aggregation, and metacognitive triggering.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from mmt.agent import MentalModelAgent
from mmt.models import (
    Constraint,
    ConstraintSet,
    ConstraintType,
    Domain,
    Entity,
    Fact,
    IconicLayer,
    Judgment,
    Model,
    Provenance,
    ReasoningResult,
    SpatialLayout,
)
from mmt.checker import ConstraintChecker


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def make_spatial_model(ordering: list[str]) -> Model:
    from mmt.builder import _derive_spatial_predicates
    spatial = SpatialLayout(ordering=ordering)
    spatial.derive_coordinates()
    return Model(
        domain=Domain.SPATIAL,
        iconic_layer=IconicLayer(spatial=spatial),
        entities=[Entity(id=e) for e in ordering],
        relations=_derive_spatial_predicates(ordering),
    )


def make_propositional_model(*facts: tuple) -> Model:
    return Model(
        domain=Domain.PROPOSITIONAL,
        relations=[
            Fact(predicate=p, args=list(a), polarity=pol,
                 provenance=Provenance.EXPLICIT)
            for p, a, pol in facts
        ],
    )


def make_mock_narrate_response(text: str) -> MagicMock:
    mock = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = text
    mock.content = [block]
    return mock


# ─────────────────────────────────────────────
# MentalModelAgent._aggregate tests
# ─────────────────────────────────────────────


class TestAggregate:
    def setup_method(self):
        self.agent = MentalModelAgent.__new__(MentalModelAgent)

    def test_all_true_is_valid(self):
        assert self.agent._aggregate([True, True, True]) == Judgment.VALID

    def test_any_false_is_invalid(self):
        assert self.agent._aggregate([True, False, True]) == Judgment.INVALID
        assert self.agent._aggregate([False]) == Judgment.INVALID

    def test_all_none_is_underdetermined(self):
        assert self.agent._aggregate([None, None]) == Judgment.UNDERDETERMINED

    def test_mixed_true_none_is_underdetermined(self):
        assert self.agent._aggregate([True, None]) == Judgment.UNDERDETERMINED

    def test_empty_is_underdetermined(self):
        assert self.agent._aggregate([]) == Judgment.UNDERDETERMINED


# ─────────────────────────────────────────────
# MentalModelAgent.should_deliberate tests
# ─────────────────────────────────────────────


class TestShouldDeliberate:
    def setup_method(self):
        self.agent = MentalModelAgent.__new__(MentalModelAgent)

    def _make_cs(self, **kwargs) -> ConstraintSet:
        defaults = {
            "domain": Domain.PROPOSITIONAL,
            "has_disjunctions": False,
            "has_counterfactuals": False,
            "has_negated_conditionals": False,
            "matches_known_illusion": False,
        }
        defaults.update(kwargs)
        return ConstraintSet(**defaults)

    def test_underdetermined_triggers_s2(self):
        cs = self._make_cs()
        assert self.agent.should_deliberate(cs, Judgment.UNDERDETERMINED, [None]) is True

    def test_none_in_verdicts_triggers_s2(self):
        cs = self._make_cs()
        assert self.agent.should_deliberate(cs, Judgment.VALID, [True, None]) is True

    def test_known_illusion_triggers_s2(self):
        cs = self._make_cs(matches_known_illusion=True)
        assert self.agent.should_deliberate(cs, Judgment.VALID, [True]) is True

    def test_counterfactuals_trigger_s2(self):
        cs = self._make_cs(has_counterfactuals=True)
        assert self.agent.should_deliberate(cs, Judgment.VALID, [True]) is True

    def test_negated_conditionals_trigger_s2(self):
        cs = self._make_cs(has_negated_conditionals=True)
        assert self.agent.should_deliberate(cs, Judgment.VALID, [True]) is True

    def test_disjunctions_trigger_s2(self):
        cs = self._make_cs(has_disjunctions=True)
        assert self.agent.should_deliberate(cs, Judgment.VALID, [True]) is True

    def test_simple_valid_no_triggers(self):
        # Valid verdict, no special patterns: should not require s2
        # BUT: has_disjunctions=False, so check purely valid case
        cs = ConstraintSet(
            domain=Domain.SPATIAL,
            has_disjunctions=False,
            has_counterfactuals=False,
            has_negated_conditionals=False,
            matches_known_illusion=False,
        )
        # With a clear VALID result and no triggers, should_deliberate depends
        # on has_unexamined_branches — with no constraints, returns False
        result = self.agent.should_deliberate(cs, Judgment.VALID, [True])
        # No disjunctions, no illusion flags → False (no deliberation needed)
        # (has_unexamined_branches checks constraints list, which is empty)
        assert result is False


# ─────────────────────────────────────────────
# MentalModelAgent._evaluate_model tests
# ─────────────────────────────────────────────


class TestEvaluateModel:
    def setup_method(self):
        self.agent = MentalModelAgent.__new__(MentalModelAgent)
        self.agent.checker = ConstraintChecker()

    def test_spatial_evaluation(self):
        model = make_spatial_model(["A", "B", "C"])
        # A is left of C (emergent inference from iconic structure)
        result = self.agent._evaluate_model(model, "left_of", ["A", "C"], True, "atomic")
        assert result is True

    def test_spatial_evaluation_false(self):
        model = make_spatial_model(["A", "B", "C"])
        result = self.agent._evaluate_model(model, "left_of", ["C", "A"], True, "atomic")
        assert result is False

    def test_unknown_fact(self):
        model = make_propositional_model()
        result = self.agent._evaluate_model(model, "raining", [], True, "atomic")
        assert result is None

    def test_existential_evaluation(self):
        model = Model(
            domain=Domain.QUANTIFICATIONAL,
            entities=[Entity(id="e1")],
            relations=[
                Fact(predicate="artist", args=["e1"], polarity=True,
                     provenance=Provenance.EXPLICIT)
            ],
        )
        result = self.agent._evaluate_model(model, "artist", ["artist"], True, "existential")
        assert result is True


# ─────────────────────────────────────────────
# MentalModelAgent._compute_confidence tests
# ─────────────────────────────────────────────


class TestComputeConfidence:
    def setup_method(self):
        self.agent = MentalModelAgent.__new__(MentalModelAgent)

    def test_all_true_verdicts_full_confidence(self):
        conf = self.agent._compute_confidence([True, True, True], Judgment.VALID)
        assert conf == 1.0

    def test_some_none_reduces_confidence(self):
        conf = self.agent._compute_confidence([True, None], Judgment.VALID)
        assert 0 < conf < 1.0

    def test_invalid_confidence(self):
        conf = self.agent._compute_confidence([False, True], Judgment.INVALID)
        assert 0 < conf <= 1.0

    def test_empty_verdicts_zero_confidence(self):
        conf = self.agent._compute_confidence([], Judgment.UNDERDETERMINED)
        assert conf == 0.0


# ─────────────────────────────────────────────
# MentalModelAgent integration (fully mocked)
# ─────────────────────────────────────────────


class TestAgentIntegration:
    """
    Test the full reasoning loop with all dependencies mocked.
    Verifies the System 1/System 2 control flow.
    """

    def _make_agent(
        self,
        models: list[Model],
        constraints: ConstraintSet,
        query_parsed: dict,
        narration: str = "Test explanation.",
    ) -> MentalModelAgent:
        """Build an agent with all dependencies mocked."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_mock_narrate_response(narration)

        agent = MentalModelAgent.__new__(MentalModelAgent)
        agent.client = mock_client
        agent.model_id = "claude-opus-4-6"

        # Mock compiler
        agent.compiler = MagicMock()
        agent.compiler.extract.return_value = constraints
        agent.compiler.parse_query.return_value = query_parsed

        # Mock builder
        agent.builder = MagicMock()
        agent.builder.construct.return_value = models

        # Real checker and enforcer
        agent.checker = ConstraintChecker()
        agent.enforcer = MagicMock()

        # Mock counterexample finder
        from mmt.models import SearchResult
        agent.ce_finder = MagicMock()
        agent.ce_finder.search.return_value = SearchResult(found=False, coverage=1)

        return agent

    def test_spatial_valid_reasoning(self):
        """A is left of B, B is left of C → A is left of C (valid)."""
        model = make_spatial_model(["A", "B", "C"])
        cs = ConstraintSet(
            domain=Domain.SPATIAL,
            constraints=[
                Constraint(type=ConstraintType.SPATIAL_RELATION,
                           entity1="A", relation="left_of", entity2="B"),
                Constraint(type=ConstraintType.SPATIAL_RELATION,
                           entity1="B", relation="left_of", entity2="C"),
            ],
        )
        query = {"predicate": "left_of", "args": ["A", "C"],
                 "polarity": True, "query_type": "atomic"}

        agent = self._make_agent([model], cs, query)
        result = agent.reason(
            premises=["A is to the left of B", "B is to the left of C"],
            query="Is A to the left of C?",
            deliberate=False,  # Force System 1
        )

        assert result.judgment == Judgment.VALID
        assert result.system_used == "system1"
        assert result.models_checked == 1

    def test_invalid_conclusion_found(self):
        """Conclusion contradicts the spatial model."""
        model = make_spatial_model(["A", "B", "C"])
        cs = ConstraintSet(domain=Domain.SPATIAL)
        query = {"predicate": "left_of", "args": ["C", "A"],
                 "polarity": True, "query_type": "atomic"}

        agent = self._make_agent([model], cs, query)
        result = agent.reason(
            premises=["A is to the left of B"],
            query="Is C to the left of A?",
            deliberate=False,
        )
        assert result.judgment == Judgment.INVALID

    def test_underdetermined_with_open_world(self):
        """No information about query → underdetermined."""
        model = make_propositional_model()
        cs = ConstraintSet(domain=Domain.PROPOSITIONAL)
        query = {"predicate": "raining", "args": [],
                 "polarity": True, "query_type": "atomic"}

        agent = self._make_agent([model], cs, query)
        result = agent.reason(
            premises=["Something happened"],
            query="Is it raining?",
            deliberate=False,
        )
        assert result.judgment == Judgment.UNDERDETERMINED

    def test_system2_triggered_for_illusion_pattern(self):
        """Known illusion pattern should trigger System 2."""
        model = make_propositional_model(
            ("king", [], True),
            ("ace", [], True),
        )
        cs = ConstraintSet(
            domain=Domain.PROPOSITIONAL,
            matches_known_illusion=True,  # Triggers System 2
        )
        query = {"predicate": "ace", "args": [],
                 "polarity": True, "query_type": "atomic"}

        agent = self._make_agent([model], cs, query)
        result = agent.reason(
            premises=[
                "If there is a king then there is an ace",
                "If there is not a king then there is an ace",
            ],
            query="Is there an ace?",
            deliberate=None,  # Auto-detect
        )
        # System 2 should be invoked due to illusion pattern
        assert result.system_used == "system2"

    def test_explicit_system1_mode(self):
        """When deliberate=False, always use System 1."""
        model = make_propositional_model(("A", [], True))
        cs = ConstraintSet(
            domain=Domain.PROPOSITIONAL,
            matches_known_illusion=True,  # Would normally trigger S2
        )
        query = {"predicate": "A", "args": [],
                 "polarity": True, "query_type": "atomic"}

        agent = self._make_agent([model], cs, query)
        result = agent.reason(
            premises=["A"],
            query="A?",
            deliberate=False,  # Force System 1
        )
        assert result.system_used == "system1"

    def test_counterexample_found_in_system2(self):
        """System 2 counterexample search finds a falsifying model."""
        model = make_propositional_model(
            ("some_A", [], True),
            ("some_B", [], True),
        )
        cs = ConstraintSet(
            domain=Domain.PROPOSITIONAL,
            has_disjunctions=True,  # Triggers System 2
        )
        query = {"predicate": "some_A_B", "args": [],
                 "polarity": True, "query_type": "atomic"}

        # Counterexample model that falsifies conclusion
        ce_model = make_propositional_model(("some_A_B", [], False))

        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_mock_narrate_response("CE found.")

        agent = MentalModelAgent.__new__(MentalModelAgent)
        agent.client = mock_client
        agent.model_id = "claude-opus-4-6"
        agent.compiler = MagicMock()
        agent.compiler.extract.return_value = cs
        agent.compiler.parse_query.return_value = query
        agent.builder = MagicMock()
        agent.builder.construct.return_value = [model]
        agent.checker = ConstraintChecker()
        agent.enforcer = MagicMock()

        from mmt.models import SearchResult
        agent.ce_finder = MagicMock()
        agent.ce_finder.search.return_value = SearchResult(
            found=True, counterexample=ce_model, method="test"
        )

        result = agent.reason(
            premises=["Some A are B", "Some B are C"],
            query="Are some A also C?",
            deliberate=True,
        )

        assert result.judgment == Judgment.INVALID
        assert result.counterexample is not None
        assert result.system_used == "system2"

    def test_inconsistent_premises_detected(self):
        """Contradictory premises return INCONSISTENT_PREMISES."""
        # A model with an antisymmetry violation
        from mmt.models import ConditionalTrigger
        model = Model(
            domain=Domain.PROPOSITIONAL,
            relations=[
                Fact(predicate="left_of", args=["A", "B"], polarity=True,
                     provenance=Provenance.EXPLICIT),
                Fact(predicate="left_of", args=["B", "A"], polarity=True,  # contradiction
                     provenance=Provenance.EXPLICIT),
            ],
        )
        cs = ConstraintSet(domain=Domain.SPATIAL)
        query = {"predicate": "left_of", "args": ["A", "B"],
                 "polarity": True, "query_type": "atomic"}

        agent = self._make_agent([model], cs, query)
        result = agent.reason(
            premises=["A is left of B", "B is left of A"],
            query="Is A left of B?",
            deliberate=False,
        )
        assert result.judgment == Judgment.INCONSISTENT_PREMISES
