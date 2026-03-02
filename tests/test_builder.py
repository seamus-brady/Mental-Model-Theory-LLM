"""
Unit tests for ModelBuilder — specifically the pure-Python parts
(iconic predicate derivation) that don't require API calls.
"""

import pytest
from mmt.builder import (
    _derive_causal_predicates,
    _derive_spatial_predicates,
    _derive_temporal_predicates,
    _extract_rules_from_constraints,
)
from mmt.models import (
    CausalLayout,
    Constraint,
    ConstraintSet,
    ConstraintType,
    Domain,
    Provenance,
    TemporalLayout,
)


# ─────────────────────────────────────────────
# Spatial predicate derivation
# ─────────────────────────────────────────────


class TestDeriveSpatialPredicates:
    def test_two_entities(self):
        relations = _derive_spatial_predicates(["A", "B"])
        predicates = [(f.predicate, f.args, f.polarity) for f in relations]

        # A left_of B
        assert ("left_of", ["A", "B"], True) in predicates
        # B right_of A
        assert ("right_of", ["B", "A"], True) in predicates

    def test_three_entities_transitive(self):
        relations = _derive_spatial_predicates(["A", "B", "C"])
        predicates = [(f.predicate, f.args) for f in relations if f.polarity]

        # All ordered pairs from ordering
        assert ("left_of", ["A", "B"]) in predicates
        assert ("left_of", ["A", "C"]) in predicates  # emergent inference
        assert ("left_of", ["B", "C"]) in predicates
        assert ("right_of", ["B", "A"]) in predicates
        assert ("right_of", ["C", "A"]) in predicates  # emergent
        assert ("right_of", ["C", "B"]) in predicates

    def test_provenance_is_iconic(self):
        relations = _derive_spatial_predicates(["A", "B"])
        for r in relations:
            assert r.provenance == Provenance.ICONIC

    def test_empty_ordering(self):
        relations = _derive_spatial_predicates([])
        assert relations == []

    def test_single_entity(self):
        relations = _derive_spatial_predicates(["A"])
        assert relations == []


# ─────────────────────────────────────────────
# Temporal predicate derivation
# ─────────────────────────────────────────────


class TestDeriveTemporalPredicates:
    def test_before_after(self):
        temporal = TemporalLayout(intervals={
            "E1": {"start": 0.0, "end": 5.0},
            "E2": {"start": 6.0, "end": 10.0},
        })
        relations = _derive_temporal_predicates(temporal, ["E1", "E2"])
        predicates = [(f.predicate, f.args) for f in relations if f.polarity]

        assert ("before", ["E1", "E2"]) in predicates
        assert ("after", ["E2", "E1"]) in predicates
        # Should NOT have before(E2, E1)
        before_e2_e1 = [f for f in relations
                        if f.predicate == "before" and f.args == ["E2", "E1"]]
        assert all(not f.polarity for f in before_e2_e1)

    def test_overlaps(self):
        temporal = TemporalLayout(intervals={
            "E1": {"start": 0.0, "end": 5.0},
            "E2": {"start": 3.0, "end": 8.0},
        })
        relations = _derive_temporal_predicates(temporal, ["E1", "E2"])
        predicates = [(f.predicate, f.args) for f in relations if f.polarity]
        assert ("overlaps", ["E1", "E2"]) in predicates

    def test_during(self):
        temporal = TemporalLayout(intervals={
            "outer": {"start": 0.0, "end": 10.0},
            "inner": {"start": 2.0, "end": 7.0},
        })
        relations = _derive_temporal_predicates(temporal, ["outer", "inner"])
        predicates = [(f.predicate, f.args) for f in relations if f.polarity]
        assert ("during", ["inner", "outer"]) in predicates

    def test_provenance_is_iconic(self):
        temporal = TemporalLayout(intervals={
            "E1": {"start": 0.0, "end": 5.0},
            "E2": {"start": 6.0, "end": 10.0},
        })
        relations = _derive_temporal_predicates(temporal, ["E1", "E2"])
        for r in relations:
            assert r.provenance == Provenance.ICONIC


# ─────────────────────────────────────────────
# Causal predicate derivation
# ─────────────────────────────────────────────


class TestDeriveCausalPredicates:
    def test_direct_edges(self):
        causal = CausalLayout(
            nodes=["A", "B", "C"],
            edges=[
                {"from": "A", "to": "B", "type": "causes"},
                {"from": "B", "to": "C", "type": "enables"},
            ],
        )
        relations = _derive_causal_predicates(causal)
        predicates = [(f.predicate, f.args) for f in relations if f.polarity]

        assert ("causes", ["A", "B"]) in predicates
        assert ("enables", ["B", "C"]) in predicates

    def test_transitive_influence(self):
        causal = CausalLayout(
            nodes=["A", "B", "C"],
            edges=[
                {"from": "A", "to": "B", "type": "causes"},
                {"from": "B", "to": "C", "type": "causes"},
            ],
        )
        relations = _derive_causal_predicates(causal)
        predicates = [(f.predicate, f.args) for f in relations if f.polarity]

        # Transitive: A causally_influences C
        assert ("causally_influences", ["A", "C"]) in predicates

    def test_provenance_is_iconic(self):
        causal = CausalLayout(
            nodes=["A", "B"],
            edges=[{"from": "A", "to": "B", "type": "causes"}],
        )
        relations = _derive_causal_predicates(causal)
        for r in relations:
            assert r.provenance == Provenance.ICONIC


# ─────────────────────────────────────────────
# _extract_rules_from_constraints
# ─────────────────────────────────────────────


class TestExtractRules:
    def test_universal_becomes_schema_rule(self):
        cs = ConstraintSet(
            domain=Domain.QUANTIFICATIONAL,
            constraints=[
                Constraint(
                    type=ConstraintType.UNIVERSAL,
                    antecedent_pred="musician",
                    consequent_pred="creative",
                )
            ],
        )
        schema_rules, cond_triggers = _extract_rules_from_constraints(cs)
        assert len(schema_rules) == 1
        assert schema_rules[0].antecedent_pred == "musician"
        assert schema_rules[0].consequent_pred == "creative"
        assert len(cond_triggers) == 0

    def test_conditional_becomes_trigger(self):
        cs = ConstraintSet(
            domain=Domain.PROPOSITIONAL,
            constraints=[
                Constraint(
                    type=ConstraintType.CONDITIONAL,
                    antecedent="raining",
                    consequent="ground_wet",
                )
            ],
        )
        schema_rules, cond_triggers = _extract_rules_from_constraints(cs)
        assert len(schema_rules) == 0
        assert len(cond_triggers) == 1
        assert cond_triggers[0].antecedent_pred == "raining"
        assert cond_triggers[0].consequent_pred == "ground_wet"

    def test_non_rule_constraints_ignored(self):
        cs = ConstraintSet(
            domain=Domain.SPATIAL,
            constraints=[
                Constraint(
                    type=ConstraintType.SPATIAL_RELATION,
                    entity1="A", relation="left_of", entity2="B",
                )
            ],
        )
        schema_rules, cond_triggers = _extract_rules_from_constraints(cs)
        assert schema_rules == []
        assert cond_triggers == []

    def test_mixed_constraints(self):
        cs = ConstraintSet(
            domain=Domain.MIXED,
            constraints=[
                Constraint(
                    type=ConstraintType.UNIVERSAL,
                    antecedent_pred="A",
                    consequent_pred="B",
                ),
                Constraint(
                    type=ConstraintType.CONDITIONAL,
                    antecedent="X",
                    consequent="Y",
                ),
                Constraint(
                    type=ConstraintType.ATOMIC,
                    predicate="some_fact",
                    args=[],
                    polarity=True,
                ),
            ],
        )
        schema_rules, cond_triggers = _extract_rules_from_constraints(cs)
        assert len(schema_rules) == 1
        assert len(cond_triggers) == 1
