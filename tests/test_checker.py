"""
Unit tests for ConstraintChecker and ProvenanceEnforcer.
No API calls — pure Python constraint checking logic.
"""

import pytest
from mmt.checker import ConstraintChecker, ProvenanceEnforcer
from mmt.models import (
    Constraint,
    ConstraintSet,
    ConstraintType,
    ConditionalTrigger,
    Domain,
    Entity,
    Fact,
    IconicLayer,
    Model,
    Provenance,
    SchemaRule,
    SpatialLayout,
    TemporalLayout,
    CausalLayout,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────


def make_spatial_model(ordering: list[str]) -> Model:
    """Build a spatial model with derived predicates from ordering."""
    from mmt.builder import _derive_spatial_predicates
    spatial = SpatialLayout(ordering=ordering)
    spatial.derive_coordinates()
    entities = [Entity(id=e) for e in ordering]
    relations = _derive_spatial_predicates(ordering)
    return Model(
        domain=Domain.SPATIAL,
        iconic_layer=IconicLayer(spatial=spatial),
        entities=entities,
        relations=relations,
    )


def make_simple_model(*facts: tuple) -> Model:
    """Build a propositional model from (predicate, args, polarity) tuples."""
    relations = [
        Fact(predicate=p, args=list(a), polarity=pol, provenance=Provenance.EXPLICIT)
        for p, a, pol in facts
    ]
    return Model(domain=Domain.PROPOSITIONAL, relations=relations)


# ─────────────────────────────────────────────
# ConstraintChecker.evaluate tests
# ─────────────────────────────────────────────


class TestEvaluate:
    def setup_method(self):
        self.checker = ConstraintChecker()

    def test_atomic_true(self):
        model = make_simple_model(("raining", [], True))
        assert self.checker.evaluate(model, "raining", [], True) is True

    def test_atomic_false(self):
        model = make_simple_model(("raining", [], True))
        # Querying for polarity=False when fact is True → False
        assert self.checker.evaluate(model, "raining", [], False) is False

    def test_open_world_unknown(self):
        model = make_simple_model()
        # No fact about "raining" → unknown (not false)
        assert self.checker.evaluate(model, "raining", [], True) is None

    def test_negated_fact(self):
        model = make_simple_model(("raining", [], False))
        assert self.checker.evaluate(model, "raining", [], False) is True
        assert self.checker.evaluate(model, "raining", [], True) is False

    def test_spatial_left_of_from_iconic(self):
        model = make_spatial_model(["A", "B", "C"])
        assert self.checker.evaluate(model, "left_of", ["A", "B"]) is True
        assert self.checker.evaluate(model, "left_of", ["B", "A"]) is False
        # Emergent inference: A left of C (not explicit in premises)
        assert self.checker.evaluate(model, "left_of", ["A", "C"]) is True

    def test_temporal_before_from_iconic(self):
        temporal = TemporalLayout(intervals={
            "E1": {"start": 0.0, "end": 5.0},
            "E2": {"start": 6.0, "end": 10.0},
        })
        model = Model(
            domain=Domain.TEMPORAL,
            iconic_layer=IconicLayer(temporal=temporal),
            entities=[Entity(id="E1"), Entity(id="E2")],
        )
        assert self.checker.evaluate(model, "before", ["E1", "E2"]) is True
        assert self.checker.evaluate(model, "before", ["E2", "E1"]) is False

    def test_schema_rule_derivation(self):
        model = Model(
            domain=Domain.QUANTIFICATIONAL,
            entities=[Entity(id="alice")],
            relations=[
                Fact(predicate="musician", args=["alice"], polarity=True,
                     provenance=Provenance.EXPLICIT),
            ],
            schema_rules=[SchemaRule(antecedent_pred="musician", consequent_pred="creative")],
        )
        # "creative(alice)" should follow from schema rule
        result = self.checker.evaluate(model, "creative", ["alice"])
        assert result is True

    def test_with_args(self):
        model = make_simple_model(("likes", ["alice", "bob"], True))
        assert self.checker.evaluate(model, "likes", ["alice", "bob"], True) is True
        assert self.checker.evaluate(model, "likes", ["bob", "alice"], True) is None


# ─────────────────────────────────────────────
# ConstraintChecker.check_consistency tests
# ─────────────────────────────────────────────


class TestCheckConsistency:
    def setup_method(self):
        self.checker = ConstraintChecker()

    def test_consistent_spatial_model(self):
        model = make_spatial_model(["A", "B", "C"])
        result = self.checker.check_consistency(model)
        assert result.consistent is True
        assert result.violations == []

    def test_antisymmetry_violation(self):
        model = make_simple_model(
            ("left_of", ["A", "B"], True),
            ("left_of", ["B", "A"], True),  # contradiction
        )
        result = self.checker.check_consistency(model)
        assert result.consistent is False
        assert any("antisymmetry" in v.type for v in result.violations)

    def test_transitivity_violation(self):
        model = make_simple_model(
            ("left_of", ["A", "B"], True),
            ("left_of", ["B", "C"], True),
            ("left_of", ["A", "C"], False),  # contradicts transitivity
        )
        result = self.checker.check_consistency(model)
        assert result.consistent is False
        assert any("transitivity" in v.type for v in result.violations)

    def test_iconic_symbolic_mismatch(self):
        spatial = SpatialLayout(ordering=["A", "B"], coordinates={"A": 0, "B": 1})
        model = Model(
            domain=Domain.SPATIAL,
            iconic_layer=IconicLayer(spatial=spatial),
            entities=[Entity(id="A"), Entity(id="B")],
            relations=[
                # Contradicts ordering: A is at 0, B at 1, so A is left of B
                Fact(predicate="left_of", args=["A", "B"], polarity=False,
                     provenance=Provenance.EXPLICIT),
            ],
        )
        result = self.checker.check_consistency(model)
        assert result.consistent is False
        assert any("iconic_mismatch" in v.type for v in result.violations)

    def test_consistent_with_conditionals(self):
        model = Model(
            domain=Domain.PROPOSITIONAL,
            relations=[
                Fact(predicate="raining", args=[], polarity=True,
                     provenance=Provenance.EXPLICIT),
                Fact(predicate="ground_wet", args=[], polarity=True,
                     provenance=Provenance.EXPLICIT),
            ],
            conditional_triggers=[
                ConditionalTrigger(
                    antecedent_pred="raining", antecedent_args=[],
                    consequent_pred="ground_wet", consequent_args=[],
                )
            ],
        )
        result = self.checker.check_consistency(model)
        assert result.consistent is True

    def test_conditional_trigger_violation(self):
        model = Model(
            domain=Domain.PROPOSITIONAL,
            relations=[
                Fact(predicate="raining", args=[], polarity=True,
                     provenance=Provenance.EXPLICIT),
                Fact(predicate="ground_wet", args=[], polarity=False,  # violates conditional
                     provenance=Provenance.EXPLICIT),
            ],
            conditional_triggers=[
                ConditionalTrigger(
                    antecedent_pred="raining", antecedent_args=[],
                    consequent_pred="ground_wet", consequent_args=[],
                )
            ],
        )
        result = self.checker.check_consistency(model)
        assert result.consistent is False
        assert any("conditional" in v.type for v in result.violations)


# ─────────────────────────────────────────────
# ConstraintChecker.evaluate_existential
# ─────────────────────────────────────────────


class TestEvaluateExistential:
    def setup_method(self):
        self.checker = ConstraintChecker()

    def test_some_artist_exists(self):
        model = Model(
            domain=Domain.QUANTIFICATIONAL,
            entities=[
                Entity(id="e1", properties={"artist": True}),
                Entity(id="e2", properties={"artist": False}),
            ],
            relations=[
                Fact(predicate="artist", args=["e1"], polarity=True,
                     provenance=Provenance.EXPLICIT),
                Fact(predicate="artist", args=["e2"], polarity=False,
                     provenance=Provenance.EXPLICIT),
            ],
        )
        assert self.checker.evaluate_existential(model, "artist") is True

    def test_no_artist_exists(self):
        model = Model(
            domain=Domain.QUANTIFICATIONAL,
            entities=[Entity(id="e1")],
            relations=[
                Fact(predicate="artist", args=["e1"], polarity=False,
                     provenance=Provenance.EXPLICIT),
            ],
        )
        assert self.checker.evaluate_existential(model, "artist") is False

    def test_unknown_existential(self):
        model = Model(
            domain=Domain.QUANTIFICATIONAL,
            entities=[Entity(id="e1")],
            relations=[],  # no facts about "artist"
        )
        assert self.checker.evaluate_existential(model, "artist") is None


# ─────────────────────────────────────────────
# ConstraintChecker.evaluate_universal
# ─────────────────────────────────────────────


class TestEvaluateUniversal:
    def setup_method(self):
        self.checker = ConstraintChecker()

    def test_all_musicians_are_students_true(self):
        model = Model(
            domain=Domain.QUANTIFICATIONAL,
            entities=[Entity(id="alice"), Entity(id="bob")],
            relations=[
                Fact(predicate="musician", args=["alice"], polarity=True,
                     provenance=Provenance.EXPLICIT),
                Fact(predicate="student", args=["alice"], polarity=True,
                     provenance=Provenance.EXPLICIT),
                Fact(predicate="musician", args=["bob"], polarity=True,
                     provenance=Provenance.EXPLICIT),
                Fact(predicate="student", args=["bob"], polarity=True,
                     provenance=Provenance.EXPLICIT),
            ],
        )
        assert self.checker.evaluate_universal(model, "musician", "student") is True

    def test_all_musicians_are_students_false(self):
        model = Model(
            domain=Domain.QUANTIFICATIONAL,
            entities=[Entity(id="alice"), Entity(id="bob")],
            relations=[
                Fact(predicate="musician", args=["alice"], polarity=True,
                     provenance=Provenance.EXPLICIT),
                Fact(predicate="student", args=["alice"], polarity=True,
                     provenance=Provenance.EXPLICIT),
                Fact(predicate="musician", args=["bob"], polarity=True,
                     provenance=Provenance.EXPLICIT),
                Fact(predicate="student", args=["bob"], polarity=False,  # bob not a student
                     provenance=Provenance.EXPLICIT),
            ],
        )
        assert self.checker.evaluate_universal(model, "musician", "student") is False


# ─────────────────────────────────────────────
# ProvenanceEnforcer tests
# ─────────────────────────────────────────────


class TestProvenanceEnforcer:
    def setup_method(self):
        self.enforcer = ProvenanceEnforcer()

    def _make_cs(self) -> ConstraintSet:
        return ConstraintSet(
            domain=Domain.PROPOSITIONAL,
            constraints=[
                Constraint(
                    type=ConstraintType.ATOMIC,
                    predicate="raining",
                    args=[],
                    polarity=True,
                )
            ],
        )

    def test_explicit_fact_allowed(self):
        cs = self._make_cs()
        model = Model(
            domain=Domain.PROPOSITIONAL,
            relations=[
                Fact(predicate="raining", args=[], polarity=True,
                     provenance=Provenance.EXPLICIT),
            ],
        )
        violations = self.enforcer.validate_model(model, cs)
        assert violations == []

    def test_iconic_fact_allowed(self):
        cs = self._make_cs()
        model = Model(
            domain=Domain.SPATIAL,
            relations=[
                Fact(predicate="left_of", args=["A", "B"], polarity=True,
                     provenance=Provenance.ICONIC),
            ],
        )
        violations = self.enforcer.validate_model(model, cs)
        assert violations == []

    def test_assumed_fact_flagged(self):
        cs = self._make_cs()
        model = Model(
            domain=Domain.PROPOSITIONAL,
            relations=[
                Fact(predicate="sunny", args=[], polarity=True,
                     provenance=Provenance.ASSUMED),  # not derivable
            ],
        )
        violations = self.enforcer.validate_model(model, cs)
        assert len(violations) > 0
        assert any("unsupported_fact" in v.type for v in violations)

    def test_strip_unsupported(self):
        cs = self._make_cs()
        assumed_fact = Fact(predicate="sunny", args=[], polarity=True,
                            provenance=Provenance.ASSUMED)
        model = Model(
            domain=Domain.PROPOSITIONAL,
            relations=[
                Fact(predicate="raining", args=[], polarity=True,
                     provenance=Provenance.EXPLICIT),
                assumed_fact,
            ],
        )
        violations = self.enforcer.validate_model(model, cs)
        cleaned = self.enforcer.strip_unsupported(model, violations)
        # Assumed fact removed
        assert cleaned.get_fact("sunny", []) is None
        # Explicit fact preserved
        assert cleaned.get_fact("raining", []) is not None


# ─────────────────────────────────────────────
# ConstraintChecker.satisfies_constraints
# ─────────────────────────────────────────────


class TestSatisfiesConstraints:
    def setup_method(self):
        self.checker = ConstraintChecker()

    def test_spatial_constraint_satisfied(self):
        model = make_spatial_model(["A", "B", "C"])
        cs = ConstraintSet(
            domain=Domain.SPATIAL,
            constraints=[
                Constraint(
                    type=ConstraintType.SPATIAL_RELATION,
                    entity1="A", relation="left_of", entity2="B",
                ),
            ],
        )
        assert self.checker.satisfies_constraints(model, cs) is True

    def test_spatial_constraint_violated(self):
        model = make_spatial_model(["B", "A", "C"])  # B is left of A
        cs = ConstraintSet(
            domain=Domain.SPATIAL,
            constraints=[
                Constraint(
                    type=ConstraintType.SPATIAL_RELATION,
                    entity1="A", relation="left_of", entity2="B",
                ),
            ],
        )
        assert self.checker.satisfies_constraints(model, cs) is False

    def test_universal_constraint_satisfied(self):
        model = Model(
            domain=Domain.QUANTIFICATIONAL,
            entities=[Entity(id="e1")],
            relations=[
                Fact(predicate="musician", args=["e1"], polarity=True,
                     provenance=Provenance.EXPLICIT),
            ],
            schema_rules=[SchemaRule(antecedent_pred="musician", consequent_pred="creative")],
        )
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
        assert self.checker.satisfies_constraints(model, cs) is True
