"""
Unit tests for core data models.
No API calls — pure Python tests.
"""

import pytest
from mmt.models import (
    CausalLayout,
    Constraint,
    ConstraintSet,
    ConstraintType,
    ConditionalTrigger,
    Domain,
    Entity,
    Fact,
    IconicLayer,
    Judgment,
    Model,
    Provenance,
    SchemaRule,
    SpatialLayout,
    TemporalLayout,
)


# ─────────────────────────────────────────────
# SpatialLayout tests
# ─────────────────────────────────────────────


class TestSpatialLayout:
    def test_derive_coordinates(self):
        layout = SpatialLayout(ordering=["A", "B", "C"])
        layout.derive_coordinates()
        assert layout.coordinates == {"A": 0, "B": 1, "C": 2}

    def test_is_left_of_true(self):
        layout = SpatialLayout(ordering=["A", "B", "C"], coordinates={"A": 0, "B": 1, "C": 2})
        assert layout.is_left_of("A", "B") is True
        assert layout.is_left_of("A", "C") is True  # transitive
        assert layout.is_left_of("B", "C") is True

    def test_is_left_of_false(self):
        layout = SpatialLayout(ordering=["A", "B", "C"], coordinates={"A": 0, "B": 1, "C": 2})
        assert layout.is_left_of("B", "A") is False
        assert layout.is_left_of("C", "A") is False

    def test_is_right_of(self):
        layout = SpatialLayout(ordering=["A", "B", "C"], coordinates={"A": 0, "B": 1, "C": 2})
        assert layout.is_right_of("B", "A") is True
        assert layout.is_right_of("C", "A") is True

    def test_unknown_entity_returns_none(self):
        layout = SpatialLayout(ordering=["A", "B"], coordinates={"A": 0, "B": 1})
        assert layout.is_left_of("A", "X") is None
        assert layout.is_right_of("X", "B") is None

    def test_position_of(self):
        layout = SpatialLayout(ordering=["A", "B"], coordinates={"A": 0, "B": 1})
        assert layout.position_of("A") == 0
        assert layout.position_of("Z") is None


# ─────────────────────────────────────────────
# TemporalLayout tests
# ─────────────────────────────────────────────


class TestTemporalLayout:
    def setup_method(self):
        # E1: 0-5, E2: 3-8 (overlaps E1), E3: 10-12 (after both)
        self.layout = TemporalLayout(intervals={
            "E1": {"start": 0.0, "end": 5.0},
            "E2": {"start": 3.0, "end": 8.0},
            "E3": {"start": 10.0, "end": 12.0},
        })

    def test_before(self):
        assert self.layout.before("E1", "E3") is True
        assert self.layout.before("E2", "E3") is True
        assert self.layout.before("E3", "E1") is False

    def test_after(self):
        assert self.layout.after("E3", "E1") is True
        assert self.layout.after("E1", "E3") is False

    def test_overlaps(self):
        assert self.layout.overlaps("E1", "E2") is True
        assert self.layout.overlaps("E1", "E3") is False
        assert self.layout.overlaps("E2", "E3") is False

    def test_during(self):
        inner = TemporalLayout(intervals={
            "outer": {"start": 0.0, "end": 10.0},
            "inner": {"start": 2.0, "end": 7.0},
        })
        assert inner.during("inner", "outer") is True
        assert inner.during("outer", "inner") is False

    def test_meets(self):
        meet = TemporalLayout(intervals={
            "A": {"start": 0.0, "end": 5.0},
            "B": {"start": 5.0, "end": 10.0},
        })
        assert meet.meets("A", "B") is True
        assert meet.meets("B", "A") is False

    def test_unknown_event(self):
        assert self.layout.before("E1", "X") is None


# ─────────────────────────────────────────────
# CausalLayout tests
# ─────────────────────────────────────────────


class TestCausalLayout:
    def setup_method(self):
        self.layout = CausalLayout(
            nodes=["switch", "circuit", "light"],
            edges=[
                {"from": "switch", "to": "circuit", "type": "causes"},
                {"from": "circuit", "to": "light", "type": "enables"},
            ],
            states={"switch": "on", "circuit": "closed", "light": "on"},
        )

    def test_get_children(self):
        assert self.layout.get_children("switch") == ["circuit"]
        assert self.layout.get_children("circuit") == ["light"]
        assert self.layout.get_children("light") == []

    def test_get_parents(self):
        assert self.layout.get_parents("light") == ["circuit"]
        assert self.layout.get_parents("switch") == []

    def test_reachable_from(self):
        reachable = self.layout.reachable_from("switch")
        assert "circuit" in reachable
        assert "light" in reachable
        assert "switch" not in reachable

    def test_topological_order(self):
        order = self.layout.topological_order()
        assert order.index("switch") < order.index("circuit")
        assert order.index("circuit") < order.index("light")


# ─────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────


class TestModel:
    def _make_model(self) -> Model:
        return Model(
            domain=Domain.SPATIAL,
            entities=[Entity(id="A"), Entity(id="B"), Entity(id="C")],
            relations=[
                Fact(predicate="left_of", args=["A", "B"], polarity=True,
                     provenance=Provenance.EXPLICIT),
                Fact(predicate="left_of", args=["B", "C"], polarity=True,
                     provenance=Provenance.EXPLICIT),
            ],
        )

    def test_get_fact_found(self):
        model = self._make_model()
        fact = model.get_fact("left_of", ["A", "B"])
        assert fact is not None
        assert fact.polarity is True

    def test_get_fact_not_found(self):
        model = self._make_model()
        fact = model.get_fact("left_of", ["C", "A"])
        assert fact is None

    def test_add_fact_new(self):
        model = self._make_model()
        new_fact = Fact(predicate="right_of", args=["B", "A"], polarity=True,
                        provenance=Provenance.DERIVED)
        model.add_fact(new_fact)
        assert model.get_fact("right_of", ["B", "A"]) is not None

    def test_add_fact_replace(self):
        model = self._make_model()
        # Replace existing fact
        replacement = Fact(predicate="left_of", args=["A", "B"], polarity=False,
                           provenance=Provenance.DERIVED)
        model.add_fact(replacement)
        fact = model.get_fact("left_of", ["A", "B"])
        assert fact is not None
        assert fact.polarity is False

    def test_copy_is_independent(self):
        model = self._make_model()
        copy = model.copy()
        copy.add_fact(Fact(predicate="left_of", args=["A", "C"], polarity=True,
                           provenance=Provenance.DERIVED))
        # Original should not be affected
        assert model.get_fact("left_of", ["A", "C"]) is None

    def test_has_entity(self):
        model = self._make_model()
        assert model.has_entity("A") is True
        assert model.has_entity("X") is False

    def test_without_fact(self):
        model = self._make_model()
        fact = model.relations[0]
        new_model = model.without_fact(fact)
        assert new_model.get_fact("left_of", ["A", "B"]) is None
        # Original unchanged
        assert model.get_fact("left_of", ["A", "B"]) is not None

    def test_without_entity(self):
        model = self._make_model()
        new_model = model.without_entity("A")
        assert not new_model.has_entity("A")
        # Facts involving A should be removed
        assert new_model.get_fact("left_of", ["A", "B"]) is None


# ─────────────────────────────────────────────
# ConstraintSet tests
# ─────────────────────────────────────────────


class TestConstraintSet:
    def _make_cs(self) -> ConstraintSet:
        return ConstraintSet(
            domain=Domain.PROPOSITIONAL,
            constraints=[
                Constraint(
                    type=ConstraintType.CONDITIONAL,
                    antecedent="raining",
                    consequent="ground_wet",
                ),
                Constraint(
                    type=ConstraintType.DISJUNCTION,
                    disjuncts=["soup", "salad"],
                    exclusive=True,
                ),
                Constraint(
                    type=ConstraintType.UNIVERSAL,
                    antecedent_pred="musician",
                    consequent_pred="creative",
                ),
            ],
            entities=["e1"],
            has_disjunctions=True,
        )

    def test_get_conditionals(self):
        cs = self._make_cs()
        conds = cs.get_conditionals()
        assert len(conds) == 1
        assert conds[0].antecedent == "raining"

    def test_get_disjunctions(self):
        cs = self._make_cs()
        disjs = cs.get_disjunctions()
        assert len(disjs) == 1
        assert disjs[0].exclusive is True

    def test_get_universals(self):
        cs = self._make_cs()
        univs = cs.get_universals()
        assert len(univs) == 1
        assert univs[0].antecedent_pred == "musician"

    def test_has_unexamined_branches_empty_models(self):
        cs = self._make_cs()
        # With no models, disjunction branches are unexamined
        assert cs.has_unexamined_branches([]) is True

    def test_has_unexamined_branches_with_models(self):
        cs = self._make_cs()
        # With 2 models for 2 disjuncts, fully covered
        # (simplified: just checks len(models) < total_branches)
        dummy_model = Model(domain=Domain.PROPOSITIONAL)
        assert cs.has_unexamined_branches([dummy_model, dummy_model]) is False
