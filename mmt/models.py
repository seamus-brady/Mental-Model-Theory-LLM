"""
Core data structures for the Mental Model Theory agent.

Design principles from the plan:
- Iconic layer is primary: spatial/temporal/causal layouts come first
- Polarity + open-world: absence means unknown, not false
- Provenance tracking: block confabulation
- Dual-world representation: factual vs counterfactual models
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────


class InconsistentConstraintsError(ValueError):
    """Raised when a set of constraints is detected to be logically inconsistent."""


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────


class Domain(str, Enum):
    PROPOSITIONAL = "propositional"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    QUANTIFICATIONAL = "quantificational"
    MIXED = "mixed"


class Judgment(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNDERDETERMINED = "underdetermined"
    INCONSISTENT_PREMISES = "inconsistent_premises"


class ConstraintType(str, Enum):
    CONDITIONAL = "conditional"
    COUNTERFACTUAL = "counterfactual"
    DISJUNCTION = "disjunction"
    EXCLUSIVE_DISJUNCTION = "exclusive_disjunction"
    CONJUNCTION = "conjunction"
    CAUSAL_CAUSES = "causal_causes"
    CAUSAL_ENABLES = "causal_enables"
    SPATIAL_RELATION = "spatial_relation"
    TEMPORAL_RELATION = "temporal_relation"
    UNIVERSAL = "universal"
    EXISTENTIAL = "existential"
    NEGATION = "negation"
    ATOMIC = "atomic"


class Provenance(str, Enum):
    EXPLICIT = "explicit"      # directly stated in a premise
    DERIVED = "derived"        # logically derived from explicit facts
    ICONIC = "iconic"          # derived from the iconic layout structure
    ASSUMED = "assumed"        # added without license (confabulation risk)


# ─────────────────────────────────────────────
# Iconic Layer
# ─────────────────────────────────────────────


class SpatialLayout(BaseModel):
    """
    Supramodal spatial layout model (Byrne & Johnson-Laird, 1989).
    Ordering is primary; coordinates derived from it.
    """
    model_config = ConfigDict(extra="forbid")

    ordering: list[str] = Field(default_factory=list, description="Left-to-right entity ordering")
    coordinates: dict[str, int] = Field(default_factory=dict, description="Entity -> position index")

    def derive_coordinates(self) -> None:
        """Sync coordinates from ordering."""
        self.coordinates = {e: i for i, e in enumerate(self.ordering)}

    def position_of(self, entity: str) -> Optional[int]:
        return self.coordinates.get(entity)

    def is_left_of(self, a: str, b: str) -> Optional[bool]:
        pa, pb = self.position_of(a), self.position_of(b)
        if pa is None or pb is None:
            return None
        return pa < pb

    def is_right_of(self, a: str, b: str) -> Optional[bool]:
        pa, pb = self.position_of(a), self.position_of(b)
        if pa is None or pb is None:
            return None
        return pa > pb


class TemporalLayout(BaseModel):
    """
    Interval-based timeline (Schaeken & Johnson-Laird, 1996).
    Allen's 13 temporal relations are computed from intervals.
    """
    model_config = ConfigDict(extra="forbid")

    intervals: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="event_id -> {start, end}"
    )

    def before(self, a: str, b: str) -> Optional[bool]:
        ia, ib = self.intervals.get(a), self.intervals.get(b)
        if not ia or not ib:
            return None
        return ia["end"] < ib["start"]

    def after(self, a: str, b: str) -> Optional[bool]:
        return self.before(b, a)

    def overlaps(self, a: str, b: str) -> Optional[bool]:
        ia, ib = self.intervals.get(a), self.intervals.get(b)
        if not ia or not ib:
            return None
        return ia["start"] < ib["end"] and ib["start"] < ia["end"]

    def during(self, a: str, b: str) -> Optional[bool]:
        """a is contained within b."""
        ia, ib = self.intervals.get(a), self.intervals.get(b)
        if not ia or not ib:
            return None
        return ib["start"] <= ia["start"] and ia["end"] <= ib["end"]

    def meets(self, a: str, b: str) -> Optional[bool]:
        """a ends exactly when b starts."""
        ia, ib = self.intervals.get(a), self.intervals.get(b)
        if not ia or not ib:
            return None
        return ia["end"] == ib["start"]


class CausalLayout(BaseModel):
    """
    Causal DAG for kinematic simulation (Goldvarg & Johnson-Laird, 2001).
    Deterministic cause-effect chains; states propagate forward.
    """
    model_config = ConfigDict(extra="forbid")

    nodes: list[str] = Field(default_factory=list)
    edges: list[dict[str, str]] = Field(
        default_factory=list,
        description="[{from, to, type: causes|enables}]"
    )
    states: dict[str, Any] = Field(default_factory=dict)

    def get_children(self, node: str) -> list[str]:
        return [e["to"] for e in self.edges if e["from"] == node]

    def get_parents(self, node: str) -> list[str]:
        return [e["from"] for e in self.edges if e["to"] == node]

    def reachable_from(self, source: str) -> set[str]:
        """All nodes causally reachable from source."""
        visited: set[str] = set()
        stack = [source]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(self.get_children(node))
        visited.discard(source)
        return visited

    def topological_order(self) -> list[str]:
        """Kahn's algorithm for topological sort."""
        in_degree: dict[str, int] = {n: 0 for n in self.nodes}
        for edge in self.edges:
            in_degree[edge["to"]] = in_degree.get(edge["to"], 0) + 1
        queue = [n for n in self.nodes if in_degree[n] == 0]
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for child in self.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return result


class IconicLayer(BaseModel):
    """
    Domain-specific iconic representations.
    The iconic layer is primary: predicates are DERIVED from it, not added to it.
    """
    model_config = ConfigDict(extra="forbid")

    spatial: Optional[SpatialLayout] = None
    temporal: Optional[TemporalLayout] = None
    causal: Optional[CausalLayout] = None

    @property
    def is_empty(self) -> bool:
        return self.spatial is None and self.temporal is None and self.causal is None


# ─────────────────────────────────────────────
# Symbolic Layer
# ─────────────────────────────────────────────


class Fact(BaseModel):
    """
    A single fact in the symbolic layer.
    Polarity: True = asserted true, False = asserted false.
    Open-world: absence of a fact means UNKNOWN, not false.
    """
    model_config = ConfigDict(extra="forbid")

    predicate: str
    args: list[str]
    polarity: bool
    provenance: Provenance = Provenance.EXPLICIT
    source_premise: Optional[int] = None


class Entity(BaseModel):
    """An entity (individual) in the model."""
    model_config = ConfigDict(extra="forbid")

    id: str
    entity_type: str = "generic"
    properties: dict[str, bool] = Field(default_factory=dict)
    skolem: bool = False  # True if introduced by existential (Skolem entity)


class SchemaRule(BaseModel):
    """A universal rule: forall x: antecedent(x) -> consequent(x)."""
    model_config = ConfigDict(extra="forbid")

    antecedent_pred: str
    consequent_pred: str


class ConditionalTrigger(BaseModel):
    """If antecedent holds then consequent must hold."""
    model_config = ConfigDict(extra="forbid")

    antecedent_pred: str
    antecedent_args: list[str]
    consequent_pred: str
    consequent_args: list[str]


# ─────────────────────────────────────────────
# Mental Model
# ─────────────────────────────────────────────


class Model(BaseModel):
    """
    A single mental model representing one possible state of affairs.

    Structure:
    - iconic_layer: primary spatial/temporal/causal layout
    - entities: individuals introduced
    - relations: explicit facts (polarity + provenance)
    - schema_rules: universal rules (for quantificational reasoning)
    - conditional_triggers: if-then rules active in this model
    - implicit_note: what cases are left unspecified (principle of truth)
    """
    model_config = ConfigDict(extra="forbid")

    domain: Domain
    iconic_layer: IconicLayer = Field(default_factory=IconicLayer)
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Fact] = Field(default_factory=list)
    schema_rules: list[SchemaRule] = Field(default_factory=list)
    conditional_triggers: list[ConditionalTrigger] = Field(default_factory=list)
    is_fleshed_out: bool = False
    implicit_note: str = ""
    # Counterfactual support: maintain factual + counterfactual worlds
    world_label: str = "actual"   # "actual" or "counterfactual"

    def get_fact(self, predicate: str, args: list[str]) -> Optional[Fact]:
        for fact in self.relations:
            if fact.predicate == predicate and fact.args == args:
                return fact
        return None

    def add_fact(self, fact: Fact) -> None:
        """Add or replace a fact."""
        for i, f in enumerate(self.relations):
            if f.predicate == fact.predicate and f.args == fact.args:
                self.relations[i] = fact
                return
        self.relations.append(fact)

    def has_entity(self, entity_id: str) -> bool:
        return any(e.id == entity_id for e in self.entities)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        for e in self.entities:
            if e.id == entity_id:
                return e
        return None

    def copy(self) -> "Model":
        return self.model_copy(deep=True)

    def without_fact(self, fact: Fact) -> "Model":
        m = self.copy()
        m.relations = [f for f in m.relations if not (
            f.predicate == fact.predicate and f.args == fact.args
        )]
        return m

    def without_entity(self, entity_id: str) -> "Model":
        m = self.copy()
        m.entities = [e for e in m.entities if e.id != entity_id]
        m.relations = [
            f for f in m.relations
            if entity_id not in f.args
        ]
        return m


# ─────────────────────────────────────────────
# Constraints (extracted from premises)
# ─────────────────────────────────────────────


class Constraint(BaseModel):
    """A single logical constraint extracted from a premise."""
    model_config = ConfigDict(extra="forbid")

    type: ConstraintType

    # Conditional / Counterfactual: IF antecedent THEN consequent
    antecedent: Optional[str] = None
    consequent: Optional[str] = None

    # Disjunction: A or B [or C...]
    disjuncts: Optional[list[str]] = None
    exclusive: bool = False  # exclusive disjunction

    # Spatial: entity1 left_of entity2
    entity1: Optional[str] = None
    relation: Optional[str] = None   # left_of, right_of, before, after, overlaps, causes, enables
    entity2: Optional[str] = None

    # Universal: forall x: antecedent_pred(x) -> consequent_pred(x)
    antecedent_pred: Optional[str] = None
    consequent_pred: Optional[str] = None

    # Existential / Atomic: entities and facts
    entities: Optional[list[str]] = None
    properties: Optional[list[str]] = None
    predicate: Optional[str] = None
    args: Optional[list[str]] = None
    polarity: Optional[bool] = None

    # Cause/effect
    cause: Optional[str] = None
    effect: Optional[str] = None
    causal_type: Optional[str] = None  # "causes" or "enables"


class ConstraintSet(BaseModel):
    """
    The full set of constraints extracted from a set of premises.
    This is the output of the SemanticCompiler.
    """
    model_config = ConfigDict(extra="forbid")

    domain: Domain
    constraints: list[Constraint] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    background_knowledge: list[str] = Field(default_factory=list)
    has_disjunctions: bool = False
    has_counterfactuals: bool = False
    has_negated_conditionals: bool = False
    matches_known_illusion: bool = False
    raw_premises: list[str] = Field(default_factory=list)

    def get_spatial_constraints(self) -> list[Constraint]:
        return [c for c in self.constraints if c.type == ConstraintType.SPATIAL_RELATION]

    def get_temporal_constraints(self) -> list[Constraint]:
        return [c for c in self.constraints if c.type == ConstraintType.TEMPORAL_RELATION]

    def get_causal_constraints(self) -> list[Constraint]:
        return [c for c in self.constraints
                if c.type in (ConstraintType.CAUSAL_CAUSES, ConstraintType.CAUSAL_ENABLES)]

    def get_disjunctions(self) -> list[Constraint]:
        return [c for c in self.constraints
                if c.type in (ConstraintType.DISJUNCTION, ConstraintType.EXCLUSIVE_DISJUNCTION)]

    def get_conditionals(self) -> list[Constraint]:
        return [c for c in self.constraints if c.type == ConstraintType.CONDITIONAL]

    def get_universals(self) -> list[Constraint]:
        return [c for c in self.constraints if c.type == ConstraintType.UNIVERSAL]

    def has_unexamined_branches(self, examined_models: list[Model]) -> bool:
        """Check if disjunctions have unexplored branches."""
        disjs = self.get_disjunctions()
        if not disjs:
            return False
        total_branches = sum(len(c.disjuncts or []) for c in disjs)
        return len(examined_models) < total_branches


# ─────────────────────────────────────────────
# Reasoning Results
# ─────────────────────────────────────────────


class Violation(BaseModel):
    """A constraint violation found in a model."""
    model_config = ConfigDict(extra="forbid")

    type: str
    detail: str
    severity: str = "warning"  # "warning" or "block"


class ConsistencyResult(BaseModel):
    """Result of checking a model's consistency."""
    model_config = ConfigDict(extra="forbid")

    consistent: bool
    violations: list[Violation] = Field(default_factory=list)


class SearchResult(BaseModel):
    """Result of a counterexample search."""
    model_config = ConfigDict(extra="forbid")

    found: bool
    counterexample: Optional[Model] = None
    method: str = ""
    coverage: int = 0  # number of models examined


class ReasoningResult(BaseModel):
    """
    Final output of the MentalModelAgent.reason() method.

    judgment: VALID | INVALID | UNDERDETERMINED | INCONSISTENT_PREMISES
    explanation: natural language explanation
    models_checked: number of models examined
    counterexample: a model falsifying the conclusion (if INVALID)
    supporting_model: a model supporting the conclusion (if VALID)
    system_used: "system1" or "system2"
    """
    model_config = ConfigDict(extra="forbid")

    judgment: Judgment
    explanation: str
    models_checked: int
    counterexample: Optional[Model] = None
    supporting_model: Optional[Model] = None
    system_used: str = "system1"
    confidence: float = 1.0


# ─────────────────────────────────────────────
# LLM Output Schemas (structured output models)
# ─────────────────────────────────────────────


class LLMConstraint(BaseModel):
    """Simplified constraint schema for LLM structured output."""
    model_config = ConfigDict(extra="ignore")

    type: str
    antecedent: Optional[str] = None
    consequent: Optional[str] = None
    disjuncts: Optional[list[str]] = None
    exclusive: Optional[bool] = None
    entity1: Optional[str] = None
    relation: Optional[str] = None
    entity2: Optional[str] = None
    antecedent_pred: Optional[str] = None
    consequent_pred: Optional[str] = None
    properties: Optional[list[str]] = None
    predicate: Optional[str] = None
    args: Optional[list[str]] = None
    polarity: Optional[bool] = None
    cause: Optional[str] = None
    effect: Optional[str] = None
    causal_type: Optional[str] = None


class LLMConstraintSet(BaseModel):
    """LLM output schema for the SemanticCompiler."""
    model_config = ConfigDict(extra="ignore")

    domain: str
    constraints: list[LLMConstraint]
    entities: list[str]
    has_disjunctions: bool
    has_counterfactuals: bool
    has_negated_conditionals: bool
    matches_known_illusion: bool
    reasoning_notes: str


class LLMSpatialModel(BaseModel):
    """LLM output for spatial domain model construction."""
    model_config = ConfigDict(extra="ignore")

    ordering: list[str]
    entity_types: dict[str, str]
    implicit_note: str


class LLMTemporalModel(BaseModel):
    """LLM output for temporal domain model construction."""
    model_config = ConfigDict(extra="ignore")

    events: list[str]
    event_start: list[float]
    event_end: list[float]
    entity_types: dict[str, str]
    implicit_note: str


class LLMCausalModel(BaseModel):
    """LLM output for causal domain model construction."""
    model_config = ConfigDict(extra="ignore")

    nodes: list[str]
    edge_from: list[str]
    edge_to: list[str]
    edge_type: list[str]
    initial_states: dict[str, str]
    entity_types: dict[str, str]
    implicit_note: str


class LLMEntityOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    entity_type: str
    properties_true: list[str]


class LLMFactOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    predicate: str
    args: list[str]
    polarity: bool


class LLMPropositionalModel(BaseModel):
    """LLM output for propositional/quantificational model construction."""
    model_config = ConfigDict(extra="ignore")

    entities: list[LLMEntityOut]
    facts: list[LLMFactOut]
    implicit_note: str


class LLMCounterexampleAttempt(BaseModel):
    """LLM attempt to construct a counterexample."""
    model_config = ConfigDict(extra="ignore")

    possible: bool
    blocking_constraint: Optional[str] = None
    model: Optional[LLMPropositionalModel] = None
    reasoning: str


class LLMQueryParse(BaseModel):
    """LLM parse of a query string into a structured predicate."""
    model_config = ConfigDict(extra="ignore")

    predicate: str
    args: list[str]
    polarity: bool
    query_type: str  # "atomic", "existential", "universal"
    subject: Optional[str] = None
    property_pred: Optional[str] = None
