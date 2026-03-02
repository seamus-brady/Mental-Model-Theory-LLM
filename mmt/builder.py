"""
Model Builder: Constructs mental models from a ConstraintSet.

Architecture:
- Spatial/Temporal/Causal domains: ICONIC-FIRST construction
  → LLM proposes layout → Python derives predicates from structure
- Propositional/Quantificational: LLM proposes entities + facts
  → ConstraintChecker validates

Key principles:
- Minimality: add as few entities/facts as possible (Principle of Truth)
- Iconicity: for layout domains, read inferences off structure
- Bounded resources: System1/System2 configs limit model count

Reference: Khemlani & Johnson-Laird (2021), Johnson-Laird (1983)
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import anthropic

from ._logging import get_logger
from .checker import ConstraintChecker, ProvenanceEnforcer
from .models import (
    CausalLayout,
    Constraint,
    ConstraintSet,
    ConstraintType,
    ConditionalTrigger,
    Domain,
    Entity,
    Fact,
    IconicLayer,
    InconsistentConstraintsError,
    LLMCausalModel,
    LLMPropositionalModel,
    LLMSpatialModel,
    LLMTemporalModel,
    Model,
    Provenance,
    SchemaRule,
    SpatialLayout,
    TemporalLayout,
)

if TYPE_CHECKING:
    pass

_logger = get_logger(__name__)

# ─────────────────────────────────────────────
# System 1 / System 2 configurations
# ─────────────────────────────────────────────

SYSTEM1_CONFIG = {
    "max_models": 1,
    "max_entities_per_type": 3,
    "flesh_out_negations": False,
    "branch_on_disjunction": False,  # take first disjunct only
    "counterexample_search": False,
}

SYSTEM2_CONFIG = {
    "max_models": 8,
    "max_entities_per_type": 6,
    "flesh_out_negations": True,
    "branch_on_disjunction": True,
    "counterexample_search": True,
    "max_search_depth": 3,
}

# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

_SPATIAL_MODEL_SYSTEM = """\
You are constructing a SPATIAL mental model following Johnson-Laird's Principle of Truth.

Rules:
1. ICONIC FIRST: Give the left-to-right ordering of entities.
2. MINIMAL: Only include entities mentioned or required.
3. PRINCIPLE OF TRUTH: Only represent explicitly true spatial relations.
4. OPEN WORLD: Do not infer relations not stated.

The ordering is the PRIMARY representation. Predicates like left_of are DERIVED from it.

OUTPUT FORMAT: Return ONLY a valid JSON object, no markdown, no code fences.
{
  "ordering": ["entity1", "entity2", ...],
  "entity_types": {"entity1": "object", ...},
  "implicit_note": "description of what is left unspecified"
}
"""

_SPATIAL_MODEL_USER = """\
Spatial constraints:
{constraints}

Entities: {entities}

Construct the INITIAL spatial model (System 1: one minimal model, principle of truth).
Give entities a left-to-right ordering consistent with the constraints.
Use implicit_note to say what is left unspecified.
"""

_TEMPORAL_MODEL_SYSTEM = """\
You are constructing a TEMPORAL mental model following Johnson-Laird's Principle of Truth.

Rules:
1. ICONIC FIRST: Give events as intervals {start, end} on a number line.
2. MINIMAL: Only include events mentioned or required.
3. Use concrete numbers (e.g., start=0, end=5 for event lasting 5 units).
4. Allen's relations (before, overlaps, during) are DERIVED from intervals.

OUTPUT FORMAT: Return ONLY a valid JSON object, no markdown, no code fences.
{
  "events": ["event1", "event2", ...],
  "event_start": [0, 5, ...],
  "event_end": [3, 10, ...],
  "entity_types": {"event1": "event", ...},
  "implicit_note": "description of what is left unspecified"
}
events, event_start, and event_end must all be the same length.
"""

_TEMPORAL_MODEL_USER = """\
Temporal constraints:
{constraints}

Entities/Events: {entities}

Construct the INITIAL temporal model. Assign intervals consistent with the constraints.
events, event_start, and event_end must all be the same length.
"""

_CAUSAL_MODEL_SYSTEM = """\
You are constructing a CAUSAL mental model following Johnson-Laird's Principle of Truth.

Rules:
1. ICONIC FIRST: Give the causal DAG (nodes + directed edges).
2. edge_type: "causes" (sufficient) or "enables" (necessary but not sufficient)
3. initial_states: current state of each node ("on"/"off", "true"/"false", etc.)
4. MINIMAL: Only include what is stated.

OUTPUT FORMAT: Return ONLY a valid JSON object, no markdown, no code fences.
{
  "nodes": ["A", "B", ...],
  "edge_from": ["A", ...],
  "edge_to": ["B", ...],
  "edge_type": ["causes" or "enables", ...],
  "initial_states": {"A": "on", ...},
  "entity_types": {"A": "node", ...},
  "implicit_note": "description of what is left unspecified"
}
edge_from, edge_to, and edge_type must all be the same length.
"""

_CAUSAL_MODEL_USER = """\
Causal constraints:
{constraints}

Entities: {entities}

Construct the INITIAL causal model. Include all mentioned nodes and edges.
edge_from, edge_to, and edge_type must all be the same length.
"""

_PROPOSITIONAL_MODEL_SYSTEM = """\
You are constructing a PROPOSITIONAL mental model following Johnson-Laird's Principle of Truth.

Rules:
1. MINIMAL ENTITIES: Introduce only the individuals required.
2. PRINCIPLE OF TRUTH: Only add facts that are explicitly true in the premises.
3. OPEN WORLD: Leave unmentioned properties unspecified (not false).
4. SKOLEM: For "Some A is B", create one individual that is both A and B.
5. For universals "All A are B", add a schema rule and ONE witness entity.
6. Conditional "If A then B": initial model shows A∧B; leave ¬A cases implicit.

OUTPUT FORMAT: Return ONLY a valid JSON object, no markdown, no code fences.
{
  "entities": [
    {"id": "e1", "entity_type": "person", "properties_true": ["artist", "beekeeper"]}
  ],
  "facts": [
    {"predicate": "is_artist", "args": ["e1"], "polarity": true}
  ],
  "implicit_note": "description of what is left unspecified"
}
"""

_PROPOSITIONAL_MODEL_USER = """\
Constraints:
{constraints}

Entities introduced: {entities}

Construct the INITIAL propositional model (System 1: single model, principle of truth).
Use implicit_note to describe what cases are left unspecified.
"""

_FLESH_OUT_SYSTEM = """\
You are expanding a mental model to make ALL implicit possibilities explicit (System 2 / deliberative reasoning).

For each conditional "If A then B", add the missing cases:
  - ¬A ∧ B (A false, B true — allowed)
  - ¬A ∧ ¬B (both false — allowed)
  Note: A ∧ ¬B is EXCLUDED (violates the conditional).

For each disjunction "A or B", create separate models for each branch.

Return the ADDITIONAL entities and facts needed to flesh out the model.
Do not duplicate facts already in the current model.
"""

_FLESH_OUT_USER = """\
Current model:
{model_json}

Constraints to flesh out:
{constraints}

What additional facts complete the implicit cases?
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


# ─────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────


class ModelBuilder:
    """
    Builds mental models from a ConstraintSet.

    For spatial/temporal/causal: iconic-first construction.
    For propositional/quantificational: LLM-guided symbolic construction.
    """

    def __init__(self, client: anthropic.Anthropic, model_id: str = "claude-opus-4-6"):
        self.client = client
        self.model_id = model_id
        self.checker = ConstraintChecker()
        self.enforcer = ProvenanceEnforcer()

    def construct(
        self,
        constraints: ConstraintSet,
        mode: str = "system1",
        existing_models: list[Model] | None = None,
    ) -> list[Model]:
        """
        Build mental models from a constraint set.

        Args:
            constraints: Extracted constraint set
            mode: "system1" (fast, single model) or "system2" (deliberative)
            existing_models: Models already built (for System 2 expansion)

        Returns:
            List of mental models
        """
        config = SYSTEM1_CONFIG if mode == "system1" else SYSTEM2_CONFIG
        domain = constraints.domain

        if domain == Domain.SPATIAL:
            models = self._construct_spatial(constraints, config)
        elif domain == Domain.TEMPORAL:
            models = self._construct_temporal(constraints, config)
        elif domain == Domain.CAUSAL:
            models = self._construct_causal(constraints, config)
        else:
            # Propositional, quantificational, or mixed
            models = self._construct_propositional(constraints, config)

        # System 2: flesh out alternatives from conditionals/disjunctions
        if mode == "system2" and config.get("flesh_out_negations"):
            models = self._flesh_out_alternatives(models, constraints, config)

        # Apply minimality (strip unsupported assumed facts)
        models = [self._minimize(m, constraints) for m in models]

        return models[: config["max_models"]]

    # ─── Spatial (iconic-first) ───────────────────────────────────────────

    @staticmethod
    def _detect_spatial_cycle(spatial_cs: list[Constraint]) -> bool:
        """
        Return True if the spatial constraints form a directed cycle (contradiction).

        Builds a directed graph where an edge A→B means "A is left of B" and
        checks for cycles via DFS. A cycle (e.g. A<B, B<A or A<B, B<C, C<A)
        means the premises are contradictory.
        """
        # Build left-of adjacency: A -> {entities that A is left of}
        graph: dict[str, set[str]] = {}
        for c in spatial_cs:
            e1, rel, e2 = c.entity1, (c.relation or ""), c.entity2
            if not e1 or not e2:
                continue
            rel_lower = rel.lower().replace(" ", "_")
            if "left" in rel_lower:
                graph.setdefault(e1, set()).add(e2)
            elif "right" in rel_lower:
                graph.setdefault(e2, set()).add(e1)

        visited: set[str] = set()
        rec_stack: set[str] = set()

        def _has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if _has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        for node in list(graph.keys()):
            if node not in visited:
                if _has_cycle(node):
                    return True
        return False

    def _construct_spatial(
        self, constraints: ConstraintSet, config: dict
    ) -> list[Model]:
        spatial_cs = [
            c for c in constraints.constraints
            if c.type == ConstraintType.SPATIAL_RELATION
        ]

        if self._detect_spatial_cycle(spatial_cs):
            raise InconsistentConstraintsError(
                "Spatial constraints are contradictory: left-of ordering forms a cycle."
            )

        cs_str = self._format_constraints(spatial_cs)

        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=2048,
            system=_SPATIAL_MODEL_SYSTEM,
            messages=[{"role": "user", "content": _SPATIAL_MODEL_USER.format(
                constraints=cs_str,
                entities=", ".join(constraints.entities),
            )}],
        )
        raw = _parse_json_response(response)
        llm_out = LLMSpatialModel.model_validate(raw)

        return [self._build_from_spatial(llm_out, constraints)]

    def _build_from_spatial(
        self, llm_out: LLMSpatialModel, constraints: ConstraintSet
    ) -> Model:
        """Build a Model from a spatial LLM output, deriving predicates from ordering."""
        ordering = llm_out.ordering

        spatial = SpatialLayout(ordering=ordering)
        spatial.derive_coordinates()

        # Derive ALL spatial predicates from iconic structure
        relations = _derive_spatial_predicates(ordering)

        entities = [
            Entity(id=e, entity_type=llm_out.entity_types.get(e, "object"))
            for e in ordering
        ]

        # Add schema rules / conditional triggers from constraints
        schema_rules, cond_triggers = _extract_rules_from_constraints(constraints)

        return Model(
            domain=Domain.SPATIAL,
            iconic_layer=IconicLayer(
                spatial=spatial
            ),
            entities=entities,
            relations=relations,
            schema_rules=schema_rules,
            conditional_triggers=cond_triggers,
            is_fleshed_out=False,
            implicit_note=llm_out.implicit_note,
        )

    # ─── Temporal (iconic-first) ──────────────────────────────────────────

    def _construct_temporal(
        self, constraints: ConstraintSet, config: dict
    ) -> list[Model]:
        temporal_cs = [
            c for c in constraints.constraints
            if c.type == ConstraintType.TEMPORAL_RELATION
        ]
        cs_str = self._format_constraints(temporal_cs)

        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=2048,
            system=_TEMPORAL_MODEL_SYSTEM,
            messages=[{"role": "user", "content": _TEMPORAL_MODEL_USER.format(
                constraints=cs_str,
                entities=", ".join(constraints.entities),
            )}],
        )
        raw = _parse_json_response(response)
        llm_out = LLMTemporalModel.model_validate(raw)

        return [self._build_from_temporal(llm_out, constraints)]

    def _build_from_temporal(
        self, llm_out: LLMTemporalModel, constraints: ConstraintSet
    ) -> Model:
        events = llm_out.events
        starts = llm_out.event_start
        ends = llm_out.event_end

        intervals = {}
        for ev, s, e in zip(events, starts, ends):
            intervals[ev] = {"start": s, "end": e}

        temporal = TemporalLayout(intervals=intervals)
        relations = _derive_temporal_predicates(temporal, events)

        entities = [
            Entity(id=ev, entity_type=llm_out.entity_types.get(ev, "event"))
            for ev in events
        ]

        schema_rules, cond_triggers = _extract_rules_from_constraints(constraints)

        return Model(
            domain=Domain.TEMPORAL,
            iconic_layer=IconicLayer(temporal=temporal),
            entities=entities,
            relations=relations,
            schema_rules=schema_rules,
            conditional_triggers=cond_triggers,
            is_fleshed_out=False,
            implicit_note=llm_out.implicit_note,
        )

    # ─── Causal (iconic-first) ────────────────────────────────────────────

    def _construct_causal(
        self, constraints: ConstraintSet, config: dict
    ) -> list[Model]:
        causal_cs = [
            c for c in constraints.constraints
            if c.type in (ConstraintType.CAUSAL_CAUSES, ConstraintType.CAUSAL_ENABLES)
        ]
        cs_str = self._format_constraints(causal_cs)

        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=2048,
            system=_CAUSAL_MODEL_SYSTEM,
            messages=[{"role": "user", "content": _CAUSAL_MODEL_USER.format(
                constraints=cs_str,
                entities=", ".join(constraints.entities),
            )}],
        )
        raw = _parse_json_response(response)
        llm_out = LLMCausalModel.model_validate(raw)

        return [self._build_from_causal(llm_out, constraints)]

    def _build_from_causal(
        self, llm_out: LLMCausalModel, constraints: ConstraintSet
    ) -> Model:
        edges = [
            {"from": f, "to": t, "type": ty}
            for f, t, ty in zip(
                llm_out.edge_from, llm_out.edge_to, llm_out.edge_type
            )
        ]
        causal = CausalLayout(
            nodes=llm_out.nodes,
            edges=edges,
            states=llm_out.initial_states,
        )
        relations = _derive_causal_predicates(causal)
        entities = [
            Entity(id=n, entity_type=llm_out.entity_types.get(n, "node"))
            for n in llm_out.nodes
        ]
        schema_rules, cond_triggers = _extract_rules_from_constraints(constraints)

        return Model(
            domain=Domain.CAUSAL,
            iconic_layer=IconicLayer(causal=causal),
            entities=entities,
            relations=relations,
            schema_rules=schema_rules,
            conditional_triggers=cond_triggers,
            is_fleshed_out=False,
            implicit_note=llm_out.implicit_note,
        )

    # ─── Propositional / Quantificational ────────────────────────────────

    def _construct_propositional(
        self, constraints: ConstraintSet, config: dict
    ) -> list[Model]:
        cs_str = self._format_constraints(constraints.constraints)

        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=2048,
            system=_PROPOSITIONAL_MODEL_SYSTEM,
            messages=[{"role": "user", "content": _PROPOSITIONAL_MODEL_USER.format(
                constraints=cs_str,
                entities=", ".join(constraints.entities) or "none yet",
            )}],
        )
        try:
            raw = _parse_json_response(response)
            llm_out = LLMPropositionalModel.model_validate(raw)
        except Exception:
            _logger.exception(
                "ModelBuilder._construct_propositional: failed to parse LLM response"
            )
            return []

        return [self._build_from_propositional(llm_out, constraints)]

    def _build_from_propositional(
        self, llm_out: LLMPropositionalModel, constraints: ConstraintSet
    ) -> Model:
        entities = []
        for e in llm_out.entities:
            props = {p: True for p in e.properties_true}
            entities.append(Entity(
                id=e.id,
                entity_type=e.entity_type,
                properties=props,
            ))

        relations = []
        for f in llm_out.facts:
            relations.append(Fact(
                predicate=f.predicate,
                args=f.args,
                polarity=f.polarity,
                provenance=Provenance.EXPLICIT,
            ))

        schema_rules, cond_triggers = _extract_rules_from_constraints(constraints)

        domain = constraints.domain
        if domain == Domain.MIXED:
            domain = Domain.PROPOSITIONAL

        return Model(
            domain=domain,
            iconic_layer=IconicLayer(),
            entities=entities,
            relations=relations,
            schema_rules=schema_rules,
            conditional_triggers=cond_triggers,
            is_fleshed_out=False,
            implicit_note=llm_out.implicit_note,
        )

    # ─── System 2: flesh out alternatives ────────────────────────────────

    def _flesh_out_alternatives(
        self,
        models: list[Model],
        constraints: ConstraintSet,
        config: dict,
    ) -> list[Model]:
        """
        Expand models by making implicit possibilities explicit.
        For conditionals: add ¬A∧B and ¬A∧¬B models.
        For disjunctions: branch into separate models per disjunct.
        """
        result = list(models)
        max_models = config.get("max_models", 8)

        # Branch disjunctions
        if config.get("branch_on_disjunction", True):
            for model in list(result):
                if len(result) >= max_models:
                    break
                branches = self._branch_disjunctions(model, constraints)
                for branch in branches:
                    if len(result) < max_models:
                        branch.is_fleshed_out = True
                        result.append(branch)

        # Flesh out conditionals
        for model in list(result):
            if len(result) >= max_models:
                break
            cond_expansions = self._expand_conditionals(model, constraints)
            for exp in cond_expansions:
                if len(result) < max_models:
                    exp.is_fleshed_out = True
                    result.append(exp)

        return result

    def _branch_disjunctions(
        self, model: Model, constraints: ConstraintSet
    ) -> list[Model]:
        """Create one model per disjunct branch."""
        branches = []
        for c in constraints.get_disjunctions():
            if not c.disjuncts:
                continue
            for disjunct in c.disjuncts:
                branch = model.copy()
                # Add the disjunct as true
                pred = disjunct.replace(" ", "_").lower()
                branch.add_fact(Fact(
                    predicate=pred,
                    args=[],
                    polarity=True,
                    provenance=Provenance.DERIVED,
                ))
                # For exclusive disjunction, negate others
                if c.exclusive:
                    for other in c.disjuncts:
                        if other != disjunct:
                            other_pred = other.replace(" ", "_").lower()
                            branch.add_fact(Fact(
                                predicate=other_pred,
                                args=[],
                                polarity=False,
                                provenance=Provenance.DERIVED,
                            ))
                branches.append(branch)
        return branches

    def _expand_conditionals(
        self, model: Model, constraints: ConstraintSet
    ) -> list[Model]:
        """
        For each conditional "If A then B", add:
        - ¬A ∧ B model
        - ¬A ∧ ¬B model
        (Note: A ∧ ¬B is excluded — violates conditional)
        """
        expansions = []
        for c in constraints.get_conditionals():
            if not c.antecedent or not c.consequent:
                continue
            ant = c.antecedent.replace(" ", "_").lower()
            con = c.consequent.replace(" ", "_").lower()

            # ¬A ∧ B
            alt1 = model.copy()
            alt1.add_fact(Fact(predicate=ant, args=[], polarity=False,
                               provenance=Provenance.DERIVED))
            alt1.add_fact(Fact(predicate=con, args=[], polarity=True,
                               provenance=Provenance.DERIVED))
            expansions.append(alt1)

            # ¬A ∧ ¬B
            alt2 = model.copy()
            alt2.add_fact(Fact(predicate=ant, args=[], polarity=False,
                               provenance=Provenance.DERIVED))
            alt2.add_fact(Fact(predicate=con, args=[], polarity=False,
                               provenance=Provenance.DERIVED))
            expansions.append(alt2)

        return expansions

    # ─── Minimality ───────────────────────────────────────────────────────

    def _minimize(self, model: Model, constraints: ConstraintSet) -> Model:
        """
        Remove assumed facts that are unnecessary for satisfying constraints.
        Implements structural minimality enforcement.
        """
        violations = self.enforcer.validate_model(model, constraints)
        if violations:
            model = self.enforcer.strip_unsupported(model, violations)
        return model

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _format_constraints(self, constraints: list[Constraint]) -> str:
        lines = []
        for c in constraints:
            lines.append(f"- type={c.type.value}: {c.model_dump(exclude_none=True)}")
        return "\n".join(lines) if lines else "(none)"


# ─────────────────────────────────────────────
# Iconic predicate derivation (pure Python)
# ─────────────────────────────────────────────


def _derive_spatial_predicates(ordering: list[str]) -> list[Fact]:
    """
    Derive all spatial predicates from the iconic ordering.
    This is "reading off" the structure — not logical deduction.
    """
    relations = []
    for i, e1 in enumerate(ordering):
        for j, e2 in enumerate(ordering):
            if i == j:
                continue
            if i < j:
                relations.append(Fact(
                    predicate="left_of",
                    args=[e1, e2],
                    polarity=True,
                    provenance=Provenance.ICONIC,
                ))
                relations.append(Fact(
                    predicate="right_of",
                    args=[e2, e1],
                    polarity=True,
                    provenance=Provenance.ICONIC,
                ))
    return relations


def _derive_temporal_predicates(
    temporal: TemporalLayout, events: list[str]
) -> list[Fact]:
    """Derive temporal predicates from interval structure."""
    relations = []
    for i, e1 in enumerate(events):
        for j, e2 in enumerate(events):
            if i == j:
                continue
            before = temporal.before(e1, e2)
            if before is True:
                relations.append(Fact(
                    predicate="before",
                    args=[e1, e2],
                    polarity=True,
                    provenance=Provenance.ICONIC,
                ))
                relations.append(Fact(
                    predicate="after",
                    args=[e2, e1],
                    polarity=True,
                    provenance=Provenance.ICONIC,
                ))
            overlaps = temporal.overlaps(e1, e2)
            if overlaps is True and i < j:
                relations.append(Fact(
                    predicate="overlaps",
                    args=[e1, e2],
                    polarity=True,
                    provenance=Provenance.ICONIC,
                ))
            during = temporal.during(e1, e2)
            if during is True:
                relations.append(Fact(
                    predicate="during",
                    args=[e1, e2],
                    polarity=True,
                    provenance=Provenance.ICONIC,
                ))
    return relations


def _derive_causal_predicates(causal: CausalLayout) -> list[Fact]:
    """Derive causal predicates from DAG structure (transitive closure)."""
    relations = []

    # Direct edges
    for edge in causal.edges:
        pred = "causes" if edge.get("type") == "causes" else "enables"
        relations.append(Fact(
            predicate=pred,
            args=[edge["from"], edge["to"]],
            polarity=True,
            provenance=Provenance.ICONIC,
        ))

    # Transitive influence
    for node in causal.nodes:
        reachable = causal.reachable_from(node)
        for target in reachable:
            # Avoid duplicating direct edges
            if not any(
                f.predicate in ("causes", "enables")
                and f.args == [node, target]
                for f in relations
            ):
                relations.append(Fact(
                    predicate="causally_influences",
                    args=[node, target],
                    polarity=True,
                    provenance=Provenance.ICONIC,
                ))

    return relations


# ─────────────────────────────────────────────
# Helper: extract schema rules from constraints
# ─────────────────────────────────────────────


def _extract_rules_from_constraints(
    constraints: ConstraintSet,
) -> tuple[list[SchemaRule], list[ConditionalTrigger]]:
    """Extract schema rules and conditional triggers from a ConstraintSet."""
    schema_rules = []
    cond_triggers = []

    for c in constraints.constraints:
        if c.type == ConstraintType.UNIVERSAL and c.antecedent_pred and c.consequent_pred:
            schema_rules.append(SchemaRule(
                antecedent_pred=c.antecedent_pred,
                consequent_pred=c.consequent_pred,
            ))

        if c.type == ConstraintType.CONDITIONAL and c.antecedent and c.consequent:
            ant = c.antecedent.replace(" ", "_").lower()
            con = c.consequent.replace(" ", "_").lower()
            cond_triggers.append(ConditionalTrigger(
                antecedent_pred=ant,
                antecedent_args=[],
                consequent_pred=con,
                consequent_args=[],
            ))

    return schema_rules, cond_triggers
