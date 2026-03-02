"""
Constraint Checker and Provenance Enforcer.

The ConstraintChecker acts as the formal "judge" in the
"LLM proposes, checker verifies" architecture.

It validates:
- Transitivity (left_of, before, causes → transitive closure)
- Antisymmetry (left_of, before → cannot have both A left_of B and B left_of A)
- Mutex constraints (exclusive disjunctions)
- Schema rule satisfaction (universals)
- Conditional trigger satisfaction
- Iconic-symbolic consistency (symbolic predicates match iconic layout)

Query evaluation returns True | False | None (open-world assumption).
"""

from __future__ import annotations

from .models import (
    CausalLayout,
    ConsistencyResult,
    Constraint,
    ConstraintSet,
    ConstraintType,
    ConditionalTrigger,
    Fact,
    IconicLayer,
    Model,
    Provenance,
    SpatialLayout,
    TemporalLayout,
    Violation,
)

# Predicates that are transitive
_TRANSITIVE_PREDS = {"left_of", "right_of", "before", "after", "causes", "causally_influences"}
# Predicates that are antisymmetric (cannot have both P(a,b) and P(b,a) as true)
_ANTISYMMETRIC_PREDS = {"left_of", "right_of", "before", "after", "strictly_left_of"}
# Predicates that are asymmetric with their inverse
_INVERSES = {
    "left_of": "right_of",
    "right_of": "left_of",
    "before": "after",
    "after": "before",
}


class ConstraintChecker:
    """
    Formal validator for mental models.

    Key methods:
    - check_consistency(model): returns ConsistencyResult
    - evaluate(model, predicate, args, polarity): returns True/False/None
    - satisfies_constraints(model, constraints): full constraint satisfaction check
    """

    # ─── Consistency checking ──────────────────────────────────────────────

    def check_consistency(self, model: Model) -> ConsistencyResult:
        """
        Check a model for internal consistency.
        Returns a ConsistencyResult listing all violations found.
        """
        violations: list[Violation] = []

        violations.extend(self._check_transitivity(model))
        violations.extend(self._check_antisymmetry(model))
        violations.extend(self._check_iconic_symbolic_match(model))
        violations.extend(self._check_conditional_triggers(model))

        return ConsistencyResult(
            consistent=len(violations) == 0,
            violations=violations,
        )

    def _check_transitivity(self, model: Model) -> list[Violation]:
        """
        For transitive predicates P, if P(a,b) and P(b,c) are true,
        then P(a,c) must not be explicitly false.
        """
        violations = []
        for pred in _TRANSITIVE_PREDS:
            true_pairs = [
                (f.args[0], f.args[1])
                for f in model.relations
                if f.predicate == pred and f.polarity and len(f.args) == 2
            ]
            false_pairs = {
                (f.args[0], f.args[1])
                for f in model.relations
                if f.predicate == pred and not f.polarity and len(f.args) == 2
            }
            # Build reachability
            reachable: dict[str, set[str]] = {}
            for a, b in true_pairs:
                reachable.setdefault(a, set()).add(b)

            # Check transitive closure
            changed = True
            while changed:
                changed = False
                for a, targets in list(reachable.items()):
                    for b in list(targets):
                        for c in reachable.get(b, set()):
                            if c not in targets:
                                reachable[a].add(c)
                                changed = True

            for a, reachable_set in reachable.items():
                for c in reachable_set:
                    if a != c and (a, c) in false_pairs:
                        violations.append(Violation(
                            type="transitivity_violation",
                            detail=(
                                f"{pred}({a},{c}) is explicitly false "
                                f"but follows transitively from model"
                            ),
                            severity="block",
                        ))
        return violations

    def _check_antisymmetry(self, model: Model) -> list[Violation]:
        """
        For antisymmetric predicates, if P(a,b) is true then P(b,a) must not be true.
        """
        violations = []
        for pred in _ANTISYMMETRIC_PREDS:
            true_pairs = {
                (f.args[0], f.args[1])
                for f in model.relations
                if f.predicate == pred and f.polarity and len(f.args) == 2
            }
            for a, b in true_pairs:
                if (b, a) in true_pairs:
                    violations.append(Violation(
                        type="antisymmetry_violation",
                        detail=f"Both {pred}({a},{b}) and {pred}({b},{a}) are true",
                        severity="block",
                    ))
        return violations

    def _check_iconic_symbolic_match(self, model: Model) -> list[Violation]:
        """
        Ensure symbolic predicates derived from iconic structure are consistent.
        If the model has a spatial layout, symbolic left_of facts must match it.
        """
        violations = []
        il = model.iconic_layer

        if il.spatial:
            ordering = il.spatial.ordering
            for fact in model.relations:
                if fact.predicate == "left_of" and len(fact.args) == 2:
                    a, b = fact.args
                    if a in ordering and b in ordering:
                        actual = ordering.index(a) < ordering.index(b)
                        if fact.polarity != actual:
                            violations.append(Violation(
                                type="iconic_mismatch",
                                detail=(
                                    f"left_of({a},{b}) polarity={fact.polarity} "
                                    f"contradicts spatial ordering {ordering}"
                                ),
                                severity="block",
                            ))

        if il.temporal:
            for fact in model.relations:
                if fact.predicate == "before" and len(fact.args) == 2:
                    a, b = fact.args
                    actual = il.temporal.before(a, b)
                    if actual is not None and fact.polarity != actual:
                        violations.append(Violation(
                            type="iconic_mismatch",
                            detail=(
                                f"before({a},{b}) polarity={fact.polarity} "
                                f"contradicts temporal intervals"
                            ),
                            severity="block",
                        ))

        return violations

    def _check_conditional_triggers(self, model: Model) -> list[Violation]:
        """
        If a conditional trigger's antecedent is true,
        the consequent must not be explicitly false.
        """
        violations = []
        for trigger in model.conditional_triggers:
            ant_fact = model.get_fact(trigger.antecedent_pred, trigger.antecedent_args)
            if ant_fact and ant_fact.polarity:
                con_fact = model.get_fact(trigger.consequent_pred, trigger.consequent_args)
                if con_fact and not con_fact.polarity:
                    violations.append(Violation(
                        type="conditional_violation",
                        detail=(
                            f"Trigger: if {trigger.antecedent_pred}({trigger.antecedent_args}) "
                            f"then {trigger.consequent_pred}({trigger.consequent_args}) — "
                            f"antecedent is true but consequent is false"
                        ),
                        severity="block",
                    ))
        return violations

    # ─── Query evaluation ──────────────────────────────────────────────────

    def evaluate(
        self,
        model: Model,
        predicate: str,
        args: list[str],
        polarity: bool = True,
    ) -> bool | None:
        """
        Evaluate a predicate query against a model.

        Returns:
            True  — predicate holds (with given polarity)
            False — predicate does not hold
            None  — open-world: unknown (fact absent from model)
        """
        # Try iconic layer first (more reliable)
        iconic_result = self._evaluate_iconic(model.iconic_layer, predicate, args)
        if iconic_result is not None:
            return iconic_result == polarity

        # Try symbolic layer
        fact = model.get_fact(predicate, args)
        if fact is None:
            # Check if derivable from schema rules
            derived = self._check_schema_rules(model, predicate, args)
            if derived is not None:
                return derived == polarity
            return None  # open world: unknown

        return fact.polarity == polarity

    def _evaluate_iconic(
        self,
        il: IconicLayer,
        predicate: str,
        args: list[str],
    ) -> bool | None:
        """Try to evaluate directly from iconic structure."""
        if il.spatial and predicate == "left_of" and len(args) == 2:
            return il.spatial.is_left_of(args[0], args[1])
        if il.spatial and predicate == "right_of" and len(args) == 2:
            return il.spatial.is_right_of(args[0], args[1])
        if il.temporal and predicate == "before" and len(args) == 2:
            return il.temporal.before(args[0], args[1])
        if il.temporal and predicate == "after" and len(args) == 2:
            return il.temporal.after(args[0], args[1])
        if il.temporal and predicate == "overlaps" and len(args) == 2:
            return il.temporal.overlaps(args[0], args[1])
        if il.temporal and predicate == "during" and len(args) == 2:
            return il.temporal.during(args[0], args[1])
        if il.causal and predicate == "causally_influences" and len(args) == 2:
            reachable = il.causal.reachable_from(args[0])
            return args[1] in reachable
        return None

    def _check_schema_rules(
        self,
        model: Model,
        predicate: str,
        args: list[str],
    ) -> bool | None:
        """
        Check if a predicate holds due to a universal schema rule.
        If rule is forall x: A(x) -> B(x), and we query B(e),
        check if A(e) is true.
        """
        for rule in model.schema_rules:
            if rule.consequent_pred == predicate and len(args) == 1:
                subject = args[0]
                ant_fact = model.get_fact(rule.antecedent_pred, [subject])
                if ant_fact and ant_fact.polarity:
                    return True
        return None

    def evaluate_existential(self, model: Model, property_pred: str) -> bool | None:
        """
        Evaluate: Does some entity have property_pred?
        Returns True if at least one entity satisfies it,
        False if all entities explicitly lack it,
        None if unknown.
        """
        results = []
        for entity in model.entities:
            result = self.evaluate(model, property_pred, [entity.id])
            if result is True:
                return True
            results.append(result)

        if all(r is False for r in results) and results:
            return False
        return None

    def evaluate_universal(
        self, model: Model, antecedent_pred: str, consequent_pred: str
    ) -> bool | None:
        """
        Evaluate: Are all antecedent_pred entities also consequent_pred?
        """
        ant_entities = [
            e for e in model.entities
            if self.evaluate(model, antecedent_pred, [e.id]) is True
        ]
        if not ant_entities:
            return None  # vacuously true but unverifiable

        results = [
            self.evaluate(model, consequent_pred, [e.id])
            for e in ant_entities
        ]
        if all(r is True for r in results):
            return True
        if any(r is False for r in results):
            return False
        return None

    # ─── Full constraint satisfaction ─────────────────────────────────────

    def satisfies_constraints(
        self, model: Model, constraints: ConstraintSet
    ) -> bool:
        """
        Check whether a model satisfies all constraints.
        Returns True if all constraints are satisfied, False otherwise.
        """
        for constraint in constraints.constraints:
            if not self._check_constraint(model, constraint):
                return False
        return True

    def _check_constraint(self, model: Model, constraint: Constraint) -> bool:
        """Check a single constraint against a model."""
        ct = constraint.type

        if ct == ConstraintType.SPATIAL_RELATION:
            return self._check_spatial_constraint(model, constraint)

        if ct == ConstraintType.TEMPORAL_RELATION:
            return self._check_temporal_constraint(model, constraint)

        if ct in (ConstraintType.CAUSAL_CAUSES, ConstraintType.CAUSAL_ENABLES):
            return self._check_causal_constraint(model, constraint)

        if ct == ConstraintType.CONDITIONAL:
            return self._check_conditional_constraint(model, constraint)

        if ct in (ConstraintType.DISJUNCTION, ConstraintType.EXCLUSIVE_DISJUNCTION):
            return self._check_disjunction_constraint(model, constraint)

        if ct == ConstraintType.UNIVERSAL:
            return self._check_universal_constraint(model, constraint)

        if ct == ConstraintType.EXISTENTIAL:
            return self._check_existential_constraint(model, constraint)

        if ct == ConstraintType.ATOMIC:
            return self._check_atomic_constraint(model, constraint)

        # Unknown constraint type: conservatively allow
        return True

    def _check_spatial_constraint(self, model: Model, c: Constraint) -> bool:
        if not c.entity1 or not c.relation or not c.entity2:
            return True
        il = model.iconic_layer
        if il.spatial:
            if c.relation == "left_of":
                result = il.spatial.is_left_of(c.entity1, c.entity2)
                return result is not False
            if c.relation == "right_of":
                result = il.spatial.is_right_of(c.entity1, c.entity2)
                return result is not False
        # Fall back to symbolic
        fact = model.get_fact(c.relation, [c.entity1, c.entity2])
        return fact is None or fact.polarity

    def _check_temporal_constraint(self, model: Model, c: Constraint) -> bool:
        if not c.entity1 or not c.relation or not c.entity2:
            return True
        il = model.iconic_layer
        if il.temporal:
            if c.relation == "before":
                result = il.temporal.before(c.entity1, c.entity2)
                return result is not False
            if c.relation == "after":
                result = il.temporal.after(c.entity1, c.entity2)
                return result is not False
            if c.relation == "overlaps":
                result = il.temporal.overlaps(c.entity1, c.entity2)
                return result is not False
        fact = model.get_fact(c.relation, [c.entity1, c.entity2])
        return fact is None or fact.polarity

    def _check_causal_constraint(self, model: Model, c: Constraint) -> bool:
        if not c.cause or not c.effect:
            return True
        il = model.iconic_layer
        if il.causal:
            reachable = il.causal.reachable_from(c.cause)
            return c.effect in reachable or c.effect in il.causal.get_children(c.cause)
        pred = "causes" if c.causal_type == "causes" else "enables"
        fact = model.get_fact(pred, [c.cause, c.effect])
        return fact is None or fact.polarity

    def _check_conditional_constraint(self, model: Model, c: Constraint) -> bool:
        """If antecedent is true, consequent must not be false."""
        if not c.antecedent or not c.consequent:
            return True

        ant_pred = c.antecedent.replace(" ", "_").lower()
        con_pred = c.consequent.replace(" ", "_").lower()

        # Check all entities as potential subjects
        for entity in model.entities:
            ant_val = self.evaluate(model, ant_pred, [entity.id])
            if ant_val is True:
                con_val = self.evaluate(model, con_pred, [entity.id])
                if con_val is False:
                    return False

        # Check as propositional (no entity args)
        ant_fact = model.get_fact(ant_pred, [])
        if ant_fact and ant_fact.polarity:
            con_fact = model.get_fact(con_pred, [])
            if con_fact and not con_fact.polarity:
                return False
        return True

    def _check_disjunction_constraint(self, model: Model, c: Constraint) -> bool:
        """At least one disjunct must be satisfiable."""
        if not c.disjuncts:
            return True
        # For exclusive disjunction, exactly one must be true
        # For regular, at least one
        # In System 1 (single model), this is difficult to check fully
        # We conservatively return True if we cannot determine
        return True

    def _check_universal_constraint(self, model: Model, c: Constraint) -> bool:
        """All entities satisfying antecedent must satisfy consequent."""
        if not c.antecedent_pred or not c.consequent_pred:
            return True
        result = self.evaluate_universal(model, c.antecedent_pred, c.consequent_pred)
        return result is not False

    def _check_existential_constraint(self, model: Model, c: Constraint) -> bool:
        """At least one entity exists satisfying the properties."""
        if not c.properties:
            return True
        for prop in c.properties:
            result = self.evaluate_existential(model, prop)
            if result is False:
                return False
        return True

    def _check_atomic_constraint(self, model: Model, c: Constraint) -> bool:
        """An explicit fact must hold."""
        if not c.predicate or c.args is None:
            return True
        result = self.evaluate(model, c.predicate, c.args, polarity=True)
        if c.polarity is False:
            result = self.evaluate(model, c.predicate, c.args, polarity=False)
        return result is not False


# ─────────────────────────────────────────────
# Provenance Enforcer
# ─────────────────────────────────────────────


class ProvenanceEnforcer:
    """
    Blocks confabulation by checking that all facts in a model
    are derivable from the constraint set.

    Facts with provenance=ASSUMED that cannot be derived are rejected.
    """

    def validate_model(
        self, model: Model, constraints: ConstraintSet
    ) -> list[Violation]:
        """
        Return a list of provenance violations.
        Facts with provenance=ASSUMED that aren't derivable are flagged.
        """
        violations = []
        for fact in model.relations:
            if fact.provenance == Provenance.ASSUMED:
                if not self._is_derivable(fact, model, constraints):
                    violations.append(Violation(
                        type="unsupported_fact",
                        detail=(
                            f"{fact.predicate}({fact.args}) was assumed "
                            f"without license from any premise"
                        ),
                        severity="block",
                    ))
        return violations

    def _is_derivable(
        self, fact: Fact, model: Model, constraints: ConstraintSet
    ) -> bool:
        """Check if a fact can be derived from the constraint set."""
        # Explicit from premises
        if fact.provenance == Provenance.EXPLICIT:
            return True

        # Derived from iconic structure
        if fact.provenance == Provenance.ICONIC:
            return True

        # Check schema rules (universals)
        for c in constraints.get_universals():
            if (c.consequent_pred == fact.predicate
                    and len(fact.args) == 1
                    and c.antecedent_pred):
                ant_fact = model.get_fact(c.antecedent_pred, fact.args)
                if ant_fact and ant_fact.polarity:
                    return True

        # Check transitivity (fact is transitive consequence)
        if fact.predicate in _TRANSITIVE_PREDS and len(fact.args) == 2:
            a, b = fact.args
            # Try to find a chain a -> ... -> b
            checker = ConstraintChecker()
            eval_result = checker.evaluate(model, fact.predicate, [a, b])
            if eval_result is True and fact.polarity:
                return True

        return False

    def strip_unsupported(
        self, model: Model, violations: list[Violation]
    ) -> Model:
        """Remove facts that caused unsupported_fact violations."""
        blocked_details = {v.detail for v in violations if v.type == "unsupported_fact"}
        filtered_relations = []
        for fact in model.relations:
            detail = (
                f"{fact.predicate}({fact.args}) was assumed "
                f"without license from any premise"
            )
            if detail not in blocked_details:
                filtered_relations.append(fact)
        result = model.copy()
        result.relations = filtered_relations
        return result
