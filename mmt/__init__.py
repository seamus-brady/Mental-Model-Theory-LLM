"""
Mental Model Theory (MMT) Agent
================================
An LLM-based agent implementing Johnson-Laird's Mental Model Theory.

Based on the unified architecture from the plan:
- Semantic Compiler: LLM parses natural language premises into constraints
- Model Builder: Constructs iconic-first mental models (spatial/temporal/causal)
- Constraint Checker: Validates models and evaluates queries (true/false/unknown)
- Counterexample Finder: Searches for models falsifying a conclusion
- Mental Model Agent: Orchestrates System 1/System 2 dual-process reasoning
"""

from .agent import MentalModelAgent
from .models import (
    Domain,
    Judgment,
    Model,
    ConstraintSet,
    ReasoningResult,
    Fact,
    Entity,
    Provenance,
    IconicLayer,
    SpatialLayout,
    TemporalLayout,
    CausalLayout,
)

__all__ = [
    "MentalModelAgent",
    "Domain",
    "Judgment",
    "Model",
    "ConstraintSet",
    "ReasoningResult",
    "Fact",
    "Entity",
    "Provenance",
    "IconicLayer",
    "SpatialLayout",
    "TemporalLayout",
    "CausalLayout",
]
