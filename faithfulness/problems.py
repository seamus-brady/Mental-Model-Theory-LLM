"""
Augmented problem set for the extraction-faithfulness experiment.

Each problem carries GOLD relation triples — the ground-truth set of atomic
relations a correct extraction must recover, no more and no fewer. Gold triples
are hand-verifiable because the templates are deliberately simple.

A triple is (relation, arg1, arg2) for binary relations, using a canonical
vocabulary shared by both extraction arms so scoring is arm-neutral:
    spatial:  left_of
    temporal: before
    causal:   causes, enables

Directionality matters: left_of(A,B) != left_of(B,A). Converse phrasings in the
surface premise ("B is to the right of A") still normalise to the canonical
left_of(A,B) gold triple, so an extractor is scored on the *relation it encodes*,
not the words it happened to see.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field


@dataclass
class FaithfulnessProblem:
    name: str
    domain: str                      # "spatial" | "temporal" | "causal"
    premises: list[str]
    gold_triples: set[tuple[str, str, str]] = field(default_factory=set)


# ─────────────────────────────────────────────────────────────────────────────
# Template banks. Each surface form maps to a canonical triple so that converse
# and synonym phrasings are held to the same gold standard.
# ─────────────────────────────────────────────────────────────────────────────

# (surface_template, canonical_relation, flips_direction)
_SPATIAL_FORMS = [
    ("{a} is to the left of {b}.", "left_of", False),
    ("{a} is left of {b}.", "left_of", False),
    ("{b} is to the right of {a}.", "left_of", True),   # right_of(b,a) == left_of(a,b)
    ("{a} sits somewhere to the left of {b}.", "left_of", False),
    ("{b} is positioned to the right of {a}.", "left_of", True),
]

_TEMPORAL_FORMS = [
    ("{a} happens before {b}.", "before", False),
    ("{a} occurs before {b}.", "before", False),
    ("{b} happens after {a}.", "before", True),          # after(b,a) == before(a,b)
    ("{a} takes place earlier than {b}.", "before", False),
    ("{b} comes after {a}.", "before", True),
]

_CAUSAL_FORMS = [
    ("{a} causes {b}.", "causes", False),
    ("{a} brings about {b}.", "causes", False),
    ("{a} leads to {b}.", "causes", False),
    ("{a} enables {b}.", "enables", False),
    ("{a} makes {b} possible.", "enables", False),
]

_FORMS = {"spatial": _SPATIAL_FORMS, "temporal": _TEMPORAL_FORMS, "causal": _CAUSAL_FORMS}
_ENTITIES = ["A", "B", "C", "D", "E", "F", "G"]


def _emit(a: str, b: str, form) -> tuple[str, tuple[str, str, str]]:
    """Render one premise and its canonical gold triple."""
    template, rel, flips = form
    surface = template.format(a=a, b=b)
    # `a`, `b` are the canonical chain order and the template already encodes any
    # converse phrasing in the surface text, so the gold triple is ALWAYS
    # (rel, a, b). `flips` selects the surface wording only — applying it to the
    # gold too would reverse the ground truth for every converse-phrased premise.
    triple = (rel, a, b)
    return surface, triple


def generate(seed: int = 0) -> list[FaithfulnessProblem]:
    """
    Build the augmented set: for each domain, chains of length 2–4, each rendered
    with a mix of canonical and converse/synonym surface forms so the extractor
    cannot win by pattern-matching a single phrasing.
    """
    rng = random.Random(seed)
    problems: list[FaithfulnessProblem] = []

    for domain, forms in _FORMS.items():
        counter = 0
        for chain_len in (2, 3, 4):
            # A handful of distinct entity chains per length.
            for combo in itertools.islice(
                itertools.permutations(_ENTITIES, chain_len), 6
            ):
                premises: list[str] = []
                gold: set[tuple[str, str, str]] = set()
                for a, b in zip(combo, combo[1:]):
                    form = rng.choice(forms)
                    surface, triple = _emit(a, b, form)
                    premises.append(surface)
                    gold.add(triple)
                counter += 1
                problems.append(
                    FaithfulnessProblem(
                        name=f"{domain}_len{chain_len}_{counter:02d}",
                        domain=domain,
                        premises=premises,
                        gold_triples=gold,
                    )
                )
    return problems


if __name__ == "__main__":
    ps = generate()
    by_domain: dict[str, int] = {}
    for p in ps:
        by_domain[p.domain] = by_domain.get(p.domain, 0) + 1
    print(f"Generated {len(ps)} problems: {by_domain}")
    for p in ps[:3]:
        print(f"\n{p.name}")
        for prem in p.premises:
            print(f"   {prem}")
        print(f"   gold: {sorted(p.gold_triples)}")
