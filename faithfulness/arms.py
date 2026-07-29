"""
The two extraction arms under test, plus normalisers that reduce each arm's
output to the SAME canonical triple vocabulary so triple-level scoring is fair.

Arm A (iconic):  reuse the repo's SemanticCompiler unchanged -> ConstraintSet,
                 then read canonical triples off the extracted constraints.
Arm B (fol):     a matched-effort FOL extractor prompted to emit atoms in a
                 fixed predicate vocabulary, parsed back to the same triples.

Design guards:
  * Same model, same max_tokens, same temperature for both arms.
  * The FOL prompt is given as much structure and as many examples as the
    iconic one, and is reported verbatim (see FOL_SYSTEM) so nobody can say the
    baseline was strawmanned.
  * Converse surface forms are normalised on BOTH sides, so neither arm is
    rewarded for surface pattern-matching.
"""

from __future__ import annotations

import json
import re

from mmt.compiler import SemanticCompiler
from mmt.models import ConstraintType

# Canonical relation vocabulary shared across arms and gold labels.
_CANON = {"left_of", "before", "causes", "enables"}

# Converse map: how a non-canonical relation flips to canonical form.
_CONVERSE = {
    "right_of": ("left_of", True),
    "after": ("before", True),
    "left_of": ("left_of", False),
    "before": ("before", False),
    "causes": ("causes", False),
    "enables": ("enables", False),
}


# ─────────────────────────────────────────────────────────────────────────────
# Arm A: iconic (repo's own compiler)
# ─────────────────────────────────────────────────────────────────────────────

def iconic_triples(compiler: SemanticCompiler, premises: list[str]) -> set[tuple[str, str, str]]:
    """Run the repo compiler and read canonical triples off the ConstraintSet."""
    cs = compiler.extract(premises)
    out: set[tuple[str, str, str]] = set()
    for c in cs.constraints:
        if c.type in (ConstraintType.SPATIAL_RELATION, ConstraintType.TEMPORAL_RELATION):
            rel, e1, e2 = c.relation, c.entity1, c.entity2
            out |= _normalise(rel, e1, e2)
        elif c.type in (ConstraintType.CAUSAL_CAUSES, ConstraintType.CAUSAL_ENABLES):
            rel = c.causal_type or ("causes" if c.type == ConstraintType.CAUSAL_CAUSES else "enables")
            out |= _normalise(rel, c.cause, c.effect)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Arm B: FOL baseline (Logic-LM style), matched effort
# ─────────────────────────────────────────────────────────────────────────────

FOL_SYSTEM = """\
You are a first-order-logic semantic parser. Translate natural-language premises
into ground FOL atoms using EXACTLY this predicate vocabulary:

  left_of(x, y)   — x is spatially left of y
  before(x, y)    — event x occurs before event y
  causes(x, y)    — x causes y
  enables(x, y)   — x enables y

RULES:
  * Use ONLY these four predicates. Normalise converse phrasings:
      "y is to the right of x"  -> left_of(x, y)
      "y happens after x"       -> before(x, y)
  * One atom per stated relation. Do not add transitive or inferred atoms.
  * Entities are the capital-letter tokens exactly as written.

Return ONLY a JSON object, no prose, no code fences:
{ "atoms": ["left_of(A, B)", "before(C, D)", ...] }
"""

_ATOM_RE = re.compile(r"([a-z_]+)\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)")


def fol_triples(compiler: SemanticCompiler, premises: list[str]) -> set[tuple[str, str, str]]:
    """Run the FOL arm through the same client/model as the iconic arm."""
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(premises))
    resp = compiler.client.messages.create(
        model=compiler.model,
        max_tokens=4096,
        system=FOL_SYSTEM,
        messages=[{"role": "user", "content": f"Premises:\n{numbered}\n\nReturn ONLY the JSON."}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        atoms = json.loads(text).get("atoms", [])
    except json.JSONDecodeError:
        atoms = _ATOM_RE.findall(text)  # tolerant fallback: pull atoms from raw text
        return _from_matches(atoms)
    out: set[tuple[str, str, str]] = set()
    for atom in atoms:
        m = _ATOM_RE.match(atom.strip())
        if m:
            out |= _normalise(m.group(1), m.group(2).strip(), m.group(3).strip())
    return out


def _from_matches(matches) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for rel, a, b in matches:
        out |= _normalise(rel, a.strip(), b.strip())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Shared normaliser
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(rel, e1, e2) -> set[tuple[str, str, str]]:
    """Reduce a (relation, e1, e2) to canonical directed triple form, or drop it."""
    if not rel or not e1 or not e2:
        return set()
    rel = rel.strip().lower().replace(" ", "_")
    if rel not in _CONVERSE:
        return set()  # off-vocabulary relation: counts as an extraction miss
    canon, flip = _CONVERSE[rel]
    a, b = (e2, e1) if flip else (e1, e2)
    return {(canon, a.strip(), b.strip())}
