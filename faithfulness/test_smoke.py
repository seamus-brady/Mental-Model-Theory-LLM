"""
No-API smoke test. Mocks the Anthropic client so both arms and the scorer run
end to end, proving the plumbing works before any real spend.

Run:  python -m faithfulness.test_smoke
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from mmt.compiler import SemanticCompiler
from .problems import generate
from .arms import iconic_triples, fol_triples
from .run import prf


class _Block:
    def __init__(self, text): self.type, self.text = "text", text


class _Resp:
    def __init__(self, text): self.content = [_Block(text)]


class MockClient:
    """
    Returns a canned iconic-compiler JSON when it sees the MMT system prompt,
    and a canned FOL JSON when it sees the FOL system prompt. Perfect extractions,
    so a correct harness must score F1 = 1.0 for both arms on a simple problem.
    """
    def __init__(self):
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, model, max_tokens, system, messages):
        if "first-order-logic" in system:
            return _Resp(json.dumps({"atoms": ["left_of(A, B)", "left_of(B, C)"]}))
        # iconic compiler path
        return _Resp(json.dumps({
            "domain": "spatial",
            "constraints": [
                {"type": "spatial_relation", "entity1": "A", "relation": "left_of", "entity2": "B"},
                {"type": "spatial_relation", "entity1": "B", "relation": "left_of", "entity2": "C"},
            ],
            "entities": ["A", "B", "C"],
            "has_disjunctions": False, "has_counterfactuals": False,
            "has_negated_conditionals": False, "matches_known_illusion": False,
            "reasoning_notes": "",
        }))


def main():
    compiler = SemanticCompiler(client=MockClient())
    premises = ["A is to the left of B.", "B is to the left of C."]
    gold = {("left_of", "A", "B"), ("left_of", "B", "C")}

    icon = iconic_triples(compiler, premises)
    fol = fol_triples(compiler, premises)

    print("iconic triples:", sorted(icon))
    print("fol triples:   ", sorted(fol))

    ip, ir, iff = prf(icon, gold)
    fp, fr, ff = prf(fol, gold)
    print(f"iconic  P={ip:.2f} R={ir:.2f} F1={iff:.2f}")
    print(f"fol     P={fp:.2f} R={fr:.2f} F1={ff:.2f}")

    # Also prove the converse-normalisation guard works: "B right of A" -> left_of(A,B)
    conv = ["B is to the right of A."]

    class ConvMock(MockClient):
        def _create(self, model, max_tokens, system, messages):
            if "first-order-logic" in system:
                return _Resp(json.dumps({"atoms": ["left_of(A, B)"]}))
            return _Resp(json.dumps({
                "domain": "spatial",
                "constraints": [{"type": "spatial_relation", "entity1": "B",
                                 "relation": "right_of", "entity2": "A"}],
                "entities": ["A", "B"], "has_disjunctions": False,
                "has_counterfactuals": False, "has_negated_conditionals": False,
                "matches_known_illusion": False, "reasoning_notes": "",
            }))

    cc = SemanticCompiler(client=ConvMock())
    ci = iconic_triples(cc, conv)
    cf = fol_triples(cc, conv)
    print("\nconverse guard — iconic:", sorted(ci), " fol:", sorted(cf))

    ok = (iff == 1.0 and ff == 1.0
          and ci == {("left_of", "A", "B")} and cf == {("left_of", "A", "B")})
    print(f"\nProblem set size: {len(generate())}")
    print("SMOKE TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
