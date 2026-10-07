"""Bottom-up saturation with joins over known ground atoms.

No backward solver, grounding library, or logic engine is called here.
Depth(fact)=0; depth(rule application)=1+max(depth(premise)).
"""

from __future__ import annotations

from .model import Atom, Proof, Sentence, Solution, variable


def solve(sentences: tuple[Sentence, ...], goal: Atom) -> Solution:
    if not goal.ground:
        raise ValueError("A ground goal is required")
    known: dict[Atom, list[Proof]] = {}

    def insert(atom: Atom, candidate: Proof) -> bool:
        previous = known.setdefault(atom, [])
        if any(p.depth <= candidate.depth and p.sentence_ids <= candidate.sentence_ids for p in previous):
            return False
        previous[:] = [p for p in previous
                       if not (candidate.depth <= p.depth and candidate.sentence_ids <= p.sentence_ids)]
        previous.append(candidate)
        return True

    for sentence in sentences:
        if sentence.is_fact:
            insert(sentence.head, Proof(0, frozenset({sentence.id})))

    changed = True
    while changed:
        changed = False
        # A round sees one immutable snapshot; sentence ordering cannot alter depth.
        index: dict[tuple[str, int], list[tuple[Atom, Proof]]] = {}
        for atom, proofs in known.items():
            index.setdefault((atom.predicate, len(atom.arguments)), []).extend((atom, p) for p in proofs)
        for rule in sentences:
            if rule.is_fact:
                continue
            partial: list[tuple[dict[str, str], frozenset[str], int]] = [({}, frozenset(), -1)]
            for premise in rule.body:
                following = []
                for binding, ids, maximum in partial:
                    for fact, proof in index.get((premise.predicate, len(premise.arguments)), []):
                        extended = dict(binding)
                        matched = True
                        for term, value in zip(premise.arguments, fact.arguments):
                            if variable(term):
                                if term in extended and extended[term] != value:
                                    matched = False
                                    break
                                extended[term] = value
                            elif term != value:
                                matched = False
                                break
                        if matched:
                            following.append((extended, ids | proof.sentence_ids, max(maximum, proof.depth)))
                partial = following
                if not partial:
                    break
            for binding, ids, maximum in partial:
                conclusion = Atom(rule.head.predicate,
                                  tuple(binding[a] if variable(a) else a for a in rule.head.arguments))
                changed |= insert(conclusion, Proof(maximum + 1, ids | {rule.id}))
    return Solution(tuple(known.get(goal, ())))
