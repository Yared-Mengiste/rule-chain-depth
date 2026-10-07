"""Independent top-down proof search, from a ground goal toward facts.

Unlike the forward solver, this enumerates substitutions for a rule after
matching its conclusion. Repeated ground goals on one branch are cut. Any
cycle-containing proof is dominated by the proof with that cycle removed.
Memoization includes the ancestor set, avoiding context-dependent cycle bugs.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product

from .model import Atom, Proof, Sentence, Solution


def solve(sentences: tuple[Sentence, ...], goal: Atom) -> Solution:
    if not goal.ground:
        raise ValueError("A ground goal is required")
    constants = tuple(sorted({term for s in sentences for p in (s.head, *s.body)
                              for term in p.arguments if not term.startswith("?")} | set(goal.arguments)))

    @lru_cache(maxsize=None)
    def visit(target: Atom, ancestors: frozenset[Atom]) -> tuple[Proof, ...]:
        if target in ancestors:
            return ()
        found: set[Proof] = set()
        for sentence in sentences:
            if sentence.is_fact:
                if sentence.head == target:
                    found.add(Proof(0, frozenset([sentence.id])))
                continue
            if (sentence.head.predicate != target.predicate or
                    len(sentence.head.arguments) != len(target.arguments)):
                continue
            assignment: dict[str, str] = {}
            fits = True
            for name, constant in zip(sentence.head.arguments, target.arguments):
                if name.startswith("?"):
                    if name in assignment and assignment[name] != constant:
                        fits = False
                        break
                    assignment[name] = constant
                elif name != constant:
                    fits = False
                    break
            if not fits:
                continue
            free = sorted({arg for atom in sentence.body for arg in atom.arguments
                           if arg.startswith("?") and arg not in assignment})
            for values in product(constants, repeat=len(free)):
                substitution = assignment | dict(zip(free, values))
                premise_proofs = []
                for atom in sentence.body:
                    subgoal = Atom(atom.predicate, tuple(substitution.get(a, a) for a in atom.arguments))
                    choices = visit(subgoal, ancestors | {target})
                    if not choices:
                        break
                    premise_proofs.append(choices)
                else:
                    for combination in product(*premise_proofs):
                        sources = frozenset([sentence.id]).union(*(p.sentence_ids for p in combination))
                        found.add(Proof(1 + max(p.depth for p in combination), sources))
        # Independently implemented dominance, after enumerating proofs of this goal.
        keep = []
        for candidate in found:
            dominated = False
            for other in found:
                if other != candidate and other.depth <= candidate.depth and other.sentence_ids.issubset(candidate.sentence_ids):
                    dominated = True
                    break
            if not dominated:
                keep.append(candidate)
        return tuple(keep)

    return Solution(visit(goal, frozenset()))
