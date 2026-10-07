"""Twenty authored worlds for the balanced 40-question experiment.

Proof annotations come from explicit clause positions below, never a solver.
The two independent oracles check them before fixtures or runs are admitted.
"""

from collections import Counter
from dataclasses import replace
import random

from .language import NAMES, UNARY, authored_sentence, question_text
from .model import Atom, Question, World
from .oracle import validate


def a(predicate, *args):
    return Atom(predicate, tuple(args))


def full_dataset() -> tuple[tuple[World, ...], dict]:
    worlds, expected = [], {}
    sequences = (
        ("sort_parcel", "check_label", "pack_crate", "seal_crate"),
        ("weave_mat", "fold_cloth", "carry_basket", "lead_team"),
    )
    # Interleave depths: even a four-world trial visits every depth bucket.
    for family in range(5):
        for depth in range(4):
            number = family * 4 + depth + 1
            world_id = f"bench-{number:02}"
            people = [NAMES[(number + offset) % len(NAMES)] for offset in range(6)]
            person, foil, middle, end, other, distractor = people
            kind = "fact" if depth == 0 else ("chain", "chain", "conjunction", "relational", "mixed")[family]
            if depth == 0:
                pred = ("sort_parcel", "weave_mat", "light_lamp", "read_note", "bake_bread")[family]
                clauses = [(a(pred, person),)]
                target, control, repair = a(pred, person), a(pred, foil), a(pred, foil)
            elif family < 2:
                sequence = sequences[family]
                clauses = [(a(sequence[0], person),)] + [
                    (a(sequence[i], "?x"), a(sequence[i - 1], "?x")) for i in range(1, depth + 1)]
                target, control = a(sequence[depth], person), a(sequence[depth], foil)
                repair = a(sequence[0], foil)
            elif family == 2:
                sequence = sequences[0]
                clauses = [(a(sequence[0], person),)] + [
                    (a(sequence[i], "?x"), a(sequence[i - 1], "?x")) for i in range(1, depth)]
                clauses += [(a("weave_mat", person),)]
                branch = "weave_mat"
                if depth == 3:
                    clauses += [(a("fold_cloth", "?x"), a("weave_mat", "?x"))]
                    branch = "fold_cloth"
                clauses += [(a(sequence[depth], "?x"), a(sequence[depth - 1], "?x"), a(branch, "?x"))]
                target, control = a(sequence[depth], person), a(sequence[depth], foil)
                repair = a("weave_mat", foil)
            elif family == 3:
                clauses = [(a("mentor", person, middle),), (a("mentor", middle, end),),
                           (a("train", "?x", "?z"), a("mentor", "?x", "?y"), a("mentor", "?y", "?z"))]
                pred = "train"
                for next_pred in ("guide", "help")[:depth - 1]:
                    clauses += [(a(next_pred, "?x", "?y"), a(pred, "?x", "?y"))]
                    pred = next_pred
                target, control = a(pred, person, end), a(pred, person, other)
                repair = a("mentor", middle, other)
            else:
                if depth == 1:
                    clauses = [(a("mentor", person, middle),),
                               (a("lead_team", "?x"), a("mentor", "?x", "?y"))]
                else:
                    clauses = [(a("mentor", person, middle),), (a("mentor", middle, end),),
                               (a("train", "?x", "?z"), a("mentor", "?x", "?y"), a("mentor", "?y", "?z")),
                               (a("lead_team", "?x"), a("train", "?x", "?y"))]
                pred = "lead_team"
                if depth == 3:
                    clauses += [(a("open_gate", "?x"), a("lead_team", "?x"))]
                    pred = "open_gate"
                target, control = a(pred, person), a(pred, foil)
                repair = a("mentor", foil, middle)

            # Every clause constructed so far is necessary for the target.
            proof_numbers = set(range(1, len(clauses) + 1))
            if family == 2 and depth:
                clauses += [(a(sequences[0][0], foil),)]
            # Same vocabulary appears near the foil without supplying its missing fact.
            used = {atom.predicate for clause in clauses for atom in clause}
            spare = [p for p in UNARY if p not in used]
            filler = [(a(spare[0], foil),),
                      (a(spare[1], "?x"), a(spare[0], "?x")),
                      (a(spare[2], distractor),),
                      (a(spare[3], "?x"), a(spare[2], "?x")),
                      (a(spare[4], other),),
                      (a(spare[5], "?x"), a(spare[4], "?x"))]
            filler += [(a(spare[i % len(spare)], people[(i + 2) % len(people)]),) for i in range(6, 18)]
            for clause in filler:
                if clause not in clauses and len(clauses) < 12:
                    clauses.append(clause)
            authored = [authored_sentence(i, *clause) for i, clause in enumerate(clauses, 1)]
            critical = {f"S{i:02}" for i in proof_numbers}
            seed = 4000 + number
            rng = random.Random(seed)
            shuffled = list(authored)
            while True:
                rng.shuffle(shuffled)
                if shuffled[-1].id not in critical and shuffled != authored:
                    break
            ids = {s.id: f"S{i:02}" for i, s in enumerate(shuffled, 1)}
            positive, negative = world_id + "-supported", world_id + "-control"
            questions = (
                Question(positive, question_text(target), target, "SUPPORTED", depth, kind),
                Question(negative, question_text(control), control, "NOT_ESTABLISHED", depth, kind, positive, repair),
            )
            worlds.append(World(world_id, tuple(replace(s, id=ids[s.id]) for s in shuffled), questions, seed))
            expected[positive] = {"label": "SUPPORTED", "depth": depth,
                                  "minimum_depth_proofs": [sorted(ids[s] for s in critical)]}
            expected[negative] = {"label": "NOT_ESTABLISHED", "depth": None, "minimum_depth_proofs": []}
    return tuple(worlds), expected


def validate_full(worlds, expected):
    if len(worlds) != 20 or any(len(w.questions) != 2 for w in worlds):
        raise ValueError("The full benchmark requires 20 worlds with two questions each")
    counts = Counter((q.depth_bucket, q.expected_label) for w in worlds for q in w.questions)
    if counts != Counter({(d, label): 5 for d in range(4) for label in ("SUPPORTED", "NOT_ESTABLISHED")}):
        raise ValueError("The full benchmark requires five supported/control pairs per depth")
    for world in worlds:
        if [q.expected_label for q in world.questions] != ["SUPPORTED", "NOT_ESTABLISHED"]:
            raise ValueError("Each world must list its supported question then its control")
    return validate(worlds, expected)
