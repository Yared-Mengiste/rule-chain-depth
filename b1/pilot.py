"""Eight hand-authored worlds, with hand-annotated proofs, not solver output.

The first four isolate depth 0/1/2/3. The next four cover conjunction (including
unequal branch depths), relational composition, and a mixed proof. Each world
has one supported target and one missing-premise control. This pilot is separate
from the full 40-question benchmark in dataset.py.
"""

from dataclasses import replace
import random

from .language import authored_sentence, question_text
from .model import Atom, Question, World


def a(predicate: str, *people: str) -> Atom:
    return Atom(predicate, tuple(people))


def handmade_pilot() -> tuple[tuple[World, ...], dict]:
    worlds = []
    expectations = {}

    def add_world(number, clauses, target, foil, repair, depth, kind, proof_numbers):
        world_id = f"pilot-{number:02}"
        authored = tuple(authored_sentence(i, *clause) for i, clause in enumerate(clauses, 1))
        critical = {f"S{i:02}" for i in proof_numbers}
        seed = 1700 + number
        rng = random.Random(seed)
        order = list(authored)
        while True:
            rng.shuffle(order)
            if order[-1].id not in critical and order != list(authored):
                break
        id_map = {sentence.id: f"S{i:02}" for i, sentence in enumerate(order, 1)}
        shuffled = tuple(replace(s, id=id_map[s.id]) for s in order)
        positive_id, negative_id = f"{world_id}-supported", f"{world_id}-control"
        questions = (
            Question(positive_id, question_text(target), target, "SUPPORTED", depth, kind),
            Question(negative_id, question_text(foil), foil, "NOT_ESTABLISHED", depth, kind,
                     positive_id, repair),
        )
        worlds.append(World(world_id, shuffled, questions, seed))
        expectations[positive_id] = {
            "label": "SUPPORTED", "depth": depth,
            "minimum_depth_proofs": [sorted(id_map[sid] for sid in critical)],
        }
        expectations[negative_id] = {
            "label": "NOT_ESTABLISHED", "depth": None, "minimum_depth_proofs": [],
        }

    for number, depth, person in [(1, 0, "Hana"), (2, 1, "Abebe"),
                                   (3, 2, "Selam"), (4, 3, "Aster")]:
        sequence = ("sort_parcel", "check_label", "pack_crate", "seal_crate")
        clauses = [
            (a("sort_parcel", person),),
            (a("check_label", "?x"), a("sort_parcel", "?x")),
            (a("pack_crate", "?x"), a("check_label", "?x")),
            (a("seal_crate", "?x"), a("pack_crate", "?x")),
            (a("paint_gate", "Bekele"),),
            (a("clean_bench", "?x"), a("paint_gate", "?x")),
            (a("make_tea", "Meron"),),
            (a("supervise", "Bekele", "Dawit"),),
            (a("mentor", "Bekele", "Rahel"),),
            (a("fold_cloth", "?x"), a("weave_mat", "?x")),
            (a("weave_mat", "Genet"),),
            (a("carry_drum", "Rahel"),),
        ]
        add_world(number, clauses, a(sequence[depth], person), a(sequence[depth], "Meron"),
                  a("sort_parcel", "Meron"), depth, "fact" if depth == 0 else "chain",
                  list(range(1, depth + 2)))

    add_world(5, [
        (a("sort_parcel", "Hana"),),
        (a("check_label", "Hana"),),
        (a("pack_crate", "?x"), a("sort_parcel", "?x"), a("check_label", "?x")),
        (a("sort_parcel", "Meron"),),
        (a("paint_gate", "Meron"),),
        (a("light_lamp", "?x"), a("check_label", "?x")),
        (a("clean_bench", "?x"), a("paint_gate", "?x")),
        (a("mentor", "Bekele", "Rahel"),),
        (a("supervise", "Bekele", "Abebe"),),
        (a("fold_cloth", "Rahel"),),
        (a("carry_basket", "?x"), a("weave_mat", "?x")),
        (a("weave_mat", "Selam"),),
    ], a("pack_crate", "Hana"), a("pack_crate", "Meron"), a("check_label", "Meron"),
        1, "conjunction", [1, 2, 3])

    add_world(6, [
        (a("sort_parcel", "Hana"),),
        (a("check_label", "?x"), a("sort_parcel", "?x")),
        (a("pack_crate", "?x"), a("check_label", "?x")),
        (a("weave_mat", "Hana"),),
        (a("fold_cloth", "?x"), a("weave_mat", "?x")),
        (a("seal_crate", "?x"), a("pack_crate", "?x"), a("fold_cloth", "?x")),
        (a("sort_parcel", "Meron"),),
        (a("paint_gate", "Meron"),),
        (a("carry_basket", "?x"), a("paint_gate", "?x")),
        (a("mentor", "Bekele", "Rahel"),),
        (a("supervise", "Bekele", "Selam"),),
        (a("clean_bench", "?x"), a("light_lamp", "?x")),
    ], a("seal_crate", "Hana"), a("seal_crate", "Meron"), a("weave_mat", "Meron"),
        3, "conjunction", [1, 2, 3, 4, 5, 6])

    hop = (a("train", "?x", "?z"), a("mentor", "?x", "?y"), a("mentor", "?y", "?z"))
    add_world(7, [
        (a("mentor", "Hana", "Bekele"),),
        (a("mentor", "Bekele", "Abebe"),), hop,
        (a("supervise", "Bekele", "Selam"),),
        (a("mentor", "Meron", "Rahel"),),
        (a("supervise", "Rahel", "Dawit"),),
        (a("guide", "?x", "?y"), a("train", "?x", "?y")),
        (a("clean_bench", "?x"), a("paint_gate", "?x")),
        (a("paint_gate", "Rahel"),),
        (a("fold_cloth", "?x"), a("weave_mat", "?x")),
        (a("carry_basket", "Selam"),),
        (a("make_tea", "Dawit"),),
    ], a("train", "Hana", "Abebe"), a("train", "Hana", "Selam"),
        a("mentor", "Bekele", "Selam"), 1, "relational", [1, 2, 3])

    add_world(8, [
        (a("mentor", "Hana", "Bekele"),),
        (a("mentor", "Bekele", "Abebe"),), hop,
        (a("lead_team", "?x"), a("train", "?x", "?y")),
        (a("mentor", "Meron", "Rahel"),),
        (a("supervise", "Rahel", "Selam"),),
        (a("guide", "?x", "?y"), a("supervise", "?x", "?y")),
        (a("paint_gate", "Rahel"),),
        (a("clean_bench", "?x"), a("paint_gate", "?x")),
        (a("fold_cloth", "?x"), a("weave_mat", "?x")),
        (a("carry_basket", "Selam"),),
        (a("make_tea", "Dawit"),),
    ], a("lead_team", "Hana"), a("lead_team", "Meron"), a("mentor", "Rahel", "Selam"),
        2, "mixed", [1, 2, 3, 4])

    return tuple(worlds), expectations
