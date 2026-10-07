from dataclasses import replace
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from b1 import backward, forward
from b1.io import read_worlds
from b1.language import authored_sentence
from b1.model import Atom, Proof, Question, Sentence, Solution
from b1.oracle import OracleDisagreement, consensus, fingerprint, validate
from b1.pilot import handmade_pilot

ROOT = Path(__file__).resolve().parents[1]


def a(predicate, *args):
    return Atom(predicate, tuple(args))


def s(number, head, *body):
    return Sentence(f"S{number:02}", "logic-only unit fixture", head, tuple(body))


class HandmadeWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worlds = read_worlds(ROOT / "data/pilot/worlds.json")
        cls.expected = json.loads((ROOT / "data/pilot/hand_expected.json").read_text())

    def test_all_eight_worlds_against_hand_annotations(self):
        result = validate(self.worlds, self.expected)
        self.assertEqual(result["status"], "AGREED")
        self.assertEqual(result["world_count"], 8)
        self.assertEqual(result["question_count"], 16)
        self.assertEqual({a["depth"] for a in result["answers"]}, {None, 0, 1, 2, 3})
        self.assertEqual({a["rule_kind"] for a in result["answers"]},
                         {"fact", "chain", "conjunction", "relational", "mixed"})

    def test_frozen_data_reproduces_from_hand_authored_source(self):
        authored, expected = handmade_pilot()
        self.assertEqual(fingerprint(authored), fingerprint(self.worlds))
        self.assertEqual(expected, self.expected)

    def test_every_canonical_proof_sentence_is_necessary(self):
        for world in self.worlds:
            query = world.questions[0]
            result = consensus(world.sentences, query.goal)
            for proof in result.proofs:
                subset = tuple(s for s in world.sentences if s.id in proof.sentence_ids)
                for removed in subset:
                    with self.subTest(world=world.id, removed=removed.id):
                        remainder = tuple(s for s in subset if s != removed)
                        self.assertEqual(consensus(remainder, query.goal).label, "NOT_ESTABLISHED")

    def test_relational_hop_is_one_rule_application(self):
        world = self.worlds[6]
        result = consensus(world.sentences, world.questions[0].goal)
        self.assertEqual(result.depth, 1)
        self.assertEqual(len(result.proofs[0].sentence_ids), 3)

    def test_conjunction_uses_longest_branch_not_total_applications(self):
        world = self.worlds[5]
        result = consensus(world.sentences, world.questions[0].goal)
        self.assertEqual(result.depth, 3)
        used = [s for s in world.sentences if s.id in result.proofs[0].sentence_ids]
        self.assertEqual(sum(not s.is_fact for s in used), 4)

    def test_shuffle_does_not_change_answers(self):
        for world in self.worlds:
            for query in world.questions:
                original = consensus(world.sentences, query.goal)
                self.assertEqual(original.signature(), consensus(tuple(reversed(world.sentences)), query.goal).signature())

    def test_shortcut_rejected(self):
        world = self.worlds[3]
        target = world.questions[0]
        proof_ids = set(self.expected[target.id]["minimum_depth_proofs"][0])
        index = next(i for i, sentence in enumerate(world.sentences[:-1]) if sentence.id not in proof_ids)
        changed = list(world.sentences)
        changed[index] = replace(authored_sentence(50, target.goal), id=changed[index].id)
        with self.assertRaisesRegex(ValueError, "shortcut"):
            validate((replace(world, sentences=tuple(changed)),))

    def test_provable_unsupported_control_rejected(self):
        world = self.worlds[1]
        positive, control = world.questions
        changed = replace(control, text=positive.text, goal=positive.goal)
        with self.assertRaisesRegex(ValueError, "expected NOT_ESTABLISHED, found SUPPORTED"):
            validate((replace(world, questions=(positive, changed)),))

    def test_wrong_depth_rejected(self):
        world = self.worlds[2]
        query = replace(world.questions[0], depth_bucket=3)
        with self.assertRaisesRegex(ValueError, "exact depth 2"):
            validate((replace(world, questions=(query,)),))

    def test_english_cannot_silently_diverge_from_logic(self):
        world = self.worlds[0]
        changed = replace(world.sentences[0], text="Hana can bake bread.")
        with self.assertRaisesRegex(ValueError, "English and logical form disagree"):
            validate((replace(world, sentences=(changed, *world.sentences[1:])),))


class SolverEdgeCases(unittest.TestCase):
    def test_conjunction_does_not_join_different_people(self):
        clauses = (s(1, a("p", "Hana")), s(2, a("q", "Bekele")),
                   s(3, a("r", "?x"), a("p", "?x"), a("q", "?x")))
        self.assertEqual(consensus(clauses, a("r", "Hana")).label, "NOT_ESTABLISHED")

    def test_relational_middle_person_must_match(self):
        clauses = (s(1, a("mentor", "Hana", "Bekele")), s(2, a("mentor", "Meron", "Selam")),
                   s(3, a("train", "?x", "?z"), a("mentor", "?x", "?y"), a("mentor", "?y", "?z")))
        self.assertEqual(consensus(clauses, a("train", "Hana", "Selam")).label, "NOT_ESTABLISHED")

    def test_repeated_head_variable_and_repeated_premise_variable(self):
        clauses = (s(1, a("p", "Hana")), s(2, a("r", "?x", "?x"), a("p", "?x")),
                   s(3, a("q", "?x"), a("r", "?x", "?x")))
        self.assertEqual(consensus(clauses, a("r", "Hana", "Bekele")).label, "NOT_ESTABLISHED")
        self.assertEqual(consensus(clauses, a("q", "Hana")).depth, 2)

    def test_cycle_without_seed_proves_nothing(self):
        clauses = (s(1, a("p", "?x"), a("q", "?x")), s(2, a("q", "?x"), a("p", "?x")))
        self.assertEqual(consensus(clauses, a("p", "Hana")).label, "NOT_ESTABLISHED")

    def test_seeded_cycle_terminates_and_keeps_shortest_proof(self):
        clauses = (s(1, a("p", "Hana")), s(2, a("p", "?x"), a("q", "?x")),
                   s(3, a("q", "?x"), a("p", "?x")))
        self.assertEqual(consensus(clauses, a("p", "Hana")).signature(), ((0, ("S01",)),))
        self.assertEqual(consensus(clauses, a("q", "Hana")).signature(), ((1, ("S01", "S03")),))

    def test_alternative_provenance_at_equal_global_depth_is_retained(self):
        clauses = (
            s(1, a("p", "Hana")), s(2, a("q", "Hana")),
            s(3, a("p", "?x"), a("q", "?x")),
            s(4, a("r", "?x"), a("q", "?x")),
            s(5, a("t", "?x"), a("r", "?x")),
            s(6, a("u", "?x"), a("p", "?x"), a("t", "?x")),
        )
        result = consensus(clauses, a("u", "Hana"))
        self.assertEqual(result.depth, 3)
        self.assertEqual(result.signature(), (
            (3, ("S01", "S02", "S04", "S05", "S06")),
            (3, ("S02", "S03", "S04", "S05", "S06")),
        ))

    def test_shorter_proof_can_be_found_after_longer_proof(self):
        clauses = (s(1, a("p", "Hana")), s(2, a("q", "?x"), a("p", "?x")),
                   s(3, a("r", "?x"), a("q", "?x")),
                   s(4, a("r", "?x"), a("p", "?x")))
        result = consensus(clauses, a("r", "Hana"))
        self.assertEqual(result.depth, 1)
        self.assertEqual(len(result.proofs), 2)

    def test_retrieved_alternative_can_prove_goal_without_canonical_sources(self):
        clauses = (s(1, a("p", "Hana")), s(2, a("q", "Hana")),
                   s(3, a("r", "?x"), a("p", "?x")),
                   s(4, a("r", "?x"), a("q", "?x")))
        self.assertEqual(consensus(clauses[1::2], a("r", "Hana")).depth, 1)

    def test_range_restriction_prevents_unbound_conclusions(self):
        with self.assertRaisesRegex(ValueError, "head variable"):
            s(1, a("p", "?z"), a("q", "?x"))

    def test_deliberate_solver_disagreement_stops(self):
        clauses = (s(1, a("p", "Hana")),)
        with patch("b1.backward.solve", return_value=Solution(())):
            with self.assertRaises(OracleDisagreement):
                consensus(clauses, a("p", "Hana"))

    def test_one_hundred_seeded_small_theories(self):
        rng = random.Random(1801)
        predicates = ("p", "q", "r", "t")
        people = ("Hana", "Bekele")
        for trial in range(100):
            clauses = [s(1, a(rng.choice(predicates), rng.choice(people))),
                       s(2, a(rng.choice(predicates), rng.choice(people))),
                       s(3, a("edge", *rng.choices(people, k=2)))]
            for number in range(4, 9):
                if rng.random() < 0.25:
                    clauses.append(s(number, a(rng.choice(predicates), "?x"),
                                     a("edge", "?x", "?y"), a(rng.choice(predicates), "?y")))
                else:
                    body = [a(rng.choice(predicates), "?x") for _ in range(rng.choice((1, 2)))]
                    clauses.append(s(number, a(rng.choice(predicates), "?x"), *body))
            for predicate in predicates:
                for person in people:
                    with self.subTest(trial=trial, predicate=predicate, person=person):
                        goal = a(predicate, person)
                        self.assertEqual(forward.solve(tuple(clauses), goal).signature(),
                                         backward.solve(tuple(clauses), goal).signature())


if __name__ == "__main__":
    unittest.main()
