import tempfile
from pathlib import Path
import unittest

from b1.environment import file_sha256, redact
from b1.records import error_record, is_correct, log_errors, pln_label, proof_sources, validate_record
from b1.snapshot import restore_snapshot


def response(proof="[]", status="well_aligned", answer="Yes, definitely."):
    return {"raw_proof": proof, "query_status": status,
            "executed_query": "(: $prf (Ready hana) $tv)", "answer": answer}


class ResultContractTests(unittest.TestCase):
    def test_answer_wording_cannot_create_a_proof(self):
        self.assertEqual(pln_label(response(), []), ("NOT_ESTABLISHED", None))

    def test_answer_wording_cannot_remove_a_proof(self):
        self.assertEqual(pln_label(response("['(: fact (Ready hana) (STV 1 1))']", answer="I do not know"), []),
                         ("SUPPORTED", None))

    def test_swallowed_reasoner_timeout_is_error(self):
        self.assertEqual(pln_label(response(), [{"kind": "timeout"}]), ("ERROR", "timeout"))
        log = "[Reasoner] Query failed for 'q': PeTTa query timed out after 30 seconds"
        self.assertEqual(log_errors(log)[0]["kind"], "timeout")

    def test_unparsed_question_is_error(self):
        self.assertEqual(pln_label(response(status="no_query"), []), ("ERROR", "question_parse_failure"))

    def test_malformed_proof_is_error(self):
        for value in ("no proof", "None", "{}", "[None]", "['']", "__import__('os').getcwd()"):
            with self.subTest(proof=value):
                self.assertEqual(pln_label(response(value), [])[0], "ERROR")

    def test_error_never_counts_as_correct_abstention(self):
        record = error_record("w", "q", 1.0, "timeout", "limit reached")
        self.assertFalse(is_correct(record, "NOT_ESTABLISHED"))

    def test_invented_sentence_ids_rejected(self):
        record = error_record("w", "q", 1.0, "timeout", "limit reached")
        for field in ("used_sentence_ids", "retrieved_sentence_ids"):
            changed = {**record, field: ["S99"]}
            with self.assertRaisesRegex(ValueError, "invented"):
                validate_record(changed, {"S01", "S02"})

    def test_invalid_latency_rejected(self):
        for elapsed in (float("nan"), float("inf"), -1, True):
            with self.assertRaisesRegex(ValueError, "latency"):
                validate_record(error_record("w", "q", elapsed, "timeout", "limit"), set())

    def test_provenance_matches_complete_atom_name_only(self):
        learned = [{"sentence_id": "S01", "atoms": ["(: f1 (Ready hana) (STV 1 1))"]},
                   {"sentence_id": "S02", "atoms": ["(: f10 (Ready bekele) (STV 1 1))"]}]
        result = proof_sources("['(: (rule f10) (Able bekele) (STV 1 1))']", learned)
        self.assertEqual(result["used_sentence_ids"], ["S02"])

    def test_colliding_proof_names_are_disclosed_not_guessed(self):
        learned = [{"sentence_id": "S01", "atoms": ["(: f (Ready hana) (STV 1 1))"]},
                   {"sentence_id": "S02", "atoms": ["(: f (Ready bekele) (STV 1 1))"]}]
        result = proof_sources("['(: f (Ready hana) (STV 1 1))']", learned)
        self.assertEqual(result["source_resolution"], "ambiguous")
        self.assertEqual(result["used_sentence_ids"], [])

    def test_snapshot_restores_query_mutation_and_checks_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            source, active = Path(temp) / "learned.metta", Path(temp) / "question/kb.metta"
            source.write_text("learned fact\n")
            digest = file_sha256(source)
            restore_snapshot(source, active, digest)
            active.write_text("learned fact\nquery assertion\n")
            restore_snapshot(source, active, digest)
            self.assertEqual(active.read_text(), "learned fact\n")
            source.write_text("corrupt snapshot\n")
            with self.assertRaisesRegex(RuntimeError, "changed before restore"):
                restore_snapshot(source, active, digest)

    def test_key_redaction(self):
        self.assertEqual(redact("key=example-secret", ("example-secret",)), "key=[REDACTED]")


if __name__ == "__main__":
    unittest.main()
