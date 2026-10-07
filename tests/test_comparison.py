"""Offline integration tests. Fake HTTP/PLN responses are never benchmark results."""

from collections import Counter
from contextlib import nullcontext, redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from b1.baseline import load_baseline, matched_settings, run_question
from b1.comparison import annotate, run_comparison, score, skipped_record
from b1.dataset import full_dataset, validate_full
from b1.docker_runtime import inside_benchmark, launch_command, prepare_benchmark
from b1.language import authored_sentence
from b1.io import read_worlds, write_json
from b1.llm_control import LLMSettings, reserve_request
from b1.oracle import consensus, fingerprint

ROOT = Path(__file__).resolve().parents[1]
NORMAL = ROOT.parent / "normal-rag"
ENV = {"OPENAI_MODEL": "openrouter/example/test", "OPENROUTER_PROVIDER": "example",
       "OPENAI_API_KEY": "offline-private-test-key", "PARSER": "canonical_pln",
       "OLLAMA_MODEL": "nomic-embed-text", "OLLAMA_URL": "http://127.0.0.1:11434/api/embeddings",
       "B1_MAX_LLM_CALLS": "200", "B1_LLM_MIN_INTERVAL": "0"}


class FullDatasetTests(unittest.TestCase):
    def test_frozen_40_questions_and_independent_annotations(self):
        worlds = read_worlds(ROOT / "data/benchmark/worlds.json")
        expected = json.loads((ROOT / "data/benchmark/hand_expected.json").read_text())
        authored, annotations = full_dataset()
        self.assertEqual(fingerprint(authored), fingerprint(worlds))
        self.assertEqual(annotations, expected)
        result = validate_full(worlds, expected)
        self.assertEqual(result["question_count"], 40)
        self.assertEqual(Counter(q.depth_bucket for w in worlds for q in w.questions), {0: 10, 1: 10, 2: 10, 3: 10})
        self.assertEqual([w.questions[0].depth_bucket for w in worlds[:4]], [0, 1, 2, 3])
        self.assertEqual(sum(len(w.sentences) for w in worlds), 240)
        for w in worlds:
            proof_ids = expected[w.questions[0].id]["minimum_depth_proofs"][0]
            proof = tuple(s for s in w.sentences if s.id in proof_ids)
            for removed in proof:
                with self.subTest(world=w.id, removed=removed.id):
                    self.assertEqual(consensus(tuple(s for s in proof if s != removed), w.questions[0].goal).label, "NOT_ESTABLISHED")

    def test_incomplete_full_dataset_rejected(self):
        worlds, expected = full_dataset()
        with self.assertRaises(ValueError):
            validate_full(worlds[:8], expected)


@unittest.skipUnless((NORMAL / "normal_rag/core.py").exists(), "Install the sibling normal-rag baseline for integration tests")
class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.core, self.config = load_baseline(NORMAL)
        self.llm = LLMSettings.from_env(ENV)
        self.settings = matched_settings(self.config, ENV, self.llm)

    def response(self, payload):
        if "prompt" in payload:
            self.assertEqual(payload["model"], "nomic-embed-text")
            return self.core.Response(200, b'{"embedding":[1,0]}')
        return self.core.Response(200, json.dumps({"model": "example/test", "provider": "Example",
            "choices": [{"finish_reason": "stop", "message": {"content": ' {"label":"SUPPORTED","cited":["S01"]}\n'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}).encode())

    def run_fake(self, root, **kwargs):
        return run_comparison(root, normal_repo=NORMAL, worlds_path=ROOT / "data/benchmark/worlds.json",
            expected_path=ROOT / "data/benchmark/hand_expected.json", **kwargs)

    def test_full_local_orchestration_all_120_records_same_inputs_no_key_leak(self):
        worlds, expected = full_dataset()
        seen_jobs, seen_http = [], []
        def job(python, root, job, output_dir, environment, timeout):
            seen_jobs.append(job)
            reserve_request(Path(job["request_budget_path"]), 200, 0)
            if job["mode"] == "learn":
                return {"world_id": job["world_id"], "errors": [], "sentences": [],
                    "snapshot_path": "offline", "snapshot_sha256": "offline", "learning_seconds": 1,
                    "parse_failure_count": 0}, 1, "", None
            return {"world_id": job["world_id"], "question_id": job["question_id"],
                "label": "NOT_ESTABLISHED", "errors": [], "answer_seconds": 1,
                "used_sentence_ids": [], "retrieved_sentence_ids": []}, 1, "", None
        def post(http, url, payload, headers, timeout):
            seen_http.append(payload)
            return self.response(payload)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("b1.comparison.configured_environment", return_value=dict(ENV)), \
                 patch("b1.pln_runner.repository_manifest", return_value={"offline": True}), \
                 patch("b1.pln_runner.private_qdrant", return_value=nullcontext()), \
                 patch("b1.pln_runner.process_job", side_effect=job), \
                 patch.object(self.core.HTTP, "post", post), redirect_stdout(io.StringIO()):
                out = self.run_fake(root, pln_runtime="local", repo=ROOT, python=Path(__file__), run_id="offline")
            records = json.loads((out / "records.json").read_text())
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(len(records), 120)
            self.assertEqual(Counter(r["system"] for r in records), {s: 40 for s in ("pln_rag", "rag", "full_context")})
            self.assertTrue(summary["complete"])
            self.assertTrue(summary["full_dataset_selected"])
            self.assertEqual(summary["requests_reserved"], 140)  # 60 fake PLN requests and 80 real baseline-path fake HTTP requests.
            self.assertEqual(summary["systems"]["rag"]["accuracy"], 0.5)
            learned = [j for j in seen_jobs if j["mode"] == "learn"]
            self.assertEqual(len(learned), 20)
            for w, j in zip(worlds, learned):
                self.assertEqual(j["sentences"], [{"id": s.id, "text": s.text} for s in w.sentences])
            for j in seen_jobs:
                self.assertFalse({"expected_label", "depth_bucket", "goal", "repair_fact"} & j.keys())
            chats = [p for p in seen_http if "messages" in p]
            self.assertEqual(len(chats), 80)
            for p in chats:
                message = json.loads(p["messages"][1]["content"])
                self.assertEqual(set(message), {"sentences", "question"})
                self.assertEqual(p["temperature"], self.llm.temperature)
                self.assertEqual(p["max_tokens"], self.llm.max_tokens)
                self.assertEqual(p["provider"], self.llm.kwargs()["extra_body"]["provider"])
            self.assertEqual(chats[0]["messages"][0], chats[40]["messages"][0])
            for w in worlds:
                for q in w.questions:
                    op = out / "full_context" / w.id / q.id
                    data = json.loads((op / "input.json").read_text())
                    self.assertEqual(data["sentences"], [{"id": s.id, "text": s.text} for s in w.sentences])
                    self.assertEqual(data["question"], q.text)
                    self.assertEqual((op / "raw_model_output.txt").read_bytes(), b' {"label":"SUPPORTED","cited":["S01"]}\n')
            for p in out.rglob("*"):
                if p.is_file():
                    self.assertNotIn(ENV["OPENAI_API_KEY"].encode(), p.read_bytes())

    def test_shared_budget_and_error_denominator(self):
        sent = []
        def post(http, url, payload, headers, timeout):
            sent.append(payload)
            return self.response(payload)
        with tempfile.TemporaryDirectory() as folder:
            env = {**ENV, "B1_MAX_LLM_CALLS": "1"}
            with patch("b1.comparison.configured_environment", return_value=env), \
                 patch.object(self.core.HTTP, "post", post), redirect_stdout(io.StringIO()):
                out = self.run_fake(Path(folder), systems=("full_context", "rag"), world_limit=2)
            summary = json.loads((out / "summary.json").read_text())
            records = json.loads((out / "records.json").read_text())
            self.assertEqual(len([p for p in sent if "messages" in p]), 1)
            self.assertEqual(summary["requests_reserved"], 1)
            self.assertEqual(len(records), 8)
            self.assertEqual(sum(r["label"] == "ERROR" for r in records), 7)
            self.assertEqual(summary["systems"]["full_context"]["accuracy"], 0.25)
            self.assertEqual(summary["systems"]["rag"]["accuracy"], 0)
            self.assertFalse(summary["full_dataset_selected"])
            self.assertFalse(summary["execution_complete"])

    def test_learning_failure_does_not_claim_questions_were_attempted(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch("b1.comparison.configured_environment", return_value=dict(ENV)), \
             patch("b1.pln_runner.repository_manifest", return_value={"offline": True}), \
             patch("b1.pln_runner.private_qdrant", return_value=nullcontext()), \
             patch("b1.pln_runner.process_job", return_value=(None, 1.0, "", "worker_crash")), \
             redirect_stdout(io.StringIO()):
            out = self.run_fake(Path(folder), systems=("pln_rag",), world_limit=1,
                                pln_runtime="local", repo=ROOT, python=Path(__file__))
            summary = json.loads((out / "summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertFalse(summary["execution_complete"])
            self.assertEqual(summary["systems"]["pln_rag"]["not_attempted"], 2)

    def test_prepare_does_not_call_services(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch("b1.comparison.configured_environment", return_value=dict(ENV)), \
             patch.object(self.core.HTTP, "post", side_effect=AssertionError("No network in prepare")), \
             redirect_stdout(io.StringIO()):
            out = self.run_fake(Path(folder), systems=("rag", "full_context"), prepare_only=True)
            self.assertEqual(json.loads((out / "manifest.json").read_text())["stage"], "prepared_only")
            self.assertFalse((out / "records.json").exists())

    def test_malformed_completion_is_untouched_error_without_repair(self):
        sent = []
        raw = b'{"provider":"Example","choices":[{"message":{"content":"not json"}}]}'
        class Transport:
            def post(inner, *args):
                sent.append(args)
                return self.core.Response(200, raw)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            record = run_question(self.core, self.settings, self.llm, root / "budget.json",
                [{"id": "S01", "text": "Hana can read a note."}], "Can Hana read a note?", "full_context", root / "op", transport=Transport())
            self.assertEqual(len(sent), 1)
            self.assertEqual(record["label"], "ERROR")
            self.assertEqual((root / "op/response.body").read_bytes(), raw)
            self.assertEqual((root / "op/raw_model_output.txt").read_text(), "not json")
            self.assertEqual(record["usage"]["total_tokens"], "unmeasured")

    def test_pln_skipped_coverage_is_not_false(self):
        worlds, _ = full_dataset()
        w = worlds[0]
        row = annotate(skipped_record("pln_rag", "not_run", "offline"), "pln_rag", w, w.questions[0], {"proof_sentence_ids": ["S01"]})
        self.assertEqual(row["proof_coverage"]["retrieved_contains_required"], "unmeasured")

    def test_alternative_retrieved_proof_is_sufficient_without_supplied_ids(self):
        worlds, expected = full_dataset()
        w = worlds[0]
        q = w.questions[0]
        # This diagnostic intentionally uses a second direct proof, outside the frozen fixture.
        alternate = authored_sentence(99, q.goal)
        w = replace(w, sentences=(*w.sentences, alternate))
        record = {"label": "SUPPORTED", "attempted": True, "retrieved": [{"id": "S99", "score": 1.0}],
                  "context": [{"id": "S99", "text": alternate.text}]}
        row = annotate(record, "rag", w, q, {"proof_sentence_ids": expected[q.id]["minimum_depth_proofs"][0]})
        self.assertFalse(row["proof_coverage"]["retrieved_contains_required"])
        self.assertTrue(row["proof_coverage"]["retrieved_proves_claim"])

    def test_failed_pln_launch_still_records_all_questions_and_runs_host_baseline(self):
        captured = []
        def docker(command, **kwargs):
            if command[:2] == ["docker", "inspect"]:
                return type("Result", (), {"returncode": 0, "stdout": json.dumps([
                    {"Image": "sha256:fake", "State": {"Running": True}, "Mounts": []}])})()
            captured.append((command, kwargs["env"]))
            return type("Result", (), {"returncode": 125})()
        def post(http, url, payload, headers, timeout):
            return self.response(payload)
        with tempfile.TemporaryDirectory() as folder, \
             patch("b1.comparison.configured_environment", return_value=dict(ENV)), \
             patch("b1.docker_runtime.subprocess.run", side_effect=docker), \
             patch.object(self.core.HTTP, "post", post), redirect_stdout(io.StringIO()):
            out = self.run_fake(Path(folder), systems=("pln_rag", "full_context"), world_limit=1, container="offline-container")
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["record_count"], 4)
            self.assertEqual(summary["systems"]["pln_rag"]["not_attempted"], 2)
            self.assertEqual(summary["systems"]["full_context"]["errors"], 0)
            self.assertEqual(summary["driver_errors"][0]["exit_code"], 125)
            command, environment = captured[0]
            self.assertEqual(environment["OPENAI_API_KEY"], ENV["OPENAI_API_KEY"])
            self.assertNotIn(ENV["OPENAI_API_KEY"], " ".join(command))
            self.assertEqual(environment["B1_MAX_LLM_CALLS"], "200")
            job = json.loads((out / "pln_job.json").read_text())
            self.assertEqual(Path(job["worlds_path"]), out / "worlds.json")
            self.assertEqual(Path(job["request_budget_path"]), Path(folder) / "runtime/runs" / out.name / "request_budget.json")


class SharedDockerTests(unittest.TestCase):
    def test_shared_run_matches_launcher_identity_including_sudo(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            owner = root.stat()
            info = {"Image": "sha256:offline-image", "State": {"Running": True}, "Mounts": []}
            for uid, gid in ((0, 0), (owner.st_uid + 1, owner.st_gid + 2)):
                with self.subTest(uid=uid, gid=gid), \
                     patch("b1.docker_runtime.os.geteuid", return_value=uid), \
                     patch("b1.docker_runtime.os.getegid", return_value=gid):
                    command = launch_command(root, info, "offline", 1, "q", "o", job_path=root / "job.json")
                    self.assertEqual(command[command.index("--user") + 1], f"{uid}:{gid}")
                    pilot = launch_command(root, info, "pilot", 1, "q", "o")
                    self.assertEqual(pilot[pilot.index("--user") + 1], f"{owner.st_uid}:{owner.st_gid}")

    def test_shared_worker_writes_private_artifacts_and_budget_in_either_order(self):
        # Use a real subprocess with Docker's selected UID/GID. Only the
        # checkout ownership is mocked; file permissions and budget writes are real.
        worker = """
import json, os, sys
from pathlib import Path
from b1.io import write_json
from b1.llm_control import reserve_request
os.umask(0o077)
job_path = Path(sys.argv[1])
job = json.loads(job_path.read_text())
write_json(job_path.parent / 'docker.doctor.json', {'ready': True})
write_json(Path(job['output_dir']) / 'world/question/record.json', {'offline': True})
reserve_request(Path(job['request_budget_path']), 4, 0)
"""
        for baseline_first in (False, True):
            with self.subTest(baseline_first=baseline_first), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                out = root / "artifacts/benchmark/offline"
                out.mkdir(parents=True, mode=0o700)
                budget = root / "runtime/runs/offline/request_budget.json"
                job_path = out / "pln_job.json"
                write_json(job_path, {"output_dir": str(out / "pln_rag"), "request_budget_path": str(budget)})
                job_path.chmod(0o600)
                if baseline_first:
                    reserve_request(budget, 4, 0)
                    budget.chmod(0o600)
                info = {"Image": "sha256:offline-image", "State": {"Running": True}, "Mounts": []}
                other_owner = SimpleNamespace(st_uid=os.geteuid() + 1, st_gid=os.getegid() + 1)
                with patch.object(Path, "stat", return_value=other_owner):
                    command = launch_command(root, info, "offline", 1, "q", "o", job_path=job_path)
                uid, gid = map(int, command[command.index("--user") + 1].split(":"))
                result = subprocess.run([sys.executable, "-c", worker, str(job_path)], cwd=ROOT,
                                        user=uid, group=gid, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads((out / "docker.doctor.json").read_text()), {"ready": True})
                record = out / "pln_rag/world/question/record.json"
                write_json(record, {"host_annotated": True})
                self.assertEqual(reserve_request(budget, 4, 0)[0], 3 if baseline_first else 2)
                self.assertEqual(out.stat().st_mode & 0o777, 0o700)
                self.assertEqual(budget.stat().st_mode & 0o777, 0o600)

    def test_image_job_and_env_names_without_secret_values(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            info = {"Image": "sha256:offline-image", "State": {"Running": True}, "Mounts": []}
            cmd = launch_command(root, info, "offline", 20, "http://127.0.0.1:6333", "http://127.0.0.1:11434/api/embeddings",
                job_path=root / "job.json", forwarded_env=("OPENAI_API_KEY", "OPENAI_MODEL"))
            self.assertIn("--inside-benchmark", cmd)
            self.assertIn("OPENAI_API_KEY", cmd)
            self.assertNotIn(ENV["OPENAI_API_KEY"], " ".join(cmd))
            self.assertNotIn("--volumes-from", cmd)
            self.assertEqual(cmd[cmd.index("--entrypoint") + 1], "python3")

    def test_inside_dispatch_uses_staged_dataset_output_and_shared_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "job.json"
            job = {"worlds_path": "staged-worlds.json", "expected_path": "staged-key.json",
                   "run_id": "offline", "world_limit": 20, "dataset_kind": "benchmark",
                   "output_dir": "pln-output", "request_budget_path": "shared-budget.json"}
            path.write_text(json.dumps(job))
            with patch("b1.doctor.doctor", return_value={"ready": True}), patch("b1.pln_runner.run_pln") as run:
                self.assertEqual(inside_benchmark(path), 0)
            self.assertEqual(run.call_args.args[1], Path("/app"))
            self.assertEqual(run.call_args.args[3], Path("staged-worlds.json"))
            self.assertEqual(run.call_args.kwargs["request_budget_path"], Path("shared-budget.json"))
            self.assertEqual(run.call_args.kwargs["dataset_kind"], "benchmark")

    def test_wilson_intervals_and_empty_groups(self):
        rows = [{"correct": True, "label": "SUPPORTED", "attempted": True, "answer_seconds": 2}] * 5
        result = score(rows)
        self.assertAlmostEqual(result["accuracy_wilson_95"][0], 0.5655, places=3)
        self.assertAlmostEqual(result["accuracy_wilson_95"][1], 1.0)
        self.assertEqual(score([])["accuracy"], "unmeasured")


if __name__ == "__main__":
    unittest.main()
