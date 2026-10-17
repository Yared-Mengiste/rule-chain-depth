"""Offline validity tests: no model, embedding, Docker or evaluator service calls."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import signal
import time
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from b1.llm_control import LLMSettings, reserve_request
from b2 import baseline
from b2.dataset import author, digest, graph, read, selected_ids, validate, write
from b2.evidence import evaluate, map_atoms, map_proof, overlap
from b2.pln_worker import block_query_learning, run as run_native
from b2.runner import benchmark, environment
from b2.runtime import Runtime, docker_command, run_baseline, inspect_container
from b2.worker import validate_job

ROOT = Path(__file__).resolve().parents[1]
ENV = {"OPENAI_API_KEY": "offline-secret-do-not-record", "OPENAI_MODEL": "openrouter/example/model",
       "OPENROUTER_PROVIDER": "Example", "PARSER": "canonical_pln", "OLLAMA_MODEL": "nomic-embed-text",
       "OLLAMA_URL": "http://127.0.0.1:11434/api/embeddings", "B1_QDRANT_URL": "http://127.0.0.1:6333",
       "B1_LLM_MIN_INTERVAL": "0", "B1_MAX_LLM_CALLS": "2000"}


class Failure(Exception):
    def __init__(self, kind):
        self.kind = kind


class FakeCore:
    """Only external normal-RAG dependencies are replaced; B2 adapter stays real."""
    Failure = Failure
    strict_json = staticmethod(json.loads)

    @staticmethod
    def validate_sentences(sentences):
        if any(set(s) != {"id", "text"} for s in sentences):
            raise ValueError()
        return sentences

    @staticmethod
    def embed(text, settings, http):
        return json.loads(http.post(settings.ollama_url, {"prompt": text}, {}, 1).body)["embedding"]


class Transport:
    def __init__(self, content=None, status=200):
        self.requests = []
        self.content, self.status = content, status

    def post(self, url, payload, headers, timeout):
        self.requests.append(payload)
        if "messages" not in payload:
            return SimpleNamespace(status=200, body=b'{"embedding":[1.0,0.0]}')
        body = {"model": "actual-model", "provider": "actual-provider",
                "usage": {"prompt_tokens": 12, "completion_tokens": 0, "total_tokens": 12},
                "choices": [{"finish_reason": "stop", "message": {"content": self.content if self.content is not None else
                    ' {"explanation":"The cited sentence supports the hypothesis.","evidence_ids":["sent1"]}\n'}}]}
        if self.status != 200:
            body = {"error": {"message": "maximum context length exceeded"}}
        return SimpleNamespace(status=self.status, body=json.dumps(body).encode())


def baseline_job(world, folder, system="full_context"):
    return {"system": system, "mode": "answer", "world_id": world["id"],
            "sentences": world["sentences"], "hypothesis": world["hypothesis"],
            "normal_rag_repo": "unused", "ollama_url": ENV["OLLAMA_URL"], "embedding_timeout": 1,
            "llm_timeout": 1, "top_k": 4, "llm_config": asdict(LLMSettings.from_env(ENV)),
            "request_budget_path": str(folder / "budget.json")}


def native_job(world, folder, mode="learn"):
    return {"system": "pln_rag", "world_id": world["id"], "mode": mode,
            "llm_config": asdict(LLMSettings.from_env(ENV)), "llm_timeout": 1,
            "request_budget_path": str(folder / "request_budget.json"), "repo_path": "/app",
            "work_dir": str(folder / "work"), "qdrant_url": ENV["B1_QDRANT_URL"],
            "collection": "offline_collection", "ollama_url": ENV["OLLAMA_URL"],
            "embedding_model": "nomic-embed-text", "pln_top_k": 10, "reasoning_timeout": 30,
            "sentences": world["sentences"]}


class BundleTests(unittest.TestCase):
    def test_bundle_original_inputs_counts_and_depth_discrepancy(self):
        bundle = validate()
        self.assertEqual(bundle["depth_counts"], dict.fromkeys(range(1, 8), 6))
        self.assertEqual(len({w["id"] for w in bundle["worlds"]}), 42)
        for w, r in zip(bundle["worlds"], bundle["references"]):
            self.assertEqual(len(w["sentences"]), 25)
            self.assertEqual({s["id"]: s["text"] for s in w["sentences"]}, r["source_record"]["meta"]["triples"])
            self.assertEqual(w["hypothesis"], r["source_record"]["hypothesis"])
        changed = [r for r in bundle["references"] if r["discrepancies"]]
        self.assertEqual([r["source_id"] for r in changed], ["Mercury_400025"])
        self.assertEqual(changed[0]["source_record"]["depth_of_proof"], 5)
        self.assertEqual(changed[0]["calculated_depth"], 4)

    def test_deterministic_order(self):
        source = validate()["source"]
        self.assertEqual(selected_ids(source["source_order_groups"]), source["selected_source_ids"])
        self.assertEqual(source["calculated_depths"], list(range(1, 8)) * 6)
        self.assertEqual(selected_ids(deepcopy(source["source_order_groups"])), source["selected_source_ids"])

    def test_damaged_bundle_and_inconsistent_metadata(self):
        for mutation in ("bytes", "depth", "missing", "public", "order", "recipe"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                dest = Path(folder) / "bundle"
                shutil.copytree(ROOT / "data/b2", dest)
                refs, worlds, source = read(dest / "references.json"), read(dest / "worlds.json"), read(dest / "source.json")
                if mutation == "bytes":
                    (dest / "worlds.json").write_text("damaged")
                elif mutation == "depth":
                    refs[0]["calculated_depth"] = 7
                    write(dest / "references.json", refs)
                    source["checksums"]["references.json"] = digest(dest / "references.json")
                elif mutation == "missing":
                    write(dest / "references.json", refs[:-1])
                    source["checksums"]["references.json"] = digest(dest / "references.json")
                elif mutation == "public":
                    worlds[0]["sentences"][0]["text"] = "changed"
                    write(dest / "worlds.json", worlds)
                    source["checksums"]["worlds.json"] = digest(dest / "worlds.json")
                elif mutation == "order":
                    source["selected_source_ids"].reverse()
                else:
                    source["seed"] = 18
                write(dest / "source.json", source)
                with self.assertRaises(ValueError):
                    validate(dest)

    def test_graph_branching_and_malformed_references(self):
        record = deepcopy(validate()["references"][6]["source_record"])
        result = graph(record)
        self.assertEqual(result["calculated_depth"], 7)
        self.assertTrue(any(len(p) > 1 for p in result["steps"].values()))
        mutations = [lambda r: r.update(proof=r["proof"].replace("sent", "missing", 1)),
                     lambda r: r["meta"]["intermediate_conclusions"].update(int999="unreachable"),
                     lambda r: r["meta"]["intermediate_conclusions"].update({r["meta"]["hypothesis_id"]: "wrong text"}),
                     lambda r: r.update(proof="int1 -> int1: " + r["meta"]["intermediate_conclusions"]["int1"] + ";"),
                     lambda r: r.update(proof=r["proof"] + r["proof"])]
        for mutate in mutations:
            changed = deepcopy(record)
            mutate(changed)
            with self.assertRaises(ValueError):
                graph(changed)

    def test_author_refuses_wrong_pinned_source(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "source checksum"):
                author(ROOT / "data/b2/worlds.json", ROOT / "data/b2/LICENSE", folder)


class EvidenceTests(unittest.TestCase):
    def test_perfect_partial_empty_unavailable_unknown_and_unseen(self):
        self.assertEqual(overlap(["a", "b"], ["a", "b"])["f1"], 1)
        self.assertAlmostEqual(overlap(["a"], ["a", "b"])["f1"], 2/3)
        self.assertEqual(overlap([], ["a"])["f1"], 0)
        self.assertIsNone(overlap(None, ["a"], reason="timeout")["f1"])
        self.assertEqual(overlap(["a", "unknown"], ["a"])["precision"], .5)
        self.assertEqual(overlap(["a", "b"], ["a", "b"], eligible=["a"])["precision"], .5)

    def test_pln_exact_terms_collisions_unresolved_and_empty(self):
        learned = [{"sentence_id": "sent1", "atoms": ["(: a (IsA x y) (STV 1 1))"]},
                   {"sentence_id": "sent2", "atoms": ["(: b (IsA y z) (STV 1 1))"]}]
        self.assertEqual(map_proof("['(: a (IsA b x) (STV 1 1))']", learned)["ids"], ["sent1"])
        self.assertEqual(map_proof("['(: (ModusPonens a b) (IsA x z) (STV 1 1))']", learned)["ids"], ["sent1", "sent2"])
        self.assertEqual(map_proof("[]", learned)["ids"], [])
        self.assertIsNone(map_proof("['(: unknown (IsA x z) (STV 1 1))']", learned)["ids"])
        self.assertIsNone(map_proof("not a trace", learned)["ids"])
        ambiguous = learned + [{"sentence_id": "sent3", "atoms": learned[0]["atoms"]}]
        self.assertIsNone(map_proof("['(: a (IsA x y) (STV 1 1))']", ambiguous)["ids"])
        self.assertIsNone(map_atoms(learned[0]["atoms"], ambiguous)["ids"])
        self.assertEqual(map_atoms(learned[0]["atoms"], learned)["ids"], ["sent1"])

    def test_full_context_retrieval_unavailable_evidence_independent(self):
        bundle = validate()
        w, ref = bundle["worlds"][0], bundle["references"][0]
        result = evaluate({"system": "full_context", "context_ids": [s["id"] for s in w["sentences"]],
                           "used_ids": []}, ref, w, 4)
        self.assertIsNone(result["evidence"]["retrieved"]["recall"])
        self.assertEqual(result["evidence"]["retrieved"]["reason"], "retrieval_not_used")
        self.assertEqual(result["evidence"]["context"]["recall"], 1)
        self.assertEqual(result["evidence"]["used"]["f1"], 0)

    def test_failed_native_proof_is_unavailable_not_empty_or_successful(self):
        bundle = validate()
        w, ref = bundle["worlds"][0], bundle["references"][0]
        row = evaluate({"system": "pln_rag", "status": "ERROR", "error_kind": "timeout",
                        "used_ids": ref["gold_leaves"], "empty_proof": False}, ref, w, 4)
        self.assertIsNone(row["evidence"]["used"]["f1"])
        self.assertEqual(row["evidence"]["used"]["reason"], "timeout")
        self.assertIsNone(row["diagnostics"]["empty_proof"])


class AdapterTests(unittest.TestCase):
    def test_same_prompt_settings_full_inputs_and_no_embeddings(self):
        world = validate()["worlds"][0]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            transports, rows = [], []
            for mode in ("rag", "full_context"):
                transport = Transport()
                row = baseline.run(baseline_job(world, root, mode), ENV, root / mode, core=FakeCore, transport=transport)
                transports.append(transport)
                rows.append(row)
                self.assertEqual(row["status"], "OK")
                self.assertEqual(row["llm_http_calls"][0]["usage"]["completion_tokens"], 0)
            chats = [t.requests[-1] for t in transports]
            self.assertEqual(chats[0]["messages"][0], chats[1]["messages"][0])
            for key in ("model", "temperature", "max_tokens", "response_format", "provider", "transforms"):
                self.assertEqual(chats[0][key], chats[1][key])
            data = json.loads(chats[1]["messages"][1]["content"])
            self.assertEqual(data, {"sentences": world["sentences"], "hypothesis": world["hypothesis"]})
            self.assertEqual(rows[0]["embedding_calls"], 26)
            self.assertEqual(rows[1]["embedding_calls"], 0)
            self.assertEqual(len(transports[1].requests), 1)
            self.assertIsNone(rows[1]["index_seconds"])
            self.assertEqual(len(rows[0]["retrieved_ids"]), 4)
            self.assertTrue((root / "full_context/response.body").exists())
            self.assertTrue((root / "full_context/raw_model_output.txt").read_text().startswith(" {"))

    def test_empty_malformed_unknown_and_overlong_responses_are_retained(self):
        world = validate()["worlds"][0]
        cases = [(' {"explanation":"","evidence_ids":[]}', 200, "OK"),
                 ('{"explanation":"support","evidence_ids":["sent999"]}', 200, "OK"),
                 ("not json", 200, "ERROR"), (None, 400, "ERROR")]
        for raw, status, expected in cases:
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                transport = Transport(raw, status)
                row = baseline.run(baseline_job(world, root), ENV, root / "op", core=FakeCore, transport=transport)
                self.assertEqual(row["status"], expected)
                self.assertEqual(len(transport.requests), 1)
                self.assertEqual(len(json.loads(transport.requests[0]["messages"][1]["content"])["sentences"]), 25)
                self.assertTrue((root / "op/response.body").exists())
                if status == 400:
                    self.assertEqual(row["error_kind"], "context_window_exceeded")

    def test_worker_payload_whitelist_and_query_guard(self):
        world = validate()["worlds"][0]
        job = baseline_job(world, Path("/tmp"))
        validate_job(job)
        for key in ("depth", "gold_leaves", "references", "proof", "question", "answer"):
            with self.assertRaises(ValueError):
                validate_job({**job, key: "secret evaluator value"})
        parser = SimpleNamespace(parse_query=lambda *a: SimpleNamespace(statements=["hypothesis-as-fact"], queries=["native query"]))
        handler = SimpleNamespace(add_atom=lambda *a: self.fail("must not ingest target"))
        reasoner = SimpleNamespace(_handler=handler)
        observations = []
        block_query_learning(parser, reasoner, observations)
        parsed = parser.parse_query(world["hypothesis"], [])
        self.assertEqual(parsed.statements, [])
        self.assertEqual(parsed.queries, ["native query"])
        self.assertEqual(observations[0]["blocked_statements"], ["hypothesis-as-fact"])
        with self.assertRaises(RuntimeError):
            reasoner.add_statements(["hypothesis"])
        with self.assertRaises(RuntimeError):
            handler.add_atom("hypothesis")

    def test_absent_api_usage_is_unavailable_and_raw_response_survives(self):
        class NoUsage(Transport):
            def post(self, *args):
                response = super().post(*args)
                body = json.loads(response.body)
                body["usage"] = None
                response.body = json.dumps(body).encode()
                return response
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = baseline.run(baseline_job(validate()["worlds"][0], root), ENV, root / "op",
                               core=FakeCore, transport=NoUsage())
            self.assertEqual(row["status"], "OK")
            self.assertEqual(row["llm_http_calls"][0]["usage"]["total_tokens"], "unmeasured")
            self.assertTrue((root / "op/response.body").exists())


class NativeWorkerTests(unittest.TestCase):
    def test_real_adapter_restores_snapshot_blocks_query_writes_and_observes_swallowed_timeout(self):
        """Run the adapter end to end with only native dependencies substituted."""
        atom = "(: fact (IsA x y) (STV 1 1))"
        class Handler:
            timeout = False
            def add_atom(self, value):
                pass
            def query(self, *args, **kwargs):
                if self.timeout:
                    raise TimeoutError("mock native deadline")
                return [atom]
        class Parser:
            def parse_query(self, text, context):
                return SimpleNamespace(statements=["hypothesis-as-fact"], queries=["native query"])
        class Service:
            def __init__(self, parser):
                self.parser = parser
                self._context_top_k = 10
                handler = Handler()
                handler.add_atom(atom)
                self._reasoner = SimpleNamespace(_handler=handler)
                self._vector_store = SimpleNamespace(
                    embed=lambda text: [1.0], store=lambda *a: None,
                    retrieve_context=lambda text, top_k: ([atom], [1.0]),
                    _client=SimpleNamespace(event_hooks={}))
            async def query(self, hypothesis):
                context, _ = self._vector_store.retrieve_context(hypothesis, 10)
                parsed = self.parser.parse_query(hypothesis, context)
                if parsed.statements:
                    self._reasoner.add_statements(parsed.statements)
                try:
                    traces = self._reasoner._handler.query(parsed.queries[0])
                except TimeoutError:  # The real native reasoner also swallows failures.
                    traces = []
                response = {"executed_query": parsed.queries[0], "query_status": "ok", "raw_proof": str(traces)}
                return SimpleNamespace(model_dump=lambda **kwargs: response)
        settings = SimpleNamespace(chaining_timeout=30, query_candidate_max_tries=5,
            chaining_max_steps=100, answer_generation_enabled=False, conceptnet_enabled=False,
            parser_batch_sentences=1, chunk_size=512, chunk_overlap=64)
        get_settings = lambda: settings
        get_settings.cache_clear = lambda: None
        class LM:
            def __init__(self, *args, **kwargs):
                pass
        modules = {"dspy": SimpleNamespace(LM=LM, configure=lambda **kw: None),
            "config": SimpleNamespace(get_settings=get_settings),
            "core.service": SimpleNamespace(PLNRAGService=Service),
            "core.reasoner": SimpleNamespace(PeTTaChainer=Handler),
            "parsers": SimpleNamespace(get_parser=Parser)}
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, modules), \
                patch.dict(os.environ, ENV), patch("b2.pln_worker.DurableAudit.install"), \
                patch.object(sys, "path", list(sys.path)), patch.object(sys, "dont_write_bytecode", True):
            root = Path(folder)
            snapshot = root / "learned.metta"
            snapshot.write_text(atom + "\n")
            snapshot.chmod(0o444)
            write(root / "learning.json", {"sentences": [{"sentence_id": "sent1", "atoms": [atom]}]})
            for timeout in (False, True):
                Handler.timeout = timeout
                out = root / str(timeout)
                out.mkdir()
                job = {"mode": "query", "repo_path": str(root), "work_dir": str(out / "work"),
                    "llm_config": asdict(LLMSettings.from_env(ENV)), "qdrant_url": ENV["B1_QDRANT_URL"],
                    "collection": "isolated", "ollama_url": ENV["OLLAMA_URL"], "embedding_model": "nomic-embed-text",
                    "pln_top_k": 10, "reasoning_timeout": 30, "request_budget_path": str(root / "budget"),
                    "llm_timeout": 1, "snapshot_path": str(snapshot), "snapshot_sha256": digest(snapshot),
                    "learn_output_path": str(root / "learning.json"), "hypothesis": "x is y"}
                row = run_native(job, out)
                self.assertEqual(row["status"], "ERROR" if timeout else "OK", row)
                self.assertEqual(row["timed_out"], timeout)
                self.assertEqual(row["used_ids"], None if timeout else ["sent1"])
                self.assertEqual(row["blocked_query_statements"][0]["blocked_statements"], ["hypothesis-as-fact"])
                self.assertEqual(row["before_query_sha256"], row["after_query_sha256"])
                self.assertEqual(row["parser_context_mapping"]["ids"], ["sent1"])
                self.assertTrue((out / "trace.json").exists())
                with self.assertRaises(RuntimeError):
                    # The original public guard covers vector writes as well.
                    vector = SimpleNamespace(store=lambda *a: None)
                    block_query_learning(Parser(), SimpleNamespace(_handler=Handler()), [], vector)
                    vector.store("query fact")


class MockRuntime(Runtime):
    instances = []
    fail_learning = False

    def __init__(self, root, output, env, **kwargs):
        super().__init__(root, output, env, **{**kwargs, "pln_runtime": "local"})
        self.jobs = []
        self.instances.append(self)

    def execute(self, job, operation, timeout):
        validate_job(job)
        self.jobs.append(deepcopy(job))
        operation.mkdir(parents=True)
        write(operation / "job.json", job)
        if self.fail_learning and job["mode"] == "learn":
            return {"status": "ERROR", "error_kind": "timeout", "worker_seconds": 1}
        llm = LLMSettings(**job["llm_config"])
        calls = []
        for _ in range(25 if job["mode"] == "learn" else 1):
            number, wait = reserve_request(Path(job["request_budget_path"]), llm.max_calls, llm.minimum_interval)
            calls.append({"request_number": number, "phase": job["mode"], "model": "native-returned-model",
                          "provider": "native-returned-provider", "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}})
        row = {"status": "OK", "attempted": True, "llm_http_calls": calls, "embedding_calls": len(calls)}
        if job["mode"] == "learn":
            row.update(snapshot_path=str(operation / "snapshot"), snapshot_sha256="mocked", learning_seconds=.2,
                       parse_failure_count=0, sentences=[{"sentence_id": s["id"], "atoms": ["mock atom"]} for s in job["sentences"]])
        else:
            row.update(context_ids=[f"sent{i}" for i in range(1, 26)], retrieved_ids=["sent1"], used_ids=[],
                       empty_proof=True, unresolved_sources=[], answer_seconds=.1, raw_response={"raw_proof": "[]"})
            write(operation / "trace.json", row["raw_response"])
        write(operation / "requests.json", calls)
        write(operation / "output.json", row)
        return row


class RunnerTests(unittest.TestCase):
    def run_benchmark(self, root, **kwargs):
        with patch("b2.baseline.load_baseline", return_value=(FakeCore, None)), patch.object(FakeCore, "HTTP", Transport, create=True):
            return benchmark(root, dataset_path=ROOT / "data/b2", runtime_factory=MockRuntime, **kwargs)

    def test_complete_126_records_matched_inputs_settings_and_audits(self):
        with tempfile.TemporaryDirectory() as folder, patch("b2.runner.environment", return_value=dict(ENV)):
            out = self.run_benchmark(Path(folder))
            records, summary = read(out / "records.json"), read(out / "summary.json")
            self.assertEqual(len(records), 126)
            self.assertEqual(Counter(r["system"] for r in records), {s: 42 for s in ("rag", "pln_rag", "full_context")})
            self.assertEqual(summary["requests_reserved"], 1176)
            self.assertEqual(summary["request_audits"], 1176)
            self.assertTrue(summary["execution_complete"])
            self.assertEqual(summary["overall"]["world_count"], 42)
            self.assertEqual(summary["overall"]["record_count"], 126)
            self.assertEqual(summary["by_depth"]["1"]["world_count"], 6)
            self.assertEqual(summary["systems"]["full_context"]["overall"]["metrics"]["context.recall"]["mean"], 1)
            jobs = MockRuntime.instances[-1].jobs + [read(p) for system in ("rag", "full_context")
                for p in sorted((out / system).glob("*/job.json"))]
            by_id = {w["id"]: w for w in validate()["worlds"]}
            for job in jobs:
                self.assertEqual(job["llm_config"], asdict(LLMSettings.from_env(ENV)))
                if "sentences" in job:
                    self.assertEqual(job["sentences"], by_id[job["world_id"]]["sentences"])
                if job["mode"] == "learn":
                    self.assertNotIn("hypothesis", job)
                else:
                    self.assertEqual(job["hypothesis"], by_id[job["world_id"]]["hypothesis"])
            for p in out.rglob("*"):
                if p.is_file():
                    self.assertNotIn(ENV["OPENAI_API_KEY"].encode(), p.read_bytes())
            self.assertTrue((out / "pln_rag/b2-001/query/trace.json").exists())
            self.assertIn("RAG used F1/status", (out / "report.md").read_text())

    def test_system_first_order_main_process_baselines_and_output_paths(self):
        events, jobs = [], []
        original_execute, original_baseline = MockRuntime.execute, baseline.run
        def execute(runtime, job, operation, timeout):
            events.append((job["system"], job["world_id"], job["mode"]))
            jobs.append(job)
            return original_execute(runtime, job, operation, timeout)
        def answer(job, env, operation, **kwargs):
            self.assertEqual(os.getpid(), parent_pid)
            events.append((job["system"], job["world_id"], job["mode"]))
            jobs.append(job)
            return original_baseline(job, env, operation, **kwargs)
        parent_pid = os.getpid()
        systems = ("rag", "pln_rag", "full_context")
        with tempfile.TemporaryDirectory() as folder, patch("b2.runner.environment", return_value=dict(ENV)), \
             patch.object(MockRuntime, "execute", execute), patch("b2.baseline.run", side_effect=answer):
            out = self.run_benchmark(Path(folder), world_limit=2, systems=systems)
            expected = [(s, w, mode) for s in systems for w in ("b2-001", "b2-002")
                        for mode in (("learn", "query") if s == "pln_rag" else ("answer",))]
            self.assertEqual(events, expected)
            self.assertEqual(read(out / "manifest.json")["system_order"], list(systems))
            self.assertEqual({j["request_budget_path"] for j in jobs}, {str(out / "request_budget.json")})
            self.assertFalse((out / "workers").exists())
            self.assertFalse((out / "worker-code").exists())
            collections = [j["collection"] for j in jobs if j["mode"] == "learn"]
            self.assertEqual(len(set(collections)), 2)
            for system, world, mode in events:
                operation = out / system / world
                if system == "pln_rag":
                    operation /= mode
                for name in ("job.json", "output.json", "requests.json"):
                    self.assertTrue((operation / name).is_file(), operation / name)
            for job in (j for j in jobs if j["mode"] == "query"):
                learned = read(Path(job["learn_output_path"]))
                self.assertEqual(job["snapshot_path"], learned["snapshot_path"])
                self.assertEqual(job["snapshot_sha256"], learned["snapshot_sha256"])
                self.assertEqual(job["collection"], next(j["collection"] for j in jobs
                    if j["mode"] == "learn" and j["world_id"] == job["world_id"]))
            calls = sorted(read(out / "llm_requests.json"), key=lambda c: c["request_number"])
            self.assertEqual([c["system"] for c in calls], ["rag"] * 2 + ["pln_rag"] * 52 + ["full_context"] * 2)

    def test_pln_consumes_shared_budget_before_baselines(self):
        with tempfile.TemporaryDirectory() as folder, patch("b2.runner.environment", return_value={**ENV, "B1_MAX_LLM_CALLS": "52"}):
            out = self.run_benchmark(Path(folder), world_limit=2)
            rows, summary = read(out / "records.json"), read(out / "summary.json")
            self.assertEqual((summary["requests_reserved"], summary["request_audits"]), (52, 52))
            self.assertTrue(all(r["status"] == "OK" for r in rows if r["system"] == "pln_rag"))
            self.assertTrue(all(r["error_kind"] == "request_budget_exceeded" and not r["attempted"]
                                for r in rows if r["system"] != "pln_rag"))

    def test_baseline_only_never_constructs_runtime_or_launches_processes(self):
        with tempfile.TemporaryDirectory() as folder, patch("b2.runner.environment", return_value=dict(ENV)), \
             patch("b2.baseline.load_baseline", return_value=(FakeCore, None)), \
             patch.object(FakeCore, "HTTP", Transport, create=True), \
             patch("b2.runtime.subprocess.Popen", side_effect=AssertionError("No subprocess")), \
             patch("b2.runner.implementation", return_value={}):
            factory = Mock(side_effect=AssertionError("No PLN runtime"))
            out = benchmark(Path(folder), dataset_path=ROOT / "data/b2", systems=("rag", "full_context"),
                            world_limit=2, runtime_factory=factory)
            factory.assert_not_called()
            self.assertTrue(read(out / "summary.json")["execution_complete"])
            self.assertFalse((out / "docker.launch.json").exists())
            for system in ("rag", "full_context"):
                self.assertEqual(read(out / system / "b2-001/launch.json")["execution"], "main_process")
                self.assertTrue((out / system / "b2-001/raw.log").exists())

    def test_pln_setup_failure_still_runs_baselines(self):
        with tempfile.TemporaryDirectory() as folder, patch("b2.runner.environment", return_value=dict(ENV)), \
             patch.object(MockRuntime, "run_pln", side_effect=RuntimeError("Docker unavailable")):
            out = self.run_benchmark(Path(folder), world_limit=2)
            rows = read(out / "records.json")
            self.assertTrue(all(r["status"] == "OK" for r in rows if r["system"] != "pln_rag"))
            self.assertTrue(all(r["error_kind"] == "pln_not_run" for r in rows if r["system"] == "pln_rag"))
            self.assertEqual(read(out / "manifest.json")["driver_errors"][0]["detail"], "Docker unavailable")

    def test_interrupt_recovers_durable_audit_and_reports_all_pairs(self):
        def interrupted(transport, url, payload, headers, timeout):
            raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as folder, patch("b2.runner.environment", return_value=dict(ENV)), \
             patch.object(Transport, "post", interrupted):
            out = self.run_benchmark(Path(folder), world_limit=2, systems=("full_context",))
            summary = read(out / "summary.json")
            self.assertEqual(summary["status"], "INTERRUPTED")
            self.assertEqual((summary["requests_reserved"], summary["request_audits"]), (1, 1))
            rows = read(out / "records.json")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["error_kind"], "interrupted")
            self.assertFalse(rows[1]["attempted"])
            self.assertEqual(read(out / "full_context/b2-001/output.json")["error_kind"], "interrupted")

    def test_dotenv_b2_overrides_reach_all_public_jobs(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            root = Path(folder)
            (root / ".env").write_text("\n".join(f"{k}={v}" for k, v in ENV.items()) +
                "\nB1_LLM_MAX_TOKENS=500\nB2_LLM_MAX_TOKENS=77\nB2_MAX_LLM_CALLS=300\n"
                "B2_LLM_TEMPERATURE=0.3\nB2_QUERY_TIMEOUT=17\nB2_LEARN_TIMEOUT=25\n"
                "B2_LLM_TIMEOUT=9\nB2_REASONING_TIMEOUT=7\nB2_EMBEDDING_TIMEOUT=8\n"
                "B2_QDRANT_URL=http://qdrant:6333\nB2_LLM_MIN_INTERVAL=0.2\n")
            with patch.dict(os.environ, {"B2_LLM_MAX_TOKENS": "99"}):
                env = environment(root)
            self.assertEqual(env["B1_LLM_MIN_INTERVAL"], "0.2")
            # Verify propagation without actually pacing the offline requests.
            reserve = reserve_request
            def unpaced(path, limit, interval):
                return reserve(path, limit, interval, sleep=lambda _: None)
            with patch(__name__ + ".reserve_request", unpaced), patch("b2.baseline.reserve_request", unpaced), \
                 patch("b2.runner.environment", return_value=env):
                out = self.run_benchmark(root, world_limit=1, top_k=3, pln_top_k=11)
            jobs = [read(p) for p in out.rglob("job.json")]
            for job in jobs:
                self.assertEqual(job["llm_config"]["max_tokens"], 99)
                self.assertEqual(job["llm_config"]["max_calls"], 300)
                self.assertEqual(job["llm_config"]["minimum_interval"], .2)
                self.assertEqual(job["llm_config"]["temperature"], .3)
                self.assertEqual(job["llm_timeout"], 9)
                if job["system"] == "pln_rag":
                    self.assertEqual(job["reasoning_timeout"], 7)
                    self.assertEqual(job["pln_top_k"], 11)
                    self.assertEqual(job["qdrant_url"], "http://qdrant:6333")
                else:
                    self.assertEqual(job["embedding_timeout"], 8)
                    self.assertEqual(job["top_k"], 3)
            self.assertEqual(read(out / "manifest.json")["timeouts"]["LEARN_TIMEOUT"], 25)

    def test_shared_cap_and_missing_usage_remain_honest(self):
        with tempfile.TemporaryDirectory() as folder, patch("b2.runner.environment", return_value={**ENV, "B1_MAX_LLM_CALLS": "1"}):
            out = self.run_benchmark(Path(folder), systems=("full_context", "rag"), world_limit=2)
            rows, summary = read(out / "records.json"), read(out / "summary.json")
            self.assertEqual(len(rows), 4)
            self.assertEqual(summary["requests_reserved"], 1)
            self.assertEqual(sum(r["status"] == "NOT_ATTEMPTED" for r in rows), 3)
            for r in rows[1:]:
                self.assertIsNone(r["resources"]["total_tokens"])
                self.assertIsNone(r["evidence"]["used"]["f1"])

    def test_learning_timeout_and_missing_credentials_account_for_all_pairs(self):
        with tempfile.TemporaryDirectory() as folder, patch("b2.runner.environment", return_value=dict(ENV)), patch.object(MockRuntime, "fail_learning", True):
            out = self.run_benchmark(Path(folder), world_limit=2)
            rows = read(out / "records.json")
            self.assertEqual(len(rows), 6)
            for row in [r for r in rows if r["system"] == "pln_rag"]:
                self.assertTrue(row["attempted"])
                self.assertFalse(row["answer_attempted"])
                self.assertTrue(row["timed_out"])
            summary = read(out / "summary.json")
            timeouts = summary["systems"]["pln_rag"]["overall"]["metrics"]["timeout_rate"]
            self.assertEqual(timeouts["mean"], 1)
            self.assertEqual((timeouts["measured"], timeouts["total"]), (2, 2))
            self.assertEqual(sum(r["status"] == "OK" for r in rows), 4)
        with tempfile.TemporaryDirectory() as folder, patch("b2.runner.environment", return_value={**ENV, "OPENAI_API_KEY": ""}):
            out = self.run_benchmark(Path(folder), world_limit=2)
            self.assertEqual(len(read(out / "records.json")), 6)
            self.assertEqual(read(out / "summary.json")["requests_reserved"], 0)

    def test_offline_plans_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder, patch("b2.runner.environment", return_value={**ENV, "OPENAI_API_KEY": ""}), patch("b2.runtime.inspect_container", side_effect=AssertionError("No Docker")):
            root = Path(folder)
            for count, systems, expected in [(42, ("pln_rag", "rag", "full_context"), 1176), (2, ("pln_rag", "rag", "full_context"), 56), (2, ("full_context",), 2)]:
                out = self.run_benchmark(root, prepare_only=True, world_limit=count, systems=systems)
                plan = read(out / "plan.json")
                self.assertEqual(plan["request_estimate"]["chat_calls"], expected)
                self.assertEqual(plan["status"], "PREPARED")
                self.assertFalse((out / "worker-code").exists())
                self.assertFalse((out / "request_budget.json").exists())
            with self.assertRaises(FileExistsError):
                self.run_benchmark(root, prepare_only=True, output_dir=out)


class IsolationTests(unittest.TestCase):
    def test_docker_supervisor_and_children_preserve_image_pythonpath(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            code, native = root / "repository code", root / "image dependencies"
            output = code / "artifacts/b2/run"
            for directory in (code / "b2", native / "b2", output):
                directory.mkdir(parents=True)
            (code / "b2/__init__.py").write_text("")
            (code / "b2/runtime.py").write_text(
                "import json, os, subprocess, sys, petta\n"
                "child = subprocess.check_output([sys.executable, '-m', 'b2.worker'], text=True)\n"
                "print(json.dumps({'worker': __file__, 'native': petta.__file__, "
                "'args': sys.argv[1:], 'pythonpath': os.environ['PYTHONPATH'], 'child': json.loads(child)}))\n")
            (code / "b2/worker.py").write_text(
                "import json, os, petta\nprint(json.dumps({'native': petta.__file__, 'pythonpath': os.environ['PYTHONPATH']}))\n")
            (native / "petta.py").write_text("# Native dependency supplied by the image.\n")
            (native / "b2/__init__.py").write_text("raise RuntimeError('Must load the mounted repository')\n")
            identity = {"image": "sha256:immutable", "parser_mounts": []}
            command = docker_command(identity, code, output, "b2-offline")
            image_index = command.index(identity["image"])
            env = {"PATH": os.environ["PATH"], "PYTHONPATH": str(native)}
            for index, argument in enumerate(command[:image_index]):
                if argument == "--env":
                    key, separator, value = command[index + 1].partition("=")
                    if separator:
                        env[key] = value
            result = subprocess.run([sys.executable, *command[image_index + 1:]], cwd=code,
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(result.stdout)
            self.assertEqual(value["worker"], str(code / "b2/runtime.py"))
            self.assertEqual(value["native"], str(native / "petta.py"))
            self.assertEqual(value["args"], ["--inside-pln", str(output / "pln_job.json")])
            self.assertEqual(value["pythonpath"], str(native))
            self.assertEqual(value["child"], {"native": str(native / "petta.py"), "pythonpath": str(native)})

    def test_repository_parser_and_external_output_mounts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "repository"
            parser = {"source": str(Path(folder) / "parser.json"), "destination": "/app/data/simba_all.json"}
            identity = {"image": "sha256:immutable", "parser_mounts": [parser]}
            for output, count in ((root / "artifacts/b2/run", 2), (Path(folder) / "external run", 3)):
                command = docker_command(identity, root, output, "b2-offline")
                mounts = [command[i+1] for i, arg in enumerate(command) if arg == "--mount"]
                self.assertEqual(len(mounts), count)
                self.assertIn(f"type=bind,src={root},dst={root}", mounts)
                self.assertEqual(command[command.index("--workdir") + 1], str(root))
                self.assertIn(f"type=bind,src={parser['source']},dst={parser['destination']},readonly", mounts)
                self.assertNotIn("--volumes-from", command)
                self.assertIn("OPENAI_API_KEY", command)
                self.assertNotIn(ENV["OPENAI_API_KEY"], " ".join(command))
                self.assertFalse(any("PYTHONPATH" in arg or "SWI_HOME_DIR" in arg or "LD_LIBRARY_PATH" in arg for arg in command))
                self.assertEqual(command[command.index("--user") + 1], f"{os.geteuid()}:{os.getegid()}")
                self.assertEqual(command[command.index("--pull") + 1], "never")
            with self.assertRaises(ValueError):
                docker_command(identity, root, Path(folder) / "comma,path", "b2-offline")

    def test_inspection_only_reuses_immutable_image_and_parser_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            parser = Path(folder) / "parser.json"
            parser.write_text('{}')
            info = {"Image": "sha256:immutable", "State": {"Running": True}, "Mounts": [
                {"Type": "volume", "Source": "/live", "Destination": "/app/data"},
                {"Type": "bind", "Source": str(parser), "Destination": "/app/data/simba_all.json"}]}
            result = SimpleNamespace(returncode=0, stdout=json.dumps([info]))
            with patch("b2.runtime.subprocess.run", return_value=result):
                identity = inspect_container("source")
            self.assertEqual(identity["image"], "sha256:immutable")
            self.assertEqual(len(identity["parser_mounts"]), 1)
            self.assertEqual(identity["parser_mounts"][0]["sha256"], digest(parser))
            for changes in ({"Image": "mutable:tag"}, {"State": {"Running": False}}):
                result.stdout = json.dumps([{**info, **changes}])
                with patch("b2.runtime.subprocess.run", return_value=result), self.assertRaises(ValueError):
                    inspect_container("source")

    def test_one_container_for_all_worlds_and_cleanup_on_exit_or_interrupt(self):
        for outcome in (0, 1, KeyboardInterrupt()):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                job = {"run_id": "offline", "worlds": validate()["worlds"][:2]}
                runtime = Runtime(ROOT, output, ENV, pln_python=Path(sys.executable), pln_runtime="docker", container="source")
                process = Mock(pid=1234)
                process.wait.side_effect = [outcome, 0]
                process.poll.return_value = None if isinstance(outcome, KeyboardInterrupt) else outcome
                identity = {"image": "sha256:immutable", "parser_mounts": []}
                with patch("b2.runtime.inspect_container", return_value=identity) as inspect, \
                     patch("b2.runtime.subprocess.Popen", return_value=process) as popen, \
                     patch("b2.runtime.subprocess.run") as cleanup, patch("b2.runtime.os.killpg") as kill:
                    if isinstance(outcome, KeyboardInterrupt):
                        with self.assertRaises(KeyboardInterrupt):
                            runtime.run_pln(job)
                        kill.assert_called_once_with(1234, signal.SIGKILL)
                    elif outcome:
                        with self.assertRaises(RuntimeError):
                            runtime.run_pln(job)
                    else:
                        runtime.run_pln(job)
                    inspect.assert_called_once_with("source")
                    popen.assert_called_once()
                    command = popen.call_args.args[0]
                    self.assertEqual(command[:3], ["docker", "run", "--rm"])
                    name = command[command.index("--name") + 1]
                    self.assertTrue(name.startswith("b2-offline-"))
                    cleanup.assert_called_once_with(["docker", "rm", "-f", name], capture_output=True, timeout=30, check=False)
                self.assertEqual(read(output / "pln_job.json")["worlds"], job["worlds"])
                self.assertTrue((output / "docker.log").exists())
                self.assertFalse((output / "worker-code").exists())

    def test_real_process_timeout_retains_job_and_failure(self):
        world = validate()["worlds"][0]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            runtime = Runtime(ROOT, output, dict(os.environ), pln_python=Path(sys.executable),
                              pln_runtime="local", container="unused")
            job = native_job(world, output)
            original = subprocess.Popen
            def sleeper(*args, **kwargs):
                return original([sys.executable, "-c", "import time; time.sleep(5)"], **kwargs)
            operation = output / "pln_rag/b2-001/learn"
            with patch("b2.runtime.subprocess.Popen", side_effect=sleeper):
                row = runtime.execute(job, operation, .05)
            self.assertEqual(row["error_kind"], "timeout")
            self.assertIsNone(row["used_ids"])
            self.assertEqual(read(operation / "output.json"), row)
            self.assertTrue((operation / "job.json").exists())

    def test_baseline_deadline_retains_partial_audit_and_restores_timer(self):
        class SlowTransport(Transport):
            def post(self, *args):
                time.sleep(2)
                return super().post(*args)
        world = validate()["worlds"][0]
        previous = signal.getsignal(signal.SIGALRM)
        for system in ("rag", "full_context"):
            with self.subTest(system=system), tempfile.TemporaryDirectory() as folder, \
                 patch("b2.baseline.load_baseline", return_value=(FakeCore, None)), \
                 patch.object(FakeCore, "HTTP", SlowTransport, create=True):
                output = Path(folder)
                row = run_baseline(baseline_job(world, output, system), output / system, ENV, .05)
                self.assertEqual(row["error_kind"], "timeout")
                self.assertTrue(row["timed_out"])
                self.assertIsNone(row["used_ids"])
                self.assertEqual(len(row["llm_http_calls"]), 0 if system == "rag" else 1)
                self.assertEqual(read(output / system / "output.json"), row)
                self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
                self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))

    def test_fresh_worker_processes_use_repository_and_native_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            native, output = root / "native dependencies", root / "results"
            (root / "b2").mkdir()
            native.mkdir()
            (native / "petta.py").write_text("")
            (root / "b2/__init__.py").write_text("")
            (root / "b2/worker.py").write_text(
                "import json, os, pathlib, sys, petta\n"
                "job = pathlib.Path(sys.argv[2])\n"
                "(job.parent / 'output.json').write_text(json.dumps({'status': 'OK', 'pid': os.getpid(), "
                "'native': petta.__file__, 'cwd': os.getcwd(), 'pythonpath': os.environ['PYTHONPATH']}))\n")
            env = {**os.environ, "PYTHONPATH": str(native)}
            runtime = Runtime(root, output, env, pln_python=Path(sys.executable), pln_runtime="local", container=None)
            job = native_job(validate()["worlds"][0], output)
            learned = runtime.execute(job, output / "pln_rag/b2-001/learn", 5)
            job.pop("sentences")
            job.update(mode="query", hypothesis="public hypothesis", snapshot_path="unused",
                       snapshot_sha256="unused", learn_output_path="unused")
            queried = runtime.execute(job, output / "pln_rag/b2-001/query", 5)
            self.assertNotEqual(learned["pid"], queried["pid"])
            for row in (learned, queried):
                self.assertNotEqual(row["pid"], os.getpid())
                self.assertEqual(row["native"], str(native / "petta.py"))
                self.assertEqual(row["pythonpath"], str(native))
                self.assertEqual(row["cwd"], str(root))

    def test_clean_bundle_smoke_no_credentials_or_sibling_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for package in ("b1", "b2"):
                shutil.copytree(ROOT / package, root / package, ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copytree(ROOT / "data/b2", root / "data/b2")
            env = {"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1", "OPENAI_MODEL": "openai/gpt-4o-mini"}
            # sitecustomize denies sockets in the actual subprocess; preparation
            # must not inspect Docker or load sibling runtimes either.
            (root / "sitecustomize.py").write_text("import socket\ndef denied(*a, **k): raise RuntimeError('network forbidden')\nsocket.socket = denied\n")
            env["PYTHONPATH"] = str(root)
            for args in (("validate",), ("benchmark", "--prepare-only")):
                result = subprocess.run([sys.executable, "-m", "b2", *args], cwd=root, env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / ".env").exists())
            plan = read(next((root / "artifacts/b2").glob("*/plan.json")))
            self.assertEqual(plan["world_count"], 42)
            self.assertEqual(plan["request_estimate"]["chat_calls"], 1176)


if __name__ == "__main__":
    unittest.main()
