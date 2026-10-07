import asyncio
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from b1.environment import configured_environment
from b1.llm_control import HTTPAudit, LLMSettings, RequestBudgetExceeded, reserve_request, response_metadata
from b1.docker_runtime import launch_command

try:
    import httpx
except ImportError:
    httpx = None


class RequestControlTests(unittest.TestCase):
    def test_model_and_provider_are_configurable_and_routing_is_pinned(self):
        settings = LLMSettings.from_env({"OPENAI_MODEL": "openrouter/nvidia/example:free", "OPENROUTER_PROVIDER": "nvidia"})
        self.assertEqual(settings.model, "openrouter/nvidia/example:free")
        self.assertEqual(settings.max_calls, 40)
        self.assertEqual(settings.minimum_interval, 4)
        self.assertEqual(settings.kwargs()["extra_body"]["provider"], {
            "only": ["nvidia"], "order": ["nvidia"], "allow_fallbacks": False, "require_parameters": True})
        self.assertEqual(settings.kwargs()["num_retries"], 0)

    def test_free_routes_reject_unbounded_or_unpaced_calls(self):
        for extra in ({"B1_MAX_LLM_CALLS": "0"}, {"B1_LLM_MIN_INTERVAL": "0"},
                      {"B1_LLM_MAX_TOKENS": "0"}, {"B1_LLM_TEMPERATURE": "NaN"}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                LLMSettings.from_env({"OPENAI_MODEL": "openrouter/nvidia/example:free", "OPENROUTER_PROVIDER": "nvidia", **extra})
        with self.assertRaises(ValueError):
            LLMSettings.from_env({"OPENAI_MODEL": "openrouter/nvidia/example:free"})

    def test_budget_is_shared_across_reopened_files_and_stops_before_another_request(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            now = [100.0]
            def sleep(seconds):
                now[0] += seconds
            self.assertEqual(reserve_request(path, 2, 4, clock=lambda: now[0], sleep=sleep), (1, 0))
            self.assertEqual(reserve_request(path, 2, 4, clock=lambda: now[0], sleep=sleep), (2, 4))
            with self.assertRaises(RequestBudgetExceeded):
                reserve_request(path, 2, 4, clock=lambda: now[0], sleep=sleep)
            self.assertEqual(json.loads(path.read_text())["calls"], 2)

    def test_missing_usage_never_becomes_zero_or_inferred_provider(self):
        metadata = response_metadata({"usage": {"prompt_tokens": 3, "completion_tokens": 0}})
        self.assertEqual(metadata["provider"], "unmeasured")
        self.assertEqual(metadata["usage"], {"prompt_tokens": 3, "completion_tokens": "unmeasured", "total_tokens": "unmeasured"})
        metadata = response_metadata({"error": {"metadata": {"provider_name": "Nvidia"}}})
        self.assertEqual(metadata["provider"], "Nvidia")


@unittest.skipUnless(httpx, "HTTP transport tests require requirements-pln.txt")
class HTTPAuditTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.settings = LLMSettings("openrouter/nvidia/example:free", "https://openrouter.ai/api/v1", "nvidia", 0, 2000, 2, 0)
        self.errors = []
        self.audit = HTTPAudit(self.settings, Path(self.folder.name) / "budget.json", lambda: "query:q", self.errors)
        self.audit.install()

    def tearDown(self):
        self.audit.close()
        self.folder.cleanup()

    def body(self):
        return {"model": "nvidia/example:free", "messages": [{"role": "user", "content": "Question?"}],
                **self.settings.kwargs()["extra_body"]}

    def test_exact_response_metadata_and_no_auth_header_recording(self):
        raw = b' {"model":"nvidia/example:free", "provider":"Nvidia", "usage":{"total_tokens":17}}\n'
        with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=raw))) as client:
            response = client.post(self.settings.api_base + "/chat/completions", json=self.body(),
                                   headers={"Authorization": "Bearer private-secret"})
            self.assertEqual(response.content, raw)
        self.assertEqual(self.audit.calls[0]["raw_response"].encode(), raw)
        self.assertEqual(self.audit.calls[0]["provider"], "Nvidia")
        self.assertEqual(self.audit.calls[0]["usage"]["total_tokens"], 17)
        self.assertNotIn("private-secret", json.dumps(self.audit.calls))

    def test_budget_limits_actual_http_attempts(self):
        sent = []
        def handler(req):
            sent.append(req)
            return httpx.Response(200, json={})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            for _ in range(2):
                client.post(self.settings.api_base + "/chat/completions", json=self.body())
            with self.assertRaises(RequestBudgetExceeded):
                client.post(self.settings.api_base + "/chat/completions", json=self.body())
        self.assertEqual(len(sent), 2)

    def test_lost_provider_pin_is_rejected_before_network(self):
        sent = []
        with httpx.Client(transport=httpx.MockTransport(lambda req: sent.append(req))) as client:
            with self.assertRaisesRegex(RuntimeError, "provider pin"):
                client.post(self.settings.api_base + "/chat/completions", json={"model": "nvidia/example:free"})
        self.assertEqual(sent, [])
        self.assertEqual(self.audit.calls, [])

    def test_api_error_records_provider_and_prevents_sdk_retry(self):
        sent = []
        def handler(req):
            sent.append(req)
            return httpx.Response(429, json={"error": {"metadata": {"provider_name": "Nvidia"}}})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            response = client.post(self.settings.api_base + "/chat/completions", json=self.body())
            self.assertEqual(response.status_code, 429)
            with self.assertRaisesRegex(RuntimeError, "previous LLM request failed"):
                client.post(self.settings.api_base + "/chat/completions", json=self.body())
        self.assertEqual(len(sent), 1)
        self.assertEqual(self.errors[0]["kind"], "llm_api_error")
        self.assertEqual(self.audit.calls[0]["provider"], "Nvidia")

    def test_embeddings_do_not_consume_llm_budget(self):
        with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"embedding": [1, 0]}))) as client:
            client.post("http://127.0.0.1:11434/api/embeddings", json={"prompt": "sentence"})
        self.assertEqual(self.audit.calls, [])

    def test_async_path_has_the_same_accounting(self):
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"provider": "Nvidia"}))) as client:
                await client.post(self.settings.api_base + "/chat/completions", json=self.body())
        asyncio.run(run())
        self.assertEqual(self.audit.calls[0]["provider"], "Nvidia")


class DockerIsolationTests(unittest.TestCase):
    def test_launch_uses_image_digest_without_live_application_storage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parser = root / "simba_canonical_pln.json"
            parser.write_text("{}")
            info = {"Image": "sha256:fixed-image", "State": {"Running": True}, "Mounts": [
                {"Type": "volume", "Source": "/live-application-volume", "Destination": "/app/data"},
                {"Type": "bind", "Source": str(parser), "Destination": "/app/data/simba_canonical_pln.json"},
            ]}
            command = launch_command(root, info, "trial", 2, "http://127.0.0.1:6333", "http://127.0.0.1:11434/api/embeddings")
            text = " ".join(command)
            self.assertIn("sha256:fixed-image", command)
            self.assertNotIn("/live-application-volume", text)
            self.assertNotIn("--volumes-from", command)
            self.assertIn("dst=/app/data/simba_canonical_pln.json,readonly", text)
            self.assertIn("B1_USE_ROOTLESS_RUNTIME=false", command)
            self.assertNotIn("OPENAI_API_KEY", text)

    def test_stopped_container_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            launch_command(Path(folder), {"Image": "sha256:x", "State": {"Running": False}}, "trial", 2, "q", "o")


class TrialScopeTests(unittest.TestCase):
    def test_two_world_trial_cannot_be_reported_as_full_pilot_pass(self):
        from b1.pln_runner import run_pilot
        source = Path(__file__).resolve().parents[1]
        annotations = json.loads((source / "data/pilot/hand_expected.json").read_text())
        jobs = []
        def fake_job(python, root, job, output_dir, environment, timeout):
            jobs.append(job)
            if job["mode"] == "learn":
                result = {"world_id": job["world_id"], "errors": [], "sentences": [],
                          "snapshot_path": "unused-test-snapshot", "snapshot_sha256": "unused-test-hash",
                          "learning_seconds": 0.01, "parse_failure_count": 0}
            else:
                expected = annotations[job["question_id"]]
                result = {"question_id": job["question_id"], "world_id": job["world_id"],
                          "label": expected["label"], "errors": [], "answer_seconds": 0.01,
                          "used_sentence_ids": [], "retrieved_sentence_ids": []}
            return result, 0.01, "", None
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            env = {"OPENAI_MODEL": "openrouter/nvidia/example:free", "OPENROUTER_PROVIDER": "nvidia",
                   "OPENAI_API_KEY": "offline-key", "PARSER": "canonical_pln"}
            with patch("b1.pln_runner.configured_environment", return_value=env), \
                    patch("b1.pln_runner.repository_manifest", return_value={"offline": True}), \
                    patch("b1.pln_runner.private_qdrant", return_value=nullcontext()), \
                    patch("b1.pln_runner.process_job", side_effect=fake_job):
                output = run_pilot(root, root, Path("unused-python"), source / "data/pilot/worlds.json",
                                   source / "data/pilot/hand_expected.json", "offline-trial", 2)
            gate = json.loads((output / "gate.json").read_text())
            self.assertEqual(gate["status"], "TRIAL_COMPLETE")
            self.assertFalse(gate["pilot_complete"])
            self.assertEqual(gate["record_count"], 4)
            self.assertEqual(len([job for job in jobs if job["mode"] == "learn"]), 2)
            for job in jobs:
                self.assertNotIn("expected_label", job)
                self.assertNotIn("depth", job)


if __name__ == "__main__":
    unittest.main()
