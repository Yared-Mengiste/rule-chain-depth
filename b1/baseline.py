"""Load the sibling baseline without editing it; add shared request accounting."""

import hashlib
import importlib.util
from pathlib import Path
import sys
import time

from .io import write_json
from .llm_control import RequestBudgetExceeded, reserve_request, response_metadata


def load_baseline(repo: Path):
    package = repo.resolve() / "normal_rag"
    if not all((package / name).is_file() for name in ("__init__.py", "core.py", "config.py")):
        raise ValueError("--normal-rag-repo must contain the standalone normal_rag package")
    name = "_b1_baseline_" + hashlib.sha256(str(package).encode()).hexdigest()[:12]
    if name + ".core" not in sys.modules:
        old = sys.dont_write_bytecode
        sys.dont_write_bytecode = True  # No writes to the sibling project, including caches.
        try:
            for suffix, file in (("", "__init__.py"), (".config", "config.py"), (".core", "core.py")):
                spec = importlib.util.spec_from_file_location(name + suffix, package / file,
                    submodule_search_locations=[str(package)] if not suffix else None)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name + suffix] = module
                spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = old
    return sys.modules[name + ".core"], sys.modules[name + ".config"]


def matched_settings(config, env: dict, llm):
    if not llm.model.startswith("openrouter/"):
        raise ValueError("The shared comparison requires OPENAI_MODEL=openrouter/<model>")
    if env.get("OLLAMA_MODEL", "nomic-embed-text") != "nomic-embed-text":
        raise ValueError("The shared comparison requires OLLAMA_MODEL=nomic-embed-text")
    return config.Settings(api_key=env.get("OPENAI_API_KEY", ""),
        model=llm.model.removeprefix("openrouter/"), base_url=llm.api_base,
        provider=llm.provider, temperature=llm.temperature, max_tokens=llm.max_tokens,
        timeout_seconds=float(env.get("B1_LLM_TIMEOUT", "60")),
        ollama_url=env.get("OLLAMA_URL", "http://127.0.0.1:11434/api/embeddings"),
        embedding_timeout_seconds=float(env.get("B1_EMBEDDING_TIMEOUT", "30")))


class BudgetHTTP:
    def __init__(self, core, settings, llm, budget: Path, phase: str, transport=None):
        self.core, self.settings, self.llm, self.budget = core, settings, llm, budget
        self.phase, self.transport = phase, transport or core.HTTP()
        self.calls = []

    def post(self, url, payload, headers, timeout):
        if url != self.settings.base_url + "/chat/completions":
            return self.transport.post(url, payload, headers, timeout)
        if payload.get("provider") != self.llm.kwargs()["extra_body"]["provider"]:
            raise self.core.Failure("provider_pin_lost")
        try:
            number, wait = reserve_request(self.budget, self.llm.max_calls, self.llm.minimum_interval)
        except RequestBudgetExceeded:
            raise self.core.Failure("request_budget_exceeded") from None
        row = {"request_number": number, "phase": self.phase, "request": payload,
               "rate_wait_seconds": wait, **response_metadata({})}
        self.calls.append(row)
        started = time.perf_counter()
        try:
            response = self.transport.post(url, payload, headers, timeout)
            row.update(http_status=response.status, raw_response=response.body.decode("utf-8", errors="replace"))
            try:
                value = self.core.strict_json(response.body)
            except (ValueError, UnicodeError):
                value = {}
            row.update(response_metadata(value if isinstance(value, dict) else {}))
            return response
        except self.core.Failure as exc:
            row["error_kind"] = exc.kind
            raise
        finally:
            row["seconds"] = time.perf_counter() - started


def run_question(core, settings, llm, budget, sentences, question, mode, output, *, transport=None):
    """Only public sentences and question text cross into the baseline call."""
    http = BudgetHTTP(core, settings, llm, budget, mode, transport)
    answer = core.answer_question(sentences, question, mode, settings, http=http)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "input.json", {"sentences": sentences, "question": question, "mode": mode})
    if answer.request is not None:
        write_json(output / "request.json", answer.request)
    if answer.response_body is not None:
        (output / "response.body").write_bytes(answer.response_body)
    raw = answer.record.get("raw_model_output")
    if isinstance(raw, str):
        (output / "raw_model_output.txt").write_bytes(raw.encode("utf-8"))
    answer.record["llm_http_calls"] = http.calls
    answer.record["rate_wait_seconds"] = sum(c["rate_wait_seconds"] for c in http.calls)
    return answer.record
