"""Request-level accounting and pacing for isolated benchmark workers."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit


@dataclass(frozen=True)
class LLMSettings:
    model: str
    api_base: str
    provider: str | None
    temperature: float
    max_tokens: int
    max_calls: int
    minimum_interval: float

    @classmethod
    def from_env(cls, env: dict) -> LLMSettings:
        model = env["OPENAI_MODEL"]
        router = model.startswith("openrouter/")
        if not model or "/" not in model:
            raise ValueError("OPENAI_MODEL must include its DSPy provider prefix")
        provider = env.get("OPENROUTER_PROVIDER") if router else None
        if router and not provider:
            raise ValueError("Set OPENROUTER_PROVIDER to pin one upstream provider")
        api_base = (env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1") if router
                    else "https://api.openai.com/v1").rstrip("/")
        parsed = urlsplit(api_base)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("The LLM base URL must be HTTPS without embedded credentials")
        result = cls(model, api_base, provider, float(env.get("B1_LLM_TEMPERATURE", "0")),
                     int(env.get("B1_LLM_MAX_TOKENS", "2000")),
                     int(env.get("B1_MAX_LLM_CALLS", "40" if router else "0")),
                     float(env.get("B1_LLM_MIN_INTERVAL", "4" if router else "0")))
        if (not math.isfinite(result.temperature) or not 0 <= result.temperature <= 2
                or result.max_tokens < 1 or result.max_calls < 0
                or not math.isfinite(result.minimum_interval) or result.minimum_interval < 0):
            raise ValueError("Invalid LLM temperature, token limit, request budget, or pacing interval")
        if router and model.endswith(":free") and (result.max_calls == 0 or result.minimum_interval < 3.1):
            raise ValueError("Free routes require a finite request budget and at least 3.1 seconds between requests")
        return result

    def public(self) -> dict:
        return {"model": self.model, "api_base": self.api_base, "provider": self.provider,
                "temperature": self.temperature, "max_tokens": self.max_tokens,
                "max_calls": self.max_calls, "minimum_interval": self.minimum_interval,
                "allow_fallbacks": False, "num_retries": 0, "cache": False}

    def kwargs(self) -> dict:
        result = {"temperature": self.temperature, "max_tokens": self.max_tokens,
                  "api_base": self.api_base, "cache": False, "num_retries": 0}
        if self.provider:
            result["extra_body"] = {"provider": {"order": [self.provider], "only": [self.provider],
                                                  "allow_fallbacks": False, "require_parameters": True}}
        return result


class RequestBudgetExceeded(RuntimeError):
    pass


def reserve_request(path: Path, limit: int, interval: float, *, clock=time.time, sleep=time.sleep) -> tuple[int, float]:
    """Share one limit and request-start schedule across fresh worker processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        text = handle.read()
        state = json.loads(text) if text else {"calls": 0, "next_start": 0.0}
        if limit and state["calls"] >= limit:
            raise RequestBudgetExceeded(f"The run reached its {limit}-request budget; no further LLM call was sent")
        wait = max(0.0, state["next_start"] - clock())
        if wait:
            sleep(wait)
        state = {"calls": state["calls"] + 1, "next_start": clock() + interval}
        handle.seek(0)
        handle.truncate()
        json.dump(state, handle)
        handle.flush()
        return state["calls"], wait


def response_metadata(payload: dict) -> dict:
    def reported(value):
        return value if isinstance(value, str) and value.strip() else "unmeasured"
    provider = reported(payload.get("provider"))
    error = payload.get("error")
    metadata = error.get("metadata") if isinstance(error, dict) else None
    if provider == "unmeasured" and isinstance(metadata, dict):
        provider = reported(metadata.get("provider_name"))
    usage = payload.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return {"model": reported(payload.get("model")), "provider": provider,
            "usage": {key: value if type(value := usage.get(key)) is int and value > 0 else "unmeasured"
                      for key in ("prompt_tokens", "completion_tokens", "total_tokens")}}


class HTTPAudit:
    """Observe actual chat HTTP requests, including any DSPy format fallback calls.

    Installed inside a single-use worker. No authorization headers are recorded.
    Reading the response retains its content for the SDK and does not repair it.
    """
    def __init__(self, settings: LLMSettings, budget: Path, phase, errors: list):
        self.settings, self.budget, self.phase, self.errors = settings, budget, phase, errors
        self.calls = []
        self.fatal = False
        self.original_send = None
        self.original_asend = None

    def _begin(self, request):
        if request.method != "POST" or str(request.url).rstrip("/") != self.settings.api_base + "/chat/completions":
            return None
        if self.fatal:
            raise RuntimeError("The previous LLM request failed; no additional request was sent")
        body = json.loads(request.content)
        if self.settings.provider and body.get("provider") != self.settings.kwargs()["extra_body"]["provider"]:
            raise RuntimeError("The outgoing request did not preserve the configured provider pin")
        number, wait = reserve_request(self.budget, self.settings.max_calls, self.settings.minimum_interval)
        row = {"request_number": number, "phase": self.phase(), "request": body,
               "rate_wait_seconds": wait, **response_metadata({})}
        self.calls.append(row)
        return row

    def _finish(self, row: dict, response):
        row.update(http_status=response.status_code, raw_response=response.text)
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if isinstance(payload, dict):
            row.update(response_metadata(payload))
        if response.status_code >= 400 or (isinstance(payload, dict) and payload.get("error")):
            self.fatal = True
            self.errors.append({"phase": self.phase(), "kind": "llm_api_error",
                                "detail": f"LLM HTTP status {response.status_code}; see llm_http_calls raw_response"})

    def install(self):
        import httpx
        self.original_send = httpx.Client.send
        self.original_asend = httpx.AsyncClient.send
        audit = self

        def send(client, request, *args, **kwargs):
            row = audit._begin(request)
            if row is None:
                return audit.original_send(client, request, *args, **kwargs)
            started = time.perf_counter()
            try:
                response = audit.original_send(client, request, *args, **kwargs)
                response.read()
                audit._finish(row, response)
                return response
            except Exception as exc:
                audit.fatal = True
                row["exception_type"] = type(exc).__name__
                raise
            finally:
                row["seconds"] = time.perf_counter() - started

        async def asend(client, request, *args, **kwargs):
            row = audit._begin(request)
            if row is None:
                return await audit.original_asend(client, request, *args, **kwargs)
            started = time.perf_counter()
            try:
                response = await audit.original_asend(client, request, *args, **kwargs)
                await response.aread()
                audit._finish(row, response)
                return response
            except Exception as exc:
                audit.fatal = True
                row["exception_type"] = type(exc).__name__
                raise
            finally:
                row["seconds"] = time.perf_counter() - started

        httpx.Client.send, httpx.AsyncClient.send = send, asend

    def close(self):
        if self.original_send:
            import httpx
            httpx.Client.send, httpx.AsyncClient.send = self.original_send, self.original_asend
