"""Read-only setup checks. These outputs are not benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from .environment import configured_environment, redact, secret_values
from .io import write_json

SMOKE = '''
import importlib.metadata, json
from pettachainer.pettachainer import PeTTaChainer
from nl2pln import NL2PLNModule
import dspy, httpx, pydantic_settings, fastapi
handler = PeTTaChainer()
handler.add_atom("(: b1_fact (Ready hana) (STV 1 1))")
handler.add_atom("(: b1_rule (Implication (Premises (Ready $x)) (Conclusions (Able $x))) (STV 1 1))")
proof = handler.query("(: $prf (Able hana) $tv)", timeout_sec=0)
if not proof or not any("(Able hana)" in str(p) for p in proof):
    raise RuntimeError("The dependency smoke test did not return the expected one-rule proof: " + str(proof))
print("B1_DOCTOR=" + json.dumps({"proof": proof, "packages": {name: importlib.metadata.version(name) for name in ["dspy", "janus-swi", "httpx", "pydantic", "fastapi"]}}))
'''


def doctor(root: Path, repo: Path, python: Path, output: Path) -> dict:
    env = configured_environment(root)
    key = env.get("OPENAI_API_KEY", "").strip()
    result = {"stage": "setup_only_not_benchmark_results", "api_key_configured": key not in ("", "sk-...", "your-api-key"),
              "model": env["OPENAI_MODEL"], "parser": env["PARSER"], "repo_exists": (repo / "core/service.py").exists(),
              "python_exists": python.exists(), "private_qdrant_binary_exists": (root / "runtime/bin/qdrant").exists()}
    try:
        smoke = subprocess.run([str(python), "-c", SMOKE], cwd=root, env=env,
                               capture_output=True, text=True, timeout=45)
        log = redact(smoke.stdout + smoke.stderr, secret_values(env))
        result["runtime_smoke_passed"] = smoke.returncode == 0 and "B1_DOCTOR=" in smoke.stdout
        result["runtime_log"] = log
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["runtime_smoke_passed"] = False
        result["runtime_log"] = redact(str(exc), secret_values(env))
    result["ready"] = all(result[k] for k in ("api_key_configured", "repo_exists", "python_exists", "runtime_smoke_passed"))
    write_json(output, result)
    return result
