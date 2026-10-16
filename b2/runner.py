"""One fixed comparison, one command, one report."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import uuid

from b1.environment import configured_environment, redact, secret_values, file_sha256
from b1.llm_control import LLMSettings
from .dataset import validate, write
from .evidence import evaluate
from .report import save
from .runtime import Runtime

SYSTEMS = ("pln_rag", "rag", "full_context")


def implementation(root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    status = git("status", "--porcelain")
    return {"revision": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "dirty": bool(status) if status is not None else None, "status": status,
            "source_sha256": {str(p.relative_to(root)): file_sha256(p)
                for package in ("b1", "b2") for p in sorted((root / package).glob("*.py"))}}


def environment(root, qdrant_url=None, ollama_url=None, pln_runtime="docker"):
    env = configured_environment(root)
    for suffix in ("MAX_LLM_CALLS", "LLM_MIN_INTERVAL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS",
                   "LLM_TIMEOUT", "QUERY_TIMEOUT", "LEARN_TIMEOUT", "REASONING_TIMEOUT", "EMBEDDING_TIMEOUT"):
        if "B2_" + suffix in env:
            env["B1_" + suffix] = env["B2_" + suffix]
    env["B1_QDRANT_URL"] = qdrant_url or env.get("B2_QDRANT_URL") or env.get("B1_QDRANT_URL", "http://127.0.0.1:6333" if pln_runtime == "docker" else "http://127.0.0.1:16333")
    env["OLLAMA_URL"] = ollama_url or env["OLLAMA_URL"]
    return env


def benchmark(root, *, systems=SYSTEMS, world_limit=42, prepare_only=False, output_dir=None,
              run_id=None, top_k=4, pln_top_k=10, pln_runtime="docker", container="pln-rag-pln-rag-1",
              repo=None, python=None, normal_rag_repo=None, qdrant_url=None, ollama_url=None,
              dataset_path=None, runtime_factory=Runtime):
    root = Path(root).resolve()
    bundle = validate(dataset_path)  # Always the complete bundle before slicing.
    if not 1 <= world_limit <= 42 or not 1 <= top_k <= 25 or not 1 <= pln_top_k:
        raise ValueError("world-limit must be 1..42, top-k 1..25 and pln-top-k positive")
    if not systems or len(systems) != len(set(systems)) or not set(systems) <= set(SYSTEMS):
        raise ValueError("Choose distinct systems from pln_rag, rag, full_context")
    worlds, references = bundle["worlds"][:world_limit], bundle["references"][:world_limit]
    env = environment(root, qdrant_url, ollama_url, pln_runtime)
    llm = LLMSettings.from_env(env)
    if not llm.model.startswith(("openai/", "openrouter/")):
        raise ValueError("B2 supports matched openai/ or openrouter/ chat routes")
    timeouts = {key: float(env.get("B1_" + key, default)) for key, default in (
        ("QUERY_TIMEOUT", "180"), ("LEARN_TIMEOUT", "900"), ("LLM_TIMEOUT", "60"),
        ("REASONING_TIMEOUT", "30"), ("EMBEDDING_TIMEOUT", "30"))}
    if any(not math.isfinite(v) or v <= 0 for v in timeouts.values()):
        raise ValueError("Timeouts must be positive finite seconds")
    if "pln_rag" in systems and not timeouts["REASONING_TIMEOUT"].is_integer():
        raise ValueError("Native PLN reasoning timeout must be a positive whole number of seconds")
    if env.get("PARSER", "canonical_pln") != "canonical_pln" and "pln_rag" in systems:
        raise ValueError("B2 currently requires the native canonical_pln parser")
    if "rag" in systems and env.get("OLLAMA_MODEL", "nomic-embed-text") != "nomic-embed-text":
        raise ValueError("The normal-RAG embedding component requires nomic-embed-text")
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Unsafe run ID")
    out = Path(output_dir).resolve() if output_dir is not None else root / "artifacts/b2" / run_id
    out.mkdir(parents=True, exist_ok=False)
    counts = {s: sum(len(w["sentences"]) + 1 for w in worlds) if s == "pln_rag" else len(worlds) for s in systems}
    repo = Path(repo or root.parent / "PLN-RAG").resolve()
    normal_rag_repo = Path(normal_rag_repo or root.parent / "normal-rag").resolve()
    manifest = {"schema_version": 1, "status": "PREPARED" if prepare_only else "RUNNING", "run_id": run_id,
        "world_count": len(worlds), "record_count": len(worlds) * len(systems), "systems": list(systems),
        "full_dataset_selected": len(worlds) == 42, "selected_world_ids": [w["id"] for w in worlds],
        "selected_source_ids": [r["source_id"] for r in references], "bundle_sha256": bundle["bundle_sha256"],
        "bundle_checksums": bundle["source"]["checksums"], "implementation": implementation(root),
        "llm_settings": llm.public(), "timeouts": timeouts,
        "retrieval": {"rag_top_k": top_k, "pln_context_top_k": pln_top_k,
            "pln_context_unit": "native vector entries/atoms", "pln_parser_enrichment_max_atoms": 50,
            "pln_reasoner_access": "entire learned per-world atomspace", "gold_adaptive": False},
        "runtime": {"pln_runtime": pln_runtime, "container": container if pln_runtime == "docker" else None,
            "image": None, "repo": str(repo), "normal_rag_repo": str(normal_rag_repo),
            "pln_python": str(python or sys.executable), "baseline_python": sys.executable,
            "qdrant_url": env["B1_QDRANT_URL"], "ollama_url": env["OLLAMA_URL"],
            "embedding_model": env["OLLAMA_MODEL"], "answer_generation_enabled": False,
            "query_candidate_max_tries": 5, "background_enabled": False},
        "request_estimate": {"by_system": counts, "chat_calls": sum(counts.values()),
            "rag_embedding_calls": sum(len(w["sentences"]) + 1 for w in worlds) if "rag" in systems else 0,
            "pln_embedding_calls": "native ingestion/query dependent; separately observed via runtime",
            "full_context_embedding_calls": 0,
            "cap_exceeded_by_initial_estimate": bool(llm.max_calls and sum(counts.values()) > llm.max_calls),
            "note": "Initial chat estimate before parser formatting/chunking fallbacks. Output-token limits are per-call ceilings, not usage predictions. Request cap is never raised automatically."}}
    write(out / "manifest.json", manifest)
    if prepare_only:
        write(out / "plan.json", manifest)
        return out
    budget = out / "workers/request_budget.json"
    raw = [{"system": s, "world_id": w["id"], "status": "NOT_ATTEMPTED", "attempted": False,
            "error_kind": "not_attempted", "llm_http_calls": []} for w in worlds for s in systems]
    by_pair = {(r["world_id"], r["system"]): r for r in raw}
    runtime, interrupted = None, False

    def checkpoint():
        # Recover durable request observations even when a worker is killed or
        # the launcher is interrupted before receiving its final result.
        for row in raw:
            audits = []
            folder = out / "workers" / row["world_id"] / row["system"]
            for path in sorted(folder.rglob("requests.json")) if folder.exists() else []:
                audits.extend(json.loads(path.read_text()))
            if audits:
                row["llm_http_calls"] = sorted(audits, key=lambda c: c["request_number"])
        rows = [evaluate(by_pair[(w["id"], s)], ref, w, top_k) for w, ref in zip(worlds, references) for s in systems]
        reserved = json.loads(budget.read_text())["calls"] if budget.exists() else 0
        write(out / "manifest.json", manifest)
        save(out, manifest, rows, reserved)

    checkpoint()  # All pairs are present even before the first operation.
    try:
        if not env.get("OPENAI_API_KEY", "").strip():
            raise ValueError("Set OPENAI_API_KEY in this repository's .env or inherited environment")
        runtime = runtime_factory(out, env, python=Path(sys.executable), pln_python=Path(python or sys.executable),
                                  pln_runtime=pln_runtime, container=container)
        for world in worlds:
            for system in systems:
                row = by_pair[(world["id"], system)]
                if llm.max_calls and budget.exists() and json.loads(budget.read_text())["calls"] >= llm.max_calls:
                    row["error_kind"] = "request_budget_exceeded"
                    checkpoint()
                    continue
                common = {"system": system, "world_id": world["id"], "llm_config": asdict(llm),
                          "request_budget_path": str(budget), "llm_timeout": timeouts["LLM_TIMEOUT"]}
                operation = out / "workers" / world["id"] / system
                try:
                    row.pop("error_kind", None)
                    row["attempted"] = True
                    if system == "pln_rag":
                        common.update(repo_path="/app" if pln_runtime == "docker" else str(repo),
                            qdrant_url=env["B1_QDRANT_URL"], ollama_url=env["OLLAMA_URL"],
                            embedding_model=env["OLLAMA_MODEL"], pln_top_k=pln_top_k,
                            collection=f"b2_{uuid.uuid4().hex}_{world['id'].replace('-', '_')}",
                            reasoning_timeout=timeouts["REASONING_TIMEOUT"])
                        learned = runtime.execute({**common, "mode": "learn", "sentences": world["sentences"],
                            "work_dir": str(operation / "learn/work")}, operation / "learn", timeouts["LEARN_TIMEOUT"])
                        row.update(learning_seconds=learned.get("learning_seconds"),
                            learning_worker_seconds=learned.get("worker_seconds"), parse_failure_count=learned.get("parse_failure_count"),
                            embedding_calls=learned.get("embedding_calls"),
                            llm_http_calls=learned.get("llm_http_calls", []))
                        if learned.get("status") != "OK":
                            row.update(status="ERROR", answer_attempted=False, error_kind="learning_failed",
                                       learning_error=learned.get("error_kind"),
                                       timed_out=learned.get("error_kind") == "timeout")
                        else:
                            row["answer_attempted"] = True
                            result = runtime.execute({**common, "mode": "query", "hypothesis": world["hypothesis"],
                                "work_dir": str(operation / "query/work"), "snapshot_path": learned["snapshot_path"],
                                "snapshot_sha256": learned["snapshot_sha256"], "learn_output_path": str(operation / "learn/output.json")},
                                operation / "query", timeouts["QUERY_TIMEOUT"])
                            result["llm_http_calls"] = row["llm_http_calls"] + result.get("llm_http_calls", [])
                            result["embedding_calls"] = (learned["embedding_calls"] + result["embedding_calls"]
                                if type(learned.get("embedding_calls")) is int and type(result.get("embedding_calls")) is int else None)
                            row.update(result)
                    else:
                        result = runtime.execute({**common, "mode": "answer", "sentences": world["sentences"],
                            "hypothesis": world["hypothesis"], "normal_rag_repo": str(normal_rag_repo),
                            "ollama_url": env["OLLAMA_URL"], "embedding_timeout": timeouts["EMBEDDING_TIMEOUT"],
                            "top_k": top_k}, operation, timeouts["QUERY_TIMEOUT"])
                        row.update(result)
                except Exception as exc:
                    row.update(status="ERROR", error_kind=type(exc).__name__,
                        error_detail=redact(str(exc), secret_values(env)))
                row.update(world_id=world["id"], system=system)
                checkpoint()
    except KeyboardInterrupt:
        interrupted = True
        manifest["driver_error"] = "interrupted"
        for row in raw:
            if row["status"] == "NOT_ATTEMPTED" and row["attempted"]:
                row.update(status="ERROR", error_kind="interrupted")
    except Exception as exc:
        manifest["driver_error"] = redact(str(exc), secret_values(env))
        for row in raw:
            if row["status"] == "NOT_ATTEMPTED":
                row["error_kind"] = "setup_failed"
    finally:
        manifest["status"] = "INTERRUPTED" if interrupted else "COMPLETE" if all(r["status"] == "OK" for r in raw) else "COMPLETE_WITH_FAILURES"
        if runtime is not None and runtime.identity is not None:
            manifest["runtime"]["image"] = runtime.identity["image"]
        checkpoint()
    return out
