"""Run identical questions through Docker/local PLN-RAG and host baselines."""

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import uuid

from .baseline import load_baseline, matched_settings, run_question
from .dataset import validate_full
from .docker_runtime import prepare_benchmark
from .environment import configured_environment, file_sha256, redact, secret_values
from .io import read_worlds, write_json
from .llm_control import LLMSettings
from .oracle import consensus, validate
from .pln_runner import public_sentences, run_pln

SYSTEMS = ("pln_rag", "rag", "full_context")
UNMEASURED = "unmeasured"


def comparison_environment(root, *, qdrant_url=None, ollama_url=None, pln_runtime="docker"):
    env = configured_environment(root)
    llm = LLMSettings.from_env(env)
    # Materialize defaults so the Docker image and host cannot select different ones.
    defaults = {"OPENROUTER_BASE_URL": llm.api_base,
                "B1_LLM_TEMPERATURE": str(llm.temperature), "B1_LLM_MAX_TOKENS": str(llm.max_tokens),
                "B1_MAX_LLM_CALLS": str(llm.max_calls), "B1_LLM_MIN_INTERVAL": str(llm.minimum_interval),
                "B1_LLM_TIMEOUT": "60", "B1_QUERY_TIMEOUT": "180", "B1_LEARN_TIMEOUT": "900",
                "B1_REASONING_TIMEOUT": "30", "B1_EMBEDDING_TIMEOUT": "30"}
    for key, value in defaults.items():
        env.setdefault(key, value)
    env["B1_QDRANT_URL"] = qdrant_url or env.get("B1_QDRANT_URL", "http://127.0.0.1:6333" if pln_runtime == "docker" else "http://127.0.0.1:16333")
    if ollama_url:
        env["OLLAMA_URL"] = ollama_url
    for key in ("B1_LLM_TIMEOUT", "B1_QUERY_TIMEOUT", "B1_LEARN_TIMEOUT", "B1_REASONING_TIMEOUT", "B1_EMBEDDING_TIMEOUT"):
        # PLN's worker uses integer LLM/reasoning seconds; validate before spending calls.
        value = int(env[key])
        if value <= 0:
            raise ValueError(f"{key} must be a positive integer number of seconds")
    if env["PARSER"] != "canonical_pln":
        raise ValueError("The comparison requires PARSER=canonical_pln")
    return env, llm


def skipped_record(system, kind, detail):
    return {"schema_version": 1, "system": system, "mode": system,
            "label": "ERROR", "attempted": False, "error_kind": kind, "error_detail": detail,
            "answer_seconds": UNMEASURED, "index_build_seconds": UNMEASURED,
            "retrieved": [], "cited": [], "retrieved_sentence_ids": [], "used_sentence_ids": [],
            "invalid_citation_count": 0, "raw_model_output": None,
            "model": UNMEASURED, "provider": UNMEASURED,
            "usage": {key: UNMEASURED for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
            "llm_http_calls": []}


def annotate(record, system, world, question, oracle_answer):
    """Called only after the system returned or the operation was explicitly skipped."""
    record.update(system=system, world_id=world.id, question_id=question.id,
                  question=question.text, expected_label=question.expected_label,
                  depth_bucket=question.depth_bucket, rule_kind=question.rule_kind,
                  correct=record["label"] == question.expected_label)
    record.setdefault("attempted", True)
    required = oracle_answer["proof_sentence_ids"] or None
    retrieved = (record.get("retrieved_sentence_ids", []) if system == "pln_rag"
                 else [r["id"] for r in record.get("retrieved", [])])
    context = [s["id"] for s in record.get("context", [])]
    measured = record["attempted"] and bool(required)
    record["proof_coverage"] = {
        "required_ids": required,
        "retrieved_contains_required": (set(required) <= set(retrieved)
            if measured and system != "full_context" and
            (bool(record.get("retrieval_calls")) if system == "pln_rag" else bool(record.get("context"))) else UNMEASURED),
        "context_contains_required": (set(required) <= set(context)
            if measured and system != "pln_rag" and context else UNMEASURED),
        "required_sentence_count": len(required) if required else UNMEASURED,
        "fits_top_four": len(required) <= 4 if required else UNMEASURED,
    }
    # Supplied-proof coverage and logical sufficiency are different measurements.
    # The oracle accepts any proof from the retrieved subset, after the answer.
    retrieval_measured = record["proof_coverage"]["retrieved_contains_required"] != UNMEASURED
    context_measured = record["proof_coverage"]["context_contains_required"] != UNMEASURED
    for name, ids, available in (("retrieved_proves_claim", retrieved, retrieval_measured),
                                  ("context_proves_claim", context, context_measured)):
        record["proof_coverage"][name] = (consensus(tuple(s for s in world.sentences if s.id in ids), question.goal).label == "SUPPORTED"
                                               if available else UNMEASURED)
    return record


def score(rows):
    count = len(rows)
    correct = sum(r["correct"] for r in rows)
    times = [r["answer_seconds"] for r in rows if type(r.get("answer_seconds")) in (int, float)]
    if count:
        z = 1.959963984540054
        p = correct / count
        center = (p + z * z / (2 * count)) / (1 + z * z / count)
        margin = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
        interval = [max(0.0, center - margin), min(1.0, center + margin)]
    else:
        interval = UNMEASURED
    return {"questions": count, "correct": correct,
            "errors": sum(r["label"] == "ERROR" for r in rows),
            "not_attempted": sum(not r["attempted"] for r in rows),
            "accuracy": correct / count if count else UNMEASURED,
            "accuracy_wilson_95": interval,
            "timeouts": sum("timeout" in (r.get("error_kind") or "") for r in rows),
            "mean_answer_seconds": sum(times) / len(times) if times else UNMEASURED,
            "timed_questions": len(times)}


def build_summary(records, systems, worlds, calls, learning, budget):
    by_system = {}
    for system in systems:
        rows = [r for r in records if r["system"] == system]
        observed = calls.get(system, [])
        by_system[system] = {
            **score(rows),
            "by_depth": {str(d): score([r for r in rows if r["depth_bucket"] == d]) for d in range(4)},
            "supported_by_depth": {str(d): score([r for r in rows if r["depth_bucket"] == d and r["expected_label"] == "SUPPORTED"]) for d in range(4)},
            "by_rule_kind": {k: score([r for r in rows if r["rule_kind"] == k]) for k in sorted({r["rule_kind"] for r in rows})},
            "by_expected_label": {k: score([r for r in rows if r["expected_label"] == k]) for k in ("SUPPORTED", "NOT_ESTABLISHED")},
            "providers_reported": sorted({c.get("provider", UNMEASURED) for c in observed}),
            "models_reported": sorted({c.get("model", UNMEASURED) for c in observed}),
            "http_attempts_observed": len(observed),
            "usage": {key: {
                "measured_sum": sum(c["usage"][key] for c in observed if type(c.get("usage", {}).get(key)) is int)
                    if any(type(c.get("usage", {}).get(key)) is int for c in observed) else UNMEASURED,
                "unmeasured_calls": sum(type(c.get("usage", {}).get(key)) is not int for c in observed),
            } for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
            "invalid_citations": sum(r.get("invalid_citation_count", 0) for r in rows) if system != "pln_rag" else UNMEASURED,
            "retrieval_coverage": dict(Counter(str(r["proof_coverage"]["retrieved_contains_required"]) for r in rows)),
            "retrieval_sufficiency": dict(Counter(str(r["proof_coverage"]["retrieved_proves_claim"]) for r in rows)),
        }
    total = sum(len(w.questions) for w in worlds) * len(systems)
    learned_count = sum(r.get("sentence_count", len(r.get("sentences", []))) for r in learning)
    failures = sum(r.get("parse_failure_count", 0) for r in learning)
    rag = {r["question_id"]: r for r in records if r["system"] == "rag" and r["expected_label"] == "SUPPORTED"}
    full = {r["question_id"]: r for r in records if r["system"] == "full_context"}
    comparisons = [{"question_id": qid, "rag_label": r["label"], "full_context_label": full[qid]["label"],
                    "retrieval_proves_claim": r["proof_coverage"]["retrieved_proves_claim"]} for qid, r in rag.items() if qid in full]
    return {"schema_version": 1, "status": "COMPLETE_WITH_ERRORS" if any(r["label"] == "ERROR" for r in records) else "COMPLETE",
            "complete": len(records) == total, "execution_complete": len(records) == total and all(r["attempted"] for r in records),
            "world_count": len(worlds), "questions_per_system": total // len(systems),
            "record_count": len(records), "expected_records": total,
            "requests_reserved": json.loads(budget.read_text())["calls"] if budget.exists() else 0,
            "systems": by_system,
            "pln_learning_seconds": sum(r["learning_seconds"] for r in learning if type(r.get("learning_seconds")) in (int, float)) if learning else UNMEASURED,
            "pln_sentence_parse_failures": sum(r.get("parse_failure_count", 0) for r in learning) if learning else UNMEASURED,
            "pln_sentence_parse_failure_share": failures / learned_count if learned_count else UNMEASURED,
            "supported_baseline_comparisons": comparisons,
            "interval_note": "Wilson 95% binomial intervals are descriptive; paired questions and repeated templates are not independent samples.",
            "accuracy_denominator": "All selected questions, including ERROR and explicitly unattempted records",
            "timing_note": "Answer time includes LLM calls and pacing. PLN learning is separate; RAG index time is in each record."}


def render_report(out, summary, records):
    lines = ["# Shared rule-chain-depth results", "", f"Status: **{summary['status']}**", "",
             f"{summary['world_count']} worlds; {summary['questions_per_system']} questions per selected system.", "",
             "Accuracy includes errors and unattempted questions in its denominator. COMPLETE describes execution, not perfect accuracy.", "",
             "| System | Correct / total | Errors | Not attempted | Accuracy |", "|---|---:|---:|---:|---:|"]
    for system, row in summary["systems"].items():
        lines.append(f"| {system} | {row['correct']} / {row['questions']} | {row['errors']} | {row['not_attempted']} | {row['accuracy']:.1%} |")
    lines += ["", "## Accuracy by depth", "", "| System | Depth | Correct / total | Errors |", "|---|---:|---:|---:|"]
    for system, row in summary["systems"].items():
        for depth, group in row["by_depth"].items():
            lines.append(f"| {system} | {depth} | {group['correct']} / {group['questions']} | {group['errors']} |")
    lines += ["", "Unsupported questions are grouped by their supported pair's depth; their actual proof depth is null.", "",
              "See [summary.json](summary.json) for per-kind scores, providers, tokens, and timing; [records.json](records.json) for every result.", "",
              "Top-four retrieval cannot contain a supplied proof needing more than four sentences. Coverage records disclose this limit.", "",
              "## Per-question records", "", "| System | Question | Label | Expected | Record |", "|---|---|---|---|---|"]
    for r in records:
        path = f"{r['system']}/{r['world_id']}/{r['question_id']}/record.json"
        lines.append(f"| {r['system']} | {r['question_id']} | {r['label']} | {r['expected_label']} | [record]({path}) |")
    lines += ["", "PLN-RAG's `pln_rag/raw.md` shows learned atoms and proofs when its runner started. Baseline operation folders contain exact requests, response bytes, raw model text, and the retrieved/shown sentences in each record."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def run_comparison(root: Path, *, systems=SYSTEMS, dataset_kind="benchmark", worlds_path=None,
                   expected_path=None, world_limit=None, normal_repo=None, pln_runtime="docker",
                   container=None, repo=None, python=None, qdrant_url=None, ollama_url=None,
                   prepare_only=False, run_id=None) -> Path:
    root = root.resolve()
    systems = tuple(systems)
    if not systems or len(set(systems)) != len(systems) or not set(systems) <= set(SYSTEMS):
        raise ValueError("Choose unique systems from pln_rag, rag, full_context")
    if dataset_kind not in ("pilot", "benchmark") or pln_runtime not in ("docker", "local"):
        raise ValueError("Invalid dataset or PLN runtime")
    data = root / "data" / dataset_kind
    worlds_path, expected_path = Path(worlds_path or data / "worlds.json"), Path(expected_path or data / "hand_expected.json")
    all_worlds = read_worlds(worlds_path)
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", identifier)
           for w in all_worlds for identifier in (w.id, *(q.id for q in w.questions))):
        raise ValueError("World and question IDs must be safe directory names")
    expected = json.loads(expected_path.read_text())
    oracle = validate_full(all_worlds, expected) if dataset_kind == "benchmark" else validate(all_worlds, expected)
    if dataset_kind == "pilot" and len(all_worlds) != 8:
        raise ValueError("Pilot comparison requires the eight-world dataset")
    limit = len(all_worlds) if world_limit is None else world_limit
    if not 1 <= limit <= len(all_worlds):
        raise ValueError(f"world-limit must be between 1 and {len(all_worlds)}")
    worlds = all_worlds[:limit]
    env, llm = comparison_environment(root, qdrant_url=qdrant_url, ollama_url=ollama_url, pln_runtime=pln_runtime)
    normal_repo = Path(normal_repo or root.parent / "normal-rag").resolve()
    # Loading only these stdlib modules also validates the common settings for PLN-only runs.
    core, config = load_baseline(normal_repo)
    settings = matched_settings(config, env, llm)
    if "pln_rag" in systems and pln_runtime == "docker" and not container:
        raise ValueError("Supply --container for the Docker PLN-RAG runtime")
    repo = Path(repo or root.parent / "PLN-RAG").resolve()
    python = Path(python or root / ".venv/bin/python").absolute()
    if "pln_rag" in systems and pln_runtime == "local" and (not repo.is_dir() or not python.is_file()):
        raise ValueError("The local PLN-RAG repo and Python interpreter must exist")
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Unsafe run ID")
    out = root / "artifacts/benchmark" / run_id
    out.mkdir(parents=True, exist_ok=False)
    budget = root / "runtime/runs" / run_id / "request_budget.json"
    if budget.parent.exists():
        raise ValueError("Run ID already has runtime state; use a fresh ID")
    write_json(out / "worlds.json", {"schema_version": 1, "worlds": [w.to_dict() for w in all_worlds]})
    write_json(out / "hand_expected.json", expected)
    write_json(out / "oracle.json", oracle)
    job = {"run_id": run_id, "world_limit": limit, "dataset_kind": dataset_kind,
           "worlds_path": str(out / "worlds.json"), "expected_path": str(out / "hand_expected.json"),
           "output_dir": str(out / "pln_rag"), "request_budget_path": str(budget)}
    plan = prepare_benchmark(root, container, job, env) if "pln_rag" in systems and pln_runtime == "docker" else None
    manifest = {"schema_version": 1, "stage": "prepared_only" if prepare_only else "live_benchmark",
                "dataset_kind": dataset_kind, "dataset_sha256": oracle["dataset_sha256"],
                "selected_world_ids": [w.id for w in worlds], "questions_per_system": 2 * limit,
                "full_dataset_selected": limit == len(all_worlds), "systems": systems,
                "system_order": systems, "llm_settings": llm.public(), "baseline_settings": settings.public(),
                "pln_runtime": pln_runtime, "docker_image": plan["image"] if plan else None,
                "service_urls": {"qdrant": env["B1_QDRANT_URL"], "ollama": env["OLLAMA_URL"]},
                "deadlines": {k: env[k] for k in env if k.startswith("B1_") and k.endswith("TIMEOUT")},
                "request_budget_scope": "One shared cap across all selected systems, including PLN ingestion",
                "source_sha256": {str(p.relative_to(root)): file_sha256(p) for p in sorted((root / "b1").glob("*.py"))},
                "normal_rag_source_sha256": {p.name: file_sha256(p) for p in sorted((normal_repo / "normal_rag").glob("*.py"))}}
    write_json(out / "manifest.json", manifest)
    print(f"{'Prepared' if prepare_only else 'Starting'} {2 * limit} questions × {len(systems)} systems; shared cap {llm.max_calls or 'unlimited'}.", flush=True)
    estimate = limit * (15 * ("pln_rag" in systems) + 2 * ("rag" in systems) + 2 * ("full_context" in systems))
    print(f"Planning estimate: about {estimate} chat calls before extra PLN parsing calls; the cap is not raised automatically.", flush=True)
    print(f"Artifacts: {out}", flush=True)
    if prepare_only:
        return out

    answers = {a["question_id"]: a for a in oracle["answers"]}
    records, calls, learning, driver_errors = [], {}, [], []
    for system in systems:
        print(f"SYSTEM {system}", flush=True)
        calls[system] = []
        if system == "pln_rag":
            try:
                if plan:
                    code = subprocess.run(plan["command"], env=env, check=False).returncode
                    if code:
                        driver_errors.append({"system": system, "kind": "docker_exit", "exit_code": code})
                else:
                    run_pln(root, repo, python, out / "worlds.json", out / "hand_expected.json", run_id, limit,
                            dataset_kind=dataset_kind, output_dir=out / system, request_budget_path=budget, environment=env)
            except Exception as exc:
                driver_errors.append({"system": system, "kind": type(exc).__name__,
                                      "detail": redact(str(exc), secret_values(env))})
            def saved(name, default):
                path = out / system / name
                return json.loads(path.read_text()) if path.exists() else default
            pln_records = {r["question_id"]: r for r in saved("records.json", [])}
            learning = saved("learning.json", [])
            calls[system] = saved("llm_requests.json", [])
        fatal = None
        for world in worlds:
            for question in world.questions:
                operation = out / system / world.id / question.id
                if system == "pln_rag":
                    record = pln_records.get(question.id) or skipped_record(system, "pln_not_run", "No PLN question record was produced; see driver_errors and PLN logs")
                elif fatal:
                    record = skipped_record(system, fatal, "No call made after a previous transport/API/budget failure in this system")
                else:
                    try:
                        record = run_question(core, settings, llm, budget, public_sentences(world), question.text, system, operation)
                    except Exception as exc:
                        detail = redact(str(exc), secret_values(env))
                        driver_errors.append({"system": system, "question_id": question.id, "kind": type(exc).__name__, "detail": detail})
                        record = skipped_record(system, "baseline_worker_error", detail)
                        record["attempted"] = True
                        fatal = "baseline_worker_error"
                    calls[system].extend(record["llm_http_calls"])
                    if record.get("error_kind") in ("request_budget_exceeded", "api_error", "network_error", "timeout", "embedding_api_error", "provider_pin_lost"):
                        fatal = record["error_kind"]
                record = annotate(record, system, world, question, answers[question.id])
                write_json(operation / "record.json", record)
                records.append(record)
                write_json(out / "records.json", records)
                write_json(out / "llm_requests.json", calls)
        write_json(out / system / "comparison_records.json", [r for r in records if r["system"] == system])
    summary = build_summary(records, systems, worlds, calls, learning, budget)
    summary.update(driver_errors=driver_errors, full_dataset_selected=limit == len(all_worlds))
    if driver_errors:
        summary["status"] = "COMPLETE_WITH_ERRORS"
    write_json(out / "summary.json", summary)
    render_report(out, summary, records)
    print(f"Report: {out / 'report.md'}", flush=True)
    return out
