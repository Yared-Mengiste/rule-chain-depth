"""Supervise isolated PLN-RAG runs, with hard process-group timeouts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request
import uuid

from .environment import configured_environment, redact, repository_manifest, secret_values
from .io import read_worlds, write_json
from .oracle import validate
from .records import error_record, is_correct, log_errors, validate_record
from .llm_control import LLMSettings


def public_sentences(world) -> list[dict]:
    return [{"id": sentence.id, "text": sentence.text} for sentence in world.sentences]


def process_job(python: Path, root: Path, job: dict, output_dir: Path,
                environment: dict, timeout: float) -> tuple[dict | None, float, str, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "input.json", job)
    output = output_dir / "output.json"
    raw_log = Path(job["work_dir"]) / "worker.private.log"
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    failure = None
    with raw_log.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen([str(python), "-m", "b1.pln_worker", "--job",
                                 str(output_dir / "input.json"), "--output", str(output)],
                                cwd=root, env=environment, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
            if code != 0:
                failure = "worker_crash"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            failure = "timeout"
    elapsed = time.perf_counter() - started
    log = redact(raw_log.read_text(encoding="utf-8", errors="replace"), secret_values(environment))
    (output_dir / "raw.log").write_text(log, encoding="utf-8")
    value = None
    if output.exists():
        value = json.loads(output.read_text(encoding="utf-8"))
    if value is None and failure is None:
        failure = "missing_worker_output"
    return value, elapsed, log, failure


def available(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


@contextmanager
def private_qdrant(root: Path, url: str, run_id: str, environment: dict):
    process = None
    log = None
    try:
        if not available(url + "/collections"):
            if url != "http://127.0.0.1:16333":
                raise RuntimeError(f"Configured Qdrant is unavailable: {url}")
            binary = root / "runtime/bin/qdrant"
            if not binary.exists():
                raise RuntimeError("Qdrant is not running and runtime/bin/qdrant is unavailable")
            work = root / "runtime/qdrant" / run_id
            work.mkdir(parents=True, exist_ok=True)
            config = work / "config.yaml"
            config.write_text(
                f"log_level: WARN\ntelemetry_disabled: true\nstorage:\n  storage_path: {work / 'storage'}\n"
                f"  snapshots_path: {work / 'snapshots'}\nservice:\n  host: 127.0.0.1\n"
                "  http_port: 16333\n  grpc_port: 16334\n", encoding="utf-8")
            log = (work / "server.log").open("w", encoding="utf-8")
            process = subprocess.Popen([str(binary), "--config-path", str(config)], cwd=work,
                                       env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            deadline = time.monotonic() + 20
            while not available(url + "/collections"):
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"Private Qdrant failed to start; see {work / 'server.log'}")
                time.sleep(0.2)
        yield
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if log is not None:
            log.close()


def render_raw_report(worlds, records: list[dict], learning: list[dict], out: Path, gate: dict) -> None:
    by_q = {r["question_id"]: r for r in records}
    by_w = {r["world_id"]: r for r in learning}
    lines = [f"# B1: {len(worlds)}-world PLN-RAG run", "",
             "Live outputs from the selected dataset and scope.", "",
             "The label comes from the proof payload, with runtime/translation errors taking precedence.", "",
             f"Gate: **{gate['status']}**. This report contains the PLN-RAG results.", "",
             "| World | Kind | Supported depth | Target label | Control label | Parse failures |",
             "|---|---|---:|---|---|---:|"]
    for world in worlds:
        target, control = world.questions
        learned = by_w.get(world.id, {})
        lines.append(f"| {world.id} | {target.rule_kind} | {target.depth_bucket} | "
                     f"{by_q.get(target.id, {}).get('label', 'NOT RUN')} | "
                     f"{by_q.get(control.id, {}).get('label', 'NOT RUN')} | "
                     f"{learned.get('parse_failure_count', 'unknown')}/12 |")
    for world in worlds:
        lines += ["", f"## {world.id}", ""]
        lines += [f"- {s.id}: {s.text}" for s in world.sentences]
        learned = by_w.get(world.id, {})
        if learned:
            lines += ["", f"Learning: {learned.get('learning_seconds', 0):.3f} seconds.", "",
                      "Parsed atoms (unaltered):", "", "```json",
                      json.dumps([{k: row.get(k) for k in ('sentence_id', 'text', 'status', 'atoms', 'error', 'rejected_count')}
                                  for row in learned.get('sentences', [])], indent=2), "```"]
        for question in world.questions:
            record = by_q.get(question.id)
            lines += ["", f"### {question.id}", "", question.text, "",
                      f"Oracle: {question.expected_label}; exact depth: "
                      f"{question.depth_bucket if question.expected_label == 'SUPPORTED' else 'null'}."]
            if record:
                lines += [f"Result: {record['label']}; answer time: {record['answer_seconds']:.3f} seconds.", "",
                          f"Used IDs: {record['used_sentence_ids']}", "", "```json",
                          json.dumps(record.get("raw_response", {"error_kind": record.get("error_kind"),
                                                                "error_detail": record.get("error_detail")}), indent=2), "```",
                          "", f"[Full worker output and logs]({world.id}/{question.id}/output.json)"]
    (out / "raw.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pln(root: Path, repo: Path, python: Path, worlds_path: Path,
            expected_path: Path, run_id: str | None = None, world_limit: int = 8, *,
            dataset_kind: str = "pilot", output_dir: Path | None = None,
            request_budget_path: Path | None = None, environment: dict | None = None) -> Path:
    worlds = read_worlds(worlds_path)
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if dataset_kind == "benchmark":
        from .dataset import validate_full
        oracle = validate_full(worlds, expected)
    elif dataset_kind == "pilot":
        oracle = validate(worlds, expected)
        if len(worlds) != 8:
            raise ValueError("This stage accepts exactly the eight pilot worlds")
    else:
        raise ValueError("Unknown dataset kind")
    total_worlds = len(worlds)
    if not 1 <= world_limit <= total_worlds:
        raise ValueError(f"world_limit must be between 1 and {total_worlds}")
    worlds = worlds[:world_limit]  # The entire frozen dataset was validated above.
    env = dict(environment) if environment is not None else configured_environment(root)
    key = env.get("OPENAI_API_KEY", "").strip()
    if key in ("", "sk-...", "your-api-key"):
        raise RuntimeError("Set OPENAI_API_KEY in rule-chain-depth/.env before the live pilot")
    if env["PARSER"] != "canonical_pln":
        raise ValueError("This pilot requires the canonical_pln parser")
    llm_settings = LLMSettings.from_env(env)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Unsafe run ID")
    out = output_dir if output_dir is not None else root / "artifacts/pilot" / run_id
    out.mkdir(parents=True, exist_ok=False)
    original_repo = repository_manifest(repo)
    write_json(out / "oracle.json", oracle)
    write_json(out / "manifest.json", {"schema_version": 1, "run_id": run_id,
               "model": env["OPENAI_MODEL"], "parser": env["PARSER"], "pln_rag": original_repo,
               "dataset_sha256": oracle["dataset_sha256"], "python": str(python),
               "stage": dataset_kind, "full_dataset_world_count": total_worlds,
               "selected_world_ids": [w.id for w in worlds], "llm_settings": llm_settings.public(),
               "docker_image": env.get("B1_DOCKER_IMAGE"),
               "snapshot_policy": "learn once; restore atom file and fresh OS process per question; unique world vector collection"})
    request_budget_path = request_budget_path or root / "runtime/runs" / run_id / "request_budget.json"
    qdrant = env.get("B1_QDRANT_URL", "http://127.0.0.1:16333")
    records, learning = [], []
    try:
        with private_qdrant(root, qdrant, run_id, env):
            for world in worlds:
                print(f"START {world.id}: {world.questions[0].rule_kind}, depth {world.questions[0].depth_bucket}", flush=True)
                common = {"repo_path": str(repo), "model": env["OPENAI_MODEL"], "parser": env["PARSER"],
                          "world_id": world.id, "qdrant_url": qdrant,
                          "collection": f"b1_{run_id}_{world.id}".replace("-", "_"),
                          "reasoning_timeout": int(env.get("B1_REASONING_TIMEOUT", "30")),
                          "llm_config": asdict(llm_settings), "request_budget_path": str(request_budget_path),
                          "llm_timeout": int(env.get("B1_LLM_TIMEOUT", "60"))}
                learn_job = {**common, "mode": "learn", "sentences": public_sentences(world),
                             "work_dir": str(root / "runtime/runs" / run_id / world.id / "learn")}
                learn_out = out / world.id / "learn"
                learned, wall, log, failure = process_job(python, root, learn_job, learn_out, env,
                                                        float(env.get("B1_LEARN_TIMEOUT", "900")))
                learn_errors = log_errors(log)
                if learned and not learned.get("worker_error"):
                    learned["worker_wall_seconds"] = wall
                    learned["errors"].extend(learn_errors)
                    learning.append(learned)
                    write_json(learn_out / "output.json", learned)
                if learned and learned.get("errors"):
                    failure = learned["errors"][0]["kind"]
                if learned and learned.get("worker_error"):
                    failure = "learning_error"
                for question in world.questions:
                    query_out = out / world.id / question.id
                    if failure or not learned:
                        record = error_record(world.id, question.id, 0.0, failure or "learning_error",
                                              "The world could not be learned; see its learning output and raw.log")
                        record["attempted"] = False
                        record["learning_worker_wall_seconds"] = wall
                    else:
                        query_job = {**common, "mode": "query", "question_id": question.id, "question": question.text,
                                     "work_dir": str(root / "runtime/runs" / run_id / world.id / question.id),
                                     "snapshot_path": learned["snapshot_path"], "snapshot_sha256": learned["snapshot_sha256"],
                                     "learn_output_path": str(learn_out / "output.json")}
                        record, query_wall, query_log, query_failure = process_job(
                            python, root, query_job, query_out, env, float(env.get("B1_QUERY_TIMEOUT", "180")))
                        if query_failure or not record or record.get("worker_error"):
                            record = error_record(world.id, question.id, query_wall, query_failure or "worker_error",
                                                  "Query worker failed; see raw.log and worker output")
                        else:
                            record["worker_wall_seconds"] = query_wall
                            observed = log_errors(query_log)
                            if observed:
                                record["errors"].extend(observed)
                                record.update(label="ERROR", error_kind=observed[0]["kind"], error_detail=observed[0]["detail"])
                    record["depth_bucket"] = question.depth_bucket
                    record["rule_kind"] = question.rule_kind
                    record["expected_label"] = question.expected_label
                    record["correct"] = is_correct(record, question.expected_label)
                    record["parse_failure_count"] = learned.get("parse_failure_count") if learned else None
                    record["sentence_count"] = len(world.sentences)
                    validate_record(record, {s.id for s in world.sentences})
                    write_json(query_out / "record.json", record)
                    records.append(record)
                    write_json(out / "records.json", records)
                    print(f"RESULT {question.id}: {record['label']} (oracle {question.expected_label}), {record['answer_seconds']:.3f}s", flush=True)
                # Infrastructure failure cannot diagnose a rule kind. Stop burning calls.
                if failure:
                    print(f"STOP: learning/runtime failure in {world.id}: {failure}", flush=True)
                    break
    finally:
        if repository_manifest(repo) != original_repo:
            raise RuntimeError("PLN-RAG source files changed during the pilot")
        bad = [r for r in records if not r["correct"]]
        complete = len(records) == sum(len(w.questions) for w in worlds)
        calls = [call for row in learning for call in row.get("llm_http_calls", [])]
        calls += [call for row in records for call in row.get("llm_http_calls", [])]
        write_json(out / "llm_requests.json", calls)
        gate = {"status": "REVIEW_REQUIRED" if bad or not complete else (
                    "TRIAL_COMPLETE" if world_limit < total_worlds else
                    "BENCHMARK_COMPLETE" if dataset_kind == "benchmark" else "PILOT_PASSED"),
                "complete": complete, "record_count": len(records),
                "pilot_complete": dataset_kind == "pilot" and complete and world_limit == 8,
                "dataset_complete": complete and world_limit == total_worlds, "world_count": len(worlds),
                "requests_sent": json.loads(request_budget_path.read_text())["calls"] if request_budget_path.exists() else 0,
                "providers_reported": sorted({call.get("provider", "unmeasured") for call in calls}),
                "failed_question_ids": [r["question_id"] for r in bad],
                "failed_supported_rule_kinds": sorted({r["rule_kind"] for r in bad if r["expected_label"] == "SUPPORTED"}),
                "next_action": "Review raw outputs, errors, and parsing diagnostics before interpreting accuracy."}
        write_json(out / "gate.json", gate)
        write_json(out / "learning.json", learning)
        render_raw_report(worlds, records, learning, out, gate)
    return out


def run_pilot(root: Path, repo: Path, python: Path, worlds_path: Path,
              expected_path: Path, run_id: str | None = None, world_limit: int = 8) -> Path:
    return run_pln(root, repo, python, worlds_path, expected_path, run_id, world_limit)
