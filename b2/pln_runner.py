"""Supervise all selected PLN worlds with separate learning and query workers."""
import json
from pathlib import Path
import uuid

from b1.environment import redact, secret_values
from .dataset import write


def budget_exhausted(path, limit):
    return bool(limit and path.exists() and json.loads(path.read_text())["calls"] >= limit)


def run_pln(job, environment, execute):
    output = Path(job["output_dir"])
    budget = Path(job["common"]["request_budget_path"])
    limit = job["common"]["llm_config"]["max_calls"]
    rows = [{"system": "pln_rag", "world_id": w["id"], "status": "NOT_ATTEMPTED", "attempted": False,
             "error_kind": "not_attempted", "llm_http_calls": []} for w in job["worlds"]]

    def checkpoint():
        write(output / "pln_records.json", rows)

    checkpoint()
    for world, row in zip(job["worlds"], rows):
        if budget_exhausted(budget, limit):
            row["error_kind"] = "request_budget_exceeded"
            checkpoint()
            continue
        operation = output / "pln_rag" / world["id"]
        common = {**job["common"], "world_id": world["id"],
                  "collection": f"b2_{uuid.uuid4().hex}_{world['id'].replace('-', '_')}"}
        # Durable active marker allows the parent to report abrupt container loss.
        row.update(status="ERROR", attempted=True, answer_attempted=False, error_kind="worker_interrupted")
        checkpoint()
        try:
            learned = execute({**common, "mode": "learn", "sentences": world["sentences"],
                "work_dir": str(operation / "learn/work")}, operation / "learn", job["timeouts"]["LEARN_TIMEOUT"])
            row.update(learning_seconds=learned.get("learning_seconds"),
                learning_worker_seconds=learned.get("worker_seconds"), parse_failure_count=learned.get("parse_failure_count"),
                embedding_calls=learned.get("embedding_calls"), llm_http_calls=learned.get("llm_http_calls", []))
            if learned.get("status") != "OK":
                row.update(status="ERROR", error_kind="learning_failed", learning_error=learned.get("error_kind"),
                           timed_out=learned.get("error_kind") == "timeout")
            else:
                row["answer_attempted"] = True
                checkpoint()
                result = execute({**common, "mode": "query", "hypothesis": world["hypothesis"],
                    "work_dir": str(operation / "query/work"), "snapshot_path": learned["snapshot_path"],
                    "snapshot_sha256": learned["snapshot_sha256"], "learn_output_path": str(operation / "learn/output.json")},
                    operation / "query", job["timeouts"]["QUERY_TIMEOUT"])
                result["llm_http_calls"] = row["llm_http_calls"] + result.get("llm_http_calls", [])
                result["embedding_calls"] = (learned["embedding_calls"] + result["embedding_calls"]
                    if type(learned.get("embedding_calls")) is int and type(result.get("embedding_calls")) is int else None)
                row.pop("error_kind", None)
                row.update(result)
        except KeyboardInterrupt:
            row.update(status="ERROR", error_kind="interrupted")
            raise
        except Exception as exc:
            row.update(status="ERROR", error_kind=type(exc).__name__,
                       error_detail=redact(str(exc), secret_values(environment)))
        finally:
            checkpoint()
    return rows
