"""Internal single-operation worker entry point; accepts public inputs only."""
import argparse
import json
import os
from pathlib import Path
import traceback

from b1.environment import redact, secret_values
from .dataset import write

COMMON = {"system", "mode", "world_id", "llm_config", "request_budget_path", "llm_timeout"}
PLN = {"repo_path", "work_dir", "qdrant_url", "collection", "ollama_url", "embedding_model",
       "pln_top_k", "reasoning_timeout"}
BASELINE = {"normal_rag_repo", "ollama_url", "embedding_timeout", "top_k"}


def validate_job(job):
    if job["system"] == "pln_rag":
        fields = {"sentences"} if job["mode"] == "learn" else {
            "hypothesis", "snapshot_path", "snapshot_sha256", "learn_output_path"}
        expected = COMMON | PLN | fields
        if job["mode"] not in ("learn", "query"):
            raise ValueError("Invalid native mode")
    elif job["system"] in ("rag", "full_context") and job["mode"] == "answer":
        expected = COMMON | BASELINE | {"sentences", "hypothesis"}
    else:
        raise ValueError("Unknown worker system/mode")
    if set(job) != expected:
        raise ValueError("Worker payload contains missing or non-public fields")
    if "sentences" in job:
        sentences = job["sentences"]
        if len(sentences) != 25 or any(set(s) != {"id", "text"} for s in sentences):
            raise ValueError("Expected only 25 original sentence IDs/texts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    options = parser.parse_args()
    output = options.job.parent
    try:
        job = json.loads(options.job.read_text())
        validate_job(job)
        if job["system"] == "pln_rag":
            from .pln_worker import run
            row = run(job, output)
        else:
            from .baseline import run
            row = run(job, dict(os.environ), output)
    except Exception as exc:
        traceback.print_exc()
        row = {"status": "ERROR", "attempted": True, "error_kind": type(exc).__name__, "error_detail": str(exc)}
    clean = redact(json.dumps(row, ensure_ascii=False, default=str), secret_values(dict(os.environ)))
    write(output / "output.json", json.loads(clean))
    return 0 if row["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
