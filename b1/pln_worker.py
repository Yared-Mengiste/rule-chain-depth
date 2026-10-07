"""One isolated PLN-RAG operation per OS process. Upstream code stays unchanged.

Input contains only English sentences/questions, IDs, and run configuration.
The oracle's logical forms and answer labels never enter this process.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from .environment import file_sha256, redact, secret_values
from .io import write_json
from .records import pln_label, proof_sources
from .snapshot import restore_snapshot
from .llm_control import HTTPAudit, LLMSettings


def jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return json.loads(json.dumps(value, default=str))


def run(job: dict) -> dict:
    repo = Path(job["repo_path"]).resolve()
    work = Path(job["work_dir"]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo))
    os.environ.update({
        "OPENAI_MODEL": job["model"], "PARSER": job["parser"],
        "ATOMSPACE_PATH": str(work / "kb.metta"),
        "QDRANT_URL": job["qdrant_url"], "QDRANT_COLLECTION": job["collection"],
        "FAISS_PATH": str(work / "faiss"),
        "CANONICAL_PLN_NL2PLN_MODULE_PATH": str(repo / "data/simba_canonical_pln.json"),
        "NL2PLN_MODULE_PATH": str(repo / "data/simba_all.json"),
        "CONCEPTNET_ENABLED": "false", "CONCEPTNET_AUTOLOAD": "false",
        "ANSWER_GENERATION_ENABLED": "true", "SOURCE_LOOKUP_MAX_ATOMS": "0",
        "PARSER_BATCH_SENTENCES": "1", "CONTEXT_TOP_K": "12",
        "CHAINING_TIMEOUT": str(job["reasoning_timeout"]), "CHAINING_MAX_STEPS": "100",
        "QUERY_FALLBACK_ENABLED": "true", "QUERY_CANDIDATE_MAX_TRIES": "5",
    })
    import dspy
    from config import get_settings
    from core.service import PLNRAGService
    from parsers import get_parser

    get_settings.cache_clear()
    cfg = get_settings()
    errors = []
    llm_calls = []
    parser_calls = []
    reasoner_calls = []
    retrieval_calls = []
    phase = {"name": "setup"}
    llm_settings = LLMSettings(**job["llm_config"])
    audit = HTTPAudit(llm_settings, Path(job["request_budget_path"]), lambda: phase["name"], errors)
    audit.install()

    class ObservedLM(dspy.LM):
        def forward(self, prompt=None, messages=None, **kwargs):
            started = time.perf_counter()
            row = {"phase": phase["name"], "prompt": prompt, "messages": messages}
            try:
                response = super().forward(prompt=prompt, messages=messages, **kwargs)
                row["response"] = jsonable(response)
                return response
            except Exception as exc:
                kind = "timeout" if "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower() else "llm_error"
                error = {"phase": phase["name"], "kind": kind,
                         "exception_type": type(exc).__name__, "detail": str(exc)}
                errors.append(error)
                row["error"] = error
                raise
            finally:
                row["seconds"] = time.perf_counter() - started
                llm_calls.append(row)

    restore_started = time.perf_counter()
    if job["mode"] == "query":
        snapshot = Path(job["snapshot_path"])
        restore_snapshot(snapshot, work / "kb.metta", job["snapshot_sha256"])
    elif (work / "kb.metta").exists():
        raise RuntimeError("Learning must begin with an empty, fresh work directory")
    restore_seconds = time.perf_counter() - restore_started

    started = time.perf_counter()
    parser = get_parser()
    service = PLNRAGService(parser)
    # Parser and answer-generator constructors configure their own global LMs.
    # Install one explicitly configured shared LM after both constructors.
    lm = ObservedLM(cfg.openai_model, api_key=cfg.openai_api_key,
                    **llm_settings.kwargs(), timeout=job["llm_timeout"])
    dspy.configure(lm=lm)

    original_retrieve = service._vector_store.retrieve_context

    def observed_retrieve(text, top_k):
        context, vector = original_retrieve(text, top_k)
        retrieval_calls.append({"phase": phase["name"], "text": text, "top_k": top_k,
                                "context_atoms": list(context)})
        return context, vector

    service._vector_store.retrieve_context = observed_retrieve

    def observe_parser(name):
        original = getattr(parser, name)

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            parser_calls.append({"phase": phase["name"], "method": name,
                                 "input": jsonable(args), "statements": list(result.statements),
                                 "queries": list(result.queries)})
            return result
        setattr(parser, name, observed)

    for method in ("parse", "parse_batch", "parse_query"):
        if hasattr(parser, method):
            observe_parser(method)

    original_query = service._reasoner._handler.query

    def observed_query(*args, **kwargs):
        started = time.perf_counter()
        row = {"args": jsonable(args), "kwargs": jsonable(kwargs)}
        try:
            result = original_query(*args, **kwargs)
            row["result"] = jsonable(result)
            return result
        except Exception as exc:
            error = {"phase": phase["name"], "kind": "timeout" if isinstance(exc, TimeoutError) else "reasoner_error",
                     "exception_type": type(exc).__name__, "detail": str(exc)}
            errors.append(error)
            row["error"] = error
            raise  # Upstream may swallow this; the observer retains the error.
        finally:
            row["seconds"] = time.perf_counter() - started
            reasoner_calls.append(row)

    service._reasoner._handler.query = observed_query
    setup_seconds = time.perf_counter() - started
    common = {"model_requested": job["model"], "parser": type(parser).__name__, "worker_pid": os.getpid(),
              "setup_seconds": setup_seconds, "restore_seconds": restore_seconds,
              "llm_settings": {**llm_settings.public(), "timeout_seconds": job["llm_timeout"]},
              "reasoning_settings": {"timeout_seconds_per_candidate": job["reasoning_timeout"],
                                     "candidate_max_tries": 5, "steps": 100},
              "errors": errors, "llm_calls": llm_calls, "llm_http_calls": audit.calls, "parser_calls": parser_calls,
              "reasoner_calls": reasoner_calls, "retrieval_calls": retrieval_calls}
    if job["mode"] == "learn":
        learned = []
        started = time.perf_counter()
        for sentence in job["sentences"]:
            phase["name"] = f"learn:{sentence['id']}"
            t0 = time.perf_counter()
            item = asyncio.run(service.ingest_batch([sentence["text"]]))[0].model_dump(mode="json")
            item["sentence_id"] = sentence["id"]
            item["seconds"] = time.perf_counter() - t0
            item["parse_failed"] = item["status"] != "success" or not item["atoms"] or item.get("rejected_count", 0) > 0
            learned.append(item)
            print(f"LEARN {job['world_id']} {sentence['id']}: {item['status']}, atoms={len(item['atoms'])}, rejected={item.get('rejected_count', 0)}", flush=True)
            if errors:
                break
        learning_seconds = time.perf_counter() - started
        snapshot = work / "learned.metta"
        kb = work / "kb.metta"
        if not kb.exists():
            kb.write_text("", encoding="utf-8")
        shutil.copyfile(kb, snapshot)
        snapshot.chmod(0o444)
        return {**common, "mode": "learn", "world_id": job["world_id"],
                "learning_seconds": learning_seconds, "sentences": learned,
                "sentence_count": len(learned), "parse_failure_count": sum(s["parse_failed"] for s in learned),
                "fully_failed_sentence_count": sum(not s["atoms"] for s in learned),
                "snapshot_path": str(snapshot), "snapshot_sha256": file_sha256(snapshot),
                "vector_count": service._vector_store.count}

    learned = json.loads(Path(job["learn_output_path"]).read_text(encoding="utf-8"))
    before = file_sha256(work / "kb.metta")
    before_atoms = (work / "kb.metta").read_text(encoding="utf-8").splitlines()
    phase["name"] = f"query:{job['question_id']}"
    started = time.perf_counter()
    response = asyncio.run(service.query(job["question"])).model_dump(mode="json")
    answer_seconds = time.perf_counter() - started
    label, error_kind = pln_label(response, errors)
    after_atoms = (work / "kb.metta").read_text(encoding="utf-8").splitlines()
    sources = proof_sources(response["raw_proof"], learned["sentences"])
    retrieved_atoms = {atom for row in retrieval_calls for atom in row["context_atoms"]}
    retrieved_ids = sorted({row["sentence_id"] for row in learned["sentences"]
                            if any(atom in retrieved_atoms for atom in row["atoms"])})
    return {**common, "schema_version": 1, "system": "pln_rag", "mode": "query",
            "world_id": job["world_id"], "question_id": job["question_id"],
            "question": job["question"], "label": label, "answer_seconds": answer_seconds,
            "error_kind": error_kind, "error_detail": errors[0]["detail"] if errors else None,
            "retrieved_sentence_ids": retrieved_ids, **sources,
            "parse_failure_count": learned["parse_failure_count"], "sentence_count": learned["sentence_count"],
            "snapshot_sha256": job["snapshot_sha256"], "before_query_sha256": before,
            "after_query_sha256": file_sha256(work / "kb.metta"),
            "query_added_statements": [atom for atom in after_atoms if atom not in before_atoms],
            "raw_response": response}


def main() -> int:
    args = argparse.ArgumentParser()
    args.add_argument("--job", required=True, type=Path)
    args.add_argument("--output", required=True, type=Path)
    options = args.parse_args()
    try:
        result = run(json.loads(options.job.read_text(encoding="utf-8")))
        clean = redact(json.dumps(result, ensure_ascii=False), secret_values(dict(os.environ)))
        write_json(options.output, json.loads(clean))
        return 0
    except Exception as exc:
        traceback.print_exc()
        write_json(options.output, {"worker_error": type(exc).__name__,
                                    "detail": redact(str(exc), secret_values(dict(os.environ)))})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
