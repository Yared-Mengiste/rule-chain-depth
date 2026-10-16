"""Isolated native PLN-RAG adapter. No reference data or synthetic oracle."""
import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import time

from b1.environment import file_sha256
from b1.llm_control import HTTPAudit, LLMSettings
from b1.snapshot import restore_snapshot
from .dataset import write
from .evidence import map_atoms, map_proof


def block_query_learning(parser, reasoner, observations, vector_store=None):
    """Retain native queries while refusing all parser-produced query facts."""
    for name in ("parse_query", "parse", "parse_batch", "retry_parse_query"):
        if not hasattr(parser, name):
            continue
        original = getattr(parser, name)

        def guarded(*args, _original=original, _name=name, **kwargs):
            result = _original(*args, **kwargs)
            if result is not None:
                observations.append({"method": _name, "blocked_statements": list(result.statements),
                                     "queries": list(result.queries)})
                result.statements = []
            return result
        setattr(parser, name, guarded)

    def reject(*args, **kwargs):
        raise RuntimeError("query_knowledge_mutation_blocked")
    reasoner.add_statements = reject
    reasoner.add_statements_report = reject
    reasoner._handler.add_atom = reject
    if vector_store is not None:
        for name in ("store", "store_many", "reset", "delete_by_source"):
            if hasattr(vector_store, name):
                setattr(vector_store, name, reject)


def observe_failures(original, errors, phase, kind):
    """Record exceptions before native service/parser code can swallow them."""
    def observed(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            errors.append({"kind": "timeout" if isinstance(exc, TimeoutError) else kind,
                           "phase": phase["name"], "detail": str(exc)})
            raise
    return observed


class DurableAudit(HTTPAudit):
    def __init__(self, *args, path, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path

    def _begin(self, request):
        if request.method == "POST" and str(request.url).rstrip("/") == self.settings.api_base + "/chat/completions":
            body = json.loads(request.content)
            expected = {"model": self.settings.model.split("/", 1)[1],
                        "temperature": self.settings.temperature, "max_tokens": self.settings.max_tokens}
            if any(body.get(key) != value for key, value in expected.items()):
                self.errors.append({"kind": "llm_configuration_mismatch", "phase": self.phase()})
                raise RuntimeError("Native parser changed matched model/generation settings")
        row = super()._begin(request)
        write(self.path, self.calls)
        return row

    def _finish(self, row, response):
        super()._finish(row, response)
        try:
            for key, value in response.json().get("usage", {}).items():
                if key in row["usage"] and type(value) is int and value >= 0:
                    row["usage"][key] = value
        except (ValueError, AttributeError):
            pass
        write(self.path, self.calls)

    def close(self):
        write(self.path, self.calls)
        super().close()


def run(job, output):
    output, repo = Path(output), Path(job["repo_path"])
    work = Path(job["work_dir"])
    work.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo))
    # Explicit overrides prevent inherited application state/background settings.
    os.environ.update({
        "OPENAI_MODEL": job["llm_config"]["model"], "PARSER": "canonical_pln",
        "ATOMSPACE_PATH": str(work / "kb.metta"), "FAISS_PATH": str(work / "faiss"),
        "QDRANT_URL": job["qdrant_url"], "QDRANT_COLLECTION": job["collection"],
        "OLLAMA_URL": job["ollama_url"], "OLLAMA_MODEL": job["embedding_model"],
        "CANONICAL_PLN_NL2PLN_MODULE_PATH": str(repo / "data/simba_canonical_pln.json"),
        "NL2PLN_MODULE_PATH": str(repo / "data/simba_all.json"),
        "CONCEPTNET_ENABLED": "false", "CONCEPTNET_AUTOLOAD": "false",
        "ANSWER_GENERATION_ENABLED": "false", "SOURCE_LOOKUP_MAX_ATOMS": "0",
        "PARSER_BATCH_SENTENCES": "1", "CONTEXT_TOP_K": str(job["pln_top_k"]),
        "CHAINING_TIMEOUT": str(int(job["reasoning_timeout"])), "CHAINING_MAX_STEPS": "100",
        "QUERY_FALLBACK_ENABLED": "true", "QUERY_CANDIDATE_MAX_TRIES": "5"})
    import dspy
    from config import get_settings
    from core.service import PLNRAGService
    from core.reasoner import PeTTaChainer
    from parsers import get_parser

    get_settings.cache_clear()
    cfg = get_settings()
    errors, parser_calls, retrieval_calls, reasoner_calls, blocked = [], [], [], [], []
    phase = {"name": "setup"}
    llm = LLMSettings(**job["llm_config"])
    audit = DurableAudit(llm, Path(job["request_budget_path"]), lambda: phase["name"], errors,
                         path=output / "requests.json")
    audit.install()
    row = {"system": "pln_rag", "status": "ERROR", "attempted": True,
           "errors": errors, "parser_calls": parser_calls, "retrieval_calls": retrieval_calls,
           "reasoner_calls": reasoner_calls, "blocked_query_statements": blocked,
           "llm_http_calls": audit.calls, "llm_settings": llm.public()}

    class ObservedLM(dspy.LM):
        def forward(self, *args, **kwargs):
            try:
                return super().forward(*args, **kwargs)
            except Exception as exc:
                errors.append({"phase": phase["name"], "kind": "llm_error", "detail": str(exc)})
                raise

    try:
        if job["mode"] == "query":
            restore_snapshot(Path(job["snapshot_path"]), work / "kb.metta", job["snapshot_sha256"])
        started = time.perf_counter()
        original_add = PeTTaChainer.add_atom
        PeTTaChainer.add_atom = observe_failures(original_add, errors, phase, "atom_load_error")
        parser = get_parser()
        service = PLNRAGService(parser)
        PeTTaChainer.add_atom = original_add
        if errors:
            raise RuntimeError("native_initialization_failed")
        dspy.configure(lm=ObservedLM(llm.model, api_key=os.environ["OPENAI_API_KEY"],
                                    **llm.kwargs(), timeout=job["llm_timeout"]))
        row["setup_seconds"] = time.perf_counter() - started
        row["native_settings"] = {
            "parser": type(parser).__name__, "context_top_k": service._context_top_k,
            "context_unit": "native vector entries carrying learned atoms",
            "enriched_context_max_atoms": 50, "engine_access": "entire learned atomspace",
            "chaining_timeout": cfg.chaining_timeout, "candidate_max_tries": cfg.query_candidate_max_tries,
            "chaining_max_steps_configured": cfg.chaining_max_steps,
            "chaining_max_steps_enforced_by_adapter": False,
            "answer_generation_enabled": cfg.answer_generation_enabled,
            "background_enabled": cfg.conceptnet_enabled,
            "parser_batch_sentences": cfg.parser_batch_sentences,
            "chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap}
        if service._context_top_k != job["pln_top_k"] or cfg.answer_generation_enabled or cfg.conceptnet_enabled:
            raise RuntimeError("native_configuration_mismatch")
        original_retrieve = service._vector_store.retrieve_context
        original_embed = observe_failures(service._vector_store.embed, errors, phase, "embedding_error")
        row["embedding_calls"] = 0

        def embed(text):
            row["embedding_calls"] += 1
            return original_embed(text)
        service._vector_store.embed = embed
        service._vector_store.store = observe_failures(service._vector_store.store, errors, phase, "vector_store_error")

        def observe_vector_response(response):
            if response.request.method == "POST" and response.status_code >= 400:
                errors.append({"kind": "vector_or_embedding_http_error", "phase": phase["name"],
                               "http_status": response.status_code})
        service._vector_store._client.event_hooks.setdefault("response", []).append(observe_vector_response)

        def retrieve(text, top_k):
            result = observe_failures(original_retrieve, errors, phase, "retrieval_error")(text, top_k)
            retrieval_calls.append({"phase": phase["name"], "top_k": top_k, "context_atoms": list(result[0])})
            return result
        service._vector_store.retrieve_context = retrieve
        for name in ("parse", "parse_batch", "parse_query", "retry_parse_query"):
            if not hasattr(parser, name):
                continue
            original = observe_failures(getattr(parser, name), errors, phase, "parser_error")

            def observed(*args, _original=original, _name=name, **kwargs):
                result = _original(*args, **kwargs)
                if result is None:
                    return result
                parser_calls.append({"phase": phase["name"], "method": _name, "input": args,
                                     "statements": list(result.statements), "queries": list(result.queries)})
                return result
            setattr(parser, name, observed)
        original_query = service._reasoner._handler.query

        def query(*args, **kwargs):
            call = {"args": args, "kwargs": kwargs}
            reasoner_calls.append(call)
            try:
                result = original_query(*args, **kwargs)
                call["result"] = result
                return result
            except Exception as exc:
                errors.append({"kind": "timeout" if isinstance(exc, TimeoutError) else "reasoner_error", "detail": str(exc)})
                raise
        service._reasoner._handler.query = query

        if job["mode"] == "learn":
            learned = row["sentences"] = []
            started = time.perf_counter()
            try:
                for sentence in job["sentences"]:
                    phase["name"] = "learn:" + sentence["id"]
                    t0 = time.perf_counter()
                    result = asyncio.run(service.ingest_batch([sentence["text"]]))[0].model_dump(mode="json")
                    result.update(sentence_id=sentence["id"], seconds=time.perf_counter() - t0)
                    result["parse_failed"] = result["status"] != "success" or not result["atoms"] or result.get("rejected_count", 0) > 0
                    learned.append(result)
                    write(output / "learning.partial.json", learned)
                    if errors:
                        break
            finally:
                row["learning_seconds"] = time.perf_counter() - started
            row["parse_failure_count"] = sum(s["parse_failed"] for s in learned)
            row["parsed_sentence_count"] = len(learned)
            kb = work / "kb.metta"
            if not kb.exists():
                kb.write_text("")
            snapshot = work / "learned.metta"
            shutil.copyfile(kb, snapshot)
            snapshot.chmod(0o444)
            row.update(snapshot_path=str(snapshot), snapshot_sha256=file_sha256(snapshot))
        else:
            learned = json.loads(Path(job["learn_output_path"]).read_text())["sentences"]
            block_query_learning(parser, service._reasoner, blocked, service._vector_store)
            before = file_sha256(work / "kb.metta")
            phase["name"] = "query"
            started = time.perf_counter()
            try:
                response = asyncio.run(service.query(job["hypothesis"])).model_dump(mode="json")
            finally:
                row["answer_seconds"] = time.perf_counter() - started
            row["raw_response"] = response
            write(output / "trace.json", response)
            after = file_sha256(work / "kb.metta")
            row.update(before_query_sha256=before, after_query_sha256=after,
                       snapshot_sha256=file_sha256(Path(job["snapshot_path"])))
            if before != after or row["snapshot_sha256"] != job["snapshot_sha256"]:
                raise RuntimeError("query_knowledge_mutation")
            if not response.get("executed_query") or response.get("query_status") in ("no_query", "malformed"):
                raise RuntimeError("hypothesis_parse_failure")
            mapping = map_proof(response.get("raw_proof"), learned)
            retrieval = map_atoms([a for c in retrieval_calls if c["phase"] == "query" for a in c["context_atoms"]], learned)
            parser_context = map_atoms([a for c in parser_calls if c["phase"] == "query"
                                        for a in c["input"][1]], learned)
            row.update(used_ids=mapping["ids"], used_reason=mapping["reason"],
                       resolved_used_ids=mapping["resolved_ids"], unresolved_sources=mapping["unresolved"],
                       empty_proof=mapping["empty"], retrieved_ids=retrieval["ids"],
                       retrieval_reason=retrieval["reason"], retrieval_mapping=retrieval,
                       parser_context_mapping=parser_context,
                       context_ids=sorted({s["sentence_id"] for s in learned if s.get("atoms")}),
                       context_basis="all successfully learned source atoms accessible to native reasoner")
        row["status"] = "ERROR" if errors else "OK"
        if errors:
            row["error_kind"] = errors[0]["kind"]
    except Exception as exc:
        row.update(error_kind=str(exc) if str(exc) in ("hypothesis_parse_failure", "query_knowledge_mutation", "native_configuration_mismatch") else type(exc).__name__,
                   error_detail=str(exc))
    finally:
        if "original_add" in locals():
            PeTTaChainer.add_atom = original_add
        row["timed_out"] = row.get("error_kind") == "timeout" or any(e["kind"] == "timeout" for e in errors)
        if row["status"] == "ERROR":
            row["used_ids"] = None
            row["empty_proof"] = None
            row["used_reason"] = row.get("error_kind", "native_failure")
        if any(e["kind"] in ("retrieval_error", "vector_or_embedding_http_error") for e in errors):
            row["retrieved_ids"] = None
            row["retrieval_reason"] = "native_retrieval_failed"
        audit.close()
        write(output / "output.json", row)
    return row
