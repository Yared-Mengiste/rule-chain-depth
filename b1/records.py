"""The shared result contract and conservative proof/error classification."""

from __future__ import annotations

import ast
import math
import re

from .model import LABELS


def error_record(world_id: str, question_id: str, elapsed: float, kind: str, detail: str) -> dict:
    return {
        "schema_version": 1, "system": "pln_rag", "world_id": world_id,
        "question_id": question_id, "label": "ERROR", "answer_seconds": elapsed,
        "used_sentence_ids": [], "retrieved_sentence_ids": [],
        "error_kind": kind, "error_detail": detail,
    }


def validate_record(record: dict, available_ids: set[str]) -> None:
    if record["label"] not in LABELS:
        raise ValueError("Unknown result label")
    seconds = record["answer_seconds"]
    if isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid answer latency")
    for field in ("used_sentence_ids", "retrieved_sentence_ids"):
        values = record[field]
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError(f"{field} must contain string sentence IDs")
        if len(set(values)) != len(values) or not set(values) <= available_ids:
            raise ValueError(f"Duplicate or invented sentence ID in {field}")
    if record["label"] == "ERROR" and not record.get("error_kind"):
        raise ValueError("An ERROR must carry its cause")


def pln_label(response: dict, errors: list[dict]) -> tuple[str, str | None]:
    if errors:
        return "ERROR", errors[0]["kind"]
    if response.get("query_status") in ("no_query", "malformed") or not response.get("executed_query"):
        return "ERROR", "question_parse_failure"
    try:
        proofs = ast.literal_eval(response["raw_proof"])
    except (KeyError, ValueError, SyntaxError, TypeError):
        return "ERROR", "invalid_proof_payload"
    if not isinstance(proofs, list) or any(not isinstance(p, str) or not p.strip() for p in proofs):
        return "ERROR", "invalid_proof_payload"
    return ("SUPPORTED", None) if proofs else ("NOT_ESTABLISHED", None)


def proof_sources(raw_proof: str, learned: list[dict]) -> dict:
    """Resolve proof symbols to input IDs by exact token equality, never similarity.

    Ambiguous atom names are disclosed, not guessed. Raw proof and raw parser
    output remain available for manual review, including query-added assertions.
    """
    names: dict[str, set[str]] = {}
    for row in learned:
        for atom in row.get("atoms", []):
            match = re.match(r"^\(:\s+([^\s()]+)\s", atom)
            if match:
                names.setdefault(match.group(1), set()).add(row["sentence_id"])
    try:
        traces = ast.literal_eval(raw_proof)
    except (ValueError, SyntaxError):
        traces = []
    tokens = set(re.findall(r"[^\s()\[\]'\",]+", " ".join(str(t) for t in traces)))
    used = set()
    ambiguous = {}
    for name in sorted(tokens & names.keys()):
        if len(names[name]) == 1:
            used.update(names[name])
        else:
            ambiguous[name] = sorted(names[name])
    return {"used_sentence_ids": sorted(used), "ambiguous_proof_symbols": ambiguous,
            "source_resolution": "ambiguous" if ambiguous else ("mapped" if used else "none")}


def log_errors(log: str) -> list[dict]:
    patterns = (
        (r"\[Reasoner\] Query failed.*", "reasoner_error"),
        (r"\[AnswerGenerator\] Failed.*", "answer_generation_error"),
        (r"^ERROR:.*", "engine_error"),
    )
    found = []
    for pattern, kind in patterns:
        for match in re.finditer(pattern, log, re.MULTILINE):
            detail = match.group(0)
            found.append({"kind": "timeout" if re.search(r"timed out|timeout", detail, re.I) else kind,
                          "detail": detail})
    return found


def is_correct(record: dict, expected_label: str) -> bool:
    return record["label"] != "ERROR" and record["label"] == expected_label
