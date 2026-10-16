"""Small local evidence/resource reports with explicit measurement coverage."""
from collections import Counter
import json

from .dataset import write


def measurement(values, reasons=None):
    measured = [v for v in values if type(v) in (int, float, bool)]
    return {"mean": sum(measured) / len(measured) if measured else None,
            "sum": sum(measured) if measured else None,
            "measured": len(measured), "total": len(values),
            "unavailable_reasons": dict(Counter(r or "unavailable" for v, r in zip(values, reasons or [None]*len(values)) if v is None))}


def resources(row):
    calls = row.get("llm_http_calls", [])
    values = {"observed_chat_calls": len(calls), "embedding_calls": row.get("embedding_calls"),
              **{key: row.get(key) for key in ("learning_seconds", "index_seconds", "answer_seconds", "worker_seconds")}}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        tokens = [c.get("usage", {}).get(key) for c in calls]
        # No token value is invented when an API does not report it.
        values[key] = sum(tokens) if tokens and all(type(v) is int for v in tokens) else None
    return values


def group(rows):
    metrics = {}
    for stage in ("retrieved", "context", "used"):
        for key in ("precision", "recall", "f1", "count", "all_gold"):
            metrics[f"{stage}.{key}"] = measurement([r["evidence"][stage][key] for r in rows],
                                                      [r["evidence"][stage]["reason"] for r in rows])
    for key in ("gold_leaf_count", "gold_fits_retrieval_limit", "unknown_citations", "unseen_citations",
                "empty_evidence", "empty_proof", "unresolved_pln_sources", "parse_failure_count"):
        metrics[key] = measurement([r["diagnostics"][key] for r in rows],
            [r["diagnostics"].get("retrieval_limit_reason") if key == "gold_fits_retrieval_limit"
             else r.get("error_kind") or "not_applicable_or_unmeasured" for r in rows])
    for key in ("observed_chat_calls", "embedding_calls", "learning_seconds", "index_seconds",
                "answer_seconds", "worker_seconds", "prompt_tokens", "completion_tokens", "total_tokens"):
        metrics[key] = measurement([r["resources"][key] for r in rows],
            [r.get("error_kind") or "not_used_or_not_reported" for r in rows])
    calls = [c for r in rows for c in r.get("llm_http_calls", [])]
    metrics["error_rate"] = measurement([r["status"] == "ERROR" for r in rows])
    metrics["timeout_rate"] = measurement([bool(r.get("timed_out")) or r.get("error_kind") == "timeout" for r in rows])
    metrics["unattempted_rate"] = measurement([not r.get("attempted", False) for r in rows])
    return {"world_count": len({r["world_id"] for r in rows}), "record_count": len(rows),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "errors": dict(Counter(r["error_kind"] for r in rows if r.get("error_kind"))),
            "unattempted": sum(not r.get("attempted", False) for r in rows), "metrics": metrics,
            "reported_models": sorted({c.get("model", "unmeasured") for c in calls}),
            "reported_providers": sorted({c.get("provider", "unmeasured") for c in calls})}


def summarize(rows, manifest, reserved):
    calls = [c for r in rows for c in r.get("llm_http_calls", [])]
    usage = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [c.get("usage", {}).get(key) for c in calls]
        values = [v if type(v) is int else None for v in values]
        values.extend([None] * max(0, reserved - len(calls)))
        usage[key] = measurement(values, ["API usage not returned or audit incomplete"] * len(values))
    return {"status": manifest["status"], "record_count": len(rows),
            "expected_record_count": manifest["record_count"],
            "full_dataset_selected": manifest["full_dataset_selected"],
            "accounting_complete": len(rows) == manifest["record_count"],
            "execution_complete": all(r["status"] == "OK" for r in rows),
            "world_count": manifest["world_count"], "overall": group(rows),
            "systems": {s: {"overall": group([r for r in rows if r["system"] == s]),
                            "by_depth": {str(d): group([r for r in rows if r["system"] == s and r["depth"] == d]) for d in range(1, 8)}}
                        for s in manifest["systems"]},
            "by_depth": {str(d): group([r for r in rows if r["depth"] == d]) for d in range(1, 8)},
            "requests_reserved": reserved, "request_audits": len(calls),
            "request_audit_coverage": {"measured": len(calls), "total": reserved},
            "reported_usage": usage,
            "interpretation": "Evidence overlap with annotated leaves, not logical correctness or official EntailmentBank proof accuracy. No label accuracy; all targets are supported."}


def render(output, summary, rows):
    def cell(metric):
        mean = metric["mean"]
        return f"{mean:.3f} ({metric['measured']}/{metric['total']})" if mean is not None else f"unavailable (0/{metric['total']})"

    lines = ["# B2: fixed EntailmentBank evidence comparison", "",
        f"Status: **{summary['status']}**. {summary['world_count']} selected worlds; {summary['record_count']} system/world records.", "",
        "Completion describes execution, not successful evidence use or reasoning. Evidence overlap measures agreement with annotated leaves; valid alternative evidence may differ. This is not official proof accuracy. The balanced subset does not represent the full dataset distribution.", "",
        "Means below include measured cases only; every value shows measured/total coverage. Empty measured evidence scores zero; failures and unresolved sources remain unavailable. Counts and reasons, diagnostics and resource coverage are in summary.json.", "",
        "| System | Depth | Worlds | Retrieved F1 | Supplied-context recall | Used/cited F1 |",
        "|---|---|---:|---|---|---|"]
    for system, data in summary["systems"].items():
        for depth, g in [("all", data["overall"]), *data["by_depth"].items()]:
            m = g["metrics"]
            lines.append(f"| {system} | {depth} | {g['world_count']} | {cell(m['retrieved.f1'])} | {cell(m['context.recall'])} | {cell(m['used.f1'])} |")
    lines += ["", "Full context uses the same generation prompt/settings with every original sentence. Its retrieval metrics are unavailable; complete supplied context does not establish correct reasoning. PLN parser context uses retrieval plus atomspace enrichment, and its reasoner can access the complete learned atomspace. Unresolved proof terms are not guessed.", "",
        "| World | Depth | RAG context recall | RAG used F1/status | Full-context used F1/status | PLN used F1/status |",
        "|---|---:|---|---|---|---|"]
    by_world = {}
    for row in rows:
        by_world.setdefault(row["world_id"], {})[row["system"]] = row

    def result(row):
        if row is None:
            return "not selected"
        value = row["evidence"]["used"]["f1"]
        return f"{value if value is not None else 'unavailable'} / {row['status']}"

    for wid, systems in by_world.items():
        rag = systems.get("rag", {})
        recall = rag.get("evidence", {}).get("context", {}).get("recall")
        depth = next(iter(systems.values()))["depth"]
        lines.append(f"| {wid} | {depth} | {recall if recall is not None else 'unavailable'} | {result(systems.get('rag'))} | {result(systems.get('full_context'))} | {result(systems.get('pln_rag'))} |")
    lines += ["", "| System | Learning seconds | Index seconds | Answer seconds | Chat calls | Input tokens | Output tokens |",
              "|---|---|---|---|---|---|---|"]
    for system, data in summary["systems"].items():
        m = data["overall"]["metrics"]
        lines.append("| " + system + " | " + " | ".join(cell(m[k]) for k in (
            "learning_seconds", "index_seconds", "answer_seconds", "observed_chat_calls", "prompt_tokens", "completion_tokens")) + " |")
    lines += ["", f"Shared requests reserved: {summary['requests_reserved']}; retained audits: {summary['request_audits']}.",
              "Reported token totals (partial totals retain coverage):", "", "```json", json.dumps(summary["reported_usage"], indent=2), "```", "",
              "Inspect records.json, llm_requests.json and workers/<world>/<system>/ for original public inputs, model requests/responses, learned snapshots and native traces. manifest.json records effective settings and implementation identity."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save(output, manifest, records, reserved):
    rows = [{**r, "resources": resources(r)} for r in records]
    summary = summarize(rows, manifest, reserved)
    write(output / "records.json", rows)
    write(output / "summary.json", summary)
    write(output / "llm_requests.json", [{"world_id": r["world_id"], "system": r["system"], **c}
        for r in rows for c in r.get("llm_http_calls", [])])
    render(output, summary, rows)
