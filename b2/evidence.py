"""Conservative source resolution and local evidence overlap (not proof accuracy)."""
import ast
import re


def sexpr(text):
    tokens = re.findall(r'\(|\)|[^\s()]+', text)
    position = 0

    def parse():
        nonlocal position
        if position >= len(tokens):
            raise ValueError("Incomplete atom")
        token = tokens[position]
        position += 1
        if token == "(":
            items = []
            while position < len(tokens) and tokens[position] != ")":
                items.append(parse())
            if position == len(tokens):
                raise ValueError("Unclosed atom")
            position += 1
            return items
        if token == ")":
            raise ValueError("Unexpected closing parenthesis")
        return token

    result = parse()
    if position != len(tokens):
        raise ValueError("Trailing atom data")
    return result


def source_index(learned):
    names, atoms = {}, {}
    for row in learned:
        for atom in row.get("atoms", []):
            atoms.setdefault(" ".join(atom.split()), set()).add(row["sentence_id"])
            try:
                parsed = sexpr(atom)
                if isinstance(parsed, list) and len(parsed) == 4 and parsed[0] == ":" and isinstance(parsed[1], str):
                    names.setdefault(parsed[1], set()).add(row["sentence_id"])
            except ValueError:
                pass
    return names, atoms


def map_atoms(context, learned):
    _, atoms = source_index(learned)
    ids, unresolved = set(), []
    for atom in context:
        candidates = atoms.get(" ".join(atom.split()), set())
        if len(candidates) == 1:
            ids.update(candidates)
        else:
            unresolved.append(atom)
    return {"ids": sorted(ids) if not unresolved else None, "resolved_ids": sorted(ids),
            "unresolved": unresolved, "reason": "ambiguous_or_unmapped_atoms" if unresolved else None}


def map_proof(raw, learned):
    """Only proof terms are inspected: predicate/type tokens are never citations.

    Unrecognized native proof syntax is deliberately unmeasured. The complete
    trace and partially resolved IDs remain available for inspection.
    """
    names, _ = source_index(learned)
    used, unresolved = set(), []

    def term(value):
        if isinstance(value, str):
            candidates = names.get(value, set())
            if len(candidates) == 1:
                used.update(candidates)
            else:
                unresolved.append(value)
        elif isinstance(value, list) and value:
            # Known composition operators carry proof arguments, not source IDs.
            if value[0] in ("ModusPonens", "modus-ponens", "And", "and", "Application", "apply"):
                for child in value[1:]:
                    term(child)
            else:
                # A named learned rule can itself be the function head.
                for child in value:
                    term(child)
        else:
            unresolved.append(repr(value))

    try:
        traces = ast.literal_eval(raw)
        if not isinstance(traces, list) or any(not isinstance(t, str) for t in traces):
            raise ValueError("Invalid trace list")
        for trace in traces:
            parsed = sexpr(trace)
            if not isinstance(parsed, list) or len(parsed) != 4 or parsed[0] != ":":
                raise ValueError("Unrecognized proof trace form")
            term(parsed[1])
        empty = not traces
    except (ValueError, SyntaxError, TypeError):
        unresolved.append("invalid_or_unrecognized_proof_trace")
        empty = None
    return {"ids": sorted(used) if not unresolved else None, "resolved_ids": sorted(used),
            "unresolved": sorted(set(unresolved)), "empty": empty,
            "reason": "unresolved_native_proof_sources" if unresolved else None}


def overlap(ids, gold, *, eligible=None, reason=None):
    if ids is None:
        return {"precision": None, "recall": None, "f1": None, "count": None,
                "all_gold": None, "reason": reason or "unavailable"}
    predicted, gold = set(ids), set(gold)
    hits = predicted & gold
    if eligible is not None:
        hits &= set(eligible)
    precision = len(hits) / len(predicted) if predicted else 0.0
    recall = len(hits) / len(gold) if gold else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "count": len(predicted), "all_gold": bool(gold) and hits == gold, "reason": None}


def evaluate(record, reference, world, top_k):
    row = dict(record)
    gold = reference["gold_leaves"]
    full = row["system"] == "full_context"
    native = row["system"] == "pln_rag"
    context, used = row.get("context_ids"), row.get("used_ids")
    if row.get("status") == "ERROR":
        used = None  # Failed/partial answers are not measured empty evidence.
    reason = row.get("error_kind") or "stage_not_measured"
    row["depth"] = reference["calculated_depth"]
    row["evidence"] = {
        "retrieved": overlap(None if full else row.get("retrieved_ids"), gold,
            reason="retrieval_not_used" if full else row.get("retrieval_reason") or reason),
        "context": overlap(context, gold, reason=row.get("context_reason") or reason),
        "used": overlap(used, gold, eligible=None if native else context or [],
            reason=row.get("used_reason") or reason)}
    all_ids = {s["id"] for s in world["sentences"]}
    row["diagnostics"] = {
        "gold_leaf_count": len(gold),
        "gold_fits_retrieval_limit": None if full or native else len(gold) <= top_k,
        "retrieval_limit_reason": "PLN limit counts vector entries/atoms, not source sentences" if native else "retrieval_not_used" if full else None,
        "unknown_citations": len(set(used) - all_ids) if used is not None else None,
        "unseen_citations": len((set(used) & all_ids) - set(context)) if not native and used is not None and context is not None else None,
        "empty_evidence": not used if used is not None else None,
        "empty_proof": row.get("empty_proof") if row.get("status") != "ERROR" else None,
        "unresolved_pln_sources": len(row["unresolved_sources"]) if "unresolved_sources" in row else None,
        "parse_failure_count": row.get("parse_failure_count"),
    }
    return row
