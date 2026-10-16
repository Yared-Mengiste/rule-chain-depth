"""Structural validation and deterministic authoring, with no logical oracle."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

REVISION = "daac2fdb7ab52ec3ef8f2953f59288c1edd7c2f0"
SOURCE_PATH = "data/public_dataset/entailment_trees_emnlp2021_data_v2/dataset/task_2/dev.jsonl"
SOURCE_SHA256 = "3271adc67c65149780adbd3729f6b19404ff288e1849905fc16c1c22814a28f7"
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
GROUP_COUNTS = {1: 51, 2: 64, 3: 28, 4: 18, 5: 13, 6: 7, 7: 6}
RECIPE = "One random.Random(17); shuffle copies of source-order groups at depths 1..7; take six each; emit six rounds of depths 1..7."


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def graph(record):
    """Check graph identity, text, dependencies, reachability and longest depth."""
    try:
        meta = record["meta"]
        sentences, intermediates = meta["triples"], meta["intermediate_conclusions"]
        root = meta["hypothesis_id"]
        require(isinstance(record["id"], str) and bool(record["id"]), "Missing source ID")
        require(set(sentences) == {f"sent{i}" for i in range(1, 26)}, "Expected 25 original sentences")
        require(all(isinstance(t, str) and t.strip() for t in sentences.values()), "Empty sentence")
        require(all(re.fullmatch(r"int[1-9][0-9]*", i) and isinstance(t, str) and t.strip()
                    for i, t in intermediates.items()), "Invalid intermediate ID/text")
        require(root in intermediates and intermediates[root] == record["hypothesis"], "Hypothesis text mismatch")
        expected_context = " ".join(f"{sid}: {text}" for sid, text in sentences.items())
        require(record["context"] == expected_context, "Original context and triples disagree")
        steps = {}
        for raw in record["proof"].split(";"):
            if not raw.strip():
                continue
            match = re.fullmatch(r"\s*((?:sent|int)[1-9][0-9]*(?:\s*&\s*(?:sent|int)[1-9][0-9]*)*)\s*->\s*(hypothesis|int[1-9][0-9]*)(?:\s*:\s*(.+?))?\s*", raw)
            require(match is not None, "Malformed proof step")
            premises = re.split(r"\s*&\s*", match[1].strip())
            target = root if match[2] == "hypothesis" else match[2]
            require(target in intermediates and target not in steps, "Unknown/duplicate conclusion")
            require(len(premises) == len(set(premises)), "Duplicate premise")
            require(all(p in sentences or p in intermediates for p in premises), "Missing premise reference")
            require(match[3] is None or match[3].strip() == intermediates[target], "Intermediate text mismatch")
            require(match[2] == "hypothesis" or match[3] is not None, "Missing intermediate text")
            steps[target] = premises
        require(set(steps) == set(intermediates), "Missing intermediate proof")
        visited, active, leaves, depths = set(), set(), set(), {}

        def visit(node):
            if node in sentences:
                leaves.add(node)
                return 0
            require(node not in active, "Cycle in reference proof")
            if node not in depths:
                active.add(node)
                depths[node] = 1 + max(visit(p) for p in steps[node])
                active.remove(node)
                visited.add(node)
            return depths[node]

        depth = visit(root)
        require(visited == set(steps), "Conclusion unreachable from hypothesis")
        require(set(meta["distractors"]) == set(sentences) - leaves, "Distractor annotations disagree with leaves")
        discrepancies = {}
        for key, actual in (("depth_of_proof", depth), ("length_of_proof", len(steps))):
            require(type(record[key]) is int, f"Invalid {key}")
            if record[key] != actual:
                discrepancies[key] = {"original": record[key], "calculated": actual}
        return {"calculated_depth": depth, "calculated_length": len(steps),
                "gold_leaves": sorted(leaves), "steps": steps, "discrepancies": discrepancies}
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed reference record") from exc


def selected_ids(groups):
    rng = random.Random(17)
    selected = {}
    for depth in range(1, 8):
        items = list(groups[str(depth)])
        rng.shuffle(items)
        selected[depth] = items[:6]
    return [selected[depth][round_] for round_ in range(6) for depth in range(1, 8)]


def public_world(record, number):
    return {"id": f"b2-{number:03d}", "hypothesis": record["hypothesis"],
            "sentences": [{"id": sid, "text": text} for sid, text in record["meta"]["triples"].items()]}


def author(source_path, license_path, output):
    require(digest(source_path) == SOURCE_SHA256, "Pinned source checksum mismatch")
    require(digest(license_path) == LICENSE_SHA256, "Pinned license checksum mismatch")
    rows = [json.loads(line) for line in Path(source_path).read_text().splitlines() if line.strip()]
    analyses = {r["id"]: graph(r) for r in rows}
    require(len(analyses) == len(rows), "Duplicate source ID")
    require(Counter(g["calculated_depth"] for g in analyses.values()) == GROUP_COUNTS, "Unexpected depth distribution")
    groups = {str(d): [r["id"] for r in rows if analyses[r["id"]]["calculated_depth"] == d] for d in range(1, 8)}
    ids, by_id = selected_ids(groups), {r["id"]: r for r in rows}
    worlds = [public_world(by_id[sid], i) for i, sid in enumerate(ids, 1)]
    references = [{"world_id": w["id"], "source_id": sid, **analyses[sid], "source_record": by_id[sid]}
                  for w, sid in zip(worlds, ids)]
    output = Path(output)
    write(output / "worlds.json", worlds)
    write(output / "references.json", references)
    (output / "LICENSE").write_bytes(Path(license_path).read_bytes())
    write(output / "source.json", {
        "dataset": "EntailmentBank Task 2 fixed 42-world benchmark", "schema_version": 1,
        "repository": "https://github.com/allenai/entailment_bank", "revision": REVISION,
        "source_path": SOURCE_PATH, "source_sha256": SOURCE_SHA256,
        "attribution": "Dalvi et al. (2021), Explaining Answers with Entailment Trees, https://aclanthology.org/2021.emnlp-main.585/",
        "seed": 17, "algorithm": RECIPE, "source_order_groups": groups,
        "selected_source_ids": ids, "selected_world_ids": [w["id"] for w in worlds],
        "calculated_depths": [r["calculated_depth"] for r in references],
        "source_annotation_discrepancies": {sid: a["discrepancies"] for sid, a in analyses.items() if a["discrepancies"]},
        "checksums": {name: digest(output / name) for name in ("worlds.json", "references.json", "LICENSE")}})
    return validate(output)


def bundle_path():
    local = Path(__file__).resolve().parents[1] / "data/b2"
    return local if local.is_dir() else Path(sys.prefix) / "share/rule-chain-depth/b2"


def validate(path=None):
    path = Path(path) if path is not None else bundle_path()
    source = read(path / "source.json")
    require((source["revision"], source["source_sha256"], source["source_path"]) ==
            (REVISION, SOURCE_SHA256, SOURCE_PATH), "Source identity mismatch")
    require(source["seed"] == 17 and source["algorithm"] == RECIPE, "Selection recipe mismatch")
    require(set(source["checksums"]) == {"worlds.json", "references.json", "LICENSE"}, "Missing bundle checksums")
    for name, expected in source["checksums"].items():
        require(digest(path / name) == expected, f"Bundle checksum mismatch: {name}")
    require(digest(path / "LICENSE") == LICENSE_SHA256, "License checksum mismatch")
    groups = source["source_order_groups"]
    require(set(groups) == {str(d) for d in range(1, 8)}, "Invalid source groups")
    require({int(d): len(ids) for d, ids in groups.items()} == GROUP_COUNTS, "Invalid source group counts")
    all_ids = [sid for ids in groups.values() for sid in ids]
    require(len(all_ids) == len(set(all_ids)), "Duplicate source inventory ID")
    ids = selected_ids(groups)
    require(ids == source["selected_source_ids"], "Selection/order mismatch")
    worlds, refs = read(path / "worlds.json"), read(path / "references.json")
    require(len(worlds) == len(refs) == len(set(ids)) == 42, "Expected 42 unique worlds")
    require([w["id"] for w in worlds] == source["selected_world_ids"], "World order mismatch")
    depths = []
    for i, (world, ref, sid) in enumerate(zip(worlds, refs, ids), 1):
        record = ref["source_record"]
        require(ref["world_id"] == world["id"] and ref["source_id"] == record["id"] == sid, "Source mapping mismatch")
        require(world == public_world(record, i), "Public inputs differ from original source")
        analysis = graph(record)
        require(all(ref[k] == v for k, v in analysis.items()), "Reference graph metadata mismatch")
        require(analysis["calculated_depth"] == (i - 1) % 7 + 1, "Depth/order mismatch")
        require(sid in groups[str(analysis["calculated_depth"])], "Source group mismatch")
        require(analysis["discrepancies"] == source["source_annotation_discrepancies"].get(sid, {}), "Discrepancy metadata mismatch")
        depths.append(analysis["calculated_depth"])
    require(depths == source["calculated_depths"], "Calculated depth metadata mismatch")
    return {"worlds": worlds, "references": refs, "source": source,
            "bundle_sha256": digest(path / "source.json"),
            "depth_counts": dict(sorted(Counter(depths).items()))}
