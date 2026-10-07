"""Fail-closed cross-checks and dataset admission rules."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import re

from . import backward, forward
from .language import authored_sentence, question_text, sentence_text
from .model import Atom, Sentence, Solution, World


class OracleDisagreement(RuntimeError):
    pass


def consensus(sentences: tuple[Sentence, ...], goal: Atom) -> Solution:
    bottom_up = forward.solve(sentences, goal)
    top_down = backward.solve(sentences, goal)
    if bottom_up.signature() != top_down.signature():
        raise OracleDisagreement(json.dumps({
            "goal": goal.to_dict(), "forward": bottom_up.to_dict(), "backward": top_down.to_dict(),
        }, sort_keys=True))
    return bottom_up


def fingerprint(worlds: tuple[World, ...]) -> str:
    payload = json.dumps([w.to_dict() for w in worlds], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate(worlds: tuple[World, ...], hand_expected: dict | None = None) -> dict:
    answers = []
    ids = set()
    if len({w.id for w in worlds}) != len(worlds):
        raise ValueError("Duplicate world ID")
    for world in worlds:
        if len(world.sentences) != 12:
            raise ValueError(f"{world.id}: B1 worlds must contain exactly 12 sentences")
        for sentence in world.sentences:
            if sentence.text != sentence_text(sentence):
                raise ValueError(f"{world.id}/{sentence.id}: English and logical form disagree")
        for text in [s.text for s in world.sentences] + [q.text for q in world.questions]:
            forbidden = [w for w in re.findall(r"[A-Za-z]+", text) if w.lower().endswith("s")]
            if forbidden:
                raise ValueError(f"{world.id}: words ending in s: {forbidden}")
        by_id = {q.id: q for q in world.questions}
        for question in world.questions:
            if question.id in ids:
                raise ValueError(f"Duplicate question ID: {question.id}")
            ids.add(question.id)
            if question.text != question_text(question.goal):
                raise ValueError(f"{question.id}: question text and goal disagree")
            result = consensus(world.sentences, question.goal)
            if result.label != question.expected_label:
                raise ValueError(f"{question.id}: expected {question.expected_label}, found {result.label}")
            if result.label == "SUPPORTED":
                if result.depth != question.depth_bucket:
                    raise ValueError(f"{question.id}: intended depth {question.depth_bucket}, exact depth {result.depth}; shortcut or invalid construction")
                for proof in result.proofs:
                    if proof.depth == result.depth and world.sentences[-1].id in proof.sentence_ids:
                        raise ValueError(f"{question.id}: a needed sentence was left last")
                    # Check that the reported sources alone really prove the goal.
                    subset = tuple(s for s in world.sentences if s.id in proof.sentence_ids)
                    if consensus(subset, question.goal).depth != proof.depth:
                        raise ValueError(f"{question.id}: invalid provenance")
            else:
                paired = by_id.get(question.paired_supported_id)
                if (not paired or paired.expected_label != "SUPPORTED" or
                        paired.depth_bucket != question.depth_bucket or paired.rule_kind != question.rule_kind):
                    raise ValueError(f"{question.id}: invalid control pair")
                repair = authored_sentence(9000, question.repair_fact)
                if any(s.id == repair.id for s in world.sentences):
                    raise ValueError("Reserved repair sentence ID already in use")
                repaired = consensus((*world.sentences, repair), question.goal)
                if repaired.depth != question.depth_bucket:
                    raise ValueError(f"{question.id}: adding its missing fact must yield depth {question.depth_bucket}")
            entry = {"world_id": world.id, "question_id": question.id,
                     "depth_bucket": question.depth_bucket, "rule_kind": question.rule_kind,
                     **result.to_dict()}
            if hand_expected is not None:
                expected = hand_expected.get(question.id)
                actual = {key: entry[key] for key in ("label", "depth", "minimum_depth_proofs")}
                if actual != expected:
                    raise ValueError(f"{question.id}: disagrees with hand-annotated answer: {actual} != {expected}")
            answers.append(entry)
    if hand_expected is not None and set(hand_expected) != ids:
        raise ValueError("Hand-annotated question IDs do not match dataset")
    return {"schema_version": 1, "dataset_sha256": fingerprint(worlds),
            "status": "AGREED", "world_count": len(worlds), "question_count": len(answers),
            "solvers": ["forward_saturation", "backward_goal_search"],
            "depth_definition": "0 for a fact; 1 + max(premise depths) for a rule application; null when unsupported",
            "control_bucket_definition": "Depth of the paired supported target, also verified after adding the control's one missing fact",
            "answers": answers}
