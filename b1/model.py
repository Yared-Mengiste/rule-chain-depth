"""Function-free positive Horn clauses. No LLM or inference library is used here."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

LABELS = frozenset({"SUPPORTED", "NOT_ESTABLISHED", "ERROR"})
RULE_KINDS = frozenset({"chain", "conjunction", "relational", "mixed"})


def variable(term: str) -> bool:
    return term.startswith("?")


@dataclass(frozen=True, order=True)
class Atom:
    predicate: str
    arguments: tuple[str, ...]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z_]*", self.predicate):
            raise ValueError(f"Invalid predicate: {self.predicate!r}")
        if len(self.arguments) not in (1, 2):
            raise ValueError("Only unary and binary atoms are allowed")
        if any(not re.fullmatch(r"\??[A-Za-z][A-Za-z_]*", a) for a in self.arguments):
            raise ValueError(f"Invalid arguments: {self.arguments!r}")

    @property
    def ground(self) -> bool:
        return not any(variable(a) for a in self.arguments)

    def to_dict(self) -> dict[str, Any]:
        return {"predicate": self.predicate, "arguments": list(self.arguments)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Atom:
        return cls(value["predicate"], tuple(value["arguments"]))


@dataclass(frozen=True)
class Sentence:
    id: str
    text: str
    head: Atom
    body: tuple[Atom, ...] = ()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"S\d{2,}", self.id):
            raise ValueError(f"Invalid sentence ID: {self.id!r}")
        if not self.body and not self.head.ground:
            raise ValueError("A fact must be ground")
        if len(self.body) > 2:
            raise ValueError("At most two rule premises are allowed")
        bound = {a for p in self.body for a in p.arguments if variable(a)}
        if any(variable(a) and a not in bound for a in self.head.arguments):
            raise ValueError("Every head variable must occur in a premise")

    @property
    def is_fact(self) -> bool:
        return not self.body

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "type": "fact" if self.is_fact else "rule",
            "head": self.head.to_dict(),
            "body": [a.to_dict() for a in self.body],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Sentence:
        result = cls(value["id"], value["text"], Atom.from_dict(value["head"]),
                     tuple(Atom.from_dict(a) for a in value["body"]))
        if value["type"] != ("fact" if result.is_fact else "rule"):
            raise ValueError("Sentence type disagrees with its logical form")
        return result


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    goal: Atom
    expected_label: str
    depth_bucket: int
    rule_kind: str
    paired_supported_id: str | None = None
    repair_fact: Atom | None = None

    def __post_init__(self) -> None:
        if not self.goal.ground:
            raise ValueError("Questions must be ground yes/no goals")
        if self.expected_label not in LABELS - {"ERROR"}:
            raise ValueError("The oracle cannot label a question ERROR")
        if self.depth_bucket not in range(4):
            raise ValueError("B1 covers depth buckets 0 through 3")
        if self.rule_kind not in RULE_KINDS | {"fact"}:
            raise ValueError("Unknown rule kind")
        if self.expected_label == "NOT_ESTABLISHED":
            if not self.paired_supported_id or not self.repair_fact or not self.repair_fact.ground:
                raise ValueError("An unsupported control needs a supported pair and one repair fact")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "text": self.text, "goal": self.goal.to_dict(),
            "expected_label": self.expected_label, "depth_bucket": self.depth_bucket,
            "rule_kind": self.rule_kind, "paired_supported_id": self.paired_supported_id,
            "repair_fact": self.repair_fact.to_dict() if self.repair_fact else None,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Question:
        return cls(value["id"], value["text"], Atom.from_dict(value["goal"]),
                   value["expected_label"], value["depth_bucket"], value["rule_kind"],
                   value.get("paired_supported_id"),
                   Atom.from_dict(value["repair_fact"]) if value.get("repair_fact") else None)


@dataclass(frozen=True)
class World:
    id: str
    sentences: tuple[Sentence, ...]
    questions: tuple[Question, ...]
    shuffle_seed: int

    def __post_init__(self) -> None:
        if len({s.id for s in self.sentences}) != len(self.sentences):
            raise ValueError("Duplicate sentence ID")
        if len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError("Duplicate question ID")
        arities: dict[str, int] = {}
        for atom in [p for s in self.sentences for p in (s.head, *s.body)] + [q.goal for q in self.questions]:
            if atom.predicate in arities and arities[atom.predicate] != len(atom.arguments):
                raise ValueError(f"Inconsistent arity for {atom.predicate}")
            arities[atom.predicate] = len(atom.arguments)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "shuffle_seed": self.shuffle_seed,
                "sentences": [s.to_dict() for s in self.sentences],
                "questions": [q.to_dict() for q in self.questions]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> World:
        return cls(value["id"], tuple(Sentence.from_dict(s) for s in value["sentences"]),
                   tuple(Question.from_dict(q) for q in value["questions"]), value["shuffle_seed"])


@dataclass(frozen=True, order=True)
class Proof:
    depth: int
    sentence_ids: frozenset[str]


@dataclass(frozen=True)
class Solution:
    """Non-dominated proofs: retain distinct sources, even at greater depth.

    A proof dominates another only when its sources are a subset AND its depth
    is no greater. Keeping this frontier avoids losing alternative proofs when
    two branches have unequal depths. Both solvers implement their own pruning.
    """

    proofs: tuple[Proof, ...]

    @property
    def depth(self) -> int | None:
        return min((p.depth for p in self.proofs), default=None)

    @property
    def label(self) -> str:
        return "SUPPORTED" if self.proofs else "NOT_ESTABLISHED"

    def signature(self) -> tuple[tuple[int, tuple[str, ...]], ...]:
        return tuple(sorted((p.depth, tuple(sorted(p.sentence_ids))) for p in self.proofs))

    def to_dict(self) -> dict[str, Any]:
        best = sorted((sorted(p.sentence_ids) for p in self.proofs if p.depth == self.depth),
                      key=lambda ids: (len(ids), ids))
        return {
            "label": self.label, "depth": self.depth,
            "proof_sentence_ids": best[0] if best else [],
            "minimum_depth_proofs": best,
            "proof_frontier": [{"depth": d, "sentence_ids": list(ids)} for d, ids in self.signature()],
        }
