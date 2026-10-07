"""A controlled everyday vocabulary; render English from the oracle structure.

This renderer is used only to author/validate datasets. Systems receive the
English text and IDs, never these structures or the oracle's answers.
"""

from .model import Atom, Sentence

UNARY = {
    "sort_parcel": "sort a parcel", "check_label": "check a label",
    "pack_crate": "pack a crate", "seal_crate": "seal a crate",
    "paint_gate": "paint a gate", "clean_bench": "clean a bench",
    "carry_drum": "carry a drum", "light_lamp": "light a lamp",
    "make_tea": "make tea", "bake_bread": "bake bread",
    "weave_mat": "weave a mat", "fold_cloth": "fold a cloth",
    "carry_basket": "carry a basket", "lead_team": "lead a team",
    "open_gate": "open a gate", "read_note": "read a note",
}
BINARY = {word: word for word in ("mentor", "supervise", "train", "guide", "help")}
NAMES = ("Hana", "Bekele", "Abebe", "Aster", "Meron", "Selam", "Dawit",
         "Rahel", "Kebede", "Lemma", "Almaz", "Tigist", "Mulu", "Emebet", "Genet")


def fact_text(atom: Atom) -> str:
    if not atom.ground:
        raise ValueError("Cannot render an unground fact")
    if any(a not in NAMES for a in atom.arguments):
        raise ValueError(f"Unknown person in {atom}")
    if len(atom.arguments) == 1:
        return f"{atom.arguments[0]} can {UNARY[atom.predicate]}."
    return f"{atom.arguments[0]} can {BINARY[atom.predicate]} {atom.arguments[1]}."


def question_text(atom: Atom) -> str:
    fact = fact_text(atom)
    person, action = fact[:-1].split(" can ", 1)
    return f"Can {person} {action}?"


def sentence_text(sentence: Sentence) -> str:
    if sentence.is_fact:
        return fact_text(sentence.head)
    head, body = sentence.head, sentence.body
    if len(body) == 1 and body[0].arguments == ("?x",) and head.arguments == ("?x",):
        return f"Everyone who can {UNARY[body[0].predicate]} can {UNARY[head.predicate]}."
    if len(body) == 2 and all(p.arguments == ("?x",) for p in (*body, head)):
        return (f"Everyone who can {UNARY[body[0].predicate]} and can "
                f"{UNARY[body[1].predicate]} can {UNARY[head.predicate]}.")
    if (len(body) == 2 and body[0].arguments == ("?x", "?y")
            and body[1].arguments == ("?y", "?z") and head.arguments == ("?x", "?z")):
        return (f"If a person can {BINARY[body[0].predicate]} a second person and that second "
                f"person can {BINARY[body[1].predicate]} a third person, then the first person "
                f"can {BINARY[head.predicate]} the third person.")
    if len(body) == 1 and body[0].arguments == ("?x", "?y"):
        if head.arguments == ("?x",):
            return f"Everyone who can {BINARY[body[0].predicate]} another person can {UNARY[head.predicate]}."
        if head.arguments == ("?x", "?y"):
            return (f"If a person can {BINARY[body[0].predicate]} another person, then the "
                    f"first person can {BINARY[head.predicate]} that other person.")
    raise ValueError(f"Unsupported English template: {sentence}")


def authored_sentence(number: int, head: Atom, *body: Atom) -> Sentence:
    structural = Sentence(f"S{number:02}", "", head, tuple(body))
    return Sentence(structural.id, sentence_text(structural), head, tuple(body))
