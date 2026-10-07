from __future__ import annotations

import json
from pathlib import Path

from .model import World


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_worlds(path: Path) -> tuple[World, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["schema_version"] != 1:
        raise ValueError("Unsupported dataset schema version")
    return tuple(World.from_dict(w) for w in value["worlds"])
