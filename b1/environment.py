"""Configure isolated subprocesses without writing to PLN-RAG's checkout."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess


def configured_environment(root: Path) -> dict[str, str]:
    result = dict(os.environ)
    dotenv = root / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.removeprefix("export ").partition("=")
            if not separator:
                raise ValueError(f"Invalid .env entry for {key!r}")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            result.setdefault(key.strip(), value)
    result.setdefault("OPENAI_MODEL", "openai/gpt-4o-mini")
    result.setdefault("PARSER", "canonical_pln")
    result.setdefault("OLLAMA_URL", "http://127.0.0.1:11434/api/embeddings")
    result.setdefault("OLLAMA_MODEL", "nomic-embed-text")
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    result["PYTHONUNBUFFERED"] = "1"
    result["DSPY_CACHEDIR"] = str(root / ".cache/dspy")
    result["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    # The rootless runtime is optional: a configured existing venv also works.
    swi = root / "runtime/swipl/usr/lib/swi-prolog"
    rootless = result.get("B1_USE_ROOTLESS_RUNTIME", "true").lower() == "true"
    if rootless and swi.exists():
        result["SWI_HOME_DIR"] = str(swi)
        library = str(root / "runtime/swipl/usr/lib/x86_64-linux-gnu")
        result["LD_LIBRARY_PATH"] = library + (":" + result["LD_LIBRARY_PATH"] if result.get("LD_LIBRARY_PATH") else "")
    dependency_dirs = []
    for pattern in ("runtime/deps/PeTTa-*/python", "runtime/deps/PeTTaChainer-*",
                    "runtime/deps/NL2PLN-*/src"):
        candidates = list(root.glob(pattern)) if rootless else []
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous dependency selection for {pattern}; keep one resolved version")
        dependency_dirs.extend(str(p) for p in candidates)
    paths = [str(root), *dependency_dirs]
    if result.get("PYTHONPATH"):
        paths.append(result["PYTHONPATH"])
    result["PYTHONPATH"] = os.pathsep.join(paths)
    return result


def secret_values(environment: dict[str, str]) -> tuple[str, ...]:
    return tuple(sorted({v for k, v in environment.items()
                         if any(word in k.upper() for word in ("API_KEY", "TOKEN", "PASSWORD", "SECRET"))
                         and len(v) >= 8}, key=len, reverse=True))


def redact(text: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        text = text.replace(secret, "[REDACTED]")
    return text


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_manifest(repo: Path) -> dict:
    files = [p for p in repo.rglob("*.py") if ".git" not in p.parts and "__pycache__" not in p.parts]
    files += [repo / "data/simba_canonical_pln.json", repo / "data/simba_all.json"]
    hashes = {str(p.relative_to(repo)): file_sha256(p) for p in sorted(files) if p.is_file()}
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return {"path": str(repo), "git_commit": commit.stdout.strip() or None, "source_sha256": hashes}
