from pathlib import Path
import shutil

from .environment import file_sha256


def restore_snapshot(source: Path, destination: Path, expected_sha256: str) -> str:
    if file_sha256(source) != expected_sha256:
        raise RuntimeError("Learned snapshot changed before restore")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    actual = file_sha256(destination)
    if actual != expected_sha256:
        raise RuntimeError("Restored atomspace differs from learned snapshot")
    return actual
