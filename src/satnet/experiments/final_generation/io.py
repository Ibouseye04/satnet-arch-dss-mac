from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

from satnet.ground.canonical import canonical_float_string, canonical_hash, canonical_json


def pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_canonical_json(path: str | Path) -> dict[str, Any]:
    data = Path(path).read_bytes()
    if not data.endswith(b"\n") or data.count(b"\n") != 1:
        raise ValueError(f"Canonical JSON must contain one newline-terminated object: {path}")
    value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs_without_duplicates)
    if not isinstance(value, dict) or (canonical_json(value) + "\n").encode("utf-8") != data:
        raise ValueError(f"JSON artifact is not canonical: {path}")
    return value


def atomic_write_bytes(path: str | Path, data: bytes, *, overwrite: bool = False) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists() and not overwrite:
            raise FileExistsError(f"Artifact already exists: {target}")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: dict[str, Any], *, overwrite: bool = False) -> None:
    atomic_write_bytes(
        path,
        (canonical_json(value) + "\n").encode("utf-8"),
        overwrite=overwrite,
    )


def file_identity(path: str | Path) -> tuple[int, str]:
    data = Path(path).read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


def parse_canonical_float(value: object, field_name: str) -> float:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a canonical float string")
    parsed = float(value)
    if canonical_float_string(parsed) != value:
        raise ValueError(f"{field_name} is not a canonical float string")
    return parsed


def canonical_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or value.startswith("/"):
        raise ValueError("Scientific inventory path is not canonical")
    path = PurePosixPath(value)
    if any(part in {".", ".."} for part in path.parts) or path.as_posix() != value:
        raise ValueError("Scientific inventory path is not canonical")
    return value


def tree_inventory(root: str | Path) -> tuple[tuple[str, int, str], ...]:
    base = Path(root)
    return tuple(
        (path.relative_to(base).as_posix(), *file_identity(path))
        for path in sorted(item for item in base.rglob("*") if item.is_file())
    )


def tree_inventory_hash(root: str | Path) -> str:
    records = [
        {"byte_length": length, "path": path, "sha256": digest}
        for path, length, digest in tree_inventory(root)
    ]
    return canonical_hash({"files": records})
