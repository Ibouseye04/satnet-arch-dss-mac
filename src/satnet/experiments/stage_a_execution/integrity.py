from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .common import payload_hash, sha256_file
from .paths import validate_no_links

INVENTORY_DOMAIN = "satnet_stage_a_artifact_inventory_v1"


def artifact_inventory(root: Path, *, excluded: Iterable[str] = ()) -> list[dict[str, Any]]:
    excluded_set = set(excluded)
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.relative_to(root).as_posix() not in excluded_set)
    validate_no_links(root, files)
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "byte_length": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def inventory_hash(records: list[dict[str, Any]]) -> str:
    return payload_hash({"artifacts": records}, domain=INVENTORY_DOMAIN)


def verify_artifact_inventory(root: Path, records: list[dict[str, Any]]) -> None:
    actual = artifact_inventory(root)
    if actual != records:
        raise ValueError("Completed artifact inventory changed")
    paths = [record["relative_path"] for record in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Artifact inventory ordering or uniqueness mismatch")
