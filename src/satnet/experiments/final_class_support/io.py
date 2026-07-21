from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from satnet.experiments.final_class_support.constants import (
    CONTRACT_SPECIFICATION_HASH,
    FREEZE_ARCHIVE_SHA256,
    FROZEN_CONTRACT_COMMIT,
    FROZEN_CONTRACT_TAG,
    GENERATION_BYTE_COUNT,
    GENERATION_FILE_COUNT,
    GENERATION_LEDGER_SHA256,
    PRODUCTION_TOOLING_SHA,
    REPLAY_BYTE_COUNT,
    REPLAY_FILE_COUNT,
    REPLAY_LEDGER_SHA256,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    source = Path(path).read_text(encoding="utf-8")
    if not source or not source.endswith("\n"):
        raise ValueError(f"Canonical nonempty JSONL required: {path}")
    records = tuple(json.loads(line) for line in source[:-1].split("\n"))
    if not records or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"JSONL object records required: {path}")
    return records


def _is_within(path: Path, root: Path) -> bool:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    return resolved_path == resolved_root or resolved_path.is_relative_to(resolved_root)


def validate_output_root(output_root: str | Path, protected_roots: Sequence[str | Path]) -> Path:
    candidate = Path(output_root)
    for root in protected_roots:
        if _is_within(candidate, Path(root)):
            raise ValueError(f"Analysis output root is inside frozen evidence: {root}")
    return candidate


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Git command failed")
    return result.stdout.strip()


def _verify_manifest(root: Path, manifest_path: Path, count: int, size: int) -> dict[str, int]:
    files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: path.as_posix())
    if len(files) != count or sum(path.stat().st_size for path in files) != size:
        raise ValueError(f"Frozen evidence cardinality mismatch: {root}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != count:
        raise ValueError(f"Frozen manifest record count mismatch: {manifest_path}")
    expected_paths = [record["relative_path"] for record in records]
    if expected_paths != sorted(expected_paths):
        raise ValueError(f"Frozen manifest is not lexicographically ordered: {manifest_path}")
    actual_paths = [path.relative_to(root).as_posix() for path in files]
    if set(actual_paths) != set(expected_paths):
        raise ValueError(f"Frozen manifest path set mismatch: {manifest_path}")
    for record in records:
        path = root / Path(record["relative_path"])
        if path.stat().st_size != int(record["length_bytes"]):
            raise ValueError(f"Frozen file length mismatch: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Frozen file hash mismatch: {path}")
    return {"file_count": count, "byte_count": size, "verified_sha256_count": count}


def verify_frozen_evidence(
    *,
    production_tooling_root: str | Path,
    generation_root: str | Path,
    replay_root: str | Path,
    freeze_root: str | Path,
    freeze_archive: str | Path,
    freeze_archive_hash_file: str | Path,
) -> dict[str, Any]:
    tooling = Path(production_tooling_root)
    generation = Path(generation_root)
    replay = Path(replay_root)
    freeze = Path(freeze_root)
    archive = Path(freeze_archive)
    archive_hash_file = Path(freeze_archive_hash_file)
    for path in (tooling, generation, replay, freeze, archive, archive_hash_file):
        if not path.exists():
            raise FileNotFoundError(f"Required frozen input is absent: {path}")
    head = _git(tooling, "rev-parse", "HEAD")
    status = _git(tooling, "status", "--short")
    tag_target = _git(tooling, "rev-list", "-n", "1", FROZEN_CONTRACT_TAG)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FROZEN_CONTRACT_COMMIT, PRODUCTION_TOOLING_SHA],
        cwd=tooling,
        check=False,
    ).returncode == 0
    if head != PRODUCTION_TOOLING_SHA or status or tag_target != FROZEN_CONTRACT_COMMIT or not ancestor:
        raise ValueError("Frozen production tooling Git identity mismatch")
    contract = read_json(
        tooling / "artifacts" / "final_integrated_dataset_contract" / "contract_specification.json"
    )
    if contract.get("contract_spec_hash") != CONTRACT_SPECIFICATION_HASH:
        raise ValueError("Frozen contract specification hash mismatch")
    generation_ledger = generation / "operational" / "generation_ledger.json"
    replay_ledger = replay / "replay_ledger.json"
    if sha256_file(generation_ledger) != GENERATION_LEDGER_SHA256:
        raise ValueError("Frozen generation ledger hash mismatch")
    if sha256_file(replay_ledger) != REPLAY_LEDGER_SHA256:
        raise ValueError("Frozen replay ledger hash mismatch")
    archive_hash = sha256_file(archive)
    expected_hash_line = f"{archive_hash}  {archive.name}"
    if archive_hash != FREEZE_ARCHIVE_SHA256:
        raise ValueError("Frozen archive hash mismatch")
    if archive_hash_file.read_text(encoding="utf-8").strip() != expected_hash_line:
        raise ValueError("Frozen archive hash record mismatch")
    generation_result = _verify_manifest(
        generation,
        freeze / "generation_file_manifest_sha256.csv",
        GENERATION_FILE_COUNT,
        GENERATION_BYTE_COUNT,
    )
    replay_result = _verify_manifest(
        replay,
        freeze / "replay_file_manifest_sha256.csv",
        REPLAY_FILE_COUNT,
        REPLAY_BYTE_COUNT,
    )
    return {
        "verification_status": "passed",
        "tooling_sha": head,
        "tooling_status_clean": True,
        "frozen_contract_tag": FROZEN_CONTRACT_TAG,
        "frozen_contract_commit": tag_target,
        "contract_specification_hash": CONTRACT_SPECIFICATION_HASH,
        "generation_ledger_sha256": GENERATION_LEDGER_SHA256,
        "replay_ledger_sha256": REPLAY_LEDGER_SHA256,
        "freeze_archive_sha256": archive_hash,
        "generation": generation_result,
        "replay": replay_result,
        "combined": {
            "file_count": generation_result["file_count"] + replay_result["file_count"],
            "byte_count": generation_result["byte_count"] + replay_result["byte_count"],
            "verified_sha256_count": generation_result["verified_sha256_count"]
            + replay_result["verified_sha256_count"],
        },
    }


def format_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> int:
    destination = Path(path)
    normalized = list(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in normalized:
            writer.writerow({column: format_value(row.get(column)) for column in columns})
    return len(normalized)


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
