from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import RUN_FILES, SCIENTIFIC_FILE_KEYS
from .contract import ensure_mode_root, validate_output_root
from .io import file_identity
from .orchestrator import run_directory


def compare_repeat(
    *, primary_root: str | Path, repeat_root: str | Path, run_id: int = 200
) -> dict[str, Any]:
    primary = ensure_mode_root(primary_root, "qualification", create=False)
    repeat = validate_output_root(repeat_root, other_roots=(primary,))
    repeat = ensure_mode_root(repeat, "qualification_repeat", create=False)
    primary_run = run_directory(primary, run_id)
    repeat_run = run_directory(repeat, run_id)
    keys = (*SCIENTIFIC_FILE_KEYS, "inventory", "result")
    comparisons: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for key in keys:
        left = primary_run / RUN_FILES[key]
        right = repeat_run / RUN_FILES[key]
        left_identity = file_identity(left)
        right_identity = file_identity(right)
        matched = left.read_bytes() == right.read_bytes()
        comparisons.append(
            {
                "artifact_key": key,
                "byte_length": left_identity[0],
                "matched": matched,
                "primary_sha256": left_identity[1],
                "repeat_sha256": right_identity[1],
            }
        )
        if not matched:
            mismatches.append(key)
    result = {
        "artifact_comparisons": comparisons,
        "deterministic_match": not mismatches,
        "final_result_hash_match": not any(key == "result" for key in mismatches),
        "mismatches": mismatches,
        "run_id": run_id,
        "scientific_inventory_match": not any(key == "inventory" for key in mismatches),
    }
    if mismatches:
        raise ValueError(f"Deterministic repeat mismatch: {mismatches}")
    return result
