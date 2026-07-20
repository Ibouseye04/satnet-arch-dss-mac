from __future__ import annotations

import math
from pathlib import Path
from statistics import pstdev
from typing import Any, Mapping, Sequence

from .constants import QUALIFICATION_RUN_IDS, TARGET_FIELDS
from .contract import ensure_mode_root, validate_frozen_contract
from .io import parse_canonical_float, read_canonical_json
from .mapping import FinalRunMapping
from .orchestrator import artifact_paths, run_directory, validate_completed_run


def _target_values(path: Path) -> dict[str, bool | float]:
    source = read_canonical_json(path)
    result: dict[str, bool | float] = {}
    for name in TARGET_FIELDS:
        value = source[name]
        result[name] = value if type(value) is bool else parse_canonical_float(value, name)
    return result


def _validate_ground_consistency(
    mappings: Sequence[FinalRunMapping], generation_root: Path
) -> None:
    by_design: dict[str, list[FinalRunMapping]] = {}
    for mapping in mappings:
        by_design.setdefault(mapping.design["design_id"], []).append(mapping)
    for design_id, values in by_design.items():
        expected = values[0].design
        invariants = (
            expected["selected_station_ids"],
            expected["ground_selection_seed"],
            expected["ground_selection_hash"],
            expected["ground_design_hash"],
            values[0].run["split_assignment"],
        )
        for mapping in values:
            if (
                mapping.design["selected_station_ids"],
                mapping.run["ground_selection_seed"],
                mapping.design["ground_selection_hash"],
                mapping.design["ground_design_hash"],
                mapping.run["split_assignment"],
            ) != invariants:
                raise ValueError(f"Ground design varies across realizations for {design_id}")
            g1 = (run_directory(generation_root, mapping.run_id) / "g1" / "ground_design.jsonl").read_text(encoding="utf-8")
            if not g1:
                raise ValueError(f"Missing G1 evidence for {mapping.run_id}")


def validate_qualification(
    *,
    mappings: Sequence[FinalRunMapping],
    generation_root: str | Path,
    replay_root: str | Path,
) -> dict[str, Any]:
    ordered = tuple(sorted(mappings, key=lambda value: value.run_id))
    if tuple(value.run_id for value in ordered) != QUALIFICATION_RUN_IDS:
        raise ValueError("Qualification mappings do not match the frozen 15-run set")
    generated = ensure_mode_root(generation_root, "qualification", create=False)
    replayed = ensure_mode_root(replay_root, "qualification_replay", create=False)
    generation_ledger = read_canonical_json(
        generated / "operational" / "generation_ledger.json"
    )
    replay_ledger = read_canonical_json(replayed / "replay_ledger.json")
    if generation_ledger["distinct_frozen_run_submission_count"] != len(ordered):
        raise ValueError("Qualification generation-ledger submission count mismatch")
    if generation_ledger["successful_generation_count"] != len(ordered):
        raise ValueError("Qualification generation-ledger success count mismatch")
    if replay_ledger["replay_submission_count"] != len(ordered):
        raise ValueError("Qualification replay-ledger submission count mismatch")
    if replay_ledger["successful_replay_count"] != len(ordered):
        raise ValueError("Qualification replay-ledger success count mismatch")
    result_hashes: list[str] = []
    for mapping in ordered:
        result = validate_completed_run(
            mapping=mapping, run_root=run_directory(generated, mapping.run_id)
        )
        result_hashes.append(result["run_result_hash"])
        replay = read_canonical_json(
            run_directory(replayed, mapping.run_id) / "replay_report.json"
        )
        if replay["replay_state"] != "succeeded" or not replay["input_tree_unchanged"]:
            raise ValueError(f"Qualification replay failed for run {mapping.run_id}")
        if replay["expected_result_hash"] != result["run_result_hash"]:
            raise ValueError(f"Replay result identity mismatch for run {mapping.run_id}")
        target = _target_values(artifact_paths(run_directory(generated, mapping.run_id))["target"])
        for name, value in target.items():
            if type(value) is float and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"Invalid target {name} for run {mapping.run_id}")
    _validate_ground_consistency(ordered, generated)
    return {
        "design_ids": ["D000", "D007", "D040"],
        "generation_count": len(result_hashes),
        "production_acceptance_claimed": False,
        "qualification_run_ids": list(QUALIFICATION_RUN_IDS),
        "qualification_validation_schema_version": "1",
        "replay_count": len(ordered),
        "result_hashes": result_hashes,
        "state": "passed",
    }


def validate_production_acceptance(
    *,
    mappings: Sequence[FinalRunMapping],
    generation_ledger: Mapping[str, Any],
    replay_ledger: Mapping[str, Any],
    generation_root: str | Path,
    protected_science_diff_empty: bool,
) -> dict[str, Any]:
    contract = validate_frozen_contract(compare_tag_blobs=True)
    gates = contract["specification"]["later_generation_acceptance_gates"]
    if len(mappings) != gates["expected_run_count"]:
        raise ValueError("Production acceptance run count mismatch")
    if generation_ledger["distinct_frozen_run_submission_count"] != gates["required_generation_attempt_count"]:
        raise ValueError("Production generation submission gate failed")
    if generation_ledger["successful_generation_count"] != gates["required_successful_generation_count"]:
        raise ValueError("Production successful-generation gate failed")
    if replay_ledger["replay_submission_count"] != gates["required_authoritative_replay_count"]:
        raise ValueError("Production replay submission gate failed")
    if replay_ledger["successful_replay_count"] != gates["required_successful_replay_count"]:
        raise ValueError("Production replay success gate failed")
    if not protected_science_diff_empty:
        raise ValueError("Protected-science diff gate failed")
    root = Path(generation_root)
    by_split: dict[str, list[dict[str, bool | float]]] = {
        "train": [], "validation": [], "test": []
    }
    pairs: set[tuple[str, str]] = set()
    run_ids: set[int] = set()
    design_splits: dict[str, str] = {}
    for mapping in mappings:
        if mapping.run_id in run_ids:
            raise ValueError("Duplicate production run ID")
        run_ids.add(mapping.run_id)
        pair = (mapping.design["design_id"], mapping.run["realization_id"])
        if pair in pairs:
            raise ValueError("Duplicate design-realization pair")
        pairs.add(pair)
        split = mapping.run["split_assignment"]
        previous = design_splits.setdefault(mapping.design["design_id"], split)
        if previous != split:
            raise ValueError("Design realizations are split across partitions")
        validate_completed_run(mapping=mapping, run_root=run_directory(root, mapping.run_id))
        by_split[split].append(
            _target_values(artifact_paths(run_directory(root, mapping.run_id))["target"])
        )
    for split, rows in by_split.items():
        classes = {row["overall_threshold_breach_any"] for row in rows}
        if classes != {False, True}:
            raise ValueError(f"Primary classification class gate failed for {split}")
        values = [float(row["failure_adjusted_overall_service_fraction_mean"]) for row in rows]
        if pstdev(values) == 0.0:
            raise ValueError(f"Primary regression spread gate failed for {split}")
        if len(set(values)) < gates["require_primary_regression_minimum_unique_values_per_split"]:
            raise ValueError(f"Primary regression unique-value gate failed for {split}")
    return {"production_acceptance": "passed", "validated_run_count": len(mappings)}
