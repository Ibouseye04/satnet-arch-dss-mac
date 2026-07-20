from __future__ import annotations

import math
from pathlib import Path
from statistics import pstdev
from typing import Any, Sequence

from .constants import QUALIFICATION_RUN_IDS, TARGET_FIELDS
from .contract import ensure_mode_root, validate_catalog, validate_frozen_contract
from .evidence import validate_generation_evidence, validate_replay_evidence
from .io import parse_canonical_float, read_canonical_json
from .mapping import FinalRunMapping
from .orchestrator import run_directory


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
    catalog = validate_catalog()
    targets, results = validate_generation_evidence(
        mappings=ordered,
        generation_root=generated,
        catalog=catalog,
        ledger=generation_ledger,
    )
    validate_replay_evidence(
        mappings=ordered,
        replay_root=replayed,
        generation_results=results,
        generation_targets=targets,
        ledger=replay_ledger,
    )
    result_hashes = [results[mapping.run_id]["run_result_hash"] for mapping in ordered]
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
    generation_root: str | Path,
    replay_root: str | Path,
    protected_science_diff_empty: bool,
) -> dict[str, Any]:
    contract = validate_frozen_contract(compare_tag_blobs=True)
    gates = contract["specification"]["later_generation_acceptance_gates"]
    ordered = tuple(sorted(mappings, key=lambda value: value.run_id))
    if tuple(mapping.run_id for mapping in ordered) != tuple(range(gates["expected_run_count"])):
        raise ValueError("Production acceptance frozen run set mismatch")
    if not protected_science_diff_empty:
        raise ValueError("Protected-science diff gate failed")
    generated = ensure_mode_root(generation_root, "production", create=False)
    replayed = ensure_mode_root(replay_root, "production_replay", create=False)
    generation_ledger = read_canonical_json(
        generated / "operational" / "generation_ledger.json"
    )
    replay_ledger = read_canonical_json(replayed / "replay_ledger.json")
    targets, results = validate_generation_evidence(
        mappings=ordered,
        generation_root=generated,
        catalog=validate_catalog(),
        ledger=generation_ledger,
    )
    validate_replay_evidence(
        mappings=ordered,
        replay_root=replayed,
        generation_results=results,
        generation_targets=targets,
        ledger=replay_ledger,
    )
    expected_counts = {
        "distinct_frozen_run_submission_count": gates["required_generation_attempt_count"],
        "successful_generation_count": gates["required_successful_generation_count"],
        "replay_submission_count": gates["required_authoritative_replay_count"],
        "successful_replay_count": gates["required_successful_replay_count"],
    }
    for field, expected in expected_counts.items():
        source = generation_ledger if field in generation_ledger else replay_ledger
        if source.get(field) != expected:
            raise ValueError(f"Production acceptance count gate failed: {field}")
    by_split: dict[str, list[dict[str, bool | float]]] = {
        "train": [], "validation": [], "test": []
    }
    design_splits: dict[str, str] = {}
    for mapping in ordered:
        split = mapping.run["split_assignment"]
        previous = design_splits.setdefault(mapping.design["design_id"], split)
        if previous != split:
            raise ValueError("Design realizations are split across partitions")
        values: dict[str, bool | float] = {}
        target = targets[mapping.run_id]
        for name in TARGET_FIELDS:
            value = target[name]
            parsed = value if type(value) is bool else parse_canonical_float(value, name)
            if type(parsed) is float and (
                not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0
            ):
                raise ValueError(f"Invalid target {name} for run {mapping.run_id}")
            values[name] = parsed
        by_split[split].append(values)
    for split, rows in by_split.items():
        classes = {row["overall_threshold_breach_any"] for row in rows}
        if classes != {False, True}:
            raise ValueError(f"Primary classification class gate failed for {split}")
        values = [float(row["failure_adjusted_overall_service_fraction_mean"]) for row in rows]
        if pstdev(values) == 0.0:
            raise ValueError(f"Primary regression spread gate failed for {split}")
        if len(set(values)) < gates["require_primary_regression_minimum_unique_values_per_split"]:
            raise ValueError(f"Primary regression unique-value gate failed for {split}")
    return {
        "derived_generation_submission_count": len(generation_ledger["records"]),
        "derived_replay_submission_count": len(replay_ledger["records"]),
        "production_acceptance": "passed",
        "validated_run_count": len(ordered),
    }
