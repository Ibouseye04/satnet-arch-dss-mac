from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from satnet.ground.canonical import canonical_hash
from satnet.ground.catalog import GroundStationCatalog

from .constants import RUN_ID_WIDTH
from .io import read_canonical_json
from .mapping import FinalRunMapping
from .orchestrator import (
    artifact_paths,
    attempt_input_identity,
    run_directory,
    validate_completed_run,
)

_REQUIRED_STAGES = (
    "satellite",
    "g1",
    "g2",
    "g3",
    "g4",
    "g5",
    "target",
    "inventory",
    "result",
)
_GENERATION_RECORD_FIELDS = frozenset(
    {
        "attempt_count",
        "contract_spec_hash",
        "design_id",
        "design_index",
        "design_record_hash",
        "failure_evidence_hashes",
        "ground_design_hash",
        "ground_failure_seed",
        "ground_selection_hash",
        "ground_selection_seed",
        "published_result_hash",
        "realization_id",
        "realization_index",
        "run_id",
        "run_key",
        "run_record_hash",
        "satellite_seed",
        "scientific_inventory_hash",
        "split",
        "state",
        "target_artifact_hash",
    }
)
_REPLAY_RECORD_FIELDS = frozenset(
    {
        "after_input_tree_inventory_hash",
        "before_input_tree_inventory_hash",
        "contract_spec_hash",
        "design_record_hash",
        "expected_result_hash",
        "first_mismatch",
        "input_result_hash",
        "input_tree_unchanged",
        "per_stage_comparison",
        "recomputed_result_hash",
        "replay_report_schema_version",
        "replay_state",
        "run_id",
        "run_key",
        "run_record_hash",
        "scientific_inventory_hash",
        "target_artifact_hash",
    }
)


def _indexed_records(
    records: object, expected_ids: set[int], name: str
) -> dict[int, dict[str, Any]]:
    if not isinstance(records, list):
        raise TypeError(f"{name} records must be a list")
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or type(record.get("run_id")) is not int:
            raise TypeError(f"{name} record has invalid run_id")
        run_id = record["run_id"]
        if run_id in indexed:
            raise ValueError(f"Duplicate {name} run ID: {run_id}")
        indexed[run_id] = record
    actual_ids = set(indexed)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(f"{name} run set mismatch; missing={missing}, extra={extra}")
    return indexed


def _validate_generation_attempts(
    *, root: Path, mapping: FinalRunMapping, result_hash: str
) -> int:
    attempt_root = root / "operational" / "attempts" / f"run_{mapping.run_id:0{RUN_ID_WIDTH}d}"
    paths = sorted(attempt_root.glob("attempt_*.json")) if attempt_root.is_dir() else []
    if not paths:
        raise ValueError(f"Missing generation attempt evidence for run {mapping.run_id}")
    expected_identity = attempt_input_identity(mapping)
    successful = 0
    for path in paths:
        record = read_canonical_json(path)
        if record.get("attempt_input_identity") != expected_identity:
            raise ValueError(f"Attempt identity mismatch for run {mapping.run_id}")
        if record.get("attempt_input_identity_hash") != canonical_hash(expected_identity):
            raise ValueError(f"Attempt identity hash mismatch for run {mapping.run_id}")
        if record.get("state") == "succeeded":
            successful += 1
            if record.get("published_result_hash") != result_hash:
                raise ValueError(f"Attempt result hash mismatch for run {mapping.run_id}")
    if successful != 1:
        raise ValueError(f"Run {mapping.run_id} must have exactly one successful attempt")
    return len(paths)


def validate_generation_evidence(
    *,
    mappings: Sequence[FinalRunMapping],
    generation_root: str | Path,
    catalog: GroundStationCatalog,
    ledger: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    ordered = tuple(sorted(mappings, key=lambda value: value.run_id))
    expected_ids = {mapping.run_id for mapping in ordered}
    indexed = _indexed_records(ledger.get("records"), expected_ids, "generation")
    targets: dict[int, dict[str, Any]] = {}
    results: dict[int, dict[str, Any]] = {}
    total_attempts = 0
    for mapping in ordered:
        record = indexed[mapping.run_id]
        if set(record) != _GENERATION_RECORD_FIELDS:
            raise ValueError(f"Generation ledger fields mismatch for run {mapping.run_id}")
        identity = attempt_input_identity(mapping)
        for field, expected in identity.items():
            if record.get(field) != expected:
                raise ValueError(f"Generation identity mismatch for run {mapping.run_id}: {field}")
        if record["state"] != "succeeded":
            raise ValueError(f"Generation did not succeed for run {mapping.run_id}")
        run_root = run_directory(generation_root, mapping.run_id)
        result = validate_completed_run(
            mapping=mapping, catalog=catalog, run_root=run_root
        )
        target = read_canonical_json(artifact_paths(run_root)["target"])
        inventory = read_canonical_json(artifact_paths(run_root)["inventory"])
        if record["published_result_hash"] != result["run_result_hash"]:
            raise ValueError(f"Generation result hash mismatch for run {mapping.run_id}")
        if record["scientific_inventory_hash"] != inventory["scientific_inventory_hash"]:
            raise ValueError(f"Generation inventory hash mismatch for run {mapping.run_id}")
        if record["target_artifact_hash"] != target["target_artifact_hash"]:
            raise ValueError(f"Generation target hash mismatch for run {mapping.run_id}")
        attempt_count = _validate_generation_attempts(
            root=Path(generation_root), mapping=mapping, result_hash=result["run_result_hash"]
        )
        if record["attempt_count"] != attempt_count:
            raise ValueError(f"Generation attempt count mismatch for run {mapping.run_id}")
        state = read_canonical_json(
            Path(generation_root)
            / "operational"
            / "current_state"
            / f"run_{mapping.run_id:0{RUN_ID_WIDTH}d}.json"
        )
        if (
            state.get("state") != "succeeded"
            or state.get("run_record_hash") != mapping.run["run_record_hash"]
            or state.get("published_result_hash") != result["run_result_hash"]
        ):
            raise ValueError(f"Generation state mismatch for run {mapping.run_id}")
        total_attempts += attempt_count
        targets[mapping.run_id] = target
        results[mapping.run_id] = result
    derived_submissions = len(indexed)
    derived_successes = sum(record["state"] == "succeeded" for record in indexed.values())
    expected_aggregates = {
        "distinct_frozen_run_submission_count": derived_submissions,
        "operational_attempt_event_count": total_attempts,
        "successful_generation_count": derived_successes,
    }
    for field, expected in expected_aggregates.items():
        if ledger.get(field) != expected:
            raise ValueError(f"Generation aggregate disagrees with records: {field}")
    expected_contract_hashes = {mapping.run["contract_spec_hash"] for mapping in ordered}
    if len(expected_contract_hashes) != 1 or ledger.get("contract_spec_hash") != next(iter(expected_contract_hashes)):
        raise ValueError("Generation ledger contract hash mismatch")
    return targets, results


def validate_replay_evidence(
    *,
    mappings: Sequence[FinalRunMapping],
    replay_root: str | Path,
    generation_results: Mapping[int, Mapping[str, Any]],
    generation_targets: Mapping[int, Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    ordered = tuple(sorted(mappings, key=lambda value: value.run_id))
    expected_ids = {mapping.run_id for mapping in ordered}
    indexed = _indexed_records(ledger.get("records"), expected_ids, "replay")
    for mapping in ordered:
        record = indexed[mapping.run_id]
        if set(record) != _REPLAY_RECORD_FIELDS:
            raise ValueError(f"Replay ledger fields mismatch for run {mapping.run_id}")
        persisted = read_canonical_json(
            run_directory(replay_root, mapping.run_id) / "replay_report.json"
        )
        if persisted != record:
            raise ValueError(f"Replay ledger/report mismatch for run {mapping.run_id}")
        result_hash = generation_results[mapping.run_id]["run_result_hash"]
        target_hash = generation_targets[mapping.run_id]["target_artifact_hash"]
        stages = record["per_stage_comparison"]
        if (
            record["contract_spec_hash"] != mapping.run["contract_spec_hash"]
            or record["design_record_hash"] != mapping.design["design_record_hash"]
            or record["run_id"] != mapping.run_id
            or record["run_key"] != mapping.run_key
            or record["run_record_hash"] != mapping.run["run_record_hash"]
            or record["replay_state"] != "succeeded"
            or record["first_mismatch"] is not None
            or record["input_result_hash"] != result_hash
            or record["expected_result_hash"] != result_hash
            or record["recomputed_result_hash"] != result_hash
            or record["target_artifact_hash"] != target_hash
            or record["scientific_inventory_hash"]
            != generation_results[mapping.run_id]["scientific_inventory_hash"]
            or not record["input_tree_unchanged"]
            or record["before_input_tree_inventory_hash"]
            != record["after_input_tree_inventory_hash"]
            or [value.get("stage") for value in stages] != list(_REQUIRED_STAGES)
            or any(value.get("state") != "matched" for value in stages)
        ):
            raise ValueError(f"Replay evidence mismatch for run {mapping.run_id}")
    derived_submissions = len(indexed)
    derived_successes = sum(record["replay_state"] == "succeeded" for record in indexed.values())
    if ledger.get("replay_submission_count") != derived_submissions:
        raise ValueError("Replay aggregate disagrees with records: replay_submission_count")
    if ledger.get("successful_replay_count") != derived_successes:
        raise ValueError("Replay aggregate disagrees with records: successful_replay_count")
    expected_contract_hashes = {mapping.run["contract_spec_hash"] for mapping in ordered}
    if len(expected_contract_hashes) != 1 or ledger.get("contract_spec_hash") != next(iter(expected_contract_hashes)):
        raise ValueError("Replay ledger contract hash mismatch")
    return indexed
