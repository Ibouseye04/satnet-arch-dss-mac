from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from satnet.ground.catalog import GroundStationCatalog

from .contract import ensure_mode_root, validate_output_root
from .io import atomic_write_json, tree_inventory, tree_inventory_hash
from .mapping import FinalRunMapping
from .orchestrator import run_directory
from .run_validation import validate_run_authoritatively


def replay_run_read_only(
    *,
    mapping: FinalRunMapping,
    catalog: GroundStationCatalog,
    input_root: str | Path,
    replay_output_root: str | Path,
    input_mode: str = "qualification",
    output_mode: str = "qualification_replay",
) -> dict[str, Any]:
    valid_pair = (input_mode, output_mode) in {
        ("qualification", "qualification_replay"),
        ("production", "production_replay"),
    }
    if not valid_pair:
        raise ValueError("Unsupported replay mode pair")
    contract_spec_hash = mapping.run["contract_spec_hash"]
    source_root = ensure_mode_root(
        input_root,
        input_mode,
        create=False,
        contract_spec_hash=contract_spec_hash,
    )
    output_root = validate_output_root(replay_output_root, other_roots=(source_root,))
    output_root = ensure_mode_root(
        output_root,
        output_mode,
        create=True,
        contract_spec_hash=contract_spec_hash,
    )
    source_run = run_directory(source_root, mapping.run_id)
    if not source_run.is_dir():
        raise FileNotFoundError(f"Source run does not exist: run_{mapping.run_id:04d}")
    report_path = run_directory(output_root, mapping.run_id) / "replay_report.json"
    if report_path.exists():
        raise FileExistsError(f"Replay report already exists for run {mapping.run_id}")
    before = tree_inventory(source_run)
    before_hash = tree_inventory_hash(source_run)
    first_mismatch: str | None = None
    state = "failed"
    validated: dict[str, Any] | None = None
    error: Exception | None = None
    try:
        validated = validate_run_authoritatively(
            mapping=mapping, catalog=catalog, run_root=source_run
        )
        state = "succeeded"
    except Exception as exc:
        error = exc
        first_mismatch = str(exc)
    after = tree_inventory(source_run)
    after_hash = tree_inventory_hash(source_run)
    input_unchanged = before == after and before_hash == after_hash
    if not input_unchanged:
        state = "failed"
        first_mismatch = first_mismatch or "Input run tree changed during replay"
    result_hash = None if validated is None else validated["result"]["run_result_hash"]
    report: dict[str, Any] = {
        "after_input_tree_inventory_hash": after_hash,
        "before_input_tree_inventory_hash": before_hash,
        "contract_spec_hash": mapping.run["contract_spec_hash"],
        "design_record_hash": mapping.design["design_record_hash"],
        "expected_result_hash": result_hash,
        "first_mismatch": first_mismatch,
        "input_result_hash": result_hash,
        "input_tree_unchanged": input_unchanged,
        "per_stage_comparison": [] if validated is None else validated["stages"],
        "recomputed_result_hash": result_hash,
        "replay_report_schema_version": "2",
        "replay_state": state,
        "run_id": mapping.run_id,
        "run_key": mapping.run_key,
        "run_record_hash": mapping.run["run_record_hash"],
        "scientific_inventory_hash": None
        if validated is None
        else validated["inventory"]["scientific_inventory_hash"],
        "target_artifact_hash": None
        if validated is None
        else validated["target"]["target_artifact_hash"],
    }
    atomic_write_json(report_path, report)
    if state != "succeeded":
        if error is not None:
            raise ValueError(f"Read-only replay failed: {first_mismatch}") from error
        raise ValueError(first_mismatch or "Read-only replay failed")
    return report


def replay_runs_read_only(
    *,
    mappings: Sequence[FinalRunMapping],
    catalog: GroundStationCatalog,
    input_root: str | Path,
    replay_output_root: str | Path,
    input_mode: str = "qualification",
    output_mode: str = "qualification_replay",
) -> tuple[dict[str, Any], ...]:
    return tuple(
        replay_run_read_only(
            mapping=mapping,
            catalog=catalog,
            input_root=input_root,
            replay_output_root=replay_output_root,
            input_mode=input_mode,
            output_mode=output_mode,
        )
        for mapping in sorted(mappings, key=lambda value: value.run_id)
    )
