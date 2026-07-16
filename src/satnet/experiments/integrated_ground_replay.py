from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from satnet.experiments.integrated_ground_manifest import (
    IntegratedPilotDesign,
    IntegratedPilotRun,
    read_pilot_design_manifest,
    read_pilot_run_manifest,
)
from satnet.experiments.integrated_ground_runner import (
    CANONICAL_ARTIFACT_KEYS,
    PILOT_RUNTIME_SCHEMA_VERSION,
    RUN_ARTIFACT_NAMES,
    _artifact_inventory,
    _atomic_write,
    _read_single_json,
    artifact_paths,
    extract_run_summary,
    make_pilot_satellite_artifact,
    pilot_json,
    read_pilot_satellite_artifact,
    run_directory,
    select_manifest_runs,
)
from satnet.ground.catalog import GroundStationCatalog, load_ground_station_catalog
from satnet.ground.failure_policy import GroundFailurePolicy
from satnet.ground.failure_service_persistence import (
    read_ground_failure_realization_manifest,
    read_ground_failure_service_run_manifest,
    read_ground_failure_service_step_manifest,
    replay_ground_failure_service_records,
)
from satnet.ground.integrated_persistence import (
    read_integrated_graph_manifest,
    replay_integrated_graph_records,
)
from satnet.ground.persistence import (
    make_enabled_ground_design_record,
    read_ground_design_manifest,
    reconstruct_ground_selection,
    validate_manifest_against_satellite_runs,
)
from satnet.ground.position_adapter import reconstruct_operational_satellite_position_sequence
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from satnet.ground.service_persistence import (
    read_ground_service_run_manifest,
    read_ground_service_step_manifest,
    replay_ground_service_records,
)
from satnet.ground.service_policy import GroundServicePolicy
from satnet.ground.visibility import GroundVisibilityPolicy
from satnet.ground.visibility_persistence import (
    read_ground_visibility_manifest,
    replay_ground_visibility_records,
)
from satnet.simulation.tier1_rollout import run_tier1_rollout

PILOT_REPLAY_RESULT_SCHEMA_VERSION = "1"
ALLOWED_RUN_FILENAMES = frozenset(
    {
        *RUN_ARTIFACT_NAMES.values(),
        "replay_result.json",
    }
)


def _require_single(values: Sequence[Any], description: str):
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {description}, found {len(values)}")
    return values[0]


def _read_runtime(path: Path, run_id: int) -> dict[str, object]:
    value = _read_single_json(path)
    expected = {
        "g1_seconds",
        "g2_seconds",
        "g3_seconds",
        "g4_seconds",
        "g5_seconds",
        "persistence_seconds",
        "pilot_runtime_schema_version",
        "run_id",
        "satellite_rollout_seconds",
        "total_generation_seconds",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Generation runtime fields are invalid")
    if value["pilot_runtime_schema_version"] != PILOT_RUNTIME_SCHEMA_VERSION:
        raise ValueError("Unsupported generation runtime schema")
    if value["run_id"] != run_id:
        raise ValueError("Generation runtime run ID mismatch")
    for name in expected - {"pilot_runtime_schema_version", "run_id"}:
        elapsed = value[name]
        if type(elapsed) is not float or not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError(f"Invalid generation runtime field: {name}")
    return value


def _read_summary(path: Path) -> dict[str, object]:
    value = _read_single_json(path)
    if not isinstance(value, dict):
        raise TypeError("Run summary must be an object")
    return value


def _validate_run_file_set(root: Path, *, allow_replay_result: bool) -> None:
    if not root.is_dir():
        raise ValueError(f"Run directory does not exist: {root}")
    names = {path.name for path in root.iterdir() if path.is_file()}
    required = set(RUN_ARTIFACT_NAMES.values())
    missing = sorted(required - names)
    allowed = set(ALLOWED_RUN_FILENAMES)
    if not allow_replay_result:
        allowed.remove("replay_result.json")
    extra = sorted(names - allowed)
    if missing or extra:
        raise ValueError(f"Run artifact set mismatch; missing={missing}, extra={extra}")


def _expected_ground_design(
    *,
    design: IntegratedPilotDesign,
    run: IntegratedPilotRun,
    catalog: GroundStationCatalog,
    satellite_config_hash: str,
):
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(
            civilian_count=design.civilian_count,
            government_count=design.government_count,
            military_count=design.military_count,
            station_selection_seed=run.ground_station_selection_seed,
        ),
    )
    return make_enabled_ground_design_record(
        run_id=run.run_id,
        satellite_config_hash=satellite_config_hash,
        selection=selection,
    )


def validate_existing_run(
    *,
    design: IntegratedPilotDesign,
    run: IntegratedPilotRun,
    catalog: GroundStationCatalog,
    run_root: str | Path,
    allow_replay_result: bool = True,
) -> dict[str, object]:
    root = Path(run_root)
    _validate_run_file_set(root, allow_replay_result=allow_replay_result)
    paths = artifact_paths(root)
    satellite = read_pilot_satellite_artifact(paths["satellite"])
    expected_config = design.satellite_config(
        satellite_seed=run.satellite_rollout_seed
    )
    if satellite.run_id != run.run_id or satellite.config != expected_config:
        raise ValueError("Persisted satellite artifact does not match run manifest")
    generated_steps, generated_summary, generated_failures = run_tier1_rollout(
        expected_config
    )
    expected_satellite = make_pilot_satellite_artifact(
        run_id=run.run_id,
        config=expected_config,
        steps=generated_steps,
        summary=generated_summary,
        failure_realization=generated_failures,
    )
    if satellite != expected_satellite:
        raise ValueError("Satellite rollout replay mismatch")

    ground_design = _require_single(
        read_ground_design_manifest(paths["g1"]), "G1 ground design"
    )
    validate_manifest_against_satellite_runs(
        (ground_design,), {run.run_id: expected_config.config_hash()}
    )
    expected_design = _expected_ground_design(
        design=design,
        run=run,
        catalog=catalog,
        satellite_config_hash=expected_config.config_hash(),
    )
    if ground_design != expected_design:
        raise ValueError("G1 ground-design replay mismatch")
    reconstruct_ground_selection(ground_design, catalog)

    visibility_policy = GroundVisibilityPolicy(design.minimum_elevation_deg)
    visibility_records = read_ground_visibility_manifest(paths["g2"])
    position_sources = reconstruct_operational_satellite_position_sequence(
        satellite_config=expected_config,
        failure_realization=generated_failures,
    )
    replay_ground_visibility_records(
        records=visibility_records,
        run_id=run.run_id,
        operational_satellite_snapshots=position_sources,
        ground_design=ground_design,
        catalog=catalog,
        policy=visibility_policy,
    )

    integrated_records = read_integrated_graph_manifest(paths["g3"])
    replay_integrated_graph_records(
        records=integrated_records,
        satellite_config=expected_config,
        failure_realization=generated_failures,
        ground_design=ground_design,
        catalog=catalog,
        visibility_policy=visibility_policy,
        visibility_records=visibility_records,
    )

    service_policy = GroundServicePolicy(
        design.space_gcc_threshold,
        design.ground_service_threshold,
    )
    g4_steps = read_ground_service_step_manifest(paths["g4_steps"])
    g4_run = _require_single(
        read_ground_service_run_manifest(paths["g4_run"]), "G4 run record"
    )
    verified_g4_steps, verified_g4_run = replay_ground_service_records(
        satellite_config=expected_config,
        failure_realization=generated_failures,
        ground_design=ground_design,
        catalog=catalog,
        visibility_policy=visibility_policy,
        visibility_records=visibility_records,
        integrated_records=integrated_records,
        service_policy=service_policy,
        step_records=g4_steps,
        run_records=(g4_run,),
    )

    ground_failure_policy = GroundFailurePolicy(design.ground_failure_probability)
    g5_realizations = read_ground_failure_realization_manifest(paths["g5_realization"])
    g5_steps = read_ground_failure_service_step_manifest(paths["g5_steps"])
    g5_runs = read_ground_failure_service_run_manifest(paths["g5_run"])
    verified_realization, verified_g5_steps, verified_g5_run = (
        replay_ground_failure_service_records(
            satellite_config=expected_config,
            satellite_failure_realization=generated_failures,
            ground_design=ground_design,
            catalog=catalog,
            visibility_policy=visibility_policy,
            visibility_records=visibility_records,
            integrated_records=integrated_records,
            ground_service_policy=service_policy,
            g4_step_records=verified_g4_steps,
            g4_run_record=verified_g4_run,
            ground_failure_policy=ground_failure_policy,
            realization_records=g5_realizations,
            g5_step_records=g5_steps,
            g5_run_records=g5_runs,
        )
    )

    runtime = _read_runtime(paths["runtime"], run.run_id)
    inventory = _artifact_inventory(paths)
    persisted_inventory = _read_single_json(paths["inventory"])
    if persisted_inventory != inventory:
        raise ValueError("Artifact inventory replay mismatch")
    expected_summary = extract_run_summary(
        design=design,
        run=run,
        satellite_artifact=satellite,
        ground_design=ground_design,
        g4_run=verified_g4_run,
        g5_realization=verified_realization,
        g5_steps=verified_g5_steps,
        g5_run=verified_g5_run,
        runtime=runtime,
        artifact_bytes=inventory["canonical_artifact_bytes"],
    )
    persisted_summary = _read_summary(paths["summary"])
    if persisted_summary != expected_summary:
        raise ValueError("Supplemental run summary replay mismatch")
    return {
        "artifact_inventory": inventory,
        "g1_ground_design_hash": ground_design.ground_design_hash,
        "g2_record_count": len(visibility_records),
        "g3_record_count": len(integrated_records),
        "g4_run_summary_hash": verified_g4_run.summary.run_summary_hash,
        "g4_step_count": len(verified_g4_steps),
        "g5_realization_hash": verified_realization.realization.ground_failure_realization_hash,
        "g5_run_summary_hash": verified_g5_run.summary.failure_adjusted_run_summary_hash,
        "g5_step_count": len(verified_g5_steps),
        "run_id": run.run_id,
        "satellite_artifact_hash": satellite.satellite_artifact_hash,
        "summary": expected_summary,
    }


def replay_run(
    *,
    design: IntegratedPilotDesign,
    run: IntegratedPilotRun,
    catalog: GroundStationCatalog,
    run_root: str | Path,
    write_result: bool = True,
) -> dict[str, object]:
    start = perf_counter()
    evidence = validate_existing_run(
        design=design,
        run=run,
        catalog=catalog,
        run_root=run_root,
        allow_replay_result=True,
    )
    elapsed = perf_counter() - start
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise RuntimeError("Replay produced invalid elapsed time")
    result = {
        "g1_ground_design_hash": evidence["g1_ground_design_hash"],
        "g2_record_count": evidence["g2_record_count"],
        "g3_record_count": evidence["g3_record_count"],
        "g4_run_summary_hash": evidence["g4_run_summary_hash"],
        "g4_step_count": evidence["g4_step_count"],
        "g5_realization_hash": evidence["g5_realization_hash"],
        "g5_run_summary_hash": evidence["g5_run_summary_hash"],
        "g5_step_count": evidence["g5_step_count"],
        "pilot_replay_result_schema_version": PILOT_REPLAY_RESULT_SCHEMA_VERSION,
        "replay_runtime_seconds": elapsed,
        "replay_status": "success",
        "run_id": run.run_id,
        "satellite_artifact_hash": evidence["satellite_artifact_hash"],
    }
    if write_result:
        path = Path(run_root) / "replay_result.json"
        if path.exists():
            persisted = _read_single_json(path)
            scientific = dict(result)
            scientific.pop("replay_runtime_seconds")
            persisted_scientific = dict(persisted) if isinstance(persisted, dict) else {}
            persisted_scientific.pop("replay_runtime_seconds", None)
            if persisted_scientific != scientific:
                raise ValueError("Existing replay result does not match verified evidence")
        else:
            _atomic_write(path, pilot_json(result) + "\n")
    return result


def _failure_record(
    *, run: IntegratedPilotRun, error: Exception
) -> dict[str, object]:
    return {
        "design_id": run.design_id,
        "error_message": str(error),
        "error_type": type(error).__name__,
        "realization_id": run.realization_id,
        "replay_status": "failed",
        "run_id": run.run_id,
    }


def replay_pilot_manifest(
    *,
    manifest_path: str | Path,
    output_root: str | Path,
    run_ids: Sequence[int] | None = None,
    repeat: bool = False,
) -> dict[str, object]:
    if type(repeat) is not bool:
        raise TypeError("repeat must be a Boolean")
    manifest = Path(manifest_path)
    input_root = manifest.parent
    designs = read_pilot_design_manifest(input_root / "pilot_designs.json")
    runs = read_pilot_run_manifest(manifest, designs)
    selected = select_manifest_runs(runs, run_ids)
    catalog = load_ground_station_catalog(input_root / "pilot_catalog.csv")
    design_by_id = {design.design_id: design for design in designs}
    successes: list[int] = []
    failures: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    for run in selected:
        try:
            result = replay_run(
                design=design_by_id[run.design_id],
                run=run,
                catalog=catalog,
                run_root=run_directory(output_root, run.run_id, repeat=repeat),
            )
            results.append(result)
            successes.append(run.run_id)
        except Exception as exc:
            failures.append(_failure_record(run=run, error=exc))
    report = {
        "attempted_replay_run_count": len(selected),
        "expected_manifest_run_count": len(runs),
        "failed_replay_run_count": len(failures),
        "failures": failures,
        "repeat": repeat,
        "results": results,
        "successful_replay_run_count": len(successes),
        "successful_run_ids": successes,
    }
    summary_root = Path(output_root) / ("repeats" if repeat else "summaries")
    summary_root.mkdir(parents=True, exist_ok=True)
    path = summary_root / (
        "repeat_replay_summary.json" if repeat else "replay_summary.json"
    )
    if path.exists():
        persisted = _read_single_json(path)
        scientific = dict(report)
        scientific["results"] = [
            {key: value for key, value in result.items() if key != "replay_runtime_seconds"}
            for result in report["results"]
        ]
        if isinstance(persisted, dict):
            persisted_scientific = dict(persisted)
            persisted_scientific["results"] = [
                {
                    key: value
                    for key, value in result.items()
                    if key != "replay_runtime_seconds"
                }
                for result in persisted_scientific.get("results", [])
            ]
        else:
            persisted_scientific = {}
        if persisted_scientific != scientific:
            raise ValueError("Existing replay summary does not match replay evidence")
    else:
        _atomic_write(path, pilot_json(report) + "\n")
    return report


def compare_repeat_run(
    *,
    output_root: str | Path,
    run_id: int,
) -> dict[str, object]:
    primary = run_directory(output_root, run_id)
    repeat = run_directory(output_root, run_id, repeat=True)
    primary_paths = artifact_paths(primary)
    repeat_paths = artifact_paths(repeat)
    mismatches: list[str] = []
    for key in CANONICAL_ARTIFACT_KEYS:
        if primary_paths[key].read_bytes() != repeat_paths[key].read_bytes():
            mismatches.append(key)
    primary_summary = _read_summary(primary_paths["summary"])
    repeat_summary = _read_summary(repeat_paths["summary"])
    excluded = {
        "artifact_bytes",
        "generation_runtime_seconds",
        "replay_runtime_seconds",
        "replay_status",
        "satellite_rollout_seconds",
        "g1_seconds",
        "g2_seconds",
        "g3_seconds",
        "g4_seconds",
        "g5_seconds",
        "persistence_seconds",
        "total_generation_seconds",
    }
    primary_scientific = {
        key: value for key, value in primary_summary.items() if key not in excluded
    }
    repeat_scientific = {
        key: value for key, value in repeat_summary.items() if key not in excluded
    }
    if primary_scientific != repeat_scientific:
        mismatches.append("run_summary_scientific_values")
    result = {
        "canonical_artifact_match": not mismatches,
        "mismatches": mismatches,
        "run_id": run_id,
        "scientific_summary_match": primary_scientific == repeat_scientific,
    }
    if mismatches:
        raise ValueError(f"Repeat-run determinism mismatch for run {run_id}: {mismatches}")
    return result
