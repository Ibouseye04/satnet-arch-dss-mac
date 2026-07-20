from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from satnet.ground.catalog import GroundStationCatalog
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
from satnet.ground.selection import select_ground_stations
from satnet.ground.service_persistence import (
    read_ground_service_run_manifest,
    read_ground_service_step_manifest,
    replay_ground_service_records,
)
from satnet.ground.visibility_persistence import (
    read_ground_visibility_manifest,
    replay_ground_visibility_records,
)
from satnet.simulation.tier1_rollout import run_tier1_rollout

from .artifacts import (
    make_run_result,
    make_satellite_artifact,
    make_scientific_inventory,
    make_target_artifact,
    read_satellite_artifact,
    validate_run_result,
    validate_scientific_inventory,
    validate_target_artifact,
)
from .contract import ensure_mode_root, validate_output_root
from .io import atomic_write_json, read_canonical_json, tree_inventory
from .mapping import FinalRunMapping
from .orchestrator import artifact_paths, run_directory


def _single(values: Sequence[Any], name: str) -> Any:
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {name}, found {len(values)}")
    return values[0]


def replay_run_read_only(
    *,
    mapping: FinalRunMapping,
    catalog: GroundStationCatalog,
    input_root: str | Path,
    replay_output_root: str | Path,
) -> dict[str, Any]:
    source_root = ensure_mode_root(input_root, "qualification", create=False)
    output_root = validate_output_root(replay_output_root, other_roots=(source_root,))
    output_root = ensure_mode_root(output_root, "qualification_replay", create=True)
    source_run = run_directory(source_root, mapping.run_id)
    if not source_run.is_dir():
        raise FileNotFoundError(f"Qualification run does not exist: run_{mapping.run_id:03d}")
    report_path = run_directory(output_root, mapping.run_id) / "replay_report.json"
    if report_path.exists():
        raise FileExistsError(f"Replay report already exists for run {mapping.run_id}")
    before = tree_inventory(source_run)
    stages: list[dict[str, str]] = []
    first_mismatch: str | None = None
    state = "failed"
    expected_result_hash: str | None = None
    recomputed_result_hash: str | None = None
    error: Exception | None = None
    try:
        paths = artifact_paths(source_run)
        persisted_satellite, config, _, _, persisted_failures = read_satellite_artifact(
            paths["satellite"]
        )
        if config != mapping.satellite_config:
            raise ValueError("Persisted satellite configuration differs from frozen mapping")
        generated_steps, generated_summary, generated_failures = run_tier1_rollout(config)
        expected_satellite = make_satellite_artifact(
            design=mapping.design,
            run=mapping.run,
            config=config,
            steps=generated_steps,
            summary=generated_summary,
            failure_realization=generated_failures,
        )
        if persisted_satellite != expected_satellite or persisted_failures != generated_failures:
            raise ValueError("Satellite artifact replay mismatch")
        stages.append({"stage": "satellite", "state": "matched"})

        ground_design = _single(read_ground_design_manifest(paths["g1"]), "G1 record")
        validate_manifest_against_satellite_runs(
            (ground_design,), {mapping.run_id: config.config_hash()}
        )
        selection = select_ground_stations(catalog=catalog, config=mapping.ground_config)
        expected_ground = make_enabled_ground_design_record(
            run_id=mapping.run_id,
            satellite_config_hash=config.config_hash(),
            selection=selection,
        )
        if ground_design != expected_ground:
            raise ValueError("G1 replay mismatch")
        reconstruct_ground_selection(ground_design, catalog)
        stages.append({"stage": "g1", "state": "matched"})

        positions = reconstruct_operational_satellite_position_sequence(
            satellite_config=config, failure_realization=generated_failures
        )
        visibility_records = read_ground_visibility_manifest(paths["g2"])
        replay_ground_visibility_records(
            records=visibility_records,
            run_id=mapping.run_id,
            operational_satellite_snapshots=positions,
            ground_design=ground_design,
            catalog=catalog,
            policy=mapping.visibility_policy,
        )
        stages.append({"stage": "g2", "state": "matched"})

        integrated_records = read_integrated_graph_manifest(paths["g3"])
        replay_integrated_graph_records(
            records=integrated_records,
            satellite_config=config,
            failure_realization=generated_failures,
            ground_design=ground_design,
            catalog=catalog,
            visibility_policy=mapping.visibility_policy,
            visibility_records=visibility_records,
        )
        stages.append({"stage": "g3", "state": "matched"})

        g4_steps = read_ground_service_step_manifest(paths["g4_steps"])
        g4_run = _single(read_ground_service_run_manifest(paths["g4_run"]), "G4 run record")
        verified_g4_steps, verified_g4_run = replay_ground_service_records(
            satellite_config=config,
            failure_realization=generated_failures,
            ground_design=ground_design,
            catalog=catalog,
            visibility_policy=mapping.visibility_policy,
            visibility_records=visibility_records,
            integrated_records=integrated_records,
            service_policy=mapping.service_policy,
            step_records=g4_steps,
            run_records=(g4_run,),
        )
        stages.append({"stage": "g4", "state": "matched"})

        g5_realizations = read_ground_failure_realization_manifest(paths["g5_realization"])
        g5_steps = read_ground_failure_service_step_manifest(paths["g5_steps"])
        g5_runs = read_ground_failure_service_run_manifest(paths["g5_run"])
        _, _, verified_g5_run = replay_ground_failure_service_records(
            satellite_config=config,
            satellite_failure_realization=generated_failures,
            ground_design=ground_design,
            catalog=catalog,
            visibility_policy=mapping.visibility_policy,
            visibility_records=visibility_records,
            integrated_records=integrated_records,
            ground_service_policy=mapping.service_policy,
            g4_step_records=verified_g4_steps,
            g4_run_record=verified_g4_run,
            ground_failure_policy=mapping.failure_policy,
            realization_records=g5_realizations,
            g5_step_records=g5_steps,
            g5_run_records=g5_runs,
        )
        stages.append({"stage": "g5", "state": "matched"})

        target = read_canonical_json(paths["target"])
        validate_target_artifact(target)
        if target != make_target_artifact(
            design=mapping.design, run=mapping.run, g5_summary=verified_g5_run.summary
        ):
            raise ValueError("Target artifact replay mismatch")
        stages.append({"stage": "target", "state": "matched"})

        inventory = read_canonical_json(paths["inventory"])
        validate_scientific_inventory(source_run, inventory)
        if inventory != make_scientific_inventory(source_run):
            raise ValueError("Scientific inventory replay mismatch")
        stages.append({"stage": "inventory", "state": "matched"})

        persisted_result = read_canonical_json(paths["result"])
        validate_run_result(persisted_result)
        recomputed_result = make_run_result(
            design=mapping.design, run=mapping.run, inventory=inventory, target=target
        )
        expected_result_hash = persisted_result["run_result_hash"]
        recomputed_result_hash = recomputed_result["run_result_hash"]
        if persisted_result != recomputed_result:
            raise ValueError("Final run-result replay mismatch")
        stages.append({"stage": "result", "state": "matched"})
        state = "succeeded"
    except Exception as exc:
        error = exc
        first_mismatch = str(exc)
    after = tree_inventory(source_run)
    input_unchanged = before == after
    if not input_unchanged:
        state = "failed"
        first_mismatch = first_mismatch or "Input run tree changed during replay"
    report: dict[str, Any] = {
        "expected_result_hash": expected_result_hash,
        "first_mismatch": first_mismatch,
        "input_tree_unchanged": input_unchanged,
        "per_stage_comparison": stages,
        "recomputed_result_hash": recomputed_result_hash,
        "replay_state": state,
        "run_id": mapping.run_id,
        "run_key": mapping.run_key,
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
) -> tuple[dict[str, Any], ...]:
    return tuple(
        replay_run_read_only(
            mapping=mapping,
            catalog=catalog,
            input_root=input_root,
            replay_output_root=replay_output_root,
        )
        for mapping in sorted(mappings, key=lambda value: value.run_id)
    )
