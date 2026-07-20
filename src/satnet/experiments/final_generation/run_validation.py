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
from .constants import RUN_FILES
from .io import read_canonical_json
from .mapping import FinalRunMapping


def _single(values: Sequence[Any], name: str) -> Any:
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {name}, found {len(values)}")
    return values[0]


def artifact_paths(run_root: str | Path) -> dict[str, Path]:
    root = Path(run_root)
    return {name: root / relative for name, relative in RUN_FILES.items()}


def validate_run_authoritatively(
    *, mapping: FinalRunMapping, catalog: GroundStationCatalog, run_root: str | Path
) -> dict[str, Any]:
    root = Path(run_root)
    paths = artifact_paths(root)
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Completed run is missing artifact: {path.name}")

    stages: list[dict[str, str]] = []
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
    expected_target = make_target_artifact(
        design=mapping.design, run=mapping.run, g5_summary=verified_g5_run.summary
    )
    if target != expected_target:
        raise ValueError("Target artifact replay mismatch")
    stages.append({"stage": "target", "state": "matched"})

    inventory = read_canonical_json(paths["inventory"])
    validate_scientific_inventory(root, inventory)
    if inventory != make_scientific_inventory(root):
        raise ValueError("Scientific inventory replay mismatch")
    stages.append({"stage": "inventory", "state": "matched"})

    result = read_canonical_json(paths["result"])
    validate_run_result(result)
    expected_result = make_run_result(
        design=mapping.design, run=mapping.run, inventory=inventory, target=target
    )
    if result != expected_result:
        raise ValueError("Final run-result replay mismatch")
    stages.append({"stage": "result", "state": "matched"})
    return {
        "inventory": inventory,
        "result": result,
        "stages": stages,
        "target": target,
    }
