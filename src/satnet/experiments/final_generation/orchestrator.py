from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from satnet.ground.canonical import canonical_hash
from satnet.ground.catalog import GroundStationCatalog
from satnet.ground.failure_service_persistence import (
    generate_verified_ground_failure_service_records,
    write_ground_failure_realization_manifest,
    write_ground_failure_service_run_manifest,
    write_ground_failure_service_step_manifest,
)
from satnet.ground.integrated_builder import build_integrated_ground_graph
from satnet.ground.integrated_persistence import (
    make_integrated_graph_record,
    write_integrated_graph_manifest,
)
from satnet.ground.persistence import (
    make_enabled_ground_design_record,
    write_ground_design_manifest,
)
from satnet.ground.position_adapter import reconstruct_operational_satellite_position_sequence
from satnet.ground.satellite_graph_adapter import reconstruct_operational_satellite_graph_sequence
from satnet.ground.selection import select_ground_stations
from satnet.ground.service_persistence import (
    generate_verified_ground_service_records,
    write_ground_service_run_manifest,
    write_ground_service_step_manifest,
)
from satnet.ground.visibility import evaluate_ground_design_visibility_sequence
from satnet.ground.visibility_persistence import (
    make_ground_visibility_record,
    write_ground_visibility_manifest,
)
from satnet.simulation.tier1_rollout import run_tier1_rollout

from .artifacts import (
    make_run_result,
    make_satellite_artifact,
    make_scientific_inventory,
    make_target_artifact,
    validate_run_result,
    validate_scientific_inventory,
    validate_target_artifact,
    write_satellite_artifact,
)
from .constants import CONTRACT_SPEC_HASH, RUN_FILES
from .contract import ensure_mode_root
from .io import atomic_write_json, file_identity, read_canonical_json
from .mapping import FinalRunMapping


def run_directory(output_root: str | Path, run_id: int) -> Path:
    if type(run_id) is not int or not 0 <= run_id <= 499:
        raise ValueError("run_id must be an integer in [0, 499]")
    return Path(output_root) / f"run_{run_id:03d}"


def artifact_paths(run_root: str | Path) -> dict[str, Path]:
    root = Path(run_root)
    return {name: root / relative for name, relative in RUN_FILES.items()}


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _attempt_number(output_root: Path, run_id: int) -> int:
    root = output_root / "operational" / "attempts" / f"run_{run_id:03d}"
    return len(tuple(root.glob("attempt_*.json"))) + 1 if root.exists() else 1


def _attempt_path(output_root: Path, run_id: int, number: int) -> Path:
    return (
        output_root
        / "operational"
        / "attempts"
        / f"run_{run_id:03d}"
        / f"attempt_{number:03d}.json"
    )


def _current_state_path(output_root: Path, run_id: int) -> Path:
    return output_root / "operational" / "current_state" / f"run_{run_id:03d}.json"


def _completed_stage_inventory(run_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for key, relative in RUN_FILES.items():
        path = run_root / relative
        if path.is_file():
            length, digest = file_identity(path)
            records.append({"artifact_key": key, "byte_length": length, "sha256": digest})
    return records


def _sanitize_message(error: Exception) -> str:
    message = " ".join(str(error).split())[:1000]
    return message or type(error).__name__


def validate_completed_run(
    *, mapping: FinalRunMapping, run_root: str | Path
) -> dict[str, Any]:
    root = Path(run_root)
    paths = artifact_paths(root)
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Completed run is missing artifact: {path.name}")
    target = read_canonical_json(paths["target"])
    validate_target_artifact(target)
    inventory = read_canonical_json(paths["inventory"])
    validate_scientific_inventory(root, inventory)
    result = read_canonical_json(paths["result"])
    validate_run_result(result)
    expected = make_run_result(
        design=mapping.design,
        run=mapping.run,
        inventory=inventory,
        target=target,
    )
    if result != expected:
        raise ValueError("Completed run result does not match frozen identities")
    return result


def generate_run(
    *,
    mapping: FinalRunMapping,
    catalog: GroundStationCatalog,
    output_root: str | Path,
    mode: str,
    retry: bool = False,
    verified_resume: bool = False,
) -> dict[str, Any]:
    root = ensure_mode_root(output_root, mode, create=True)
    final_root = run_directory(root, mapping.run_id)
    if final_root.exists():
        if not verified_resume:
            raise FileExistsError(f"Completed run already exists: run_{mapping.run_id:03d}")
        return validate_completed_run(mapping=mapping, run_root=final_root)
    attempt_number = _attempt_number(root, mapping.run_id)
    if attempt_number > 1 and not retry:
        raise ValueError("A retry requires explicit retry=True")
    temporary = Path(tempfile.mkdtemp(dir=root, prefix=f".run_{mapping.run_id:03d}.in_progress."))
    completed: list[str] = []
    stage = "input"
    attempt_id = f"run_{mapping.run_id:03d}-attempt_{attempt_number:03d}"
    try:
        atomic_write_json(temporary / "input" / "design_record.json", mapping.design)
        atomic_write_json(temporary / "input" / "run_record.json", mapping.run)
        stage = "satellite"
        satellite_steps, satellite_summary, satellite_failures = run_tier1_rollout(
            mapping.satellite_config
        )
        satellite_artifact = make_satellite_artifact(
            design=mapping.design,
            run=mapping.run,
            config=mapping.satellite_config,
            steps=satellite_steps,
            summary=satellite_summary,
            failure_realization=satellite_failures,
        )
        paths = artifact_paths(temporary)
        write_satellite_artifact(paths["satellite"], satellite_artifact)
        completed.append(stage)

        stage = "g1"
        selection = select_ground_stations(catalog=catalog, config=mapping.ground_config)
        if tuple(mapping.design["selected_station_ids"]) != selection.selected_station_ids:
            raise ValueError("G1 selected station IDs differ from frozen design")
        if mapping.design["ground_selection_hash"] != selection.selection_hash:
            raise ValueError("G1 selection hash differs from frozen design")
        ground_design = make_enabled_ground_design_record(
            run_id=mapping.run_id,
            satellite_config_hash=mapping.satellite_config.config_hash(),
            selection=selection,
        )
        if ground_design.ground_design_hash != mapping.design["ground_design_hash"]:
            raise ValueError("G1 ground-design hash differs from frozen design")
        write_ground_design_manifest((ground_design,), paths["g1"])
        completed.append(stage)

        stage = "g2"
        positions = reconstruct_operational_satellite_position_sequence(
            satellite_config=mapping.satellite_config,
            failure_realization=satellite_failures,
        )
        visibility_snapshots = evaluate_ground_design_visibility_sequence(
            ground_design=ground_design,
            catalog=catalog,
            satellite_sequence=positions,
            policy=mapping.visibility_policy,
        )
        visibility_records = tuple(
            make_ground_visibility_record(run_id=mapping.run_id, snapshot=snapshot)
            for snapshot in visibility_snapshots
        )
        write_ground_visibility_manifest(visibility_records, paths["g2"])
        completed.append(stage)

        stage = "g3"
        satellite_graphs = reconstruct_operational_satellite_graph_sequence(
            satellite_config=mapping.satellite_config,
            failure_realization=satellite_failures,
        )
        integrated_records = tuple(
            make_integrated_graph_record(
                ground_design=ground_design,
                snapshot=build_integrated_ground_graph(
                    ground_design=ground_design,
                    catalog=catalog,
                    satellite_graph_snapshot=graph,
                    verified_visibility_snapshot=visibility,
                ),
            )
            for graph, visibility in zip(satellite_graphs, visibility_snapshots, strict=True)
        )
        write_integrated_graph_manifest(integrated_records, paths["g3"])
        completed.append(stage)

        stage = "g4"
        g4_steps, g4_run = generate_verified_ground_service_records(
            satellite_config=mapping.satellite_config,
            failure_realization=satellite_failures,
            ground_design=ground_design,
            catalog=catalog,
            visibility_policy=mapping.visibility_policy,
            visibility_records=visibility_records,
            integrated_records=integrated_records,
            service_policy=mapping.service_policy,
        )
        write_ground_service_step_manifest(g4_steps, paths["g4_steps"])
        write_ground_service_run_manifest((g4_run,), paths["g4_run"])
        completed.append(stage)

        stage = "g5"
        g5_realization, g5_steps, g5_run = generate_verified_ground_failure_service_records(
            satellite_config=mapping.satellite_config,
            satellite_failure_realization=satellite_failures,
            ground_design=ground_design,
            catalog=catalog,
            visibility_policy=mapping.visibility_policy,
            visibility_records=visibility_records,
            integrated_records=integrated_records,
            ground_service_policy=mapping.service_policy,
            g4_step_records=g4_steps,
            g4_run_record=g4_run,
            ground_failure_policy=mapping.failure_policy,
            ground_failure_seed=mapping.run["ground_failure_seed"],
        )
        write_ground_failure_realization_manifest((g5_realization,), paths["g5_realization"])
        write_ground_failure_service_step_manifest(g5_steps, paths["g5_steps"])
        write_ground_failure_service_run_manifest((g5_run,), paths["g5_run"])
        completed.append(stage)

        stage = "target"
        target = make_target_artifact(
            design=mapping.design, run=mapping.run, g5_summary=g5_run.summary
        )
        validate_target_artifact(target)
        atomic_write_json(paths["target"], target)
        completed.append(stage)

        stage = "inventory"
        inventory = make_scientific_inventory(temporary)
        atomic_write_json(paths["inventory"], inventory)
        completed.append(stage)

        stage = "result"
        result = make_run_result(
            design=mapping.design, run=mapping.run, inventory=inventory, target=target
        )
        validate_run_result(result)
        atomic_write_json(paths["result"], result)
        completed.append(stage)
        attempt = {
            "attempt_id": attempt_id,
            "completed_stages": completed,
            "contract_spec_hash": CONTRACT_SPEC_HASH,
            "run_id": mapping.run_id,
            "run_key": mapping.run_key,
            "run_record_hash": mapping.run["run_record_hash"],
            "state": "succeeded",
            "timestamp_utc": _timestamp(),
        }
        atomic_write_json(temporary / "operational" / "attempt.json", attempt)
        os.replace(temporary, final_root)
        atomic_write_json(_attempt_path(root, mapping.run_id, attempt_number), attempt)
        atomic_write_json(
            _current_state_path(root, mapping.run_id),
            {
                "attempt_count": attempt_number,
                "published_result_hash": result["run_result_hash"],
                "run_id": mapping.run_id,
                "run_record_hash": mapping.run["run_record_hash"],
                "state": "succeeded",
            },
            overwrite=True,
        )
        return result
    except Exception as error:
        failure = {
            "attempt_id": attempt_id,
            "completed_stage_inventory": _completed_stage_inventory(temporary),
            "completed_stages": completed,
            "contract_spec_hash": CONTRACT_SPEC_HASH,
            "exception_class": type(error).__name__,
            "failed_stage": stage,
            "failed_stage_identity": f"{stage}:{mapping.run['run_record_hash']}",
            "run_id": mapping.run_id,
            "run_key": mapping.run_key,
            "run_record_hash": mapping.run["run_record_hash"],
            "sanitized_exception_message": _sanitize_message(error),
            "state": "failed",
            "timestamp_utc": _timestamp(),
        }
        failure["failure_evidence_hash"] = canonical_hash(failure)
        atomic_write_json(_attempt_path(root, mapping.run_id, attempt_number), failure)
        atomic_write_json(
            _current_state_path(root, mapping.run_id),
            {
                "attempt_count": attempt_number,
                "failure_evidence_hash": failure["failure_evidence_hash"],
                "run_id": mapping.run_id,
                "run_record_hash": mapping.run["run_record_hash"],
                "state": "failed",
            },
            overwrite=True,
        )
        failed_root = root / "operational" / "failed_attempts" / attempt_id
        failed_root.parent.mkdir(parents=True, exist_ok=True)
        if failed_root.exists():
            raise RuntimeError("Immutable failed-attempt evidence already exists") from error
        os.replace(temporary, failed_root)
        raise


def generate_runs(
    *,
    mappings: Sequence[FinalRunMapping],
    catalog: GroundStationCatalog,
    output_root: str | Path,
    mode: str,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        generate_run(mapping=value, catalog=catalog, output_root=output_root, mode=mode)
        for value in sorted(mappings, key=lambda item: item.run_id)
    )
