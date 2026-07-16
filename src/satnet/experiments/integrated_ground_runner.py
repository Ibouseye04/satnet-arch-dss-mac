from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

from satnet.experiments.integrated_ground_manifest import (
    IntegratedPilotDesign,
    IntegratedPilotRun,
    read_pilot_design_manifest,
    read_pilot_run_manifest,
)
from satnet.ground.canonical import canonical_hash, canonical_json
from satnet.ground.catalog import GroundStationCatalog, load_ground_station_catalog
from satnet.ground.failure_policy import GroundFailurePolicy
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
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from satnet.ground.service_persistence import (
    generate_verified_ground_service_records,
    write_ground_service_run_manifest,
    write_ground_service_step_manifest,
)
from satnet.ground.service_policy import GroundServicePolicy
from satnet.ground.visibility import (
    GroundVisibilityPolicy,
    evaluate_ground_design_visibility_sequence,
)
from satnet.ground.visibility_persistence import (
    make_ground_visibility_record,
    write_ground_visibility_manifest,
)
from satnet.simulation.tier1_rollout import (
    Tier1FailureRealization,
    Tier1RolloutConfig,
    Tier1RolloutStep,
    Tier1RolloutSummary,
    run_tier1_rollout,
)

PILOT_SATELLITE_ARTIFACT_SCHEMA_VERSION = "1"
PILOT_SATELLITE_ARTIFACT_DOMAIN = "satnet_integrated_ground_pilot_satellite_artifact"
PILOT_SATELLITE_ARTIFACT_VERSION = "1"
PILOT_SATELLITE_RECORD_DOMAIN = "satnet_integrated_ground_pilot_satellite_record"
PILOT_SATELLITE_RECORD_VERSION = "1"
PILOT_RUNTIME_SCHEMA_VERSION = "1"
PILOT_RUN_SUMMARY_SCHEMA_VERSION = "1"
PILOT_ARTIFACT_INVENTORY_SCHEMA_VERSION = "1"
RUN_ARTIFACT_NAMES = {
    "satellite": "satellite_rollout.json",
    "g1": "ground_design.jsonl",
    "g2": "ground_visibility.jsonl",
    "g3": "integrated_graphs.jsonl",
    "g4_steps": "ground_service_steps.jsonl",
    "g4_run": "ground_service_run.jsonl",
    "g5_realization": "ground_failure_realization.jsonl",
    "g5_steps": "ground_failure_service_steps.jsonl",
    "g5_run": "ground_failure_service_run.jsonl",
    "runtime": "generation_runtime.json",
    "inventory": "artifact_inventory.json",
    "summary": "run_summary.json",
    "log": "generation_log.jsonl",
}
CANONICAL_ARTIFACT_KEYS = (
    "satellite",
    "g1",
    "g2",
    "g3",
    "g4_steps",
    "g4_run",
    "g5_realization",
    "g5_steps",
    "g5_run",
)
RUNTIME_FIELDS = (
    "satellite_rollout_seconds",
    "g1_seconds",
    "g2_seconds",
    "g3_seconds",
    "g4_seconds",
    "g5_seconds",
    "persistence_seconds",
    "total_generation_seconds",
)


@dataclass(frozen=True)
class PilotSatelliteArtifact:
    run_id: int
    config: Tier1RolloutConfig
    steps: tuple[Tier1RolloutStep, ...]
    summary: Tier1RolloutSummary
    failure_realization: Tier1FailureRealization
    satellite_artifact_hash: str
    record_hash: str

    def __post_init__(self) -> None:
        if type(self.run_id) is not int or self.run_id < 0:
            raise TypeError("run_id must be a nonnegative integer")
        if not isinstance(self.config, Tier1RolloutConfig):
            raise TypeError("config must be Tier1RolloutConfig")
        if not isinstance(self.steps, tuple) or any(
            not isinstance(value, Tier1RolloutStep) for value in self.steps
        ):
            raise TypeError("steps must contain Tier1RolloutStep values")
        if len(self.steps) != self.config.num_steps:
            raise ValueError("Satellite steps do not match configured timestep count")
        if [step.t for step in self.steps] != list(range(self.config.num_steps)):
            raise ValueError("Satellite steps must be contiguous from zero")
        if not isinstance(self.summary, Tier1RolloutSummary):
            raise TypeError("summary must be Tier1RolloutSummary")
        if not isinstance(self.failure_realization, Tier1FailureRealization):
            raise TypeError("failure_realization must be Tier1FailureRealization")
        if self.summary.config_hash != self.config.config_hash():
            raise ValueError("Satellite summary config hash mismatch")
        if self.summary.num_steps != len(self.steps):
            raise ValueError("Satellite summary timestep count mismatch")
        if self.summary.num_failed_nodes != len(self.failure_realization.failed_nodes):
            raise ValueError("Satellite failed-node count mismatch")
        if self.summary.num_failed_edges != len(self.failure_realization.failed_edges):
            raise ValueError("Satellite failed-edge count mismatch")
        if self.satellite_artifact_hash != canonical_hash(self.scientific_payload()):
            raise ValueError("satellite_artifact_hash mismatch")
        if self.record_hash != canonical_hash(
            {
                "identity_domain": PILOT_SATELLITE_RECORD_DOMAIN,
                "identity_version": PILOT_SATELLITE_RECORD_VERSION,
                "run_id": self.run_id,
                "satellite_artifact_hash": self.satellite_artifact_hash,
            }
        ):
            raise ValueError("Satellite record hash mismatch")

    def scientific_payload(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "config_hash": self.config.config_hash(),
            "failed_edges": [list(edge) for edge in sorted(self.failure_realization.failed_edges)],
            "failed_nodes": sorted(self.failure_realization.failed_nodes),
            "identity_domain": PILOT_SATELLITE_ARTIFACT_DOMAIN,
            "identity_version": PILOT_SATELLITE_ARTIFACT_VERSION,
            "steps": [asdict(step) for step in self.steps],
            "summary": asdict(self.summary),
        }

    def to_manifest_object(self) -> dict[str, object]:
        return {
            "pilot_satellite_artifact_schema_version": PILOT_SATELLITE_ARTIFACT_SCHEMA_VERSION,
            "record_hash": self.record_hash,
            "run_id": self.run_id,
            "satellite_artifact_hash": self.satellite_artifact_hash,
            "scientific_payload": self.scientific_payload(),
        }


def make_pilot_satellite_artifact(
    *,
    run_id: int,
    config: Tier1RolloutConfig,
    steps: Sequence[Tier1RolloutStep],
    summary: Tier1RolloutSummary,
    failure_realization: Tier1FailureRealization,
) -> PilotSatelliteArtifact:
    values = {
        "config": asdict(config),
        "config_hash": config.config_hash(),
        "failed_edges": [list(edge) for edge in sorted(failure_realization.failed_edges)],
        "failed_nodes": sorted(failure_realization.failed_nodes),
        "identity_domain": PILOT_SATELLITE_ARTIFACT_DOMAIN,
        "identity_version": PILOT_SATELLITE_ARTIFACT_VERSION,
        "steps": [asdict(step) for step in steps],
        "summary": asdict(summary),
    }
    artifact_hash = canonical_hash(values)
    record_hash = canonical_hash(
        {
            "identity_domain": PILOT_SATELLITE_RECORD_DOMAIN,
            "identity_version": PILOT_SATELLITE_RECORD_VERSION,
            "run_id": run_id,
            "satellite_artifact_hash": artifact_hash,
        }
    )
    return PilotSatelliteArtifact(
        run_id=run_id,
        config=config,
        steps=tuple(steps),
        summary=summary,
        failure_realization=failure_realization,
        satellite_artifact_hash=artifact_hash,
        record_hash=record_hash,
    )


def _atomic_write(path: Path, text: str, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Pilot artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"Pilot artifact already exists: {path}")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_pilot_satellite_artifact(
    artifact: PilotSatelliteArtifact,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(artifact, PilotSatelliteArtifact):
        raise TypeError("artifact must be PilotSatelliteArtifact")
    output = Path(path)
    if output.suffix != ".json":
        raise ValueError("Satellite artifact must use .json")
    _atomic_write(
        output,
        canonical_json(artifact.to_manifest_object()) + "\n",
        overwrite=overwrite,
    )


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key '{key}'")
        result[key] = value
    return result


def _read_single_json(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    if not text or not text.endswith("\n") or text.count("\n") != 1:
        raise ValueError(f"Pilot JSON must contain one newline-terminated record: {path}")
    return json.loads(text, object_pairs_hook=_pairs_without_duplicates)


def read_pilot_satellite_artifact(path: str | Path) -> PilotSatelliteArtifact:
    value = _read_single_json(Path(path))
    expected = {
        "pilot_satellite_artifact_schema_version",
        "record_hash",
        "run_id",
        "satellite_artifact_hash",
        "scientific_payload",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Satellite artifact fields are invalid")
    if value["pilot_satellite_artifact_schema_version"] != PILOT_SATELLITE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported pilot satellite artifact schema")
    payload = value["scientific_payload"]
    payload_fields = {
        "config",
        "config_hash",
        "failed_edges",
        "failed_nodes",
        "identity_domain",
        "identity_version",
        "steps",
        "summary",
    }
    if not isinstance(payload, dict) or set(payload) != payload_fields:
        raise ValueError("Satellite scientific payload fields are invalid")
    if payload["identity_domain"] != PILOT_SATELLITE_ARTIFACT_DOMAIN:
        raise ValueError("Satellite artifact identity domain mismatch")
    if payload["identity_version"] != PILOT_SATELLITE_ARTIFACT_VERSION:
        raise ValueError("Satellite artifact identity version mismatch")
    config_fields = {field.name for field in fields(Tier1RolloutConfig)}
    if not isinstance(payload["config"], dict) or set(payload["config"]) != config_fields:
        raise ValueError("Satellite config fields are invalid")
    config = Tier1RolloutConfig(**payload["config"])
    if payload["config_hash"] != config.config_hash():
        raise ValueError("Satellite persisted config hash mismatch")
    step_fields = {field.name for field in fields(Tier1RolloutStep)}
    if not isinstance(payload["steps"], list):
        raise TypeError("Satellite steps must be an array")
    steps: list[Tier1RolloutStep] = []
    for source in payload["steps"]:
        if not isinstance(source, dict) or set(source) != step_fields:
            raise ValueError("Satellite step fields are invalid")
        steps.append(Tier1RolloutStep(**source))
    summary_fields = {field.name for field in fields(Tier1RolloutSummary)}
    if not isinstance(payload["summary"], dict) or set(payload["summary"]) != summary_fields:
        raise ValueError("Satellite summary fields are invalid")
    summary = Tier1RolloutSummary(**payload["summary"])
    if not isinstance(payload["failed_nodes"], list) or any(
        type(item) is not int for item in payload["failed_nodes"]
    ):
        raise TypeError("Satellite failed nodes must be an integer array")
    if payload["failed_nodes"] != sorted(set(payload["failed_nodes"])):
        raise ValueError("Satellite failed nodes must be unique and sorted")
    if not isinstance(payload["failed_edges"], list):
        raise TypeError("Satellite failed edges must be an array")
    failed_edges: list[tuple[int, int]] = []
    for edge in payload["failed_edges"]:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(type(endpoint) is not int for endpoint in edge)
            or edge[0] >= edge[1]
        ):
            raise ValueError("Satellite failed edge is invalid")
        failed_edges.append((edge[0], edge[1]))
    if failed_edges != sorted(set(failed_edges)):
        raise ValueError("Satellite failed edges must be unique and sorted")
    artifact = PilotSatelliteArtifact(
        run_id=value["run_id"],
        config=config,
        steps=tuple(steps),
        summary=summary,
        failure_realization=Tier1FailureRealization(
            failed_nodes=set(payload["failed_nodes"]),
            failed_edges=set(failed_edges),
        ),
        satellite_artifact_hash=value["satellite_artifact_hash"],
        record_hash=value["record_hash"],
    )
    if artifact.scientific_payload() != payload:
        raise ValueError("Satellite artifact payload is not canonical")
    return artifact


def run_directory(output_root: str | Path, run_id: int, *, repeat: bool = False) -> Path:
    base = "repeats" if repeat else "runs"
    return Path(output_root) / base / f"run_{run_id:03d}"


def artifact_paths(run_root: str | Path) -> dict[str, Path]:
    root = Path(run_root)
    return {key: root / name for key, name in RUN_ARTIFACT_NAMES.items()}


def _timed(callable_value):
    start = perf_counter()
    result = callable_value()
    return result, perf_counter() - start


def _bottleneck_counts(g5_steps) -> tuple[int, int, int, str]:
    counts = {"space": 0, "ground": 0, "tie": 0}
    for record in g5_steps:
        space = record.metrics.space_gcc_fraction_original
        ground = record.metrics.failure_adjusted_ground_service_fraction
        if space < ground:
            counts["space"] += 1
        elif ground < space:
            counts["ground"] += 1
        else:
            counts["tie"] += 1
    maximum = max(counts.values())
    leaders = [name for name, count in counts.items() if count == maximum]
    dominant = leaders[0] if len(leaders) == 1 else "tie"
    return counts["space"], counts["ground"], counts["tie"], dominant


def _validate_fraction_values(values: Mapping[str, object]) -> None:
    for name, value in values.items():
        if value is None:
            continue
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"Run summary contains nonfinite float: {name}")
            if (
                "fraction" in name
                or name.endswith("_probability")
                or name.endswith("_threshold")
            ) and not 0.0 <= value <= 1.0:
                raise ValueError(f"Run summary fraction is outside [0, 1]: {name}")
        if name.endswith("_count") and (type(value) is not int or value < 0):
            raise ValueError(f"Run summary count is invalid: {name}")


def extract_run_summary(
    *,
    design: IntegratedPilotDesign,
    run: IntegratedPilotRun,
    satellite_artifact: PilotSatelliteArtifact,
    ground_design,
    g4_run,
    g5_realization,
    g5_steps,
    g5_run,
    runtime: Mapping[str, float],
    artifact_bytes: int,
    replay_status: str = "pending",
    replay_runtime_seconds: float | None = None,
) -> dict[str, object]:
    if ground_design.run_id != run.run_id:
        raise ValueError("G1 run ID mismatch")
    if g4_run.run_id != run.run_id or g5_run.run_id != run.run_id:
        raise ValueError("Service run ID mismatch")
    if g5_realization.run_id != run.run_id:
        raise ValueError("Ground failure run ID mismatch")
    if satellite_artifact.run_id != run.run_id:
        raise ValueError("Satellite artifact run ID mismatch")
    if satellite_artifact.config.config_hash() != ground_design.satellite_config_hash:
        raise ValueError("Satellite and G1 identities do not match")
    g4 = g4_run.summary
    g5 = g5_run.summary
    if g5.baseline_ground_service_run_summary_hash != g4.run_summary_hash:
        raise ValueError("G5 baseline summary does not match G4")
    if g5.baseline_ground_service_step_sequence_hash != g4.step_sequence_hash:
        raise ValueError("G5 baseline sequence does not match G4")
    if g4.timestep_count != design.duration_minutes * 60 // design.step_seconds + 1:
        raise ValueError("G4 timestep count does not match fixed temporal profile")
    if g5.timestep_count != g4.timestep_count or len(g5_steps) != g4.timestep_count:
        raise ValueError("G5 timestep count does not match G4")
    if g5.ground_failure_realization_hash != g5_realization.realization.ground_failure_realization_hash:
        raise ValueError("G5 realization identity mismatch")
    space_count, ground_count, tie_count, dominant = _bottleneck_counts(g5_steps)
    summary: dict[str, object] = {
        "pilot_run_summary_schema_version": PILOT_RUN_SUMMARY_SCHEMA_VERSION,
        "run_id": run.run_id,
        "design_id": run.design_id,
        "realization_id": run.realization_id,
        "design_group_id": run.design_group_id,
        "design_hash": run.design_hash,
        "realization_manifest_hash": run.realization_manifest_hash,
        "satellite_config_hash": satellite_artifact.config.config_hash(),
        "satellite_artifact_hash": satellite_artifact.satellite_artifact_hash,
        "ground_design_hash": ground_design.ground_design_hash,
        "visibility_policy_hash": g5.visibility_policy_hash,
        "ground_service_policy_hash": g5.ground_service_policy_hash,
        "ground_failure_policy_hash": g5.ground_failure_policy_hash,
        "ground_failure_realization_hash": g5.ground_failure_realization_hash,
        "g4_run_summary_hash": g4.run_summary_hash,
        "g5_run_summary_hash": g5.failure_adjusted_run_summary_hash,
        "planes": design.num_planes,
        "satellites_per_plane": design.sats_per_plane,
        "configured_satellite_count": design.configured_satellite_count,
        "altitude_km": design.altitude_km,
        "inclination_deg": design.inclination_deg,
        "satellite_node_failure_probability": design.node_failure_probability,
        "satellite_edge_failure_probability": design.edge_failure_probability,
        "failed_satellite_node_count": satellite_artifact.summary.num_failed_nodes,
        "failed_satellite_edge_count": satellite_artifact.summary.num_failed_edges,
        "civilian_count": design.civilian_count,
        "government_count": design.government_count,
        "military_count": design.military_count,
        "total_ground_station_count": design.total_ground_station_count,
        "ground_failure_probability": design.ground_failure_probability,
        "failed_ground_station_count": g5.failed_ground_station_count,
        "operational_ground_station_count": g5.operational_ground_station_count,
        "failed_civilian_count": g5.failed_civilian_count,
        "failed_government_count": g5.failed_government_count,
        "failed_military_count": g5.failed_military_count,
        "space_gcc_fraction_original_min": g5.space_gcc_fraction_original_min,
        "space_gcc_fraction_original_mean": g5.space_gcc_fraction_original_mean,
        "space_gcc_fraction_surviving_min": g5.space_gcc_fraction_surviving_min,
        "space_gcc_fraction_surviving_mean": g5.space_gcc_fraction_surviving_mean,
        "space_threshold_breach_any": g5.space_threshold_breach_any,
        "space_threshold_breach_timestep_count": g5.space_threshold_breach_timestep_count,
        "baseline_ground_service_fraction_min": g5.baseline_ground_service_fraction_min,
        "baseline_ground_service_fraction_mean": g5.baseline_ground_service_fraction_mean,
        "baseline_overall_service_fraction_min": g5.baseline_overall_service_fraction_min,
        "baseline_overall_service_fraction_mean": g5.baseline_overall_service_fraction_mean,
        "failure_adjusted_ground_service_fraction_min": g5.failure_adjusted_ground_service_fraction_min,
        "failure_adjusted_ground_service_fraction_mean": g5.failure_adjusted_ground_service_fraction_mean,
        "failure_adjusted_overall_service_fraction_min": g5.failure_adjusted_overall_service_fraction_min,
        "failure_adjusted_overall_service_fraction_mean": g5.failure_adjusted_overall_service_fraction_mean,
        "ground_threshold_breach_any": g5.ground_threshold_breach_any,
        "overall_threshold_breach_any": g5.overall_threshold_breach_any,
        "ground_threshold_breach_timestep_count": g5.ground_threshold_breach_timestep_count,
        "overall_threshold_breach_timestep_count": g5.overall_threshold_breach_timestep_count,
        "ground_service_loss_due_to_failures_max": g5.ground_service_loss_due_to_failures_max,
        "ground_service_loss_due_to_failures_mean": g5.ground_service_loss_due_to_failures_mean,
        "overall_service_loss_due_to_ground_failures_max": g5.overall_service_loss_due_to_ground_failures_max,
        "overall_service_loss_due_to_ground_failures_mean": g5.overall_service_loss_due_to_ground_failures_mean,
        "failure_adjusted_civilian_service_fraction_min": g5.failure_adjusted_civilian_service_fraction_min,
        "failure_adjusted_civilian_service_fraction_mean": g5.failure_adjusted_civilian_service_fraction_mean,
        "failure_adjusted_government_service_fraction_min": g5.failure_adjusted_government_service_fraction_min,
        "failure_adjusted_government_service_fraction_mean": g5.failure_adjusted_government_service_fraction_mean,
        "failure_adjusted_military_service_fraction_min": g5.failure_adjusted_military_service_fraction_min,
        "failure_adjusted_military_service_fraction_mean": g5.failure_adjusted_military_service_fraction_mean,
        "space_bottleneck_timestep_count": space_count,
        "ground_bottleneck_timestep_count": ground_count,
        "tie_bottleneck_timestep_count": tie_count,
        "dominant_bottleneck": dominant,
        "timestep_count": g5.timestep_count,
        "generation_status": "success",
        "replay_status": replay_status,
        "generation_runtime_seconds": runtime["total_generation_seconds"],
        "replay_runtime_seconds": replay_runtime_seconds,
        "artifact_bytes": artifact_bytes,
    }
    for name in RUNTIME_FIELDS:
        summary[name] = runtime[name]
    if space_count + ground_count + tie_count != g5.timestep_count:
        raise ValueError("Bottleneck counts do not cover all timesteps")
    _validate_fraction_values(summary)
    return summary


def _artifact_inventory(paths: Mapping[str, Path]) -> dict[str, object]:
    artifacts: list[dict[str, object]] = []
    for key in CANONICAL_ARTIFACT_KEYS:
        path = paths[key]
        if not path.is_file():
            raise ValueError(f"Missing canonical run artifact: {path}")
        content = path.read_bytes()
        artifacts.append(
            {
                "artifact_key": key,
                "bytes": len(content),
                "filename": path.name,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {
        "artifacts": artifacts,
        "canonical_artifact_bytes": sum(value["bytes"] for value in artifacts),
        "pilot_artifact_inventory_schema_version": PILOT_ARTIFACT_INVENTORY_SCHEMA_VERSION,
    }


def generate_pilot_run(
    *,
    design: IntegratedPilotDesign,
    run: IntegratedPilotRun,
    catalog: GroundStationCatalog,
    output_root: str | Path,
    repeat: bool = False,
) -> dict[str, object]:
    if run.design_id != design.design_id or run.design_hash != design.design_hash:
        raise ValueError("Run does not match supplied design")
    if not isinstance(catalog, GroundStationCatalog):
        raise TypeError("catalog must be GroundStationCatalog")
    root = run_directory(output_root, run.run_id, repeat=repeat)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Pilot run directory already contains evidence: {root}")
    root.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(root)
    persistence_seconds = 0.0
    generation_start = perf_counter()
    stage_events: list[dict[str, object]] = []

    satellite_config = design.satellite_config(
        satellite_seed=run.satellite_rollout_seed
    )
    satellite_values, satellite_seconds = _timed(
        lambda: run_tier1_rollout(satellite_config)
    )
    satellite_steps, satellite_summary, satellite_failures = satellite_values
    satellite_artifact = make_pilot_satellite_artifact(
        run_id=run.run_id,
        config=satellite_config,
        steps=satellite_steps,
        summary=satellite_summary,
        failure_realization=satellite_failures,
    )
    _, elapsed = _timed(
        lambda: write_pilot_satellite_artifact(satellite_artifact, paths["satellite"])
    )
    persistence_seconds += elapsed
    stage_events.append({"run_id": run.run_id, "stage": "satellite", "status": "success"})

    def build_g1():
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
            satellite_config_hash=satellite_config.config_hash(),
            selection=selection,
        )

    ground_design, g1_seconds = _timed(build_g1)
    _, elapsed = _timed(
        lambda: write_ground_design_manifest((ground_design,), paths["g1"])
    )
    persistence_seconds += elapsed
    stage_events.append({"run_id": run.run_id, "stage": "g1", "status": "success"})

    visibility_policy = GroundVisibilityPolicy(design.minimum_elevation_deg)

    def build_g2():
        positions = reconstruct_operational_satellite_position_sequence(
            satellite_config=satellite_config,
            failure_realization=satellite_failures,
        )
        snapshots = evaluate_ground_design_visibility_sequence(
            ground_design=ground_design,
            catalog=catalog,
            satellite_sequence=positions,
            policy=visibility_policy,
        )
        return tuple(
            make_ground_visibility_record(run_id=run.run_id, snapshot=snapshot)
            for snapshot in snapshots
        ), snapshots

    g2_values, g2_seconds = _timed(build_g2)
    g2_records, g2_snapshots = g2_values
    _, elapsed = _timed(
        lambda: write_ground_visibility_manifest(g2_records, paths["g2"])
    )
    persistence_seconds += elapsed
    stage_events.append({"run_id": run.run_id, "stage": "g2", "status": "success"})

    def build_g3():
        graph_snapshots = reconstruct_operational_satellite_graph_sequence(
            satellite_config=satellite_config,
            failure_realization=satellite_failures,
        )
        integrated = tuple(
            build_integrated_ground_graph(
                ground_design=ground_design,
                catalog=catalog,
                satellite_graph_snapshot=graph_snapshot,
                verified_visibility_snapshot=visibility_snapshot,
            )
            for graph_snapshot, visibility_snapshot in zip(
                graph_snapshots, g2_snapshots, strict=True
            )
        )
        return tuple(
            make_integrated_graph_record(
                ground_design=ground_design,
                snapshot=snapshot,
            )
            for snapshot in integrated
        )

    g3_records, g3_seconds = _timed(build_g3)
    _, elapsed = _timed(
        lambda: write_integrated_graph_manifest(g3_records, paths["g3"])
    )
    persistence_seconds += elapsed
    stage_events.append({"run_id": run.run_id, "stage": "g3", "status": "success"})

    service_policy = GroundServicePolicy(
        design.space_gcc_threshold,
        design.ground_service_threshold,
    )
    g4_values, g4_seconds = _timed(
        lambda: generate_verified_ground_service_records(
            satellite_config=satellite_config,
            failure_realization=satellite_failures,
            ground_design=ground_design,
            catalog=catalog,
            visibility_policy=visibility_policy,
            visibility_records=g2_records,
            integrated_records=g3_records,
            service_policy=service_policy,
        )
    )
    g4_steps, g4_run = g4_values
    _, elapsed_steps = _timed(
        lambda: write_ground_service_step_manifest(g4_steps, paths["g4_steps"])
    )
    _, elapsed_run = _timed(
        lambda: write_ground_service_run_manifest((g4_run,), paths["g4_run"])
    )
    persistence_seconds += elapsed_steps + elapsed_run
    stage_events.append({"run_id": run.run_id, "stage": "g4", "status": "success"})

    ground_failure_policy = GroundFailurePolicy(design.ground_failure_probability)
    g5_values, g5_seconds = _timed(
        lambda: generate_verified_ground_failure_service_records(
            satellite_config=satellite_config,
            satellite_failure_realization=satellite_failures,
            ground_design=ground_design,
            catalog=catalog,
            visibility_policy=visibility_policy,
            visibility_records=g2_records,
            integrated_records=g3_records,
            ground_service_policy=service_policy,
            g4_step_records=g4_steps,
            g4_run_record=g4_run,
            ground_failure_policy=ground_failure_policy,
            ground_failure_seed=run.ground_failure_seed,
        )
    )
    g5_realization, g5_steps, g5_run = g5_values
    _, elapsed_realization = _timed(
        lambda: write_ground_failure_realization_manifest(
            (g5_realization,), paths["g5_realization"]
        )
    )
    _, elapsed_steps = _timed(
        lambda: write_ground_failure_service_step_manifest(g5_steps, paths["g5_steps"])
    )
    _, elapsed_run = _timed(
        lambda: write_ground_failure_service_run_manifest((g5_run,), paths["g5_run"])
    )
    persistence_seconds += elapsed_realization + elapsed_steps + elapsed_run
    stage_events.append({"run_id": run.run_id, "stage": "g5", "status": "success"})

    total_generation_seconds = perf_counter() - generation_start
    runtime = {
        "g1_seconds": g1_seconds,
        "g2_seconds": g2_seconds,
        "g3_seconds": g3_seconds,
        "g4_seconds": g4_seconds,
        "g5_seconds": g5_seconds,
        "persistence_seconds": persistence_seconds,
        "pilot_runtime_schema_version": PILOT_RUNTIME_SCHEMA_VERSION,
        "run_id": run.run_id,
        "satellite_rollout_seconds": satellite_seconds,
        "total_generation_seconds": total_generation_seconds,
    }
    if any(
        not isinstance(runtime[name], float)
        or not math.isfinite(runtime[name])
        or runtime[name] < 0.0
        for name in RUNTIME_FIELDS
    ):
        raise RuntimeError("Pilot generation produced invalid runtime evidence")
    _atomic_write(paths["runtime"], canonical_json(runtime) + "\n")
    _atomic_write(
        paths["log"],
        "\n".join(canonical_json(value) for value in stage_events) + "\n",
    )
    inventory = _artifact_inventory(paths)
    _atomic_write(paths["inventory"], canonical_json(inventory) + "\n")
    summary = extract_run_summary(
        design=design,
        run=run,
        satellite_artifact=satellite_artifact,
        ground_design=ground_design,
        g4_run=g4_run,
        g5_realization=g5_realization,
        g5_steps=g5_steps,
        g5_run=g5_run,
        runtime=runtime,
        artifact_bytes=inventory["canonical_artifact_bytes"],
    )
    _atomic_write(paths["summary"], canonical_json(summary) + "\n")
    return summary


def _failure_record(
    *, run: IntegratedPilotRun, error: Exception, stage: str
) -> dict[str, object]:
    return {
        "design_id": run.design_id,
        "error_message": str(error),
        "error_type": type(error).__name__,
        "realization_id": run.realization_id,
        "run_id": run.run_id,
        "stage": stage,
        "status": "failed",
    }


def select_manifest_runs(
    runs: Sequence[IntegratedPilotRun], run_ids: Sequence[int] | None
) -> tuple[IntegratedPilotRun, ...]:
    normalized = tuple(runs)
    if run_ids is None:
        return normalized
    requested = tuple(run_ids)
    if any(type(run_id) is not int for run_id in requested):
        raise TypeError("Run subset IDs must be integers")
    if len(requested) != len(set(requested)):
        raise ValueError("Run subset contains duplicate IDs")
    by_id = {run.run_id: run for run in normalized}
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise ValueError(f"Run subset contains unknown IDs: {missing}")
    return tuple(by_id[run_id] for run_id in sorted(requested))


def run_pilot_manifest(
    *,
    manifest_path: str | Path,
    output_root: str | Path,
    run_ids: Sequence[int] | None = None,
    resume: bool = False,
    repeat: bool = False,
) -> dict[str, object]:
    if type(resume) is not bool or type(repeat) is not bool:
        raise TypeError("resume and repeat must be Booleans")
    manifest = Path(manifest_path)
    input_root = manifest.parent
    designs = read_pilot_design_manifest(input_root / "pilot_designs.json")
    runs = read_pilot_run_manifest(manifest, designs)
    catalog = load_ground_station_catalog(input_root / "pilot_catalog.csv")
    selected = select_manifest_runs(runs, run_ids)
    design_by_id = {design.design_id: design for design in designs}
    successes: list[int] = []
    failures: list[dict[str, object]] = []
    resumed: list[int] = []
    for run in selected:
        root = run_directory(output_root, run.run_id, repeat=repeat)
        if root.exists() and any(root.iterdir()) and resume:
            try:
                from satnet.experiments.integrated_ground_replay import validate_existing_run

                validate_existing_run(
                    design=design_by_id[run.design_id],
                    run=run,
                    catalog=catalog,
                    run_root=root,
                )
                successes.append(run.run_id)
                resumed.append(run.run_id)
                continue
            except Exception as exc:
                failures.append(_failure_record(run=run, error=exc, stage="resume_validation"))
                continue
        try:
            generate_pilot_run(
                design=design_by_id[run.design_id],
                run=run,
                catalog=catalog,
                output_root=output_root,
                repeat=repeat,
            )
            successes.append(run.run_id)
        except Exception as exc:
            failures.append(_failure_record(run=run, error=exc, stage="generation"))
            root.mkdir(parents=True, exist_ok=True)
            failure_path = root / "generation_failure.json"
            if not failure_path.exists():
                _atomic_write(failure_path, canonical_json(failures[-1]) + "\n")
    result = {
        "attempted_run_count": len(selected),
        "expected_manifest_run_count": len(runs),
        "failed_generation_run_count": len(failures),
        "failures": failures,
        "repeat": repeat,
        "resumed_run_count": len(resumed),
        "resumed_run_ids": resumed,
        "successful_generation_run_count": len(successes),
        "successful_run_ids": successes,
    }
    summary_name = "repeat_generation_summary.json" if repeat else "generation_summary.json"
    summary_root = Path(output_root) / ("repeats" if repeat else "summaries")
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_path = summary_root / summary_name
    if summary_path.exists():
        if not resume:
            raise FileExistsError(f"Generation summary already exists: {summary_path}")
    else:
        _atomic_write(summary_path, canonical_json(result) + "\n")
    return result
