"""Build fixed-profile Tier 1 temporal DSS scenarios."""

from __future__ import annotations

from dataclasses import dataclass

from satnet.dss.realization import DSSRealizationSeeds
from satnet.dss.schemas import DSSArchitectureRequest, DSS_SEQUENCE_LENGTH
from satnet.ground.integrated_graph import OperationalSatelliteGraphSnapshot
from satnet.ground.position_adapter import reconstruct_operational_satellite_position_sequence
from satnet.ground.satellite_graph_adapter import reconstruct_operational_satellite_graph_sequence
from satnet.ground.coordinates import OperationalSatellitePositionSnapshot
from satnet.network.hypatia_adapter import PHYSICS_MODEL_VERSION
from satnet.simulation.tier1_rollout import (
    DEFAULT_EPOCH_ISO,
    DEFAULT_FAILURE_MODEL,
    Tier1FailureRealization,
    Tier1RolloutConfig,
    Tier1RolloutStep,
    Tier1RolloutSummary,
    run_tier1_rollout,
)

DSS_DURATION_MINUTES = 10
DSS_STEP_SECONDS = 60
DSS_PHASING_FACTOR = 1
DSS_MAX_ISL_DISTANCE_KM = 10_000.0
DSS_ISL_POLICY = "grid_fixed"
DSS_ADJACENT_SEARCH_K = 1
DSS_MAX_INTER_PLANE_LINKS_PER_SAT = 1
DSS_ORBITAL_ENGINE = "sgp4"
DSS_FIXED_ROLLOUT_THRESHOLD = 0.80


@dataclass(frozen=True)
class DSSScenario:
    rollout_config: Tier1RolloutConfig
    steps: tuple[Tier1RolloutStep, ...]
    summary: Tier1RolloutSummary
    failure_realization: Tier1FailureRealization
    graph_snapshots: tuple[OperationalSatelliteGraphSnapshot, ...]
    position_snapshots: tuple[OperationalSatellitePositionSnapshot, ...]
    seeds: DSSRealizationSeeds

    @property
    def graphs(self):
        return tuple(snapshot.to_networkx() for snapshot in self.graph_snapshots)


def build_rollout_config(
    architecture: DSSArchitectureRequest, satellite_failure_seed: int
) -> Tier1RolloutConfig:
    return Tier1RolloutConfig(
        num_planes=architecture.num_planes,
        sats_per_plane=architecture.sats_per_plane,
        inclination_deg=architecture.inclination_deg,
        altitude_km=architecture.altitude_km,
        phasing_factor=DSS_PHASING_FACTOR,
        duration_minutes=DSS_DURATION_MINUTES,
        step_seconds=DSS_STEP_SECONDS,
        max_isl_distance_km=DSS_MAX_ISL_DISTANCE_KM,
        isl_policy=DSS_ISL_POLICY,
        adjacent_search_k=DSS_ADJACENT_SEARCH_K,
        max_inter_plane_links_per_sat=DSS_MAX_INTER_PLANE_LINKS_PER_SAT,
        # This is fixed and not the user decision requirement. Rollout labels are
        # not used by the DSS inference path and cannot influence scenario identity.
        gcc_threshold=DSS_FIXED_ROLLOUT_THRESHOLD,
        node_failure_prob=architecture.satellite_node_failure_probability,
        edge_failure_prob=architecture.satellite_edge_failure_probability,
        failure_model=DEFAULT_FAILURE_MODEL,
        seed=satellite_failure_seed,
        epoch_iso=DEFAULT_EPOCH_ISO,
        orbital_engine=DSS_ORBITAL_ENGINE,
    )


def build_scenario(
    architecture: DSSArchitectureRequest, seeds: DSSRealizationSeeds
) -> DSSScenario:
    config = build_rollout_config(architecture, seeds.satellite_failure_seed)
    steps, summary, failures = run_tier1_rollout(config)
    graphs = reconstruct_operational_satellite_graph_sequence(
        satellite_config=config, failure_realization=failures
    )
    positions = reconstruct_operational_satellite_position_sequence(
        satellite_config=config, failure_realization=failures
    )
    if len(steps) != DSS_SEQUENCE_LENGTH or config.num_steps != DSS_SEQUENCE_LENGTH:
        raise RuntimeError(
            f"DSS TGNN requires exactly {DSS_SEQUENCE_LENGTH} snapshots; got {len(steps)}"
        )
    if len(graphs) != DSS_SEQUENCE_LENGTH or len(positions) != DSS_SEQUENCE_LENGTH:
        raise RuntimeError("Tier 1 graph/position sequences do not contain 11 snapshots")
    return DSSScenario(
        rollout_config=config,
        steps=tuple(steps),
        summary=summary,
        failure_realization=failures,
        graph_snapshots=graphs,
        position_snapshots=positions,
        seeds=seeds,
    )


def scenario_provenance() -> dict[str, object]:
    return {
        "duration_minutes": DSS_DURATION_MINUTES,
        "step_seconds": DSS_STEP_SECONDS,
        "sequence_length": DSS_SEQUENCE_LENGTH,
        "physics_model_version": PHYSICS_MODEL_VERSION,
        "failure_model": DEFAULT_FAILURE_MODEL,
        "isl_policy": DSS_ISL_POLICY,
        "orbital_engine": DSS_ORBITAL_ENGINE,
    }
