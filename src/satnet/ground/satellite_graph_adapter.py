from __future__ import annotations

from datetime import timedelta, timezone

from satnet.ground.integrated_graph import (
    OperationalSatelliteGraphSnapshot,
    operational_snapshot_from_networkx,
)
from satnet.ground.position_adapter import validate_failure_realization
from satnet.network.hypatia_adapter import HypatiaAdapter
from satnet.simulation.tier1_rollout import Tier1FailureRealization, Tier1RolloutConfig


def reconstruct_operational_satellite_graph_sequence(
    *,
    satellite_config: Tier1RolloutConfig,
    failure_realization: Tier1FailureRealization,
) -> tuple[OperationalSatelliteGraphSnapshot, ...]:
    if not isinstance(satellite_config, Tier1RolloutConfig):
        raise TypeError("satellite_config must be a Tier1RolloutConfig")
    epoch = satellite_config.epoch
    if epoch.tzinfo is None or epoch.utcoffset() is None:
        raise ValueError("Satellite rollout epoch must be timezone-aware")
    if epoch.utcoffset().total_seconds() != 0.0:
        raise ValueError("Satellite rollout epoch must be normalized to UTC")
    validate_failure_realization(
        failure_realization,
        total_satellites=satellite_config.total_satellites,
    )
    snapshots: list[OperationalSatelliteGraphSnapshot] = []
    config_hash = satellite_config.config_hash()
    with HypatiaAdapter(
        num_planes=satellite_config.num_planes,
        sats_per_plane=satellite_config.sats_per_plane,
        inclination_deg=satellite_config.inclination_deg,
        altitude_km=satellite_config.altitude_km,
        phasing_factor=satellite_config.phasing_factor,
        epoch=epoch,
        orbital_engine=satellite_config.orbital_engine,
    ) as adapter:
        adapter.generate_tles()
        adapter.calculate_isls(
            duration_minutes=satellite_config.duration_minutes,
            step_seconds=satellite_config.step_seconds,
            max_isl_distance_km=satellite_config.max_isl_distance_km,
            isl_policy=satellite_config.isl_policy,
            adjacent_search_k=satellite_config.adjacent_search_k,
            max_inter_plane_links_per_sat=satellite_config.max_inter_plane_links_per_sat,
        )
        for timestep_index in range(satellite_config.num_steps):
            graph = adapter.get_graph_at_step(timestep_index).copy()
            graph.remove_nodes_from(failure_realization.failed_nodes)
            for first, second in failure_realization.failed_edges:
                if graph.has_edge(first, second):
                    graph.remove_edge(first, second)
            timestamp = (
                epoch
                + timedelta(
                    seconds=timestep_index * satellite_config.step_seconds
                )
            ).astimezone(timezone.utc)
            snapshots.append(
                operational_snapshot_from_networkx(
                    timestep_index=timestep_index,
                    timestamp_utc=timestamp,
                    satellite_config_hash=config_hash,
                    graph=graph,
                )
            )
    if tuple(snapshot.timestep_index for snapshot in snapshots) != tuple(
        range(satellite_config.num_steps)
    ):
        raise RuntimeError("Operational graph sequence does not cover every configured timestep")
    if any(
        current.timestamp_utc >= following.timestamp_utc
        for current, following in zip(snapshots, snapshots[1:])
    ):
        raise RuntimeError("Operational graph timestamps must be strictly increasing")
    return tuple(snapshots)
