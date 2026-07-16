from __future__ import annotations

from datetime import timedelta, timezone

from satnet.ground.coordinates import (
    CoordinateFrame,
    FramedSatellitePosition,
    OperationalSatellitePositionSnapshot,
)
from satnet.network.hypatia_adapter import HypatiaAdapter
from satnet.simulation.tier1_rollout import Tier1FailureRealization, Tier1RolloutConfig


def _validate_satellite_id(value: object, total_satellites: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if not 0 <= value < total_satellites:
        raise ValueError(f"{field_name} must be within [0, {total_satellites - 1}]")
    return value


def _validate_failure_realization(
    failure_realization: Tier1FailureRealization,
    *,
    total_satellites: int,
) -> None:
    if not isinstance(failure_realization, Tier1FailureRealization):
        raise TypeError("failure_realization must be a Tier1FailureRealization")
    if not isinstance(failure_realization.failed_nodes, set):
        raise TypeError("failed_nodes must be a set")
    for node_id in failure_realization.failed_nodes:
        _validate_satellite_id(node_id, total_satellites, "failed node ID")
    if not isinstance(failure_realization.failed_edges, set):
        raise TypeError("failed_edges must be a set")
    for edge in failure_realization.failed_edges:
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError("Every failed edge must be a two-endpoint tuple")
        first = _validate_satellite_id(edge[0], total_satellites, "failed edge endpoint")
        second = _validate_satellite_id(edge[1], total_satellites, "failed edge endpoint")
        if first >= second:
            raise ValueError("failed edge endpoints must be distinct and ascending")


def reconstruct_operational_satellite_position_sequence(
    *,
    satellite_config: Tier1RolloutConfig,
    failure_realization: Tier1FailureRealization,
) -> tuple[OperationalSatellitePositionSnapshot, ...]:
    if not isinstance(satellite_config, Tier1RolloutConfig):
        raise TypeError("satellite_config must be a Tier1RolloutConfig")
    epoch = satellite_config.epoch
    if epoch.tzinfo is None or epoch.utcoffset() is None:
        raise ValueError("Satellite rollout epoch must be timezone-aware")
    if epoch.utcoffset().total_seconds() != 0.0:
        raise ValueError("Satellite rollout epoch must be normalized to UTC")
    _validate_failure_realization(
        failure_realization,
        total_satellites=satellite_config.total_satellites,
    )
    config_hash = satellite_config.config_hash()
    snapshots: list[OperationalSatellitePositionSnapshot] = []
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
            timestamp = (
                epoch
                + timedelta(
                    seconds=timestep_index * satellite_config.step_seconds
                )
            ).astimezone(timezone.utc)
            production_positions = adapter.get_positions_at_step(timestep_index)
            positions = tuple(
                FramedSatellitePosition(
                    satellite_id=position.sat_id,
                    timestamp_utc=timestamp,
                    x_km=position.x_km,
                    y_km=position.y_km,
                    z_km=position.z_km,
                    frame=CoordinateFrame.ECEF,
                )
                for position in sorted(production_positions, key=lambda value: value.sat_id)
                if position.sat_id not in failure_realization.failed_nodes
            )
            snapshots.append(
                OperationalSatellitePositionSnapshot(
                    timestep_index=timestep_index,
                    timestamp_utc=timestamp,
                    satellite_config_hash=config_hash,
                    positions=positions,
                )
            )
    if tuple(snapshot.timestep_index for snapshot in snapshots) != tuple(
        range(satellite_config.num_steps)
    ):
        raise RuntimeError("Operational position sequence does not cover every configured timestep")
    if any(
        current.timestamp_utc >= following.timestamp_utc
        for current, following in zip(snapshots, snapshots[1:])
    ):
        raise RuntimeError("Operational position timestamps must be strictly increasing")
    return tuple(snapshots)
