from __future__ import annotations

from datetime import timedelta

import pytest

from satnet.ground.coordinates import CoordinateFrame
from satnet.ground.position_adapter import reconstruct_operational_satellite_position_sequence
from satnet.simulation.tier1_rollout import Tier1FailureRealization, Tier1RolloutConfig


def config() -> Tier1RolloutConfig:
    return Tier1RolloutConfig(
        num_planes=2,
        sats_per_plane=2,
        duration_minutes=1,
        step_seconds=60,
        node_failure_prob=0.0,
        edge_failure_prob=0.0,
        seed=123,
    )


def failures(
    nodes: set[int] | None = None,
    edges: set[tuple[int, int]] | None = None,
) -> Tier1FailureRealization:
    return Tier1FailureRealization(
        failed_nodes=set() if nodes is None else nodes,
        failed_edges=set() if edges is None else edges,
    )


def test_reconstruction_covers_every_timestep_with_exact_timestamps() -> None:
    satellite_config = config()
    snapshots = reconstruct_operational_satellite_position_sequence(
        satellite_config=satellite_config,
        failure_realization=failures(),
    )
    assert len(snapshots) == satellite_config.num_steps == 2
    assert tuple(item.timestep_index for item in snapshots) == (0, 1)
    assert snapshots[0].timestamp_utc == satellite_config.epoch
    assert snapshots[1].timestamp_utc == satellite_config.epoch + timedelta(seconds=60)
    assert all(item.satellite_config_hash == satellite_config.config_hash() for item in snapshots)
    assert all(len(item.positions) == 4 for item in snapshots)
    assert all(
        position.frame is CoordinateFrame.ECEF
        for snapshot in snapshots
        for position in snapshot.positions
    )


def test_failed_satellites_are_excluded_without_new_sampling() -> None:
    satellite_config = config()
    snapshots = reconstruct_operational_satellite_position_sequence(
        satellite_config=satellite_config,
        failure_realization=failures(nodes={1, 3}),
    )
    assert all(
        tuple(position.satellite_id for position in snapshot.positions) == (0, 2)
        for snapshot in snapshots
    )


def test_all_satellites_failed_retains_every_empty_timestep() -> None:
    satellite_config = config()
    snapshots = reconstruct_operational_satellite_position_sequence(
        satellite_config=satellite_config,
        failure_realization=failures(nodes={0, 1, 2, 3}),
    )
    assert len(snapshots) == satellite_config.num_steps
    assert all(snapshot.positions == () for snapshot in snapshots)
    assert snapshots[0].timestamp_utc < snapshots[1].timestamp_utc


def test_failed_edges_do_not_change_geometric_positions() -> None:
    satellite_config = config()
    baseline = reconstruct_operational_satellite_position_sequence(
        satellite_config=satellite_config,
        failure_realization=failures(),
    )
    edge_failed = reconstruct_operational_satellite_position_sequence(
        satellite_config=satellite_config,
        failure_realization=failures(edges={(0, 1), (2, 3)}),
    )
    assert edge_failed == baseline


@pytest.mark.parametrize("node_id", [True, -1, 4, "1", 1.0])
def test_invalid_failed_node_ids_fail_closed(node_id: object) -> None:
    with pytest.raises((TypeError, ValueError), match="failed node ID"):
        reconstruct_operational_satellite_position_sequence(
            satellite_config=config(),
            failure_realization=failures(nodes={node_id}),
        )


@pytest.mark.parametrize(
    "edge",
    [
        (0,),
        (0, 1, 2),
        "01",
        (True, 1),
        (-1, 1),
        (0, 4),
        (1, 0),
        (1, 1),
    ],
)
def test_malformed_failed_edges_fail_closed(edge: object) -> None:
    with pytest.raises((TypeError, ValueError), match="failed edge"):
        reconstruct_operational_satellite_position_sequence(
            satellite_config=config(),
            failure_realization=failures(edges={edge}),
        )
