from __future__ import annotations

from datetime import datetime, timezone

from satnet.experiments.external_validation.phase4a import (
    EXTERNAL_RF_FEATURE_ORDER,
    EXTERNAL_TGNN_EDGE_FEATURE_ORDER,
    EXTERNAL_TGNN_NODE_FEATURE_ORDER,
    Phase4AConfig,
    choose_episode_timestamps,
    infer_raan_planes,
    scale_tgnn_edge_features,
)


def test_phase4a_contract_dimensions_and_order() -> None:
    assert EXTERNAL_RF_FEATURE_ORDER == (
        "num_planes",
        "sats_per_plane",
        "altitude_km",
        "inclination_deg",
        "satellite_node_failure_probability",
        "satellite_edge_failure_probability",
    )
    assert len(EXTERNAL_TGNN_NODE_FEATURE_ORDER) == 3
    assert len(EXTERNAL_TGNN_EDGE_FEATURE_ORDER) == 4


def test_episode_timestamps_are_deterministic_and_inclusive() -> None:
    start = datetime(2024, 2, 1, 12, tzinfo=timezone.utc)
    end = datetime(2025, 12, 31, 12, tzinfo=timezone.utc)
    lhs = choose_episode_timestamps(start, end)
    rhs = choose_episode_timestamps(start, end)
    assert lhs == rhs
    assert len(lhs) == 300
    assert lhs[0] == start
    assert lhs[-1] == end
    assert all(previous < current for previous, current in zip(lhs, lhs[1:]))


def test_raan_plane_inference_is_circular_and_target_blind() -> None:
    from satnet.experiments.external_validation.phase4a import OrbitalRecord

    def record(norad_id: int, raan: float, phase: float) -> OrbitalRecord:
        return OrbitalRecord(
            norad_id=norad_id,
            epoch=datetime(2024, 2, 1, tzinfo=timezone.utc),
            inclination_deg=53.0,
            raan_deg=raan,
            eccentricity=0.001,
            arg_perigee_deg=0.0,
            mean_anomaly_deg=phase,
            mean_motion_rev_day=15.5,
            mean_motion_dot_rev_day2=0.0,
            bstar=0.0,
            altitude_km=550.0,
            satrec=None,  # type: ignore[arg-type]
        )

    clusters = infer_raan_planes(
        [record(1, 359.8, 20), record(2, 0.2, 10), record(3, 120.0, 10), record(4, 120.4, 20)],
    )
    assert len(clusters) == 2
    assert {item.norad_id for item in clusters[0]} == {1, 2}


def test_frozen_tgnn_edge_scaling() -> None:
    assert scale_tgnn_edge_features(5000.0, 50.0, "inter_plane", "optical") == (
        0.5,
        0.5,
        0.5,
        1.0,
    )
    assert scale_tgnn_edge_features(10_000.0, -20.0, "seam_link", "rf") == (
        1.0,
        -0.2,
        1.0,
        0.0,
    )


def test_config_rejects_non_phase4_episode_count(tmp_path) -> None:
    try:
        Phase4AConfig(tmp_path, episode_count=299)
    except ValueError as exc:
        assert "exactly 300" in str(exc)
    else:
        raise AssertionError("invalid Phase 4A episode count was accepted")
