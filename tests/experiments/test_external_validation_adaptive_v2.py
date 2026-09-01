from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from satnet.experiments.external_validation import adaptive_v2
from satnet.experiments.external_validation.phase4a import (
    ADJACENT_SEARCH_K,
    FAILURE_MODEL,
    ISL_POLICY,
    MAX_INTER_PLANE_LINKS_PER_SAT,
    EXTERNAL_RF_FEATURE_ORDER,
    EXTERNAL_TGNN_EDGE_FEATURE_ORDER,
    EXTERNAL_TGNN_NODE_FEATURE_ORDER,
)


def test_adaptive_topology_identity_is_exact() -> None:
    assert ISL_POLICY == "grid_adaptive"
    assert ADJACENT_SEARCH_K == 1
    assert MAX_INTER_PLANE_LINKS_PER_SAT == 1
    assert FAILURE_MODEL == "persistent_temporal_union_edges_v1"


def test_external_feature_contract_is_exact() -> None:
    assert EXTERNAL_RF_FEATURE_ORDER == (
        "num_planes",
        "sats_per_plane",
        "altitude_km",
        "inclination_deg",
        "satellite_node_failure_probability",
        "satellite_edge_failure_probability",
    )
    assert EXTERNAL_TGNN_NODE_FEATURE_ORDER == (
        "plane_idx_normalized",
        "sat_in_plane_normalized",
        "node_exists_constant",
    )
    assert EXTERNAL_TGNN_EDGE_FEATURE_ORDER == (
        "distance_km_scaled_10000",
        "margin_db_scaled_100",
        "link_type_code_scaled_2",
        "link_mode_binary",
    )


def test_topology_identity_changes_with_graph_edges() -> None:
    edge = {"source": 0, "target": 1, "distance_km": 100.0, "margin_db": 20.0, "link_type": "inter_plane", "link_mode": "optical"}
    assert adaptive_v2.topology_identity([edge]) != adaptive_v2.topology_identity([{**edge, "target": 2}])


def test_output_root_must_be_absent_and_protected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="must be absent"):
        adaptive_v2.assert_output_root_safe(tmp_path, (tmp_path / "protected",))

    protected = tmp_path / "protected"
    protected.mkdir()
    with pytest.raises(RuntimeError, match="overlaps protected"):
        adaptive_v2.assert_output_root_safe(protected / "child", (protected,))


def test_historical_derived_artifacts_are_not_reusable(tmp_path: Path) -> None:
    historical = tmp_path / "historical"
    derived = historical / "episodes" / "tgnn_sequences" / "episode_0000.json"
    derived.parent.mkdir(parents=True)
    derived.touch()
    with pytest.raises(RuntimeError, match="topology/model-derived"):
        adaptive_v2.reject_historical_derived_artifact(derived, historical)
    assert "episodes/external_rf_dataset.csv" in adaptive_v2.FORBIDDEN_DERIVED_PATHS
    assert "episodes/tgnn_sequences" in adaptive_v2.FORBIDDEN_DERIVED_PATHS
    assert adaptive_v2.HISTORICAL_EXTERNAL_BUNDLE_SHA != adaptive_v2.ADAPTIVE_DATASET_BUNDLE_SHA


def test_adaptive_topology_metadata_fails_closed() -> None:
    adaptive_v2.validate_adaptive_topology_identity({
        "isl_policy": "grid_adaptive",
        "k": 1,
        "endpoint_capacity": 1,
        "temporal_failure_edge_policy": "persistent_temporal_union_edges_v1",
    })
    with pytest.raises(RuntimeError, match="topology identity"):
        adaptive_v2.validate_adaptive_topology_identity({
            "isl_policy": "grid_fixed",
            "k": 1,
            "endpoint_capacity": 1,
            "temporal_failure_edge_policy": "persistent_temporal_union_edges_v1",
        })


def test_frozen_model_manifest_rejects_runner_up_and_seed_mismatch() -> None:
    manifest = {
        "status": "completed",
        "task": "rf_space_regression",
        "seed": 42,
        "selected_config_id": "rf_018",
        "dataset_bundle_hash": adaptive_v2.ADAPTIVE_DATASET_BUNDLE_SHA,
        "phase3_candidate_set_hash": adaptive_v2.PHASE3_CANDIDATE_SET_SHA,
        "selection_freeze_sha256": adaptive_v2.RF_SELECTION_FREEZE_SHA,
        "phase4_tooling_git_sha": adaptive_v2.PHASE4_TOOLING_SHA,
        "test_accessed": False,
    }
    adaptive_v2.validate_frozen_model_manifest(manifest, "rf_space_regression", 42)
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        adaptive_v2.validate_frozen_model_manifest({**manifest, "selected_config_id": "rf_017"}, "rf_space_regression", 42)
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        adaptive_v2.validate_frozen_model_manifest(manifest, "rf_space_regression", 123)


def test_frozen_model_identity_contract_is_explicit() -> None:
    assert adaptive_v2.TASK_CONFIG_IDS == {
        "rf_integrated_classification": "rf_015",
        "rf_integrated_regression_mean": "rf_036",
        "rf_integrated_regression_min": "rf_018",
        "rf_space_classification": "rf_020",
        "rf_space_regression": "rf_018",
        "tgnn_space_classification": "tgnn_010",
        "tgnn_space_regression": "tgnn_012",
    }
    assert adaptive_v2.SEEDS == (42, 123, 456, 789, 2026)
    assert adaptive_v2.PRIMARY_SEED == 42


def test_preflight_source_has_no_model_inference_or_training_path() -> None:
    source = inspect.getsource(adaptive_v2)
    for forbidden in (".fit(", ".backward(", "optimizer.step", "torch.optim", "Healthy", "Watchlist", "Critical"):
        assert forbidden not in source


def test_adaptive_behavior_is_exercised_by_production_code() -> None:
    evidence = adaptive_v2.adaptive_behavioral_evidence()
    assert evidence["production_function"].endswith("_compute_grid_plus_isls")
    assert evidence["fixed_and_adaptive_differ"] is True
    assert evidence["adaptive_inter_plane_edge_count"] > 0
    assert evidence["max_observed_inter_plane_degree"] <= 1
    assert evidence["los_and_physics_filtering"] is True
    assert evidence["candidate_replacement_evidence"] is True
    assert evidence["graph_identity_propagates_to_sequence"] is True
    assert evidence["target_derived_from_adaptive_graph"] is True


def test_historical_episode_identity_is_audit_only() -> None:
    result = adaptive_v2.verify_historical_episode_identity(
        Path(r"C:\Users\johns\external\satnet-real-external-validation-v1")
    )
    assert result["episode_count"] == 300
    assert result["start"] == "2024-02-01T12:00:00Z"
    assert result["end"] == "2025-12-31T12:00:00Z"
    assert result["reuse_permitted"] is False
