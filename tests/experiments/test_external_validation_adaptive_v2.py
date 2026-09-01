from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from satnet.experiments.external_validation import adaptive_v2
from satnet.experiments.external_validation import phase4a
from satnet.experiments.external_validation.phase4a import (
    ADJACENT_SEARCH_K,
    EXTERNAL_RF_FEATURE_ORDER,
    EXTERNAL_STATISTICAL_CONTRACT,
    EXTERNAL_TASKS,
    EXTERNAL_TGNN_EDGE_FEATURE_ORDER,
    EXTERNAL_TGNN_NODE_FEATURE_ORDER,
    FAILURE_MODEL,
    ISL_POLICY,
    MAX_INTER_PLANE_LINKS_PER_SAT,
    OrbitalRecord,
    Phase4AConfig,
    SelectedSatellite,
    SatellitePosition,
    _build_graph,
    _contract,
    _json_bytes,
    _persist_source_manifest,
    _sha256_bytes,
    _write_csv,
    _write_json,
    _graph_snapshot,
    _target_from_snapshots,
    audit_constructed_adaptive_artifacts,
)
from satnet.network.hypatia_adapter import HypatiaAdapter


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
    assert evidence["evidence_scope"].startswith("synthetic representative topology only")


def _test_selected(count: int = 6, unavailable: set[int] | None = None) -> tuple[tuple[SelectedSatellite, ...], list[SatellitePosition]]:
    unavailable = unavailable or set()
    selected = tuple(
        SelectedSatellite(
            norad_id=index,
            plane_idx=index // 3,
            sat_in_plane=index % 3,
            orbit=OrbitalRecord(
                norad_id=index,
                epoch=phase4a.TARGET_START,
                inclination_deg=53.0,
                raan_deg=float(index // 3),
                eccentricity=0.001,
                arg_perigee_deg=0.0,
                mean_anomaly_deg=float(index),
                mean_motion_rev_day=15.5,
                mean_motion_dot_rev_day2=0.0,
                bstar=0.0,
                altitude_km=550.0,
                satrec=None,
            ),
            status="anomalous" if index in unavailable else "operational",
            status_available=index not in unavailable,
            is_isl_capable=True,
            tle_age_seconds=0.0,
        )
        for index in range(count)
    )
    with HypatiaAdapter(num_planes=2, sats_per_plane=3, inclination_deg=53.0, altitude_km=550.0, phasing_factor=1) as adapter:
        adapter.generate_tles()
        positions = adapter.get_positions_at_step(0)
    return selected, positions


def test_source_contract_hash_uses_canonical_bytes_before_output_exists(tmp_path: Path) -> None:
    output_root = tmp_path / "absent-output"
    manifest = {"manifest_version": "test", "sources": []}
    config = Phase4AConfig(output_root, source_root=tmp_path / "source")
    assert not (output_root / "external_source_manifest.json").exists()
    contract = _contract(config, manifest)
    assert contract["source_manifest_sha256"] == _sha256_bytes(_json_bytes(manifest))
    assert not (output_root / "external_source_manifest.json").exists()
    persisted = _persist_source_manifest(output_root, manifest)
    assert persisted == contract["source_manifest_sha256"]
    assert phase4a._sha256_file(output_root / "external_source_manifest.json") == persisted


def test_unavailable_nodes_only_remove_incident_edges_and_match_features(tmp_path: Path) -> None:
    available, positions = _test_selected()
    unavailable, _ = _test_selected(unavailable={0})
    config = Phase4AConfig(tmp_path / "output")
    nominal, _, _ = _build_graph(available, phase4a.TARGET_START, config, positions_override=positions)
    degraded, _, _ = _build_graph(unavailable, phase4a.TARGET_START, config, positions_override=positions)
    assert nominal.number_of_edges() > 0
    assert degraded.number_of_edges() < nominal.number_of_edges()
    assert all(0 not in edge for edge in degraded.edges())
    nominal_survivor_edges = {edge for edge in nominal.edges() if 0 not in edge}
    assert nominal_survivor_edges == set(degraded.edges())
    assert degraded.nodes[0]["exists"] is False
    assert degraded.number_of_nodes() == 6
    snapshot, _, _ = _graph_snapshot(unavailable, phase4a.TARGET_START, config, positions_override=positions)
    assert snapshot["nodes"][0]["features"][2] == 0.0
    assert all(node["features"][2] == 1.0 for node in snapshot["nodes"][1:])


def test_gcc_target_uses_exact_original_denominator() -> None:
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    graph.add_edge(0, 1)
    regression, classification, fractions = _target_from_snapshots([({}, graph)], 4, 0.8)
    assert fractions == [0.5]
    assert regression == 0.5
    assert classification is True


def test_multiple_unavailable_nodes_preserve_unrelated_links(tmp_path: Path) -> None:
    selected, positions = _test_selected(unavailable={0, 3})
    graph, _, _ = _build_graph(selected, phase4a.TARGET_START, Phase4AConfig(tmp_path), positions_override=positions)
    assert graph.number_of_nodes() == 6
    assert graph.nodes[0]["exists"] is False
    assert graph.nodes[3]["exists"] is False
    assert all(0 not in edge and 3 not in edge for edge in graph.edges())


def test_forced_grid_fixed_runtime_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selected, positions = _test_selected()
    original = phase4a._compute_grid_plus_isls

    def force_fixed(*args, **kwargs):
        kwargs["isl_policy"] = "grid_fixed"
        return original(*args, **kwargs)

    monkeypatch.setattr(phase4a, "_compute_grid_plus_isls", force_fixed)
    with pytest.raises(RuntimeError, match="runtime policy mismatch"):
        _build_graph(selected, phase4a.TARGET_START, Phase4AConfig(tmp_path), positions_override=positions)


def test_external_scope_and_statistical_contract_are_frozen() -> None:
    assert EXTERNAL_TASKS == (
        "rf_space_classification",
        "tgnn_space_classification",
        "rf_space_regression",
        "tgnn_space_regression",
    )
    adaptive_v2.validate_external_task_scope(list(EXTERNAL_TASKS))
    with pytest.raises(RuntimeError, match="exactly"):
        adaptive_v2.validate_external_task_scope(["rf_integrated_classification"])
    adaptive_v2.validate_external_statistical_contract(EXTERNAL_STATISTICAL_CONTRACT)
    assert EXTERNAL_STATISTICAL_CONTRACT["paired_unit"] == "episode_id"
    assert EXTERNAL_STATISTICAL_CONTRACT["replicates"] == 2000
    assert EXTERNAL_STATISTICAL_CONTRACT["bootstrap_seed"] == 20260820
    assert EXTERNAL_STATISTICAL_CONTRACT["primary_comparison"]["difference"] == "RF MAE minus TGNN MAE"
    assert EXTERNAL_STATISTICAL_CONTRACT["classification"] == "descriptive_only"
    assert EXTERNAL_STATISTICAL_CONTRACT["classification_inferential_procedure"] is None


def test_construction_module_has_no_inference_training_or_risk_tuning_path() -> None:
    source = inspect.getsource(phase4a)
    for forbidden in (".fit(", ".backward(", "optimizer.step", "torch.optim", "risk_bin", "threshold_tuning", "predict("):
        assert forbidden not in source
    assert "descriptive_only" in source
    assert "classification_inferential_procedure" in source


def test_historical_episode_identity_is_audit_only() -> None:
    result = adaptive_v2.verify_historical_episode_identity(
        Path(r"C:\Users\johns\external\satnet-real-external-validation-v1")
    )
    assert result["episode_count"] == 300
    assert result["start"] == "2024-02-01T12:00:00Z"
    assert result["end"] == "2025-12-31T12:00:00Z"
    assert result["reuse_permitted"] is False
