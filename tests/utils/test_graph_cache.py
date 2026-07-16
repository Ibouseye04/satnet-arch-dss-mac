"""Tests for satnet.utils.graph_cache."""

from __future__ import annotations

import json

import pytest

import satnet.utils.graph_cache as graph_cache
from satnet.utils.graph_cache import (
    CACHE_SCHEMA_VERSION,
    cache_exists,
    load_graph_sequence,
    make_cache_metadata,
    make_cache_telemetry,
    make_sample_cache_key,
    save_graph_sequence,
    validate_cache_entry,
)

torch = pytest.importorskip("torch")
Data = pytest.importorskip("torch_geometric.data").Data

BASE_CACHE_CONFIG = {
    "num_planes": 4,
    "sats_per_plane": 6,
    "inclination_deg": 53.0,
    "altitude_km": 550.0,
    "phasing_factor": 1,
    "duration_minutes": 2,
    "step_seconds": 60,
    "num_steps": 3,
    "max_isl_distance_km": 10000.0,
    "isl_policy": "grid_adaptive",
    "adjacent_search_k": 1,
    "max_inter_plane_links_per_sat": 1,
    "node_failure_prob": 0.01,
    "edge_failure_prob": 0.02,
    "failure_model": "persistent_temporal_union_edges_v1",
    "seed": 42,
    "epoch_iso": "2000-01-01T12:00:00+00:00",
    "failed_nodes_json": "[]",
    "failed_edges_json": "[]",
    "schema_version": 2,
    "dataset_version": "tier1_temporal_connectivity_v2",
    "orbital_engine": "sgp4",
    "physics_model_version": "physics-v2",
    "link_budget_config": {"optical_wavelength_m": 1550e-9},
}


def _cache_config(**updates):
    config = dict(BASE_CACHE_CONFIG)
    config.update(updates)
    return config


def _structural_graph(time_step: int) -> Data:
    return Data(
        x=torch.randn(5, 3),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 4), dtype=torch.float),
        time_step=torch.tensor([time_step], dtype=torch.long),
        num_nodes=5,
        isl_policy="grid_adaptive",
        adjacent_search_k=1,
        max_inter_plane_links_per_sat=1,
        failure_model="persistent_temporal_union_edges_v1",
    )


class TestMakeSampleCacheKey:
    def test_deterministic(self) -> None:
        cfg = _cache_config()
        assert make_sample_cache_key(cfg) == make_sample_cache_key(cfg)

    def test_key_order_independent(self) -> None:
        a = _cache_config()
        b = dict(reversed(list(a.items())))
        assert make_sample_cache_key(a) == make_sample_cache_key(b)

    def test_different_values_differ(self) -> None:
        assert make_sample_cache_key(_cache_config(num_planes=4)) != make_sample_cache_key(
            _cache_config(num_planes=5)
        )

    def test_ignores_unknown_fields(self) -> None:
        a = _cache_config()
        b = _cache_config(extra_field="ignored")
        assert make_sample_cache_key(a) == make_sample_cache_key(b)

    def test_isl_policy_changes_key(self) -> None:
        fixed = _cache_config(isl_policy="grid_fixed")
        adaptive = _cache_config(isl_policy="grid_adaptive")
        assert make_sample_cache_key(fixed) != make_sample_cache_key(adaptive)

    def test_failure_model_changes_key(self) -> None:
        t0 = _cache_config(failure_model="persistent_t0_edges_v1")
        temporal_union = _cache_config(
            failure_model="persistent_temporal_union_edges_v1"
        )
        assert make_sample_cache_key(t0) != make_sample_cache_key(temporal_union)


class TestSaveLoadGraphSequence:
    def test_roundtrip(self, tmp_path) -> None:
        data_list = [_structural_graph(time_step) for time_step in range(3)]
        key = "test_key_abc123"
        generator_config = _cache_config()
        metadata = make_cache_metadata(
            sample_cache_key=key,
            generator_provenance="test-generator:v1",
            generator_config=generator_config,
        )
        wt = save_graph_sequence(data_list, tmp_path, key, metadata=metadata)
        assert wt >= 0.0
        assert cache_exists(tmp_path, key)

        loaded, lt, loaded_metadata = load_graph_sequence(tmp_path, key)
        assert loaded is not None
        assert len(loaded) == 3
        assert lt >= 0.0
        validate_cache_entry(
            loaded,
            loaded_metadata,
            expected_sample_cache_key=key,
            expected_generator_provenance="test-generator:v1",
            expected_generator_config=generator_config,
        )

    def test_miss_returns_none(self, tmp_path) -> None:
        loaded, lt, metadata = load_graph_sequence(tmp_path, "nonexistent")
        assert loaded is None
        assert lt == 0.0
        assert metadata is None

    def test_metadata_written(self, tmp_path) -> None:
        data_list = [torch.tensor([1, 2, 3])]
        save_graph_sequence(data_list, tmp_path, "k", metadata={"note": "test"})
        meta_path = tmp_path / "k.meta.json"
        assert meta_path.exists()
        assert json.loads(meta_path.read_text())["note"] == "test"

    def test_failed_write_leaves_no_partial_entry(self, tmp_path, monkeypatch) -> None:
        key = "failed_write"
        metadata = make_cache_metadata(
            sample_cache_key=key,
            generator_provenance="test-generator:v1",
            generator_config=_cache_config(),
        )

        def fail_dump(*args, **kwargs):
            raise RuntimeError("simulated metadata write failure")

        monkeypatch.setattr(graph_cache.json, "dump", fail_dump)
        with pytest.raises(RuntimeError, match="simulated metadata write failure"):
            save_graph_sequence(
                [_structural_graph(time_step) for time_step in range(3)],
                tmp_path,
                key,
                metadata=metadata,
            )

        assert not (tmp_path / f"{key}.pt").exists()
        assert not (tmp_path / f"{key}.meta.json").exists()
        assert not (tmp_path / f"{key}.pt.pending").exists()
        assert not (tmp_path / f"{key}.meta.json.pending").exists()

    def test_metadata_distinguishes_sample_identity_and_generator_provenance(self) -> None:
        metadata = make_cache_metadata(
            sample_cache_key="cache-key-123",
            generator_provenance="graph-generator:v7",
            generator_config={"num_planes": 4},
        )
        assert metadata["sample_cache_key"] == "cache-key-123"
        assert metadata["generator_provenance"] == "graph-generator:v7"
        assert metadata["sample_cache_key"] != metadata["generator_provenance"]

    def test_validate_rejects_missing_metadata(self, tmp_path) -> None:
        key = "missing_meta"
        data_list = [Data(x=torch.randn(4, 2), edge_index=torch.randint(0, 4, (2, 4)))]
        save_graph_sequence(data_list, tmp_path, key)
        loaded, _, metadata = load_graph_sequence(tmp_path, key)
        assert loaded is not None

        with pytest.raises(ValueError, match="metadata is missing"):
            validate_cache_entry(
                loaded,
                metadata,
                expected_sample_cache_key=key,
                expected_generator_provenance="test-generator:v1",
                expected_generator_config={"num_planes": 4},
            )

    def test_validate_rejects_stale_schema_version(self, tmp_path) -> None:
        key = "stale_schema"
        data_list = [Data(x=torch.randn(4, 2), edge_index=torch.randint(0, 4, (2, 4)))]
        metadata = make_cache_metadata(
            sample_cache_key=key,
            generator_provenance="test-generator:v1",
            generator_config={"num_planes": 4},
        )
        save_graph_sequence(data_list, tmp_path, key, metadata=metadata)
        meta_path = tmp_path / f"{key}.meta.json"
        parsed = json.loads(meta_path.read_text())
        parsed["cache_schema_version"] = CACHE_SCHEMA_VERSION - 1
        meta_path.write_text(json.dumps(parsed))

        loaded, _, loaded_metadata = load_graph_sequence(tmp_path, key)
        assert loaded is not None
        with pytest.raises(ValueError, match="Unsupported cache schema version"):
            validate_cache_entry(
                loaded,
                loaded_metadata,
                expected_sample_cache_key=key,
                expected_generator_provenance="test-generator:v1",
                expected_generator_config={"num_planes": 4},
            )

    def test_validate_rejects_target_bound_payload(self, tmp_path) -> None:
        key = "bound_payload"
        data_list = [
            Data(
                x=torch.randn(4, 2),
                edge_index=torch.randint(0, 4, (2, 4)),
                y=torch.tensor([1.0]),
            )
        ]
        metadata = make_cache_metadata(
            sample_cache_key=key,
            generator_provenance="test-generator:v1",
            generator_config={"num_planes": 4},
        )
        save_graph_sequence(data_list, tmp_path, key, metadata=metadata)
        loaded, _, loaded_metadata = load_graph_sequence(tmp_path, key)
        assert loaded is not None

        with pytest.raises(ValueError, match="contains embedded labels"):
            validate_cache_entry(
                loaded,
                loaded_metadata,
                expected_sample_cache_key=key,
                expected_generator_provenance="test-generator:v1",
                expected_generator_config={"num_planes": 4},
            )

    def test_validate_rejects_legacy_label_payload_without_metadata(self, tmp_path) -> None:
        key = "legacy_labeled"
        data_list = [
            Data(
                x=torch.randn(4, 2),
                edge_index=torch.randint(0, 4, (2, 4)),
                y=torch.tensor([0.0]),
            )
        ]
        save_graph_sequence(data_list, tmp_path, key)
        loaded, _, loaded_metadata = load_graph_sequence(tmp_path, key)
        assert loaded is not None
        assert loaded_metadata is None

        with pytest.raises(ValueError, match="Legacy cache artifact detected"):
            validate_cache_entry(
                loaded,
                loaded_metadata,
                expected_sample_cache_key=key,
                expected_generator_provenance="test-generator:v1",
                expected_generator_config={"num_planes": 4},
            )

    def test_validate_rejects_generator_provenance_mismatch(self, tmp_path) -> None:
        key = "provenance_mismatch"
        data_list = [Data(x=torch.randn(4, 2), edge_index=torch.randint(0, 4, (2, 4)))]
        metadata = make_cache_metadata(
            sample_cache_key=key,
            generator_provenance="test-generator:v1",
            generator_config={"num_planes": 4},
        )
        save_graph_sequence(data_list, tmp_path, key, metadata=metadata)
        loaded, _, loaded_metadata = load_graph_sequence(tmp_path, key)
        assert loaded is not None

        with pytest.raises(ValueError, match="generator provenance mismatch"):
            validate_cache_entry(
                loaded,
                loaded_metadata,
                expected_sample_cache_key=key,
                expected_generator_provenance="test-generator:v2",
                expected_generator_config={"num_planes": 4},
            )


class TestCacheTelemetry:
    def test_keys(self) -> None:
        t = make_cache_telemetry("abc", hit=True, load_time=0.123)
        assert t["cache_key"] == "abc"
        assert t["cache_hit"] is True
        assert t["cache_load_time_s"] == 0.123
