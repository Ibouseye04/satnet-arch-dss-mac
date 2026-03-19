"""Tests for satnet.utils.graph_cache."""

from __future__ import annotations

import json

import pytest

from satnet.utils.graph_cache import (
    CACHE_SCHEMA_VERSION,
    make_cache_metadata,
    make_sample_cache_key,
    validate_cache_entry,
)

torch = pytest.importorskip("torch")
Data = pytest.importorskip("torch_geometric.data").Data

from satnet.utils.graph_cache import (  # noqa: E402
    cache_exists,
    load_graph_sequence,
    make_cache_telemetry,
    save_graph_sequence,
)


class TestMakeSampleCacheKey:
    def test_deterministic(self) -> None:
        cfg = {"num_planes": 4, "sats_per_plane": 6, "altitude_km": 550.0}
        assert make_sample_cache_key(cfg) == make_sample_cache_key(cfg)

    def test_key_order_independent(self) -> None:
        a = {"num_planes": 4, "altitude_km": 550.0}
        b = {"altitude_km": 550.0, "num_planes": 4}
        assert make_sample_cache_key(a) == make_sample_cache_key(b)

    def test_different_values_differ(self) -> None:
        a = {"num_planes": 4, "sats_per_plane": 6}
        b = {"num_planes": 5, "sats_per_plane": 6}
        assert make_sample_cache_key(a) != make_sample_cache_key(b)

    def test_ignores_unknown_fields(self) -> None:
        a = {"num_planes": 4}
        b = {"num_planes": 4, "extra_field": "ignored"}
        assert make_sample_cache_key(a) == make_sample_cache_key(b)


class TestSaveLoadGraphSequence:
    def test_roundtrip(self, tmp_path) -> None:
        data_list = [
            Data(x=torch.randn(5, 3), edge_index=torch.randint(0, 5, (2, 8)))
            for _ in range(3)
        ]
        key = "test_key_abc123"
        metadata = make_cache_metadata(
            sample_cache_key=key,
            generator_provenance="test-generator:v1",
            generator_config={"num_planes": 4},
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
            expected_generator_config={"num_planes": 4},
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
