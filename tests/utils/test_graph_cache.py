"""Tests for satnet.utils.graph_cache."""

from __future__ import annotations

import json

import pytest

from satnet.utils.graph_cache import make_sample_cache_key

torch = pytest.importorskip("torch")

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
        from torch_geometric.data import Data

        data_list = [
            Data(x=torch.randn(5, 3), edge_index=torch.randint(0, 5, (2, 8)))
            for _ in range(3)
        ]
        key = "test_key_abc123"
        wt = save_graph_sequence(data_list, tmp_path, key)
        assert wt >= 0.0
        assert cache_exists(tmp_path, key)

        loaded, lt = load_graph_sequence(tmp_path, key)
        assert loaded is not None
        assert len(loaded) == 3
        assert lt >= 0.0

    def test_miss_returns_none(self, tmp_path) -> None:
        loaded, lt = load_graph_sequence(tmp_path, "nonexistent")
        assert loaded is None
        assert lt == 0.0

    def test_metadata_written(self, tmp_path) -> None:
        data_list = [torch.tensor([1, 2, 3])]
        save_graph_sequence(data_list, tmp_path, "k", metadata={"note": "test"})
        meta_path = tmp_path / "k.meta.json"
        assert meta_path.exists()
        assert json.loads(meta_path.read_text())["note"] == "test"


class TestCacheTelemetry:
    def test_keys(self) -> None:
        t = make_cache_telemetry("abc", hit=True, load_time=0.123)
        assert t["cache_key"] == "abc"
        assert t["cache_hit"] is True
        assert t["cache_load_time_s"] == 0.123
