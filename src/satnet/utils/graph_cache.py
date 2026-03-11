"""Temporal graph sequence caching for the SatNet pipeline.

Provides deterministic cache-keying, serialization, and a cache
directory convention so temporal graph sequences do not need to be
rebuilt on every training run.

Cache layout::

    <cache_dir>/
        <cache_key>.pt          # serialized list[Data]
        <cache_key>.meta.json   # lightweight metadata

Cache keys are SHA-256 hashes of the normalized JSON representation
of the parameters that determine a generated graph sequence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default cache directory (relative to project root)
DEFAULT_CACHE_DIR = "artifacts/graph_cache"


# ── cache key ───────────────────────────────────────────────────────

# Fields that determine the generated temporal graph sequence.
# Order matters only for documentation; the JSON is sorted.
_CACHE_KEY_FIELDS = (
    "num_planes",
    "sats_per_plane",
    "inclination_deg",
    "altitude_km",
    "phasing_factor",
    "duration_minutes",
    "step_seconds",
    "max_isl_distance_km",
    "node_failure_prob",
    "edge_failure_prob",
    "seed",
    "epoch_iso",
    "failed_nodes_json",
    "failed_edges_json",
)


def make_sample_cache_key(sample_config: dict) -> str:
    """Build a deterministic cache key from sample parameters.

    Only the fields in ``_CACHE_KEY_FIELDS`` that are present in
    *sample_config* are included.  Values are JSON-normalized (sorted
    keys, no extra whitespace) before hashing.
    """
    subset = {
        k: sample_config[k]
        for k in _CACHE_KEY_FIELDS
        if k in sample_config
    }
    canonical = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# ── serialization ───────────────────────────────────────────────────

def _import_torch():  # noqa: ANN202
    import torch
    return torch


def save_graph_sequence(
    data_list: list[Any],
    cache_dir: str | Path,
    cache_key: str,
    metadata: dict | None = None,
) -> float:
    """Serialize a graph sequence to ``<cache_dir>/<cache_key>.pt``.

    Returns the wall-clock write time in seconds.
    """
    torch = _import_torch()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pt_path = cache_dir / f"{cache_key}.pt"
    t0 = time.monotonic()
    torch.save(data_list, pt_path)
    write_time = time.monotonic() - t0

    if metadata is not None:
        meta_path = cache_dir / f"{cache_key}.meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    logger.debug("Cache write %.3fs → %s", write_time, pt_path)
    return write_time


def load_graph_sequence(
    cache_dir: str | Path,
    cache_key: str,
) -> tuple[list[Any] | None, float]:
    """Load a cached graph sequence.

    Returns ``(data_list, load_time)`` on hit, ``(None, 0.0)`` on miss.
    """
    torch = _import_torch()
    pt_path = Path(cache_dir) / f"{cache_key}.pt"
    if not pt_path.exists():
        return None, 0.0

    t0 = time.monotonic()
    data_list = torch.load(pt_path, weights_only=False)
    load_time = time.monotonic() - t0
    logger.debug("Cache hit  %.3fs ← %s", load_time, pt_path)
    return data_list, load_time


def cache_exists(cache_dir: str | Path, cache_key: str) -> bool:
    return (Path(cache_dir) / f"{cache_key}.pt").exists()


# ── telemetry record ────────────────────────────────────────────────

def make_cache_telemetry(
    cache_key: str,
    hit: bool,
    load_time: float = 0.0,
    write_time: float = 0.0,
    generation_time: float = 0.0,
) -> dict:
    """Build a telemetry dict suitable for experiment logging."""
    return {
        "cache_key": cache_key,
        "cache_hit": hit,
        "cache_load_time_s": round(load_time, 4),
        "cache_write_time_s": round(write_time, 4),
        "graph_generation_time_s": round(generation_time, 4),
    }
