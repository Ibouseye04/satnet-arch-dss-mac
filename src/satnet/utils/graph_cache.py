"""Temporal graph sequence caching for the SatNet pipeline.

Provides deterministic cache-keying, serialization, and strict metadata
validation to prevent silent cache corruption across target labels.

Cache layout::

    <cache_dir>/
        <cache_key>.pt          # serialized target-agnostic list[Data]
        <cache_key>.meta.json   # required metadata

Cache keys are SHA-256 hashes of the normalized JSON representation of
the parameters that determine a generated graph sequence.
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

# Backward-incompatible cache contract version.
CACHE_SCHEMA_VERSION = 1

# Only target-agnostic payloads are supported.
PAYLOAD_MODE_TARGET_AGNOSTIC = "target_agnostic"

_REQUIRED_METADATA_FIELDS = (
    "cache_schema_version",
    "generator_fingerprint",
    "payload_mode",
    "generator_config",
)


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


def extract_cache_key_config(sample_config: dict[str, Any]) -> dict[str, Any]:
    """Return cache-identity fields from a sample configuration."""
    return {
        key: sample_config[key]
        for key in _CACHE_KEY_FIELDS
        if key in sample_config
    }


def make_sample_cache_key(sample_config: dict[str, Any]) -> str:
    """Build a deterministic cache key from sample parameters."""
    subset = extract_cache_key_config(sample_config)
    canonical = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def make_cache_metadata(
    *,
    generator_fingerprint: str,
    generator_config: dict[str, Any],
    payload_mode: str = PAYLOAD_MODE_TARGET_AGNOSTIC,
) -> dict[str, Any]:
    """Build metadata for a cache entry."""
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "generator_fingerprint": generator_fingerprint,
        "payload_mode": payload_mode,
        "generator_config": generator_config,
    }


# ── serialization ───────────────────────────────────────────────────

def _import_torch():  # noqa: ANN202
    import torch
    return torch


def _meta_path(cache_dir: str | Path, cache_key: str) -> Path:
    return Path(cache_dir) / f"{cache_key}.meta.json"


def save_graph_sequence(
    data_list: list[Any],
    cache_dir: str | Path,
    cache_key: str,
    metadata: dict[str, Any] | None = None,
) -> float:
    """Serialize a graph sequence to ``<cache_dir>/<cache_key>.pt``."""
    torch = _import_torch()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pt_path = cache_dir / f"{cache_key}.pt"
    t0 = time.monotonic()
    torch.save(data_list, pt_path)
    write_time = time.monotonic() - t0

    if metadata is not None:
        with open(_meta_path(cache_dir, cache_key), "w") as f:
            json.dump(metadata, f, indent=2)

    logger.debug("Cache write %.3fs -> %s", write_time, pt_path)
    return write_time


def load_cache_metadata(cache_dir: str | Path, cache_key: str) -> dict[str, Any] | None:
    """Load cache metadata if present."""
    meta_path = _meta_path(cache_dir, cache_key)
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def load_graph_sequence(
    cache_dir: str | Path,
    cache_key: str,
) -> tuple[list[Any] | None, float, dict[str, Any] | None]:
    """Load a cached graph sequence.

    Returns ``(data_list, load_time, metadata)`` on hit,
    ``(None, 0.0, None)`` on miss.
    """
    torch = _import_torch()
    pt_path = Path(cache_dir) / f"{cache_key}.pt"
    if not pt_path.exists():
        return None, 0.0, None

    t0 = time.monotonic()
    data_list = torch.load(pt_path, weights_only=False)
    load_time = time.monotonic() - t0
    metadata = load_cache_metadata(cache_dir, cache_key)
    logger.debug("Cache hit  %.3fs <- %s", load_time, pt_path)
    return data_list, load_time, metadata


def cache_exists(cache_dir: str | Path, cache_key: str) -> bool:
    return (Path(cache_dir) / f"{cache_key}.pt").exists()


def _payload_contains_labels(data_list: list[Any]) -> bool:
    for item in data_list:
        if isinstance(item, dict) and "y" in item:
            return True
        if getattr(item, "y", None) is not None:
            return True
    return False


def validate_cache_entry(
    data_list: list[Any],
    metadata: dict[str, Any] | None,
    *,
    expected_generator_fingerprint: str,
    expected_generator_config: dict[str, Any] | None = None,
) -> None:
    """Validate cache metadata and payload contract."""
    if metadata is None:
        if _payload_contains_labels(data_list):
            raise ValueError(
                "Legacy cache artifact detected: payload contains embedded labels "
                "and has no metadata. Delete this cache entry and regenerate "
                "with --write-cache."
            )
        raise ValueError(
            "Cache metadata is missing. Delete this cache entry and regenerate "
            "with --write-cache."
        )

    missing = [key for key in _REQUIRED_METADATA_FIELDS if key not in metadata]
    if missing:
        raise ValueError(
            f"Cache metadata missing required fields: {missing}. Delete this cache "
            "entry and regenerate with --write-cache."
        )

    schema_version = int(metadata["cache_schema_version"])
    if schema_version != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported cache schema version {schema_version}; expected "
            f"{CACHE_SCHEMA_VERSION}. Delete this cache entry and regenerate "
            "with --write-cache."
        )

    payload_mode = str(metadata["payload_mode"])
    if payload_mode != PAYLOAD_MODE_TARGET_AGNOSTIC:
        raise ValueError(
            f"Unsupported cache payload_mode '{payload_mode}'. Only "
            f"'{PAYLOAD_MODE_TARGET_AGNOSTIC}' is supported."
        )

    if _payload_contains_labels(data_list):
        raise ValueError(
            "Cache payload contains embedded labels (`y`). Target-bound cache "
            "payloads are not supported. Delete this cache entry and regenerate "
            "with --write-cache."
        )

    actual_fingerprint = str(metadata["generator_fingerprint"])
    if actual_fingerprint != expected_generator_fingerprint:
        raise ValueError(
            "Cache generator fingerprint mismatch. Delete this cache entry and "
            "regenerate with --write-cache."
        )

    if expected_generator_config is not None:
        cached_config = metadata.get("generator_config")
        if cached_config != expected_generator_config:
            raise ValueError(
                "Cache generator configuration mismatch. Delete this cache entry "
                "and regenerate with --write-cache."
            )


# ── telemetry record ────────────────────────────────────────────────

def make_cache_telemetry(
    cache_key: str,
    hit: bool,
    load_time: float = 0.0,
    write_time: float = 0.0,
    generation_time: float = 0.0,
) -> dict[str, Any]:
    """Build a telemetry dict suitable for experiment logging."""
    return {
        "cache_key": cache_key,
        "cache_hit": hit,
        "cache_load_time_s": round(load_time, 4),
        "cache_write_time_s": round(write_time, 4),
        "graph_generation_time_s": round(generation_time, 4),
    }
