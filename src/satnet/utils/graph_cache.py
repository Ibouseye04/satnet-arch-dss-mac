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
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default cache directory (relative to project root)
DEFAULT_CACHE_DIR = "artifacts/graph_cache"

# Backward-incompatible cache contract version.
CACHE_SCHEMA_VERSION = 5

# Only target-agnostic payloads are supported.
PAYLOAD_MODE_TARGET_AGNOSTIC = "target_agnostic"

_REQUIRED_METADATA_FIELDS = (
    "cache_schema_version",
    "sample_cache_key",
    "generator_provenance",
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
    "num_steps",
    "max_isl_distance_km",
    "isl_policy",
    "adjacent_search_k",
    "max_inter_plane_links_per_sat",
    "node_failure_prob",
    "edge_failure_prob",
    "failure_model",
    "seed",
    "epoch_iso",
    "failed_nodes_json",
    "failed_edges_json",
    "schema_version",
    "dataset_version",
    "orbital_engine",
    "physics_model_version",
    "link_budget_config",
)


def extract_cache_key_config(sample_config: dict[str, Any]) -> dict[str, Any]:
    """Return cache-identity fields from a sample configuration."""
    missing = sorted(set(_CACHE_KEY_FIELDS) - set(sample_config))
    if missing:
        raise ValueError(f"Cache identity missing required scientific fields: {missing}")
    return {key: sample_config[key] for key in _CACHE_KEY_FIELDS}


def make_sample_cache_key(sample_config: dict[str, Any]) -> str:
    """Build a deterministic cache key from sample parameters."""
    subset = extract_cache_key_config(sample_config)
    canonical = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def make_cache_metadata(
    *,
    sample_cache_key: str,
    generator_provenance: str,
    generator_config: dict[str, Any],
    payload_mode: str = PAYLOAD_MODE_TARGET_AGNOSTIC,
) -> dict[str, Any]:
    """Build metadata for a cache entry."""
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "sample_cache_key": sample_cache_key,
        "generator_provenance": generator_provenance,
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
    pending_pt_path = cache_dir / f"{cache_key}.pt.pending"
    meta_path = _meta_path(cache_dir, cache_key)
    pending_meta_path = cache_dir / f"{cache_key}.meta.json.pending"
    pending_pt_path.unlink(missing_ok=True)
    pending_meta_path.unlink(missing_ok=True)
    t0 = time.monotonic()
    try:
        torch.save(data_list, pending_pt_path)
        if metadata is not None:
            with open(pending_meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
        os.replace(pending_pt_path, pt_path)
        if metadata is not None:
            os.replace(pending_meta_path, meta_path)
    except Exception:
        pending_pt_path.unlink(missing_ok=True)
        pending_meta_path.unlink(missing_ok=True)
        raise
    write_time = time.monotonic() - t0

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


def _validate_graph_payload(
    data_list: list[Any],
    expected_generator_config: dict[str, Any] | None,
) -> None:
    torch = _import_torch()
    if not isinstance(data_list, list):
        raise ValueError("Cache payload must be a list of temporal graph Data objects")
    if expected_generator_config is not None:
        expected_count = int(expected_generator_config["num_steps"])
        if len(data_list) != expected_count:
            raise ValueError(
                "Cache payload graph count mismatch: "
                f"expected {expected_count}, got {len(data_list)}"
            )
    required_attributes = (
        "x",
        "edge_index",
        "edge_attr",
        "time_step",
        "num_nodes",
        "isl_policy",
        "adjacent_search_k",
        "max_inter_plane_links_per_sat",
        "failure_model",
    )
    for index, item in enumerate(data_list):
        missing = [name for name in required_attributes if not hasattr(item, name)]
        if missing:
            raise ValueError(
                f"Cache payload graph {index} missing required attributes: {missing}"
            )
        if not isinstance(item.x, torch.Tensor) or item.x.ndim != 2:
            raise ValueError(f"Cache payload graph {index} has invalid x tensor")
        if not isinstance(item.edge_index, torch.Tensor) or item.edge_index.ndim != 2:
            raise ValueError(f"Cache payload graph {index} has invalid edge_index tensor")
        if item.edge_index.shape[0] != 2:
            raise ValueError(f"Cache payload graph {index} edge_index must have shape [2, E]")
        if not isinstance(item.edge_attr, torch.Tensor) or item.edge_attr.ndim != 2:
            raise ValueError(f"Cache payload graph {index} has invalid edge_attr tensor")
        if item.edge_attr.shape[0] != item.edge_index.shape[1]:
            raise ValueError(f"Cache payload graph {index} edge_attr row count mismatch")
        num_nodes = int(item.num_nodes)
        if num_nodes != int(item.x.shape[0]):
            raise ValueError(f"Cache payload graph {index} num_nodes does not match x")
        if item.edge_index.numel() > 0:
            if int(item.edge_index.min()) < 0 or int(item.edge_index.max()) >= num_nodes:
                raise ValueError(f"Cache payload graph {index} edge_index is out of bounds")
        if not isinstance(item.time_step, torch.Tensor) or item.time_step.numel() != 1:
            raise ValueError(f"Cache payload graph {index} has invalid time_step")
        if int(item.time_step.item()) != index:
            raise ValueError(f"Cache payload graph {index} has noncontiguous time_step")


def validate_cache_entry(
    data_list: list[Any],
    metadata: dict[str, Any] | None,
    *,
    expected_sample_cache_key: str,
    expected_generator_provenance: str,
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

    if "cache_schema_version" not in metadata:
        raise ValueError(
            "Cache metadata missing required field 'cache_schema_version'. Delete "
            "this cache entry and regenerate with --write-cache."
        )

    schema_version = int(metadata["cache_schema_version"])
    if schema_version != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported cache schema version {schema_version}; expected "
            f"{CACHE_SCHEMA_VERSION}. Delete this cache entry and regenerate "
            "with --write-cache."
        )

    missing = [key for key in _REQUIRED_METADATA_FIELDS if key not in metadata]
    if missing:
        raise ValueError(
            f"Cache metadata missing required fields: {missing}. Delete this cache "
            "entry and regenerate with --write-cache."
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

    actual_sample_cache_key = str(metadata["sample_cache_key"])
    if actual_sample_cache_key != expected_sample_cache_key:
        raise ValueError(
            "Cache sample cache key mismatch. Delete this cache entry and "
            "regenerate with --write-cache."
        )

    actual_generator_provenance = str(metadata["generator_provenance"])
    if actual_generator_provenance != expected_generator_provenance:
        raise ValueError(
            "Cache generator provenance mismatch. Structural graph generation "
            "semantics changed or the cache is stale. Delete this cache entry and "
            "regenerate with --write-cache."
        )

    if expected_generator_config is not None:
        cached_config = metadata.get("generator_config")
        if cached_config != expected_generator_config:
            raise ValueError(
                "Cache generator configuration mismatch. Delete this cache entry "
                "and regenerate with --write-cache."
            )

    _validate_graph_payload(data_list, expected_generator_config)


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
