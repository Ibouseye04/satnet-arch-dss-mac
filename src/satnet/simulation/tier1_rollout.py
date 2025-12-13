"""Tier 1 Temporal Rollout Configuration and Results Contracts.

This module defines the stable API for temporal evaluation of satellite
network connectivity using the Tier 1 (Hypatia-based) pipeline.

Dataclasses defined here establish the contract between:
- Configuration inputs (constellation, time, ISL, labeling, failures)
- Per-step outputs (metrics at each time step)
- Run summary outputs (aggregated labels)

No toy topology dependencies. No Hypatia imports in this module —
these are pure data contracts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# Dataset versioning constants
DATASET_VERSION = "tier1_temporal_connectivity_v1"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Tier1RolloutConfig:
    """Configuration for a single Tier 1 temporal rollout.

    Attributes:
        num_planes: Number of orbital planes.
        sats_per_plane: Satellites per orbital plane.
        inclination_deg: Orbital inclination in degrees.
        altitude_km: Orbital altitude above Earth surface in km.
        phasing_factor: Walker Delta phasing factor (F parameter).
        duration_minutes: Total simulation duration in minutes.
        step_seconds: Time step interval in seconds.
        max_isl_distance_km: Maximum ISL distance (optional, uses adapter default if None).
        gcc_threshold: Threshold for partition detection (gcc_frac < threshold → partitioned).
        node_failure_prob: Probability of node failure (sampled once per run).
        edge_failure_prob: Probability of edge failure (sampled once per run from t=0 edges).
        seed: Random seed for reproducibility.
    """

    # Constellation parameters
    num_planes: int
    sats_per_plane: int
    inclination_deg: float = 53.0
    altitude_km: float = 550.0
    phasing_factor: int = 1

    # Time parameters
    duration_minutes: int = 90
    step_seconds: int = 60

    # ISL parameters
    max_isl_distance_km: Optional[float] = None

    # Labeling parameters
    gcc_threshold: float = 0.8

    # Failure parameters (v1: persistent failures sampled at t=0)
    node_failure_prob: float = 0.0
    edge_failure_prob: float = 0.0
    seed: int = 42

    def config_hash(self) -> str:
        """Compute a deterministic hash of this configuration.

        Returns:
            A hex string hash suitable for dataset identification.
        """
        config_dict = asdict(self)
        config_json = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]

    @property
    def num_steps(self) -> int:
        """Compute the number of time steps in this rollout."""
        total_seconds = self.duration_minutes * 60
        return total_seconds // self.step_seconds

    @property
    def total_satellites(self) -> int:
        """Compute total number of satellites in the constellation."""
        return self.num_planes * self.sats_per_plane


@dataclass
class Tier1RolloutStep:
    """Metrics for a single time step in a Tier 1 rollout.

    Attributes:
        t: Time step index (0-indexed).
        num_nodes: Number of nodes in the effective graph at this step.
        num_edges: Number of edges in the effective graph at this step.
        num_components: Number of connected components.
        gcc_size: Size of the Giant Connected Component (node count).
        gcc_frac: Fraction of nodes in the GCC (0.0 to 1.0).
        partitioned: 1 if gcc_frac < threshold, 0 otherwise.
    """

    t: int
    num_nodes: int
    num_edges: int
    num_components: int
    gcc_size: int
    gcc_frac: float
    partitioned: int


@dataclass
class Tier1RolloutSummary:
    """Aggregated summary of a Tier 1 temporal rollout.

    Attributes:
        gcc_frac_min: Minimum GCC fraction across all time steps.
        gcc_frac_mean: Mean GCC fraction across all time steps.
        partition_fraction: Fraction of time steps where network was partitioned.
        partition_any: 1 if partitioned at any time step, 0 otherwise.
        max_partition_streak: Longest consecutive run of partitioned steps.
        num_steps: Total number of time steps in the rollout.
        num_failed_nodes: Number of nodes that failed (persistent).
        num_failed_edges: Number of edges that failed (persistent, from t=0).
        schema_version: Schema version for dataset compatibility.
        dataset_version: Dataset version identifier.
        config_hash: Hash of the configuration for reproducibility.
    """

    gcc_frac_min: float
    gcc_frac_mean: float
    partition_fraction: float
    partition_any: int
    max_partition_streak: int
    num_steps: int
    num_failed_nodes: int = 0
    num_failed_edges: int = 0
    schema_version: int = field(default=SCHEMA_VERSION)
    dataset_version: str = field(default=DATASET_VERSION)
    config_hash: str = ""

    def to_dict(self) -> dict:
        """Convert summary to dictionary for serialization."""
        return asdict(self)
