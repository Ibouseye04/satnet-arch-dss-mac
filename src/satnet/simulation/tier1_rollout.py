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
from datetime import datetime, timezone
from typing import Optional


# Fixed epoch for reproducibility (J2000.0 epoch: 2000-01-01T12:00:00Z)
# Using a well-known astronomical reference epoch ensures determinism
DEFAULT_EPOCH_ISO = "2000-01-01T12:00:00+00:00"


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
        epoch_iso: TLE epoch as ISO 8601 string for reproducibility (default: J2000.0).
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

    # Epoch for orbital propagation (ISO 8601 string for JSON serialization)
    # Default is J2000.0 epoch for reproducibility
    epoch_iso: str = DEFAULT_EPOCH_ISO

    @property
    def epoch(self) -> datetime:
        """Parse epoch_iso string to datetime object."""
        return datetime.fromisoformat(self.epoch_iso)

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
        """Compute the number of time steps in this rollout.
        
        Uses inclusive t=0..T convention: duration_seconds // step_seconds + 1.
        Example: 3 minutes @ 60s = steps 0,1,2,3 = 4 steps.
        """
        total_seconds = self.duration_minutes * 60
        return total_seconds // self.step_seconds + 1

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


# ---------------------------------------------------------------------------
# Tier 1 Rollout Runner
# ---------------------------------------------------------------------------

def run_tier1_rollout(
    cfg: Tier1RolloutConfig,
) -> tuple[list[Tier1RolloutStep], Tier1RolloutSummary]:
    """
    Execute a Tier 1 temporal rollout using HypatiaAdapter.

    This function:
    1. Constructs a HypatiaAdapter from constellation parameters
    2. Generates TLEs and calculates ISLs over time
    3. Samples persistent failures from the t=0 graph
    4. For each time step, applies failures and computes GCC metrics
    5. Aggregates results into a summary

    v1 Assumptions:
    - Node failures are persistent (sampled once, applied at all steps)
    - Edge failures are sampled from edges present at t=0
    - Edges that appear later are not eligible to fail

    Args:
        cfg: Tier1RolloutConfig with constellation, time, and failure parameters.

    Returns:
        Tuple of (steps, summary) where:
        - steps: List of Tier1RolloutStep for each time step
        - summary: Tier1RolloutSummary with aggregated labels
    """
    import random

    from satnet.metrics.labels import (
        aggregate_partition_streaks,
        compute_gcc_frac,
        compute_gcc_size,
        compute_num_components,
        compute_partitioned,
    )
    from satnet.network.hypatia_adapter import HypatiaAdapter

    # 1. Construct HypatiaAdapter with explicit epoch for reproducibility
    adapter = HypatiaAdapter(
        num_planes=cfg.num_planes,
        sats_per_plane=cfg.sats_per_plane,
        inclination_deg=cfg.inclination_deg,
        altitude_km=cfg.altitude_km,
        phasing_factor=cfg.phasing_factor,
        epoch=cfg.epoch,
    )

    # 2. Generate TLEs and calculate ISLs
    adapter.generate_tles()

    isl_kwargs = {
        "duration_minutes": cfg.duration_minutes,
        "step_seconds": cfg.step_seconds,
    }
    if cfg.max_isl_distance_km is not None:
        isl_kwargs["max_isl_distance_km"] = cfg.max_isl_distance_km

    adapter.calculate_isls(**isl_kwargs)

    # 3. Sample persistent failures from t=0 graph
    rng = random.Random(cfg.seed)
    G0 = adapter.get_graph_at_step(0)

    # Sample failed nodes
    failed_nodes: set[int] = set()
    for node in G0.nodes():
        if rng.random() < cfg.node_failure_prob:
            failed_nodes.add(node)

    # Sample failed edges (from t=0 edges only)
    failed_edges: set[tuple[int, int]] = set()
    for u, v in G0.edges():
        if rng.random() < cfg.edge_failure_prob:
            # Store as sorted tuple for consistent lookup
            failed_edges.add((min(u, v), max(u, v)))

    # 4. Loop over time steps, apply failures, compute metrics
    steps: list[Tier1RolloutStep] = []
    partitioned_flags: list[int] = []

    for t, G_t in adapter.iter_graphs():
        # Apply persistent node failures
        G_eff = G_t.copy()
        nodes_to_remove = [n for n in failed_nodes if G_eff.has_node(n)]
        G_eff.remove_nodes_from(nodes_to_remove)

        # Apply edge failures (only if edge exists at this step)
        for u, v in failed_edges:
            if G_eff.has_edge(u, v):
                G_eff.remove_edge(u, v)

        # Compute metrics using pure label functions
        num_nodes = G_eff.number_of_nodes()
        num_edges = G_eff.number_of_edges()
        num_components = compute_num_components(G_eff)
        gcc_size = compute_gcc_size(G_eff)
        gcc_frac = compute_gcc_frac(G_eff)
        partitioned = compute_partitioned(gcc_frac, cfg.gcc_threshold)

        step = Tier1RolloutStep(
            t=t,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_components=num_components,
            gcc_size=gcc_size,
            gcc_frac=gcc_frac,
            partitioned=partitioned,
        )
        steps.append(step)
        partitioned_flags.append(partitioned)

    # 5. Aggregate into summary
    gcc_fracs = [s.gcc_frac for s in steps]
    num_steps = len(steps)

    if num_steps > 0:
        gcc_frac_min = min(gcc_fracs)
        gcc_frac_mean = sum(gcc_fracs) / num_steps
        partition_count = sum(partitioned_flags)
        partition_fraction = partition_count / num_steps
        partition_any = 1 if partition_count > 0 else 0
    else:
        gcc_frac_min = 0.0
        gcc_frac_mean = 0.0
        partition_fraction = 0.0
        partition_any = 0

    max_partition_streak = aggregate_partition_streaks(partitioned_flags)

    summary = Tier1RolloutSummary(
        gcc_frac_min=gcc_frac_min,
        gcc_frac_mean=gcc_frac_mean,
        partition_fraction=partition_fraction,
        partition_any=partition_any,
        max_partition_streak=max_partition_streak,
        num_steps=num_steps,
        num_failed_nodes=len(failed_nodes),
        num_failed_edges=len(failed_edges),
        config_hash=cfg.config_hash(),
    )

    return steps, summary
