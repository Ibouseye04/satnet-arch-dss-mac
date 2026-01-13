"""Tier 1 Temporal Monte Carlo Dataset Generator.

This module generates datasets using the Tier 1 (Hypatia-based) temporal
pipeline. It replaces the legacy toy topology generator with physics-based
satellite network simulation.

Key features:
- Uses HypatiaAdapter for Walker Delta constellations with SGP4 propagation
- Temporal evaluation: metrics computed over t=0..T time steps
- GCC-based partition labels (no leaky post-failure features)
- Reproducible with config + seed
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple
import random

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from satnet.simulation.tier1_rollout import (
    DATASET_VERSION,
    DEFAULT_EPOCH_ISO,
    SCHEMA_VERSION,
    Tier1FailureRealization,
    Tier1RolloutConfig,
    Tier1RolloutStep,
    Tier1RolloutSummary,
    run_tier1_rollout,
)


# ---------------------------------------------------------------------------
# Tier 1 Monte Carlo Configuration
# ---------------------------------------------------------------------------


@dataclass
class Tier1MonteCarloConfig:
    """Configuration for Tier 1 temporal Monte Carlo dataset generation.

    Attributes:
        num_runs: Number of failure realizations to simulate.
        num_planes_range: (min, max) range for number of orbital planes.
        sats_per_plane_range: (min, max) range for satellites per plane.
        inclination_deg: Orbital inclination in degrees.
        altitude_km: Orbital altitude in km.
        duration_minutes: Simulation duration per run.
        step_seconds: Time step interval.
        gcc_threshold: Threshold for partition detection.
        node_failure_prob_range: (min, max) range for node failure probability.
        edge_failure_prob_range: (min, max) range for edge failure probability.
        seed: Base random seed for reproducibility.
        sample_constellation: If True, sample constellation params per run.
                              If False, use midpoint of ranges for all runs.
    """

    num_runs: int = 100

    # Constellation parameter ranges
    num_planes_range: Tuple[int, int] = (3, 6)
    sats_per_plane_range: Tuple[int, int] = (4, 8)
    inclination_deg_range: Tuple[float, float] = (30.0, 98.0)
    altitude_km_range: Tuple[float, float] = (300.0, 1200.0)

    # Time parameters
    duration_minutes: int = 10
    step_seconds: int = 60

    # Labeling
    gcc_threshold: float = 0.8

    # Failure parameter ranges
    node_failure_prob_range: Tuple[float, float] = (0.0, 0.1)
    edge_failure_prob_range: Tuple[float, float] = (0.0, 0.2)

    # Reproducibility
    seed: int = 42

    # Sampling mode
    sample_constellation: bool = True


@dataclass
class Tier1RunRow:
    """One row in the runs table (per-run summary).

    Contains design-time features and temporal aggregate labels.
    No post-failure leakage (features are design inputs only).
    
    Graph Reconstruction Contract (Step 3):
        To reconstruct the exact graph sequence for ML training:
        1. Use epoch_iso, duration_minutes, step_seconds with HypatiaAdapter
        2. Parse failed_nodes_json and failed_edges_json
        3. Apply failures at each time step
        This ensures regenerated graphs match the original labels.
    """

    run_id: int

    # Design-time features (constellation architecture)
    num_planes: int
    sats_per_plane: int
    total_satellites: int
    inclination_deg: float
    altitude_km: float

    # Design-time features (failure assumptions)
    node_failure_prob: float
    edge_failure_prob: float

    # Time parameters
    duration_minutes: int
    step_seconds: int
    num_steps: int

    # Temporal aggregate labels (computed from simulation)
    gcc_frac_min: float
    gcc_frac_mean: float
    partition_fraction: float
    partition_any: int
    max_partition_streak: int

    # Failure counts (for analysis, not features)
    num_failed_nodes: int
    num_failed_edges: int

    # Metadata (non-default fields)
    seed: int
    config_hash: str

    # Failure realization (for graph reconstruction)
    # JSON-encoded lists: failed_nodes_json = "[1, 5, 12]"
    # failed_edges_json = "[[0,1], [3,4]]" (sorted tuples)
    failed_nodes_json: str = "[]"
    failed_edges_json: str = "[]"

    # Metadata (with defaults)
    epoch_iso: str = DEFAULT_EPOCH_ISO
    schema_version: int = SCHEMA_VERSION
    dataset_version: str = DATASET_VERSION


@dataclass
class Tier1StepRow:
    """One row in the steps table (per-step metrics).

    Contains time-series data for detailed analysis.
    """

    run_id: int
    t: int
    num_nodes: int
    num_edges: int
    num_components: int
    gcc_size: int
    gcc_frac: float
    partitioned: int


# ---------------------------------------------------------------------------
# Dataset Generation Functions
# ---------------------------------------------------------------------------


def generate_tier1_temporal_dataset(
    cfg: Tier1MonteCarloConfig,
) -> Tuple[List[Tier1RunRow], List[Tier1StepRow]]:
    """
    Generate a Tier 1 temporal connectivity dataset.

    For each run:
    1. Sample or use fixed constellation parameters
    2. Sample failure probabilities from configured ranges
    3. Execute temporal rollout using run_tier1_rollout()
    4. Collect per-run summary and per-step metrics

    Args:
        cfg: Tier1MonteCarloConfig with generation parameters.

    Returns:
        Tuple of (runs_rows, steps_rows) where:
        - runs_rows: List of Tier1RunRow (one per run)
        - steps_rows: List of Tier1StepRow (one per step per run)
    """
    rng = random.Random(cfg.seed)

    runs_rows: List[Tier1RunRow] = []
    steps_rows: List[Tier1StepRow] = []

    # Progress iterator
    run_iter: Iterator[int] = range(cfg.num_runs)
    if HAS_TQDM:
        run_iter = tqdm(run_iter, desc="Generating runs", unit="run")
    else:
        print(f"Generating {cfg.num_runs} runs...")

    for run_id in run_iter:
        # Print progress every 10% if no tqdm
        if not HAS_TQDM and cfg.num_runs >= 10 and run_id % max(1, cfg.num_runs // 10) == 0:
            print(f"  Progress: {run_id}/{cfg.num_runs} ({100*run_id//cfg.num_runs}%)")

        # Sample or use fixed constellation parameters
        if cfg.sample_constellation:
            num_planes = rng.randint(*cfg.num_planes_range)
            sats_per_plane = rng.randint(*cfg.sats_per_plane_range)
            altitude_km = rng.uniform(*cfg.altitude_km_range)
            inclination_deg = rng.uniform(*cfg.inclination_deg_range)
        else:
            num_planes = (cfg.num_planes_range[0] + cfg.num_planes_range[1]) // 2
            sats_per_plane = (cfg.sats_per_plane_range[0] + cfg.sats_per_plane_range[1]) // 2
            altitude_km = (cfg.altitude_km_range[0] + cfg.altitude_km_range[1]) / 2
            inclination_deg = (cfg.inclination_deg_range[0] + cfg.inclination_deg_range[1]) / 2

        # Sample failure probabilities
        node_failure_prob = rng.uniform(*cfg.node_failure_prob_range)
        edge_failure_prob = rng.uniform(*cfg.edge_failure_prob_range)

        # Create rollout config with unique seed per run
        run_seed = cfg.seed + run_id
        rollout_cfg = Tier1RolloutConfig(
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            inclination_deg=inclination_deg,
            altitude_km=altitude_km,
            duration_minutes=cfg.duration_minutes,
            step_seconds=cfg.step_seconds,
            gcc_threshold=cfg.gcc_threshold,
            node_failure_prob=node_failure_prob,
            edge_failure_prob=edge_failure_prob,
            seed=run_seed,
        )

        # Execute rollout
        steps, summary, failures = run_tier1_rollout(rollout_cfg)
        
        # Serialize failure realization for graph reconstruction (Step 3 contract)
        failed_nodes_json, failed_edges_json = failures.to_json_strings()

        # Build run row
        run_row = Tier1RunRow(
            run_id=run_id,
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            total_satellites=num_planes * sats_per_plane,
            inclination_deg=inclination_deg,
            altitude_km=altitude_km,
            node_failure_prob=node_failure_prob,
            edge_failure_prob=edge_failure_prob,
            duration_minutes=cfg.duration_minutes,
            step_seconds=cfg.step_seconds,
            num_steps=summary.num_steps,
            gcc_frac_min=summary.gcc_frac_min,
            gcc_frac_mean=summary.gcc_frac_mean,
            partition_fraction=summary.partition_fraction,
            partition_any=summary.partition_any,
            max_partition_streak=summary.max_partition_streak,
            num_failed_nodes=summary.num_failed_nodes,
            num_failed_edges=summary.num_failed_edges,
            failed_nodes_json=failed_nodes_json,
            failed_edges_json=failed_edges_json,
            seed=run_seed,
            config_hash=summary.config_hash,
            epoch_iso=rollout_cfg.epoch_iso,
        )
        runs_rows.append(run_row)

        # Build step rows
        for step in steps:
            step_row = Tier1StepRow(
                run_id=run_id,
                t=step.t,
                num_nodes=step.num_nodes,
                num_edges=step.num_edges,
                num_components=step.num_components,
                gcc_size=step.gcc_size,
                gcc_frac=step.gcc_frac,
                partitioned=step.partitioned,
            )
            steps_rows.append(step_row)

    return runs_rows, steps_rows


def runs_to_dicts(runs: List[Tier1RunRow]) -> List[dict]:
    """Convert run rows to list of dicts for DataFrame/Parquet export."""
    return [asdict(r) for r in runs]


def steps_to_dicts(steps: List[Tier1StepRow]) -> List[dict]:
    """Convert step rows to list of dicts for DataFrame/Parquet export."""
    return [asdict(s) for s in steps]


# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------

# Required columns for runs table (v1 schema)
RUNS_REQUIRED_COLUMNS = frozenset([
    "run_id",
    "num_planes",
    "sats_per_plane",
    "total_satellites",
    "inclination_deg",
    "altitude_km",
    "node_failure_prob",
    "edge_failure_prob",
    "duration_minutes",
    "step_seconds",
    "num_steps",
    "gcc_frac_min",
    "gcc_frac_mean",
    "partition_fraction",
    "partition_any",
    "max_partition_streak",
    "num_failed_nodes",
    "num_failed_edges",
    "failed_nodes_json",
    "failed_edges_json",
    "seed",
    "config_hash",
    "epoch_iso",
    "schema_version",
    "dataset_version",
])

# Required columns for steps table (v1 schema)
STEPS_REQUIRED_COLUMNS = frozenset([
    "run_id",
    "t",
    "num_nodes",
    "num_edges",
    "num_components",
    "gcc_size",
    "gcc_frac",
    "partitioned",
])


class SchemaValidationError(ValueError):
    """Raised when dataset does not conform to required schema."""
    pass


def validate_runs_schema(runs_dicts: List[dict]) -> None:
    """Validate that runs data conforms to v1 schema.

    Args:
        runs_dicts: List of run row dictionaries.

    Raises:
        SchemaValidationError: If required fields are missing or invalid.
    """
    if not runs_dicts:
        return  # Empty dataset is valid

    # Check first row for required columns
    first_row = runs_dicts[0]
    present_columns = set(first_row.keys())
    missing = RUNS_REQUIRED_COLUMNS - present_columns

    if missing:
        raise SchemaValidationError(
            f"Runs table missing required columns: {sorted(missing)}"
        )

    # Validate value ranges
    for i, row in enumerate(runs_dicts):
        # GCC fractions must be in [0, 1]
        for col in ["gcc_frac_min", "gcc_frac_mean", "partition_fraction"]:
            val = row.get(col)
            if val is not None and not (0.0 <= val <= 1.0):
                raise SchemaValidationError(
                    f"Row {i}: {col}={val} out of range [0, 1]"
                )

        # Binary columns must be 0 or 1
        if row.get("partition_any") not in (0, 1):
            raise SchemaValidationError(
                f"Row {i}: partition_any must be 0 or 1"
            )

        # Schema version must match
        if row.get("schema_version") != SCHEMA_VERSION:
            raise SchemaValidationError(
                f"Row {i}: schema_version={row.get('schema_version')}, expected {SCHEMA_VERSION}"
            )


def validate_steps_schema(steps_dicts: List[dict]) -> None:
    """Validate that steps data conforms to v1 schema.

    Args:
        steps_dicts: List of step row dictionaries.

    Raises:
        SchemaValidationError: If required fields are missing or invalid.
    """
    if not steps_dicts:
        return  # Empty dataset is valid

    # Check first row for required columns
    first_row = steps_dicts[0]
    present_columns = set(first_row.keys())
    missing = STEPS_REQUIRED_COLUMNS - present_columns

    if missing:
        raise SchemaValidationError(
            f"Steps table missing required columns: {sorted(missing)}"
        )

    # Validate value ranges
    for i, row in enumerate(steps_dicts):
        # GCC fraction must be in [0, 1]
        val = row.get("gcc_frac")
        if val is not None and not (0.0 <= val <= 1.0):
            raise SchemaValidationError(
                f"Row {i}: gcc_frac={val} out of range [0, 1]"
            )

        # Partitioned must be 0 or 1
        if row.get("partitioned") not in (0, 1):
            raise SchemaValidationError(
                f"Row {i}: partitioned must be 0 or 1"
            )


# ---------------------------------------------------------------------------
# Canonical Dataset Writer (validates before writing)
# ---------------------------------------------------------------------------


def write_tier1_dataset_csv(
    runs: List[Tier1RunRow],
    steps: List[Tier1StepRow],
    runs_path: "Path",
    steps_path: "Path",
) -> None:
    """Write Tier 1 temporal dataset to CSV with schema validation.

    This is the canonical writer that enforces schema validation before
    writing. Use this instead of manually writing CSV files.

    Args:
        runs: List of Tier1RunRow objects.
        steps: List of Tier1StepRow objects.
        runs_path: Path for runs CSV output.
        steps_path: Path for steps CSV output.

    Raises:
        SchemaValidationError: If data does not conform to v1 schema.
    """
    import csv
    from pathlib import Path

    # Convert to dicts
    runs_dicts = runs_to_dicts(runs)
    steps_dicts = steps_to_dicts(steps)

    # Validate schemas (raises SchemaValidationError if invalid)
    validate_runs_schema(runs_dicts)
    validate_steps_schema(steps_dicts)

    # Ensure output directories exist
    Path(runs_path).parent.mkdir(parents=True, exist_ok=True)
    Path(steps_path).parent.mkdir(parents=True, exist_ok=True)

    # Write runs table
    if runs_dicts:
        fieldnames = list(runs_dicts[0].keys())
        with open(runs_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(runs_dicts)

    # Write steps table
    if steps_dicts:
        fieldnames = list(steps_dicts[0].keys())
        with open(steps_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(steps_dicts)