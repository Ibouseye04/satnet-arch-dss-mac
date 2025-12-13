"""Export Tier 1 Temporal Design Dataset.

Generates diverse constellation designs with temporal GCC-based labels
using the Tier 1 (Hypatia-based) temporal pipeline.

Outputs both runs table (per-run summaries) and steps table (per-step metrics).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from satnet.simulation.monte_carlo import (  # noqa: E402
    Tier1MonteCarloConfig,
    generate_tier1_temporal_dataset,
    write_tier1_dataset_csv,
)


def main() -> None:
    """Generate diverse constellation designs with temporal labels."""

    cfg = Tier1MonteCarloConfig(
        num_runs=200,
        num_planes_range=(3, 8),
        sats_per_plane_range=(4, 12),
        inclination_deg=53.0,
        altitude_km=550.0,
        duration_minutes=10,
        step_seconds=60,
        gcc_threshold=0.8,
        node_failure_prob_range=(0.0, 0.2),
        edge_failure_prob_range=(0.0, 0.3),
        seed=42,
        sample_constellation=True,
    )

    print("Generating Tier 1 Temporal Design Dataset...")
    print(f"  Runs: {cfg.num_runs}")
    print(f"  Planes: {cfg.num_planes_range}, Sats/plane: {cfg.sats_per_plane_range}")
    print(f"  Duration: {cfg.duration_minutes} min @ {cfg.step_seconds}s steps")
    print("=" * 60)

    runs, steps = generate_tier1_temporal_dataset(cfg)

    print(f"\nGenerated {len(runs)} runs, {len(steps)} step records")

    # Save to CSV with schema validation
    out_dir = PROJECT_ROOT / "data"
    runs_path = out_dir / "tier1_design_runs.csv"
    steps_path = out_dir / "tier1_design_steps.csv"

    # Use canonical writer (validates schema before writing)
    write_tier1_dataset_csv(runs, steps, runs_path, steps_path)
    print(f"Written {len(runs)} run rows to {runs_path}")
    print(f"Written {len(steps)} step rows to {steps_path}")

    # Print summary stats
    print("\n=== Dataset Summary ===")
    partition_count = sum(r.partition_any for r in runs)
    mean_gcc_frac = sum(r.gcc_frac_mean for r in runs) / len(runs) if runs else 0
    print(f"Partition probability: {partition_count / len(runs):.3f}")
    print(f"Mean GCC fraction: {mean_gcc_frac:.3f}")


if __name__ == "__main__":
    main()