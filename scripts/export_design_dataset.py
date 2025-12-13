"""Export Tier 1 Temporal Design Dataset.

Generates diverse constellation designs with temporal GCC-based labels
using the Tier 1 (Hypatia-based) temporal pipeline.

Outputs both runs table (per-run summaries) and steps table (per-step metrics).

Usage:
    python scripts/export_design_dataset.py --num-runs 2000 --seed 12345
    python scripts/export_design_dataset.py --altitude-min 400 --altitude-max 800
"""

import argparse
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Tier 1 Temporal Design Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=200,
        help="Number of simulation runs to generate",
    )
    parser.add_argument(
        "--altitude-min",
        type=float,
        default=300.0,
        help="Minimum orbital altitude in km",
    )
    parser.add_argument(
        "--altitude-max",
        type=float,
        default=1200.0,
        help="Maximum orbital altitude in km",
    )
    parser.add_argument(
        "--inclination-min",
        type=float,
        default=30.0,
        help="Minimum orbital inclination in degrees",
    )
    parser.add_argument(
        "--inclination-max",
        type=float,
        default=98.0,
        help="Maximum orbital inclination in degrees",
    )
    parser.add_argument(
        "--planes-min",
        type=int,
        default=3,
        help="Minimum number of orbital planes",
    )
    parser.add_argument(
        "--planes-max",
        type=int,
        default=8,
        help="Maximum number of orbital planes",
    )
    parser.add_argument(
        "--sats-min",
        type=int,
        default=4,
        help="Minimum satellites per plane",
    )
    parser.add_argument(
        "--sats-max",
        type=int,
        default=12,
        help="Maximum satellites per plane",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Simulation duration in minutes",
    )
    parser.add_argument(
        "--step-seconds",
        type=int,
        default=60,
        help="Time step interval in seconds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: PROJECT_ROOT/data)",
    )

    return parser.parse_args()


def main() -> None:
    """Generate diverse constellation designs with temporal labels."""
    args = parse_args()

    cfg = Tier1MonteCarloConfig(
        num_runs=args.num_runs,
        num_planes_range=(args.planes_min, args.planes_max),
        sats_per_plane_range=(args.sats_min, args.sats_max),
        inclination_deg_range=(args.inclination_min, args.inclination_max),
        altitude_km_range=(args.altitude_min, args.altitude_max),
        duration_minutes=args.duration,
        step_seconds=args.step_seconds,
        gcc_threshold=0.8,
        node_failure_prob_range=(0.0, 0.2),
        edge_failure_prob_range=(0.0, 0.3),
        seed=args.seed,
        sample_constellation=True,
    )

    print("Generating Tier 1 Temporal Design Dataset...")
    print(f"  Runs: {cfg.num_runs}")
    print(f"  Planes: {cfg.num_planes_range}, Sats/plane: {cfg.sats_per_plane_range}")
    print(f"  Altitude: {cfg.altitude_km_range[0]:.0f}-{cfg.altitude_km_range[1]:.0f} km")
    print(f"  Inclination: {cfg.inclination_deg_range[0]:.0f}-{cfg.inclination_deg_range[1]:.0f} deg")
    print(f"  Duration: {cfg.duration_minutes} min @ {cfg.step_seconds}s steps")
    print(f"  Seed: {cfg.seed}")
    print("=" * 60)

    runs, steps = generate_tier1_temporal_dataset(cfg)

    print(f"\nGenerated {len(runs)} runs, {len(steps)} step records")

    # Save to CSV with schema validation
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data"
    runs_path = out_dir / "tier1_design_runs.csv"
    steps_path = out_dir / "tier1_design_steps.csv"

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