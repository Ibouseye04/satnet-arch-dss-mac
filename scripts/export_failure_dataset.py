"""Export Tier 1 Temporal Failure Dataset.

Generates a dataset using the Tier 1 (Hypatia-based) temporal pipeline.
Outputs both runs table (per-run summaries) and steps table (per-step metrics).
"""

from pathlib import Path
import sys
import csv

# --- Make sure Python can see the `src` folder ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from satnet.simulation.monte_carlo import (  # noqa: E402
    Tier1MonteCarloConfig,
    generate_tier1_temporal_dataset,
    runs_to_dicts,
    steps_to_dicts,
)


def main() -> None:
    cfg = Tier1MonteCarloConfig(
        num_runs=100,
        num_planes_range=(3, 6),
        sats_per_plane_range=(4, 8),
        duration_minutes=10,
        step_seconds=60,
        node_failure_prob_range=(0.0, 0.15),
        edge_failure_prob_range=(0.0, 0.25),
        seed=123,
    )

    print("Generating Tier 1 Temporal Monte Carlo dataset...")
    print(f"  Runs: {cfg.num_runs}")
    print(f"  Planes: {cfg.num_planes_range}, Sats/plane: {cfg.sats_per_plane_range}")
    print(f"  Duration: {cfg.duration_minutes} min @ {cfg.step_seconds}s steps")

    runs, steps = generate_tier1_temporal_dataset(cfg)

    out_dir = PROJECT_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export runs table
    runs_path = out_dir / "tier1_failure_runs.csv"
    runs_dicts = runs_to_dicts(runs)
    if runs_dicts:
        fieldnames = list(runs_dicts[0].keys())
        with runs_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(runs_dicts)
    print(f"Written {len(runs_dicts)} run rows to {runs_path}")

    # Export steps table
    steps_path = out_dir / "tier1_failure_steps.csv"
    steps_dicts = steps_to_dicts(steps)
    if steps_dicts:
        fieldnames = list(steps_dicts[0].keys())
        with steps_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(steps_dicts)
    print(f"Written {len(steps_dicts)} step rows to {steps_path}")

    # Print aggregate stats
    print("\n=== Aggregate Stats ===")
    partition_count = sum(r.partition_any for r in runs)
    mean_gcc_frac = sum(r.gcc_frac_mean for r in runs) / len(runs) if runs else 0
    print(f"Runs: {len(runs)}")
    print(f"Partition probability: {partition_count / len(runs):.3f}")
    print(f"Mean GCC fraction: {mean_gcc_frac:.3f}")


if __name__ == "__main__":
    main()