"""Tier 1 Temporal Failure Sweep.

Quick sweep to evaluate partition probability across failure rates
using the Tier 1 (Hypatia-based) temporal pipeline.
"""

from pathlib import Path
import sys

# --- Make sure Python can see the `src` folder ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from satnet.simulation.monte_carlo import (  # noqa: E402
    Tier1MonteCarloConfig,
    generate_tier1_temporal_dataset,
)


def main() -> None:
    cfg = Tier1MonteCarloConfig(
        num_runs=50,
        num_planes_range=(4, 6),
        sats_per_plane_range=(5, 8),
        duration_minutes=5,
        step_seconds=60,
        node_failure_prob_range=(0.0, 0.2),
        edge_failure_prob_range=(0.0, 0.3),
        seed=123,
    )

    print("=== Tier 1 Temporal Failure Sweep ===")
    print(f"Running {cfg.num_runs} simulations...")

    runs, steps = generate_tier1_temporal_dataset(cfg)

    # Compute aggregate stats
    partition_count = sum(r.partition_any for r in runs)
    mean_gcc_frac = sum(r.gcc_frac_mean for r in runs) / len(runs) if runs else 0
    mean_failed_nodes = sum(r.num_failed_nodes for r in runs) / len(runs) if runs else 0
    mean_failed_edges = sum(r.num_failed_edges for r in runs) / len(runs) if runs else 0

    print(f"\nRuns: {len(runs)}")
    print(f"Mean failed nodes: {mean_failed_nodes:.2f}")
    print(f"Mean failed edges: {mean_failed_edges:.2f}")
    print(f"Partition probability: {partition_count / len(runs):.3f}")
    print(f"Mean GCC fraction: {mean_gcc_frac:.3f}")


if __name__ == "__main__":
    main()