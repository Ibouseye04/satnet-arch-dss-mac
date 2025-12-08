from pathlib import Path
import sys

# --- Make sure Python can see the `src` folder ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# --- Now normal imports work ---
from satnet.simulation.monte_carlo import MonteCarloConfig, run_failure_sweep  # noqa: E402


def main() -> None:
    cfg = MonteCarloConfig(
        num_runs=1000,
        node_failure_prob=0.02,
        edge_failure_prob=0.05,
        seed=123,
    )
    stats = run_failure_sweep(cfg)

    print("=== Monte Carlo Failure Sweep ===")
    print(f"Runs: {stats.num_runs}")
    print(f"Mean failed nodes: {stats.mean_failed_nodes:.2f}")
    print(f"Mean failed edges: {stats.mean_failed_edges:.2f}")
    print(f"Partition probability: {stats.prob_partition:.3f}")
    print(
        "Mean largest-component ratio "
        f"(after failures / original nodes): "
        f"{stats.mean_largest_component_ratio:.3f}"
    )


if __name__ == "__main__":
    main()