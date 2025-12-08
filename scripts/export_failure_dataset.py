from pathlib import Path
import sys
import csv

# --- Make sure Python can see the `src` folder ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dataclasses import asdict  # noqa: E402
from satnet.simulation.monte_carlo import (  # noqa: E402
    MonteCarloConfig,
    generate_failure_dataset,
    summarize_samples,
)


def main() -> None:
    cfg = MonteCarloConfig(
        num_runs=2000,
        node_failure_prob=0.02,
        edge_failure_prob=0.05,
        seed=123,
    )

    print("Generating Monte Carlo dataset...")
    samples = generate_failure_dataset(cfg)
    stats = summarize_samples(samples)

    out_dir = PROJECT_ROOT / "Data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "failure_dataset.csv"

    # all samples have same keys, use the first as header
    rows = [asdict(s) for s in samples]
    fieldnames = list(rows[0].keys()) if rows else []

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {out_path}")

    print("=== Aggregate stats ===")
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