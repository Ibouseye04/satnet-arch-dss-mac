from pathlib import Path
import sys
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dataclasses import asdict  # noqa: E402
from satnet.simulation.monte_carlo import (  # noqa: E402
    DesignScenarioConfig,
    generate_designtime_dataset,
)


def main() -> None:
    # Define a grid of architectures + failure assumptions
    scenarios = []
    scenario_id = 0

    for num_sats in [16, 24, 32]:
        for num_gs in [2, 4, 6]:
            for isl_degree in [2, 4, 6]:
                for node_p in [0.01, 0.03, 0.05]:
                    for edge_p in [0.01, 0.03, 0.05]:
                        scenarios.append(
                            DesignScenarioConfig(
                                scenario_id=scenario_id,
                                num_satellites=num_sats,
                                num_ground_stations=num_gs,
                                isl_degree=isl_degree,
                                node_failure_prob=node_p,
                                edge_failure_prob=edge_p,
                                runs_per_scenario=50,
                            )
                        )
                        scenario_id += 1

    print(f"Total scenarios: {len(scenarios)}")
    samples = generate_designtime_dataset(scenarios, base_seed=2000)

    out_dir = PROJECT_ROOT / "Data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "design_failure_dataset.csv"

    rows = [asdict(s) for s in samples]
    fieldnames = list(rows[0].keys()) if rows else []

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()