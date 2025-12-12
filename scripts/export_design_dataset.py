"""
Export Design Dataset using Tier 1 Physics (SGP4 + Link Budgets)

Generates diverse constellation designs and collects failure metrics
using the SimulationEngine with HypatiaAdapter.
"""

import sys
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd  # noqa: E402

from satnet.simulation.engine import SimulationEngine  # noqa: E402
from satnet.simulation.failures import (  # noqa: E402
    FailureConfig,
    sample_failures,
    apply_failures,
    compute_impact,
)


def main() -> None:
    """Generate diverse constellation designs and collect failure metrics."""
    
    # Design space parameters - include smaller constellations that are more vulnerable
    num_planes_choices = [4, 6, 8, 12, 16, 20, 24, 32]
    sats_per_plane_choices = [4, 6, 8, 12, 16, 20, 24, 32]
    inclination_choices = [53.0, 70.0, 97.6]  # Starlink vs Polar
    
    # Failure probability ranges - higher values to induce partitions
    node_fail_probs = [0.05, 0.10, 0.15, 0.20, 0.30]
    edge_fail_probs = [0.10, 0.20, 0.30, 0.40, 0.50]
    
    num_designs = 200
    results = []
    
    random.seed(42)
    
    print(f"Generating {num_designs} constellation designs...")
    print("=" * 60)
    
    for i in range(num_designs):
        # Sample design parameters
        num_planes = random.choice(num_planes_choices)
        sats_per_plane = random.choice(sats_per_plane_choices)
        inclination = random.choice(inclination_choices)
        
        print(f"\n[{i+1}/{num_designs}] planes={num_planes}, sats/plane={sats_per_plane}, inc={inclination}°")
        
        try:
            # Initialize engine with design parameters
            engine = SimulationEngine(
                num_planes=num_planes,
                sats_per_plane=sats_per_plane,
                inclination_deg=inclination,
                altitude_km=550.0,
                duration_minutes=10,
            )
            
            # Run simulation to get bottleneck count
            sim_result = engine.run()
            
            # Get graph and run failure analysis with varied failure rates
            G = engine.get_graph()
            node_fail_prob = random.choice(node_fail_probs)
            edge_fail_prob = random.choice(edge_fail_probs)
            fail_cfg = FailureConfig(
                node_failure_prob=node_fail_prob,
                edge_failure_prob=edge_fail_prob,
                seed=42 + i,
            )
            failures = sample_failures(G, fail_cfg)
            G_failed = apply_failures(G, failures)
            impact = compute_impact(G, G_failed)
            
            # Collect results
            results.append({
                "design_id": i,
                "num_planes": num_planes,
                "sats_per_plane": sats_per_plane,
                "inclination_deg": inclination,
                "num_nodes": sim_result.num_nodes,
                "num_edges": sim_result.num_edges,
                "num_bottlenecks": sim_result.num_bottlenecks,
                "components_after_failure": impact.num_components_after,
            })
            
            print(f"  -> nodes={sim_result.num_nodes}, edges={sim_result.num_edges}, "
                  f"bottlenecks={sim_result.num_bottlenecks}, components_after={impact.num_components_after}")
            
        except Exception as e:
            print(f"  -> ERROR: {e}")
            continue
    
    print("\n" + "=" * 60)
    print(f"Successfully generated {len(results)} designs")
    
    # Save to CSV
    out_dir = PROJECT_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "design_dataset_tier1.csv"
    
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()