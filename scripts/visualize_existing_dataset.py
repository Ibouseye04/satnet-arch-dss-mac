#!/usr/bin/env python3
"""
Visualize Existing Tier 1 Dataset

Reads an existing Tier 1 dataset (runs + steps CSV files) and generates
visualizations for a specific run, showing the degraded network state.

Usage:
    python scripts/visualize_existing_dataset.py --runs data/tier1_design_runs.csv --steps data/tier1_design_steps.csv --run-id 0
"""

import sys
import argparse
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from satnet.network.hypatia_adapter import HypatiaAdapter
from satnet.simulation.tier1_rollout import Tier1FailureRealization
from datetime import datetime


def main():
    """Generate visualization from existing dataset."""
    parser = argparse.ArgumentParser(description="Visualize existing Tier 1 dataset")
    parser.add_argument("--runs", required=True, help="Path to runs CSV file")
    parser.add_argument("--steps", required=True, help="Path to steps CSV file")
    parser.add_argument("--run-id", type=int, default=0, help="Run ID to visualize (default: 0)")
    parser.add_argument("--output-dir", default="dataset_visualization", help="Output directory for images")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Visualizing Existing Tier 1 Dataset")
    print("=" * 60)
    
    # 1. Load dataset
    print(f"\n[1/5] Loading dataset...")
    print(f"  Runs file: {args.runs}")
    print(f"  Steps file: {args.steps}")
    
    runs_df = pd.read_csv(args.runs)
    steps_df = pd.read_csv(args.steps)
    
    print(f"  Total runs: {len(runs_df)}")
    print(f"  Total steps: {len(steps_df)}")
    
    # 2. Select specific run
    print(f"\n[2/5] Selecting run {args.run_id}...")
    if args.run_id >= len(runs_df):
        print(f"Error: run_id {args.run_id} out of range (max: {len(runs_df)-1})")
        return
    
    run_data = runs_df.iloc[args.run_id]
    run_steps = steps_df[steps_df['run_id'] == args.run_id].sort_values('t')
    
    print(f"  Constellation: {run_data['num_planes']} planes × {run_data['sats_per_plane']} sats")
    print(f"  Total satellites: {run_data['total_satellites']}")
    print(f"  Failed nodes: {run_data['num_failed_nodes']}")
    print(f"  Failed edges: {run_data['num_failed_edges']}")
    print(f"  GCC fraction min: {run_data['gcc_frac_min']:.3f}")
    print(f"  Steps in this run: {len(run_steps)}")
    
    # 3. Parse configuration
    print(f"\n[3/5] Reconstructing configuration...")
    
    # Parse epoch (use default if not in dataset)
    epoch_iso = run_data.get('epoch_iso', "2000-01-01T12:00:00+00:00")
    epoch = datetime.fromisoformat(epoch_iso)
    
    # Create HypatiaAdapter
    adapter = HypatiaAdapter(
        num_planes=int(run_data['num_planes']),
        sats_per_plane=int(run_data['sats_per_plane']),
        inclination_deg=float(run_data['inclination_deg']),
        altitude_km=float(run_data['altitude_km']),
        epoch=epoch,
    )
    
    # Generate TLEs and calculate ISLs
    adapter.generate_tles()
    adapter.calculate_isls(
        duration_minutes=int(run_data['duration_minutes']),
        step_seconds=int(run_data['step_seconds'])
    )
    
    # Parse failures
    failures = Tier1FailureRealization.from_json_strings(
        run_data['failed_nodes_json'],
        run_data['failed_edges_json']
    )
    
    print(f"  Reconstructed {len(failures.failed_nodes)} failed nodes")
    print(f"  Reconstructed {len(failures.failed_edges)} failed edges")
    
    # 4. Generate visualizations
    print(f"\n[4/5] Generating visualizations...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for step_idx, step_data in run_steps.iterrows():
        t = int(step_data['t'])
        print(f"  Processing step t={t}...", end=" ")
        
        # Get baseline graph and positions
        positions = adapter.get_positions_at_step(t)
        baseline_graph = adapter.get_graph_at_step(t)
        
        # Apply failures to get degraded graph
        degraded_graph = baseline_graph.copy()
        
        # Remove failed nodes
        nodes_to_remove = [n for n in failures.failed_nodes if degraded_graph.has_node(n)]
        degraded_graph.remove_nodes_from(nodes_to_remove)
        
        # Remove failed edges
        for u, v in failures.failed_edges:
            if degraded_graph.has_edge(u, v):
                degraded_graph.remove_edge(u, v)
        
        # Extract lat/lon arrays
        lons = np.array([pos.lon_deg for pos in positions])
        lats = np.array([pos.lat_deg for pos in positions])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(16, 8), dpi=150)
        
        # Set up the map extent
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude (°)", fontsize=12)
        ax.set_ylabel("Latitude (°)", fontsize=12)
        ax.set_title(
            f"Run {args.run_id} at t={t} (GCC: {step_data['gcc_frac']:.3f}, "
            f"Nodes: {step_data['num_nodes']}, Edges: {step_data['num_edges']})",
            fontsize=14,
            fontweight="bold",
        )
        
        # Draw background grid
        ax.set_facecolor("#e6f2ff")
        ax.grid(True, linestyle="--", alpha=0.5, color="gray")
        ax.set_xticks(np.arange(-180, 181, 30))
        ax.set_yticks(np.arange(-90, 91, 30))
        
        # Add equator and prime meridian
        ax.axhline(y=0, color="darkgray", linewidth=1.0, linestyle="-")
        ax.axvline(x=0, color="darkgray", linewidth=1.0, linestyle="-")
        
        # Draw ISL links
        link_count = 0
        seam_count = 0
        skipped_dateline = 0
        
        for u, v, data in degraded_graph.edges(data=True):
            lon1, lat1 = lons[u], lats[u]
            lon2, lat2 = lons[v], lats[v]
            
            # Skip links that cross the dateline
            if abs(lon1 - lon2) > 180:
                skipped_dateline += 1
                continue
            
            link_type = data.get("link_type", "isl")
            
            if link_type == "seam_link":
                ax.plot([lon1, lon2], [lat1, lat2], color="red", linewidth=0.8, alpha=0.7, zorder=2)
                seam_count += 1
            else:
                ax.plot([lon1, lon2], [lat1, lat2], color="gray", linewidth=0.3, alpha=0.5, zorder=1)
            
            link_count += 1
        
        # Draw satellites
        healthy_mask = np.array([i not in failures.failed_nodes for i in range(len(positions))])
        
        # Healthy satellites
        ax.scatter(lons[healthy_mask], lats[healthy_mask], c="blue", s=2, zorder=3, label="Healthy Satellite")
        
        # Failed satellites
        if not healthy_mask.all():
            ax.scatter(lons[~healthy_mask], lats[~healthy_mask], c="red", s=20, marker="x", zorder=4, label="Failed Satellite")
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=6, label="Healthy Satellite"),
            Line2D([0], [0], marker="x", color="red", markersize=8, label="Failed Satellite"),
            Line2D([0], [0], color="gray", linewidth=1, label="Active ISL"),
            Line2D([0], [0], color="red", linewidth=1.5, label="Seam Link"),
        ]
        ax.legend(handles=legend_elements, loc="lower left", fontsize=10)
        
        # Add stats annotation
        stats_text = (
            f"Active links: {link_count}\n"
            f"Seam links: {seam_count}\n"
            f"Failed nodes: {len(failures.failed_nodes)}\n"
            f"Failed edges: {len(failures.failed_edges)}\n"
            f"GCC frac: {step_data['gcc_frac']:.3f}\n"
            f"Partitioned: {step_data['partitioned']}"
        )
        ax.annotate(
            stats_text,
            xy=(0.99, 0.02),
            xycoords="axes fraction",
            fontsize=9,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        
        plt.tight_layout()
        
        # Save the figure
        output_path = output_dir / f"run{args.run_id}_t{t:02d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"✓")
    
    # 5. Summary
    print(f"\n[5/5] Summary:")
    print(f"  Visualizations saved to: {output_dir}")
    print(f"  Total frames: {len(run_steps)}")
    print(f"  Run ID: {args.run_id}")
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
