#!/usr/bin/env python3
"""
Visualize Degraded Satellite Constellation After Tier 1 Rollout

Generates time-series visualizations showing how the network degrades
after applying node and edge failures from a Tier 1 rollout.

Shows:
- Baseline topology (Hypatia output, no failures)
- Degraded topology (with failures applied)
- Failed nodes highlighted in red
- Failed edges removed from the graph

Usage:
    python scripts/visualize_degraded_network.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from satnet.network.hypatia_adapter import HypatiaAdapter
from satnet.simulation.tier1_rollout import Tier1RolloutConfig, run_tier1_rollout


def main():
    """Generate degraded constellation visualization."""
    print("=" * 60)
    print("Degraded Constellation Visualization (Tier 1 Rollout)")
    print("=" * 60)
    
    # 1. Setup: Configure Tier 1 rollout with failures
    print("\n[1/5] Configuring Tier 1 rollout with failures...")
    cfg = Tier1RolloutConfig(
        num_planes=12,
        sats_per_plane=12,
        inclination_deg=53.0,
        altitude_km=550.0,
        duration_minutes=10,
        step_seconds=60,
        node_failure_prob=0.05,  # 5% of satellites fail
        edge_failure_prob=0.10,  # 10% of ISLs fail
        seed=42,
    )
    print(f"  Constellation: {cfg.num_planes} planes × {cfg.sats_per_plane} sats = {cfg.total_satellites} sats")
    print(f"  Node failure prob: {cfg.node_failure_prob:.1%}")
    print(f"  Edge failure prob: {cfg.edge_failure_prob:.1%}")
    print(f"  Duration: {cfg.duration_minutes} minutes")
    print(f"  Steps: {cfg.num_steps}")
    
    # 2. Run Tier 1 rollout
    print("\n[2/5] Running Tier 1 rollout...")
    steps, summary, failures = run_tier1_rollout(cfg)
    print(f"  Failed nodes: {summary.num_failed_nodes}")
    print(f"  Failed edges: {summary.num_failed_edges}")
    print(f"  GCC fraction min: {summary.gcc_frac_min:.3f}")
    print(f"  GCC fraction mean: {summary.gcc_frac_mean:.3f}")
    print(f"  Partition any: {summary.partition_any}")
    
    # 3. Reconstruct Hypatia adapter for visualization
    print("\n[3/5] Reconstructing Hypatia adapter for visualization...")
    adapter = HypatiaAdapter(
        num_planes=cfg.num_planes,
        sats_per_plane=cfg.sats_per_plane,
        inclination_deg=cfg.inclination_deg,
        altitude_km=cfg.altitude_km,
        epoch=cfg.epoch,
    )
    adapter.generate_tles()
    adapter.calculate_isls(duration_minutes=cfg.duration_minutes, step_seconds=cfg.step_seconds)
    
    # 4. Generate visualizations for each time step
    print("\n[4/5] Generating degraded network visualizations...")
    
    output_dir = Path(__file__).parent.parent / "degraded_network_timesteps"
    output_dir.mkdir(exist_ok=True)
    
    for step_idx, step_data in enumerate(steps):
        print(f"  Processing step t={step_idx}/{len(steps)-1}...", end=" ")
        
        # Get baseline graph and positions
        positions = adapter.get_positions_at_step(step_idx)
        baseline_graph = adapter.get_graph_at_step(step_idx)
        
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
            f"Degraded Network at t={step_idx} (GCC: {step_data.gcc_frac:.3f}, "
            f"Nodes: {step_data.num_nodes}, Edges: {step_data.num_edges})",
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
        
        # Draw ISL links (only remaining ones in degraded graph)
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
        # Healthy nodes in blue, failed nodes in red
        healthy_mask = np.array([i not in failures.failed_nodes for i in range(len(positions))])
        
        # Healthy satellites
        ax.scatter(lons[healthy_mask], lats[healthy_mask], c="blue", s=2, zorder=3, label="Healthy Satellite")
        
        # Failed satellites (if any in this step)
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
            f"Failed nodes: {summary.num_failed_nodes}\n"
            f"Failed edges: {summary.num_failed_edges}\n"
            f"GCC frac: {step_data.gcc_frac:.3f}\n"
            f"Partitioned: {step_data.partitioned}"
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
        output_path = output_dir / f"degraded_t{step_idx:02d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"✓")
    
    print(f"\n[5/5] Saved {len(steps)} visualizations to: {output_dir}")
    
    # Summary
    print("\nSummary:")
    print(f"  Total satellites: {cfg.total_satellites}")
    print(f"  Failed nodes: {summary.num_failed_nodes} ({summary.num_failed_nodes/cfg.total_satellites:.1%})")
    print(f"  Failed edges: {summary.num_failed_edges}")
    print(f"  GCC fraction min: {summary.gcc_frac_min:.3f}")
    print(f"  GCC fraction mean: {summary.gcc_frac_mean:.3f}")
    print(f"  Network partitioned: {'Yes' if summary.partition_any else 'No'}")
    print(f"  Output directory: {output_dir}")
    
    print("\n" + "=" * 60)
    print("Degraded network visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
