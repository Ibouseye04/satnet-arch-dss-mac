#!/usr/bin/env python3
"""
Visualize Satellite Constellation Map

Generates a high-quality 2D map of the satellite constellation showing:
- Satellite positions as blue dots
- ISL links as gray lines (seam links in red)
- Handles dateline crossing by skipping problematic links

Usage:
    python scripts/visualize_constellation.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from satnet.network.hypatia_adapter import HypatiaAdapter


def main():
    """Generate constellation visualization."""
    print("=" * 60)
    print("Constellation Visualization")
    print("=" * 60)
    
    # 1. Setup: Initialize HypatiaAdapter with medium constellation for better connectivity visualization
    print("\n[1/4] Initializing HypatiaAdapter (12 planes x 12 sats)...")
    adapter = HypatiaAdapter(
        num_planes=12,
        sats_per_plane=12,
        inclination_deg=53.0,
        altitude_km=550.0,
    )
    print(f"  Total satellites: {adapter.total_satellites}")
    
    # 2. Data: Generate TLEs and calculate ISLs
    print("\n[2/5] Generating TLEs and calculating ISLs...")
    duration_minutes = 10  # Simulate 10 minutes
    step_seconds = 60      # 1-minute steps
    adapter.generate_tles()
    adapter.calculate_isls(duration_minutes=duration_minutes, step_seconds=step_seconds)
    
    num_steps = int(duration_minutes * 60 / step_seconds) + 1
    print(f"  Time steps: {num_steps} (t=0 to t={num_steps-1})")
    
    # 3. Generate visualization for each time step
    print("\n[3/5] Generating visualizations for each time step...")
    
    output_dir = Path(__file__).parent.parent / "constellation_timesteps"
    output_dir.mkdir(exist_ok=True)
    
    for step_idx in range(num_steps):
        print(f"  Processing step t={step_idx}/{num_steps-1}...", end=" ")
        
        # Extract positions and graph at this step
        positions = adapter.get_positions_at_step(step_idx)
        graph = adapter.get_graph_at_step(step_idx)
        
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
            f"Satellite Constellation at t={step_idx} ({adapter.config.num_planes} planes × "
            f"{adapter.config.sats_per_plane} sats = {adapter.total_satellites} satellites)",
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
        
        for u, v, data in graph.edges(data=True):
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
        
        # Draw satellites as blue dots
        ax.scatter(lons, lats, c="blue", s=2, zorder=3, label="Satellites")
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=6, label="Satellite"),
            Line2D([0], [0], color="gray", linewidth=1, label="ISL Link"),
            Line2D([0], [0], color="red", linewidth=1.5, label="Seam Link"),
        ]
        ax.legend(handles=legend_elements, loc="lower left", fontsize=10)
        
        # Add stats annotation
        stats_text = (
            f"Links: {link_count}\n"
            f"Seam: {seam_count}\n"
            f"Skipped: {skipped_dateline}"
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
        output_path = output_dir / f"constellation_t{step_idx:02d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"✓")
    
    print(f"\n[4/5] Saved {num_steps} visualizations to: {output_dir}")
    
    # 5. Summary
    print("\n[5/5] Summary:")
    print(f"  Constellation: {adapter.config.num_planes} planes × {adapter.config.sats_per_plane} sats")
    print(f"  Total satellites: {adapter.total_satellites}")
    print(f"  Duration: {duration_minutes} minutes")
    print(f"  Time steps: {num_steps}")
    print(f"  Output directory: {output_dir}")
    
    print("\n" + "=" * 60)
    print("Time-series visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
