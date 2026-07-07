#!/usr/bin/env python3
"""
Create Animated GIF from Constellation Timestep Visualizations

Combines individual PNG frames from a timestep directory into an animated GIF.

Usage:
    python scripts/create_gif.py --input degraded_network_timesteps --output degraded_network.gif --duration 500
"""

import argparse
from pathlib import Path
from PIL import Image


def create_gif(input_dir: str, output_file: str, duration_ms: int = 500):
    """Create animated GIF from PNG files in input directory."""
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist")
        return
    
    # Get all PNG files, sorted by name (t00, t01, etc.)
    png_files = sorted(input_path.glob("*.png"))
    
    if not png_files:
        print(f"Error: No PNG files found in {input_path}")
        return
    
    print(f"Found {len(png_files)} PNG files")
    
    # Load images
    images = []
    for png_file in png_files:
        print(f"  Loading {png_file.name}...")
        img = Image.open(png_file)
        images.append(img)
    
    # Save as GIF
    print(f"Creating animated GIF: {output_path}")
    print(f"  Frame duration: {duration_ms}ms")
    print(f"  Total duration: {duration_ms * len(images) / 1000:.1f}s")
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,  # Infinite loop
        optimize=True,
    )
    
    print(f"✓ Saved GIF to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Create animated GIF from timestep visualizations")
    parser.add_argument("--input", required=True, help="Input directory with PNG frames")
    parser.add_argument("--output", required=True, help="Output GIF filename")
    parser.add_argument("--duration", type=int, default=500, help="Frame duration in milliseconds (default: 500)")
    
    args = parser.parse_args()
    
    create_gif(args.input, args.output, args.duration)


if __name__ == "__main__":
    main()
