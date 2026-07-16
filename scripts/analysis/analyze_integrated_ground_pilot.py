from __future__ import annotations

import argparse
import json

from satnet.experiments.integrated_ground_analysis import analyze_pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(analyze_pilot(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
