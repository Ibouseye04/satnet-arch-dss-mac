from __future__ import annotations

import argparse
import json
from pathlib import Path

from satnet.experiments.final_class_support import run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-tooling-root", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--freeze-archive", type=Path, required=True)
    parser.add_argument("--freeze-archive-hash-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_analysis(
        production_tooling_root=args.production_tooling_root,
        generation_root=args.generation_root,
        replay_root=args.replay_root,
        freeze_root=args.freeze_root,
        freeze_archive=args.freeze_archive,
        freeze_archive_hash_file=args.freeze_archive_hash_file,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
