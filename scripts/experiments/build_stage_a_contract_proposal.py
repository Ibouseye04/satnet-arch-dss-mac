from __future__ import annotations

import argparse
import json
from pathlib import Path

from satnet.experiments.stage_a_contract.proposal import write_proposal_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root or repo_root / "artifacts/stage_a_discovery_contract_proposal"
    result = write_proposal_artifacts(repo_root, output_root.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
