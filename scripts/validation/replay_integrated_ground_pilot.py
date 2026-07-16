from __future__ import annotations

import argparse
import json

from satnet.experiments.integrated_ground_replay import replay_pilot_manifest


def _run_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("run IDs must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("run subset must not be empty")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-ids", type=_run_ids)
    parser.add_argument("--repeat", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = replay_pilot_manifest(
        manifest_path=args.manifest,
        output_root=args.output_root,
        run_ids=args.run_ids,
        repeat=args.repeat,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["failed_replay_run_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
