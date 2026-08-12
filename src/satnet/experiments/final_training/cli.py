from __future__ import annotations

import argparse
import json

from .contracts import AUTHORIZED_TASKS, verify_dataset_bundle, verify_training_plan
from .rf_runner import dry_run_rf_tasks
from .tgnn_runner import dry_run_tgnn_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="satnet-final-training")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-tasks")
    sub.add_parser("list-search-space")
    sub.add_parser("verify")
    sub.add_parser("dry-run")
    for name in ("train-validation", "select-model", "train-final", "evaluate-test"):
        sub.add_parser(name)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list-tasks":
        print(json.dumps({key: {"family": value.family, "task_type": value.task_type, "target": value.target} for key, value in AUTHORIZED_TASKS.items()}, indent=2))
        return 0
    if args.command == "list-search-space":
        from .search import candidate_counts, tgnn_configs
        print(json.dumps({"rf": candidate_counts(), "tgnn_per_task": len(tgnn_configs())}, indent=2))
        return 0
    if args.command == "verify":
        print(json.dumps({"dataset": verify_dataset_bundle(), "training_plan": verify_training_plan()}, indent=2))
        return 0
    if args.command == "dry-run":
        print(json.dumps({"rf": dry_run_rf_tasks(), "tgnn": dry_run_tgnn_tasks()}, indent=2))
        return 0
    raise RuntimeError(f"{args.command} is intentionally unavailable during no-training qualification")


if __name__ == "__main__":
    main()
