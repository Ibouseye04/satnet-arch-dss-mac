#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from satnet.models.autoresearch_rf import (  # noqa: E402
    DEFAULT_EXPERIMENTS_ROOT,
    RfExperimentConfig,
    SUPPORTED_SEARCH_TARGETS,
    load_experiment_config,
    run_rf_experiment,
    run_rf_search_loop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RF-only autoresearch v1 experiments on an existing Tier 1 runs CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("single", "search"),
        default="single",
        help="Run one RF experiment or a tiny search loop.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file. CLI args override defaults when no config file is provided.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to existing tier1_design_runs.csv",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default="gcc_frac_min",
        choices=SUPPORTED_SEARCH_TARGETS,
        help="Continuous resilience target to optimize in RF autoresearch v1.",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="tier1_full",
        choices=("tier1_full", "design_only", "custom"),
        help="Feature set used by the RF model.",
    )
    parser.add_argument(
        "--custom-feature-columns",
        type=str,
        default="",
        help="Comma-separated feature columns when --feature-mode=custom.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees for the baseline RF run.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional RF max depth.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Recorded RF random seed.",
    )
    parser.add_argument(
        "--fidelity-tier",
        type=str,
        default="screening",
        choices=("screening", "confirmatory"),
        help="Metadata only in v1; keeps search-stage separate from confirmatory runs.",
    )
    parser.add_argument(
        "--experiments-root",
        type=str,
        default=str(DEFAULT_EXPERIMENTS_ROOT),
        help="Directory that will contain per-run experiment folders and results.jsonl.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap on generated candidates in search mode.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional note stored in config metadata and notes.txt.",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        default=False,
        help="Skip writing the model.joblib artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = _build_config(args)

    if args.mode == "single":
        result = run_rf_experiment(
            config,
            experiments_root=args.experiments_root,
        )
        print(json.dumps(result, indent=2, default=str))
        return

    search_result = run_rf_search_loop(
        config,
        experiments_root=args.experiments_root,
        max_candidates=args.max_candidates,
    )
    print(json.dumps(search_result, indent=2, default=str))


def _build_config(args: argparse.Namespace) -> RfExperimentConfig:
    if args.config:
        config = load_experiment_config(args.config)
        if args.notes:
            config = RfExperimentConfig.from_dict({
                **config.to_dict(),
                "notes": args.notes,
            })
        return config

    if not args.dataset_path:
        raise SystemExit("ERROR: --dataset-path is required when --config is not provided")

    custom_columns = tuple(
        column.strip()
        for column in args.custom_feature_columns.split(",")
        if column.strip()
    )
    return RfExperimentConfig(
        dataset_path=args.dataset_path,
        target_name=args.target_name,
        feature_mode=args.feature_mode,
        custom_feature_columns=custom_columns,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        seed=args.seed,
        fidelity_tier=args.fidelity_tier,
        notes=args.notes,
        save_model_artifact=not args.no_save_model,
    )


if __name__ == "__main__":
    main()
