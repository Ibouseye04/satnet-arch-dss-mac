#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from satnet.models.autoresearch_rf import (  # noqa: E402
    DEFAULT_CANDIDATE_PATH,
    DEFAULT_EXPERIMENTS_ROOT,
    DEFAULT_LAST_RUN_PATH,
    DEFAULT_MUTATION_POLICY_PATH,
    RfExperimentConfig,
    SUPPORTED_SEARCH_TARGETS,
    load_experiment_config,
    run_agent_candidate,
    run_confirmatory_experiment,
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
        choices=("single", "search", "candidate", "confirmatory"),
        default="candidate",
        help="Run one RF experiment, a tiny search loop, a policy-validated candidate, or a confirmatory rerun.",
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
    parser.add_argument(
        "--candidate-path",
        type=str,
        default=str(DEFAULT_CANDIDATE_PATH),
        help="JSON config file used in candidate mode.",
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default=str(DEFAULT_MUTATION_POLICY_PATH),
        help="Mutation policy JSON enforced in candidate and confirmatory modes.",
    )
    parser.add_argument(
        "--source-experiment-id",
        type=str,
        default=None,
        help="Existing screening experiment ID to rerun in confirmatory mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "candidate":
        try:
            result = run_agent_candidate(
                candidate_path=args.candidate_path,
                policy_path=args.policy_path,
                experiments_root=args.experiments_root,
            )
        except Exception as exc:
            print(_write_command_failure_result(args.experiments_root, args.mode, exc))
            return
        print(_stable_command_result(args.experiments_root, result))
        return

    if args.mode == "confirmatory":
        if not args.source_experiment_id:
            exc = SystemExit("ERROR: --source-experiment-id is required when --mode=confirmatory")
            print(_write_command_failure_result(args.experiments_root, args.mode, exc))
            raise exc
        try:
            result = run_confirmatory_experiment(
                args.source_experiment_id,
                experiments_root=args.experiments_root,
                policy_path=args.policy_path,
            )
        except Exception as exc:
            print(_write_command_failure_result(args.experiments_root, args.mode, exc))
            return
        print(_stable_command_result(args.experiments_root, result))
        return

    config = _build_config(args)

    if args.mode == "single":
        result = run_rf_experiment(
            config,
            experiments_root=args.experiments_root,
        )
        print(_stable_command_result(args.experiments_root, result))
        return

    search_result = run_rf_search_loop(
        config,
        experiments_root=args.experiments_root,
        max_candidates=args.max_candidates,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "result_count": search_result["result_count"],
                "best_experiment_id": search_result["best_experiment_id"],
                "baseline_experiment_id": search_result["baseline_experiment_id"],
                "last_run_path": str(Path(args.experiments_root).expanduser().resolve() / DEFAULT_LAST_RUN_PATH.name),
            },
            indent=2,
            default=str,
        )
    )


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


def _stable_command_result(experiments_root: str, result: dict) -> str:
    experiments_root_path = Path(experiments_root).expanduser().resolve()
    return json.dumps(
        {
            "status": result.get("status"),
            "experiment_id": result.get("experiment_id"),
            "promotion_decision": result.get("promotion_decision"),
            "last_run_path": str(experiments_root_path / DEFAULT_LAST_RUN_PATH.name),
            "summary_path": result.get("artifact_paths", {}).get("summary"),
            "incumbent_path": result.get("artifact_paths", {}).get("incumbent"),
        },
        indent=2,
        default=str,
    )


def _write_command_failure_result(experiments_root: str, mode: str, exc: BaseException) -> str:
    experiments_root_path = Path(experiments_root).expanduser().resolve()
    experiments_root_path.mkdir(parents=True, exist_ok=True)
    last_run_path = experiments_root_path / DEFAULT_LAST_RUN_PATH.name
    payload = {
        "status": "failed",
        "mode": mode,
        "experiment_id": None,
        "promotion_decision": "failed",
        "promotion_reason": f"{exc.__class__.__name__}: {exc}",
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
        "summary_path": None,
        "incumbent_path": str(experiments_root_path / "incumbent.json"),
        "last_run_path": str(last_run_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(last_run_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return json.dumps(payload, indent=2, default=str)


if __name__ == "__main__":
    main()
