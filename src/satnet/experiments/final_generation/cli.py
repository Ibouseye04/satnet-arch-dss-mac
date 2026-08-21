from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from satnet.ground.canonical import canonical_json

from .acceptance import validate_production_acceptance, validate_qualification
from .constants import CONTRACT_SPEC_HASH, FINAL_DESIGN_COUNT, FINAL_RUN_COUNT, QUALIFICATION_RUN_IDS, RUN_ID_WIDTH
from .contract import (
    ensure_mode_root,
    protected_science_isolation_passes,
    runtime_preflight,
    validate_catalog,
    validate_frozen_contract,
    validate_output_root,
    validate_science_dependencies,
)
from .io import atomic_write_json
from .ledgers import materialize_generation_ledger, materialize_replay_ledger
from .mapping import FinalRunMapping, map_all_runs
from .orchestrator import generate_run, generate_runs, run_directory
from .repeat import compare_repeat
from .replay import replay_run_read_only, replay_runs_read_only


def _load(*, runtime_sha: str | None) -> tuple[dict[str, Any], tuple[FinalRunMapping, ...]]:
    if runtime_sha is not None:
        runtime_preflight(runtime_sha)
    contract = validate_frozen_contract(compare_tag_blobs=runtime_sha is not None)
    validate_catalog()
    validate_science_dependencies(contract["specification"])
    return contract, map_all_runs(contract)


def _selection(mappings: Sequence[FinalRunMapping], run_ids: Sequence[int]) -> tuple[FinalRunMapping, ...]:
    indexed = {mapping.run_id: mapping for mapping in mappings}
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Duplicate run IDs requested")
    try:
        return tuple(indexed[value] for value in sorted(run_ids))
    except KeyError as exc:
        raise ValueError(f"Unknown frozen run ID: {exc.args[0]}") from exc


def _print(value: dict[str, Any]) -> None:
    print(canonical_json(value))


def _confirm_production(args: argparse.Namespace) -> None:
    if not args.confirm_production:
        raise ValueError("Production requires --confirm-production")
    if args.confirm_contract_spec_hash != CONTRACT_SPEC_HASH:
        raise ValueError("Production contract confirmation mismatch")
    if args.confirm_run_count != FINAL_RUN_COUNT:
        raise ValueError(f"Production run-count confirmation must equal {FINAL_RUN_COUNT}")


def command_preflight(args: argparse.Namespace) -> None:
    _print(runtime_preflight(args.expected_tooling_sha))


def command_inspect(args: argparse.Namespace) -> None:
    _, mappings = _load(runtime_sha=None)
    mapping = mappings[args.run_id]
    _print(
        {
            "design": mapping.design,
            "intended_output_directory": f"run_{mapping.run_id:0{RUN_ID_WIDTH}d}",
            "run": mapping.run,
            "satellite_config_hash": mapping.satellite_config.config_hash(),
        }
    )


def command_dry_run(args: argparse.Namespace) -> None:
    contract, mappings = _load(runtime_sha=args.expected_tooling_sha)
    directories = [f"run_{mapping.run_id:0{RUN_ID_WIDTH}d}" for mapping in mappings]
    run_keys = [mapping.run_key for mapping in mappings]
    design_ids = {mapping.design["design_id"] for mapping in mappings}
    split_counts = {
        split: sum(mapping.run["split_assignment"] == split for mapping in mappings)
        for split in ("train", "validation", "test")
    }
    split_design_counts = {
        split: len({mapping.design["design_id"] for mapping in mappings if mapping.run["split_assignment"] == split})
        for split in ("train", "validation", "test")
    }
    if (
        len(set(directories)) != FINAL_RUN_COUNT
        or len(set(run_keys)) != FINAL_RUN_COUNT
        or len(design_ids) != FINAL_DESIGN_COUNT
    ):
        raise ValueError("Dry-run identity uniqueness failed")
    if split_counts != {"train": 7000, "validation": 1500, "test": 1500} or split_design_counts != {"train": 1400, "validation": 300, "test": 300}:
        raise ValueError("Dry-run split cardinality failed")
    _print(
        {
            "contract_spec_hash": contract["contract_spec_hash"],
            "design_count": len(design_ids),
            "executed_simulation_count": 0,
            "run_count": len(mappings),
            "run_id_max": mappings[-1].run_id,
            "run_id_min": mappings[0].run_id,
            "split_design_counts": split_design_counts,
            "split_run_counts": split_counts,
            "state": "passed",
            "unique_output_directory_count": len(set(directories)),
            "unique_run_key_count": len(set(run_keys)),
        }
    )


def command_qualify(args: argparse.Namespace) -> None:
    _, mappings = _load(runtime_sha=args.expected_tooling_sha)
    mapping = mappings[args.run_id]
    expected_mode = "qualification_repeat" if args.repeat else "qualification"
    result = generate_run(
        mapping=mapping,
        catalog=validate_catalog(),
        output_root=args.output_root,
        mode=expected_mode,
        retry=args.retry,
        verified_resume=args.verified_resume,
        resume_replay_root=args.resume_replay_root,
    )
    _print(result)


def command_qualify_design(args: argparse.Namespace) -> None:
    _, mappings = _load(runtime_sha=args.expected_tooling_sha)
    selected = tuple(mapping for mapping in mappings if mapping.design["design_id"] == args.design_id)
    if len(selected) != 5:
        raise ValueError("Frozen design must contain exactly five realizations")
    results = generate_runs(
        mappings=selected,
        catalog=validate_catalog(),
        output_root=args.output_root,
        mode="qualification",
    )
    _print({"design_id": args.design_id, "result_hashes": [value["run_result_hash"] for value in results]})


def command_qualify_set(args: argparse.Namespace) -> None:
    _, mappings = _load(runtime_sha=args.expected_tooling_sha)
    selected = _selection(mappings, QUALIFICATION_RUN_IDS)
    catalog = validate_catalog()
    results = generate_runs(
        mappings=selected,
        catalog=catalog,
        output_root=args.output_root,
        mode="qualification",
    )
    ledger = materialize_generation_ledger(
        output_root=args.output_root, mappings=selected, catalog=catalog
    )
    _print(
        {
            "generation_ledger": ledger,
            "result_hashes": [value["run_result_hash"] for value in results],
            "run_ids": list(QUALIFICATION_RUN_IDS),
        }
    )


def command_replay(args: argparse.Namespace) -> None:
    _, mappings = _load(runtime_sha=args.expected_tooling_sha)
    report = replay_run_read_only(
        mapping=mappings[args.run_id],
        catalog=validate_catalog(),
        input_root=args.input_root,
        replay_output_root=args.replay_output_root,
    )
    _print(report)


def command_replay_set(args: argparse.Namespace) -> None:
    _, mappings = _load(runtime_sha=args.expected_tooling_sha)
    selected = _selection(mappings, QUALIFICATION_RUN_IDS)
    reports = replay_runs_read_only(
        mappings=selected,
        catalog=validate_catalog(),
        input_root=args.input_root,
        replay_output_root=args.replay_output_root,
    )
    ledger = materialize_replay_ledger(
        replay_root=args.replay_output_root, mappings=selected
    )
    _print(
        {
            "replay_ledger": ledger,
            "run_ids": [value["run_id"] for value in reports],
            "successful_replays": len(reports),
        }
    )


def command_materialize_ledgers(args: argparse.Namespace) -> None:
    _, mappings = _load(runtime_sha=None)
    selected = _selection(mappings, QUALIFICATION_RUN_IDS if args.qualification else range(FINAL_RUN_COUNT))
    generation = materialize_generation_ledger(
        output_root=args.input_root, mappings=selected, catalog=validate_catalog()
    )
    result: dict[str, Any] = {"generation_ledger": generation}
    if args.replay_root:
        result["replay_ledger"] = materialize_replay_ledger(
            replay_root=args.replay_root, mappings=selected
        )
    _print(result)


def command_validate_qualification(args: argparse.Namespace) -> None:
    _, mappings = _load(runtime_sha=None)
    selected = _selection(mappings, QUALIFICATION_RUN_IDS)
    _print(
        validate_qualification(
            mappings=selected,
            generation_root=args.input_root,
            replay_root=args.replay_root,
        )
    )


def command_compare_repeat(args: argparse.Namespace) -> None:
    _print(
        compare_repeat(
            primary_root=args.input_root,
            repeat_root=args.repeat_root,
            run_id=args.run_id,
        )
    )


def command_generate_production(args: argparse.Namespace) -> None:
    _confirm_production(args)
    _, mappings = _load(runtime_sha=args.expected_tooling_sha)
    ensure_mode_root(args.output_root, "production", create=True)
    catalog = validate_catalog()
    results = tuple(
        generate_run(
            mapping=mapping,
            catalog=catalog,
            output_root=args.output_root,
            mode="production",
            retry=args.retry,
            verified_resume=args.verified_resume,
            resume_replay_root=args.resume_replay_root,
        )
        for mapping in mappings
    )
    ledger = materialize_generation_ledger(
        output_root=args.output_root, mappings=mappings, catalog=catalog
    )
    _print(
        {
            "generation_ledger": ledger,
            "production_generation_count": len(results),
        }
    )


def command_replay_production(args: argparse.Namespace) -> None:
    _confirm_production(args)
    _, mappings = _load(runtime_sha=args.expected_tooling_sha)
    reports = replay_runs_read_only(
        mappings=mappings,
        catalog=validate_catalog(),
        input_root=args.input_root,
        replay_output_root=args.replay_output_root,
        input_mode="production",
        output_mode="production_replay",
    )
    ledger = materialize_replay_ledger(
        replay_root=args.replay_output_root, mappings=mappings
    )
    _print(
        {
            "production_replay_count": len(reports),
            "replay_ledger": ledger,
        }
    )


def command_validate_production(args: argparse.Namespace) -> None:
    _confirm_production(args)
    _, mappings = _load(runtime_sha=args.expected_tooling_sha)
    report = validate_production_acceptance(
        mappings=mappings,
        generation_root=args.input_root,
        replay_root=args.replay_root,
        protected_science_diff_empty=protected_science_isolation_passes(),
    )
    report_path = Path(args.acceptance_report)
    validate_output_root(report_path.parent)
    atomic_write_json(report_path, report)
    _print(report)


def command_production(args: argparse.Namespace) -> None:
    command_generate_production(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="satnet-final-generation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--expected-tooling-sha", required=True)
    preflight.set_defaults(handler=command_preflight)

    inspect = subparsers.add_parser("inspect-run")
    inspect.add_argument("--run-id", type=int, required=True, choices=range(FINAL_RUN_COUNT))
    inspect.set_defaults(handler=command_inspect)

    dry = subparsers.add_parser("dry-run")
    dry.add_argument("--expected-tooling-sha", required=True)
    dry.set_defaults(handler=command_dry_run)

    qualify = subparsers.add_parser("qualify")
    qualify.add_argument("--run-id", type=int, required=True, choices=range(FINAL_RUN_COUNT))
    qualify.add_argument("--output-root", type=Path, required=True)
    qualify.add_argument("--expected-tooling-sha", required=True)
    qualify.add_argument("--retry", action="store_true")
    qualify.add_argument("--verified-resume", action="store_true")
    qualify.add_argument("--resume-replay-root", type=Path)
    qualify.add_argument("--repeat", action="store_true")
    qualify.set_defaults(handler=command_qualify)

    design = subparsers.add_parser("qualify-design")
    design.add_argument("--design-id", required=True)
    design.add_argument("--output-root", type=Path, required=True)
    design.add_argument("--expected-tooling-sha", required=True)
    design.set_defaults(handler=command_qualify_design)

    qualify_set = subparsers.add_parser("qualify-set")
    qualify_set.add_argument("--output-root", type=Path, required=True)
    qualify_set.add_argument("--expected-tooling-sha", required=True)
    qualify_set.set_defaults(handler=command_qualify_set)

    replay = subparsers.add_parser("replay")
    replay.add_argument("--run-id", type=int, required=True, choices=range(FINAL_RUN_COUNT))
    replay.add_argument("--input-root", type=Path, required=True)
    replay.add_argument("--replay-output-root", type=Path, required=True)
    replay.add_argument("--expected-tooling-sha", required=True)
    replay.set_defaults(handler=command_replay)

    replay_set = subparsers.add_parser("replay-set")
    replay_set.add_argument("--input-root", type=Path, required=True)
    replay_set.add_argument("--replay-output-root", type=Path, required=True)
    replay_set.add_argument("--expected-tooling-sha", required=True)
    replay_set.set_defaults(handler=command_replay_set)

    ledgers = subparsers.add_parser("materialize-ledgers")
    ledgers.add_argument("--input-root", type=Path, required=True)
    ledgers.add_argument("--replay-root", type=Path)
    ledgers.add_argument("--qualification", action="store_true")
    ledgers.set_defaults(handler=command_materialize_ledgers)

    validation = subparsers.add_parser("validate-qualification")
    validation.add_argument("--input-root", type=Path, required=True)
    validation.add_argument("--replay-root", type=Path, required=True)
    validation.set_defaults(handler=command_validate_qualification)

    repeat = subparsers.add_parser("compare-repeat")
    repeat.add_argument("--input-root", type=Path, required=True)
    repeat.add_argument("--repeat-root", type=Path, required=True)
    repeat.add_argument("--run-id", type=int, choices=range(FINAL_RUN_COUNT), default=4115)
    repeat.set_defaults(handler=command_compare_repeat)

    def add_production_confirmations(command: argparse.ArgumentParser) -> None:
        command.add_argument("--confirm-production", action="store_true")
        command.add_argument("--confirm-contract-spec-hash", required=True)
        command.add_argument("--confirm-run-count", type=int, required=True)
        command.add_argument("--expected-tooling-sha", required=True)

    production = subparsers.add_parser("production")
    production.add_argument("--output-root", type=Path, required=True)
    production.add_argument("--retry", action="store_true")
    production.add_argument("--verified-resume", action="store_true")
    production.add_argument("--resume-replay-root", type=Path)
    add_production_confirmations(production)
    production.set_defaults(handler=command_production)

    generate_production = subparsers.add_parser("generate-production")
    generate_production.add_argument("--output-root", type=Path, required=True)
    generate_production.add_argument("--retry", action="store_true")
    generate_production.add_argument("--verified-resume", action="store_true")
    generate_production.add_argument("--resume-replay-root", type=Path)
    add_production_confirmations(generate_production)
    generate_production.set_defaults(handler=command_generate_production)

    replay_production = subparsers.add_parser("replay-production")
    replay_production.add_argument("--input-root", type=Path, required=True)
    replay_production.add_argument("--replay-output-root", type=Path, required=True)
    add_production_confirmations(replay_production)
    replay_production.set_defaults(handler=command_replay_production)

    validate_production = subparsers.add_parser("validate-production")
    validate_production.add_argument("--input-root", type=Path, required=True)
    validate_production.add_argument("--replay-root", type=Path, required=True)
    validate_production.add_argument("--acceptance-report", type=Path, required=True)
    add_production_confirmations(validate_production)
    validate_production.set_defaults(handler=command_validate_production)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0
