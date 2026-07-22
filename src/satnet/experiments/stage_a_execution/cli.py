from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from satnet.experiments.final_generation.contract import validate_catalog

from .acceptance import evaluate_acceptance
from .artifact_contract import artifact_contract_hash
from .authorization import Authorization, load_authorization, validate_authorization
from .common import canonical_json_bytes
from .contract import CONTRACT_RELATIVE_ROOT, FrozenStageAContract, load_frozen_contract
from .generate import execute_generation, make_validated_production_adapter
from .identity import tooling_identity
from .ledger import read_ledger
from .locking import recover_stale_lock
from .plan import build_plan
from .preflight import run_preflight
from .replay import execute_replay


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _print(value: dict[str, Any]) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value))


def _context(args: argparse.Namespace, operation: str) -> tuple[FrozenStageAContract, Authorization | None, dict[str, Any]]:
    repo = _repo_root()
    contract = load_frozen_contract(repo, repo / CONTRACT_RELATIVE_ROOT)
    stable_commit, executable_hash, proposal_hash = tooling_identity(repo)
    science_contract_hash = artifact_contract_hash()
    authorization = None if args.authorization is None else load_authorization(args.authorization)
    provisional = build_plan(
        contract, partition=args.partition, operation=operation,
        stable_executable_commit=stable_commit, executable_inventory_hash=executable_hash,
        tooling_proposal_hash=proposal_hash, artifact_contract_hash=science_contract_hash,
        generation_root=args.generation_root, replay_root=args.replay_root,
        acceptance_root=args.acceptance_root, authorization=authorization,
    )
    if authorization is not None:
        validate_authorization(
            authorization, contract=contract, stable_executable_commit=stable_commit,
            executable_inventory_hash=executable_hash, tooling_proposal_hash=proposal_hash,
            artifact_contract_hash=science_contract_hash, operation=operation,
            partition=args.partition, run_ids=[row["global_run_id"] for row in provisional["runs"]],
            generation_root=args.generation_root, replay_root=args.replay_root,
            acceptance_root=args.acceptance_root,
        )
    return contract, authorization, provisional


def command_verify_contract(args: argparse.Namespace) -> None:
    contract = load_frozen_contract(_repo_root(), args.contract_root)
    _print({
        "contract_hash": contract.contract_hash,
        "design_count": len(contract.designs),
        "run_count": len(contract.runs),
        "seed_count": len(contract.seeds),
        "simulation_authorized": False,
        "verification": "PASSED",
    })


def command_validate_authorization(args: argparse.Namespace) -> None:
    _, authorization, plan = _context(args, args.operation)
    if authorization is None:
        raise PermissionError("EXECUTION NOT AUTHORIZED")
    _print({"authorization_hash": authorization.sha256, "operation": args.operation, "partition": args.partition, "run_count": plan["run_count"], "validation": "PASSED"})


def command_plan(args: argparse.Namespace) -> None:
    _, authorization, plan = _context(args, "PLAN")
    runs = plan["runs"]
    _print({
        "authorization_status": "AUTHORIZED" if authorization is not None else "EXECUTION NOT AUTHORIZED",
        "contract_hash": plan["contract_hash"],
        "design_count": plan["design_count"],
        "execution_authorized": authorization is not None,
        "first_run_identity": runs[0]["run_key"],
        "last_run_identity": runs[-1]["run_key"],
        "partition": plan["partition"],
        "plan_hash": plan["plan_hash"],
        "run_count": plan["run_count"],
        "sealed_holdout": {"design_count": 5, "run_count": 25, "identities": "REDACTED"},
        "simulation_executed": False,
    })


def command_preflight(args: argparse.Namespace) -> None:
    contract, authorization, plan = _context(args, args.operation)
    if authorization is None:
        raise PermissionError("EXECUTION NOT AUTHORIZED")
    certificate = run_preflight(
        repo_root=_repo_root(), contract=contract, plan=plan, authorization=authorization,
        generation_root=args.generation_root, replay_root=args.replay_root,
        acceptance_root=args.acceptance_root,
    )
    _print(certificate.report)


def command_generate(args: argparse.Namespace) -> None:
    contract, authorization, plan = _context(args, "GENERATE")
    if authorization is None:
        raise PermissionError("EXECUTION NOT AUTHORIZED")
    certificate = run_preflight(
        repo_root=_repo_root(), contract=contract, plan=plan, authorization=authorization,
        generation_root=args.generation_root, replay_root=args.replay_root,
        acceptance_root=args.acceptance_root, resume=args.resume,
    )
    adapter = make_validated_production_adapter(contract, validate_catalog())
    ledger = execute_generation(
        repo_root=_repo_root(), contract=contract, plan=plan,
        authorization_hash=authorization.sha256, preflight=certificate,
        campaign_root=args.generation_root, adapter=adapter,
        resume=args.resume, retry_failed=args.retry_failed,
    )
    _print(ledger)


def command_replay(args: argparse.Namespace) -> None:
    contract, authorization, plan = _context(args, "REPLAY")
    if authorization is None:
        raise PermissionError("EXECUTION NOT AUTHORIZED")
    certificate = run_preflight(
        repo_root=_repo_root(), contract=contract, plan=plan, authorization=authorization,
        generation_root=args.generation_root, replay_root=args.replay_root,
        acceptance_root=args.acceptance_root,
    )
    adapter = make_validated_production_adapter(contract, validate_catalog())
    _print(execute_replay(
        repo_root=_repo_root(), plan=plan, authorization_hash=authorization.sha256,
        preflight=certificate, generation_root=args.generation_root,
        replay_root=args.replay_root, adapter=adapter,
    ))


def command_accept(args: argparse.Namespace) -> None:
    contract, authorization, plan = _context(args, "ACCEPT")
    if authorization is None:
        raise PermissionError("EXECUTION NOT AUTHORIZED")
    certificate = run_preflight(
        repo_root=_repo_root(), contract=contract, plan=plan, authorization=authorization,
        generation_root=args.generation_root, replay_root=args.replay_root,
        acceptance_root=args.acceptance_root,
    )
    _print(evaluate_acceptance(
        repo_root=_repo_root(), plan=plan, authorization_hash=authorization.sha256,
        preflight=certificate, generation_root=args.generation_root,
        replay_root=args.replay_root, acceptance_root=args.acceptance_root,
    ))


def command_recover_lock(args: argparse.Namespace) -> None:
    _print(recover_stale_lock(
        args.lock, expected_identity=args.expected_identity, recovery_log=args.recovery_log,
    ))


def command_status(args: argparse.Namespace) -> None:
    if not args.ledger.is_file():
        _print({"campaign": "NOT_INITIALIZED", "sealed_holdout_identities": "REDACTED"})
        return
    ledger = read_ledger(args.ledger)
    counts = {state: sum(record["state"] == state for record in ledger["records"]) for state in ("PLANNED", "STARTING", "RUNNING", "SUCCEEDED", "FAILED", "INTERRUPTED")}
    _print({
        "contract_hash": ledger["contract_hash"], "partition": ledger["partition"],
        "expected_run_count": ledger["expected_run_count"], "state_counts": counts,
        "sealed_holdout_identities": "REDACTED",
    })


def _add_execution_arguments(parser: argparse.ArgumentParser, *, authorization_required: bool = False) -> None:
    parser.add_argument("--partition", required=True, choices=("development", "validation"))
    parser.add_argument("--generation-root", required=True, type=Path)
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--acceptance-root", required=True, type=Path)
    parser.add_argument("--authorization", type=Path, required=authorization_required)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="satnet-stage-a-execution")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-contract")
    verify.add_argument("--contract-root", type=Path, default=_repo_root() / CONTRACT_RELATIVE_ROOT)
    verify.set_defaults(handler=command_verify_contract)
    authorization = subparsers.add_parser("validate-authorization")
    _add_execution_arguments(authorization, authorization_required=True)
    authorization.add_argument("--operation", choices=("PLAN", "GENERATE", "REPLAY", "ACCEPT"), required=True)
    authorization.set_defaults(handler=command_validate_authorization)
    plan = subparsers.add_parser("plan")
    _add_execution_arguments(plan)
    plan.set_defaults(handler=command_plan)
    preflight = subparsers.add_parser("preflight")
    _add_execution_arguments(preflight, authorization_required=True)
    preflight.add_argument("--operation", choices=("GENERATE", "REPLAY", "ACCEPT"), required=True)
    preflight.set_defaults(handler=command_preflight)
    generate = subparsers.add_parser("generate")
    _add_execution_arguments(generate, authorization_required=True)
    generate.set_defaults(handler=command_generate, resume=False, retry_failed=False)
    resume = subparsers.add_parser("resume")
    _add_execution_arguments(resume, authorization_required=True)
    resume.add_argument("--retry-failed", action="store_true")
    resume.set_defaults(handler=command_generate, resume=True)
    replay = subparsers.add_parser("replay")
    _add_execution_arguments(replay, authorization_required=True)
    replay.set_defaults(handler=command_replay)
    accept = subparsers.add_parser("accept")
    _add_execution_arguments(accept, authorization_required=True)
    accept.set_defaults(handler=command_accept)
    recovery = subparsers.add_parser("recover-lock")
    recovery.add_argument("--lock", type=Path, required=True)
    recovery.add_argument("--expected-identity", required=True)
    recovery.add_argument("--recovery-log", type=Path, required=True)
    recovery.set_defaults(handler=command_recover_lock)
    status = subparsers.add_parser("status")
    status.add_argument("--ledger", type=Path, required=True)
    status.set_defaults(handler=command_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        args.handler(args)
    except (FileNotFoundError, FileExistsError, PermissionError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error), "status": "FAILED"}, sort_keys=True), file=sys.stderr)
        return 2
    return 0
