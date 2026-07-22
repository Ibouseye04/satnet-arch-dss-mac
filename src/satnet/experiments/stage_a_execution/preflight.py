from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any

from .authorization import Authorization, validate_authorization
from .contract import FrozenStageAContract, load_frozen_contract
from .evidence import DEFAULT_FROZEN_EVIDENCE_PATHS, FrozenEvidencePaths, verify_frozen_production_evidence
from .identity import tooling_identity, verify_executable_identity
from .ledger import read_bound_ledger
from .paths import available_bytes, probe_parent, validate_output_roots
from .plan import validate_plan, validate_plan_contract_binding
from .resume import validate_ledger_binding

MINIMUM_FREE_BYTES = 1_000_000_000
_PREFLIGHT_SEAL = object()


@dataclass(frozen=True)
class PreflightCertificate:
    report: dict[str, Any]
    plan_hash: str
    authorization_hash: str
    operation: str
    roots: dict[str, str]
    _seal: object


def require_preflight(certificate: PreflightCertificate, *, plan: dict[str, Any], authorization_hash: str) -> None:
    if not isinstance(certificate, PreflightCertificate) or certificate._seal is not _PREFLIGHT_SEAL:
        raise PermissionError("A valid mandatory preflight certificate is required")
    if certificate.report.get("preflight") != "PASSED":
        raise PermissionError("Mandatory preflight did not pass")
    if certificate.plan_hash != plan["plan_hash"] or certificate.authorization_hash != authorization_hash:
        raise PermissionError("Mandatory preflight identity mismatch")
    if certificate.operation != plan["operation"] or certificate.roots != plan["output_roots"]:
        raise PermissionError("Mandatory preflight operation or roots mismatch")


def run_preflight(
    *, repo_root: Path, contract: FrozenStageAContract, plan: dict[str, Any],
    authorization: Authorization, generation_root: Path, replay_root: Path, acceptance_root: Path,
    resume: bool = False, check_write_probe: bool = True,
    minimum_free_bytes: int = MINIMUM_FREE_BYTES,
    evidence_paths: FrozenEvidencePaths = DEFAULT_FROZEN_EVIDENCE_PATHS,
) -> PreflightCertificate:
    executable = verify_executable_identity(repo_root)
    if executable["stable_executable_commit"] != plan["stable_executable_commit"]:
        raise PermissionError("Plan stable executable identity mismatch")
    if executable["executable_inventory_sha256"] != plan["executable_inventory_hash"]:
        raise PermissionError("Plan executable inventory identity mismatch")
    reloaded = load_frozen_contract(repo_root, contract.contract_root)
    if reloaded.contract_hash != contract.contract_hash:
        raise ValueError("Frozen Stage A contract reload identity mismatch")
    validate_plan(plan)
    validate_plan_contract_binding(plan, reloaded)
    if plan.get("authorization_hash") != authorization.sha256:
        raise PermissionError("Plan authorization identity mismatch")
    validate_authorization(
        authorization, contract=reloaded,
        stable_executable_commit=plan["stable_executable_commit"],
        executable_inventory_hash=plan["executable_inventory_hash"],
        tooling_proposal_hash=plan["tooling_proposal_hash"],
        artifact_contract_hash=plan["artifact_contract_hash"],
        operation=plan["operation"], partition=plan["partition"],
        run_ids=[row["global_run_id"] for row in plan["runs"]],
        generation_root=generation_root, replay_root=replay_root, acceptance_root=acceptance_root,
    )
    roots = validate_output_roots(
        repo_root=repo_root, generation_root=generation_root, replay_root=replay_root,
        acceptance_root=acceptance_root, require_absent=False,
    )
    if roots != plan["output_roots"]:
        raise PermissionError("Preflight roots differ from plan roots")
    states = {name: Path(path).exists() for name, path in roots.items()}
    required_states = {
        "GENERATE": {"generation": resume, "replay": False, "acceptance": False},
        "REPLAY": {"generation": True, "replay": False, "acceptance": False},
        "ACCEPT": {"generation": True, "replay": True, "acceptance": False},
        "PLAN": states,
    }[plan["operation"]]
    if states != required_states:
        raise FileExistsError(f"Output-root state mismatch for {plan['operation']}: {states}")
    source_ledgers: dict[str, dict[str, Any]] = {}
    if plan["operation"] in {"REPLAY", "ACCEPT"}:
        generation, generation_identity = read_bound_ledger(
            generation_root,
            relative_path=plan["source_generation_ledger_relative_path"],
            byte_length=plan["source_generation_ledger_byte_length"],
            sha256=plan["source_generation_ledger_sha256"],
        )
        validate_ledger_binding(generation, plan, operation="GENERATE")
        source_ledgers["generation"] = generation_identity
    if plan["operation"] == "ACCEPT":
        replay, replay_identity = read_bound_ledger(
            replay_root,
            relative_path=plan["source_replay_ledger_relative_path"],
            byte_length=plan["source_replay_ledger_byte_length"],
            sha256=plan["source_replay_ledger_sha256"],
        )
        validate_ledger_binding(replay, plan, operation="REPLAY")
        replay_source = {
            "relative_path": replay["source_generation_ledger_relative_path"],
            "byte_length": replay["source_generation_ledger_byte_length"],
            "sha256": replay["source_generation_ledger_sha256"],
        }
        if replay_source != source_ledgers["generation"]:
            raise ValueError("Acceptance preflight source-ledger cross-binding mismatch")
        source_ledgers["replay"] = replay_identity
    evidence = verify_frozen_production_evidence(evidence_paths)
    required_modules = (
        "satnet.experiments.final_generation.orchestrator",
        "satnet.experiments.final_generation.run_validation",
        "satnet.simulation.tier1_rollout",
        "satnet.ground.failure_service_persistence",
    )
    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Required runtime modules unavailable: {missing}")
    free = {name: available_bytes(Path(path)) for name, path in roots.items()}
    if any(value < minimum_free_bytes for value in free.values()):
        raise OSError("Insufficient disk space for authorized execution")
    lock_root = Path(roots[{"GENERATE": "generation", "REPLAY": "replay", "ACCEPT": "acceptance", "PLAN": "generation"}[plan["operation"]]])
    if lock_root.with_name(lock_root.name + ".lock").exists():
        raise RuntimeError("Conflicting execution lock exists; explicit stale-lock recovery is required")
    if check_write_probe:
        for root in (generation_root, replay_root, acceptance_root):
            probe_parent(root)
    report = {
        "authorization": "VALID",
        "contract_hash": contract.contract_hash,
        "stable_executable_commit": executable["stable_executable_commit"],
        "executable_inventory_hash": executable["executable_inventory_sha256"],
        "tooling_proposal_hash": plan["tooling_proposal_hash"],
        "artifact_contract_hash": plan["artifact_contract_hash"],
        "dependencies": "AVAILABLE",
        "disk_free_bytes": free,
        "frozen_evidence_file_count": evidence["combined"]["file_count"],
        "frozen_evidence_byte_count": evidence["combined"]["byte_count"],
        "frozen_evidence_verified_sha256_count": evidence["combined"]["verified_sha256_count"],
        "output_roots": roots,
        "source_ledgers": source_ledgers,
        "plan_hash": plan["plan_hash"],
        "preflight": "PASSED",
        "simulation_executed": False,
    }
    return PreflightCertificate(
        report=report, plan_hash=plan["plan_hash"], authorization_hash=authorization.sha256,
        operation=plan["operation"], roots=roots, _seal=_PREFLIGHT_SEAL,
    )
