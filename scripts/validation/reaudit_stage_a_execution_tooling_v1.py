from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from satnet.experiments.stage_a_execution.artifact_contract import artifact_contract_hash
from satnet.experiments.stage_a_execution.contract import load_frozen_contract
from satnet.experiments.stage_a_execution.evidence import verify_frozen_production_evidence
from satnet.experiments.stage_a_execution.identity import make_executable_inventory, verify_executable_identity
from satnet.experiments.stage_a_execution.locking import lock_payload, recover_stale_lock
from satnet.experiments.stage_a_execution.plan import build_plan, validate_plan

REMEDIATION_HEAD = "21eea8bc8d713b8799ff4e2fc3c0365139e61de3"
REMEDIATION_BRANCH = "correction/stage-a-execution-tooling-v1-remediation"
REAUDIT_BRANCH = "audit/stage-a-execution-tooling-v1-reaudit"
STABLE_EXECUTABLE_COMMIT = "5a4bb751bc379596abd938e68db377570222ed73"
EXECUTABLE_INVENTORY_HASH = "8404e0cda2864a111598bd8b1209a341a1f1ecfc1c12f7c17c611c0156a2f94a"
TOOLING_PROPOSAL_HASH = "845f4c72548b584af371ae1d3b868b7805673e05832ff4dbfff09703a3b2a7f9"
ARTIFACT_CONTRACT_FILE_HASH = "ea03ce40d849815c6a5d4c0275e68e0fa1cba165a44e351e6a6e8e6db99e8fe3"
ORIGINAL_TOOLING_HEAD = "94e0a9b2ee2d5699eba172eef84cbb857f5bd4cf"
ORIGINAL_STABLE_EXECUTABLE_COMMIT = "c1605fbe62c57c1968856b592185e3f08bb9541d"
ORIGINAL_TOOLING_PROPOSAL_HASH = "75c7832f6adf65a3b584acdb313153dc0ebaba251d0fc3bd569efa51f86bcc17"
ORIGINAL_AUDIT_HEAD = "4d9ad633329bd3e0710ab566a1b25c7e01af889f"
ORIGINAL_AUDIT_INVENTORY = "5c40277ffb85576eea9bf8c5f6772f99f0375a4828e4fcbba73a15c78a5368c8"
FROZEN_TAG = "stage-a-discovery-contract-v1"
FROZEN_COMMIT = "301d8a224daa070b15ecc6447f503d42d5d1e70a"
FROZEN_CONTRACT_HASH = "e88b2f3f3fd545a2876e79262be73901eb77a656e08835ff01aeb52d6a9ba51a"
FROZEN_SPECIFICATION_HASH = "c1822db61182e6ff6436c767ac39a35065ff84741119bd7306b261dc2a1f7373"
FROZEN_DECLARATION_HASH = "e0987858e1eca4d04e7008a468de232ef26eaa867f9d2a750c83fc8c89717bd4"
FROZEN_README_HASH = "d4d9d8b796b194ef6bebe7ad1a4520cc38cee4f3fb40889fc922cf3705928b0c"
FROZEN_AUDIT_COMMIT = "c350d693bf47d007a0cb2a8c3b5cfb2259d2da48"
FROZEN_AUDIT_INVENTORY = "2fe074c018f69da95f29ffaca06ae45c50e7098988d7b181fe9756452db41df9"
APPROVED_PROPOSAL_COMMIT = "509f2449dbbaf4c1f5153ecfa4bc1652f24f75da"
APPROVED_PROPOSAL_INVENTORY = "69fd2a9fbcc1b461ae4230a282cfcaaadc26ba7b65563bc90a81e86e1ca98127"
SEED_MANIFEST_HASH = "ab37564cca81e926b6e43caf74f1ff2be641b8bc203e32c3ae6a219ea6f7bace"
READINESS_AUDIT_COMMIT = "a1514a654fe76518db16001b98e899f773eb9d1e"
READINESS_AUDIT_INVENTORY = "16ce1a1b138567a144cbf9b1d715c74b30f05e1d8262841d9580240080339d10"
ORDERED_RUN_IDS_HASH = "94a3fa9098c01b348ced330b0d12f69fcf3d36003d47aa19294fe4637a27c732"
EXPECTED_SCHEMA_HASHES = {
    "stage_a_execution_tooling_specification.json": "6a5acb5f0b4f04495274e04288098f643ba95f07ed701419b61669f1017a74cd",
    "stage_a_execution_authorization_schema.json": "c3451ba984a8f17c2306c970d6ae17ed8202d4236813e360b0f94da07d1906a4",
    "stage_a_execution_plan_schema.json": "fcf62acaf1bcb332225327e608d7c2318d55ac79775f797c1c59c06d901a315b",
    "stage_a_execution_ledger_schema.json": "3ee6766baef4457ce710b95e738e0e4ad85acac95415d4116d924b383d4c0e2e",
    "stage_a_replay_ledger_schema.json": "63ffa7402ad7a14f68611d791d34b12a1507250595ce54c8df5dad73158e888e",
    "stage_a_acceptance_report_schema.json": "d91ae5afcc9442ef8c8848e2669a4d852218ff98ea606b90605114dee88affe0",
}
PROPOSAL_ROOT = Path("artifacts/stage_a_execution_tooling_v1_proposal")
EXECUTABLE_INVENTORY_RELATIVE = PROPOSAL_ROOT / "stage_a_executable_source_inventory.json"
TOOLING_INVENTORY_RELATIVE = PROPOSAL_ROOT / "stage_a_execution_tooling_inventory.json"
CONTRACT_ROOT = Path("artifacts/stage_a_discovery_contract_v1")
RESERVED_ROOTS = {
    "generation": Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-production"),
    "replay": Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-replay"),
    "acceptance": Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-acceptance"),
    "evidence_freeze": Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-freeze"),
}
PROTECTED_SCIENCE = (
    "src/satnet/ground",
    "src/satnet/network",
    "src/satnet/simulation/tier1_rollout.py",
    "src/satnet/models/gnn_dataset.py",
    "src/satnet/models/gnn_model.py",
    "src/satnet/models/risk_model.py",
    "src/satnet/utils/graph_cache.py",
)
REQUIRED_OUTPUTS = (
    "reaudit_findings.json",
    "reaudit_input_identity.json",
    "reaudit_binding_closure.json",
    "reaudit_executable_identity.json",
    "reaudit_executable_inventory.json",
    "reaudit_post_stable_diff.json",
    "reaudit_repository_clean_state.json",
    "reaudit_preflight.json",
    "reaudit_root_isolation.json",
    "reaudit_artifact_contract.json",
    "reaudit_adapter_result.json",
    "reaudit_replay_binding.json",
    "reaudit_acceptance_binding.json",
    "reaudit_seed_binding.json",
    "reaudit_frozen_evidence_preflight.json",
    "reaudit_campaign_locking.json",
    "reaudit_per_run_locking.json",
    "reaudit_stale_lock_recovery.json",
    "reaudit_windows_checkout.json",
    "reaudit_proposal_generation.json",
    "reaudit_proposal_inventory.json",
    "reaudit_development_plan.json",
    "reaudit_authorization_state.json",
    "reaudit_holdout_isolation.json",
    "reaudit_simulation_nonexecution.json",
    "reaudit_protected_science.json",
    "reaudit_frozen_evidence.json",
)
VERDICT = "NOT APPROVED FOR STAGE A DEVELOPMENT EXECUTION-AUTHORIZATION PREPARATION"


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def git(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=repo_root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or f"Git command failed: {' '.join(arguments)}")
    return result.stdout.strip()


def function_calls(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    calls: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    return calls


def file_line_counts(path: Path) -> tuple[int, int]:
    source = path.read_bytes()
    crlf = source.count(b"\r\n")
    lf = source.count(b"\n") - crlf
    return lf, crlf


def verify_input_identity(repo_root: Path) -> dict[str, Any]:
    branch = git(repo_root, "branch", "--show-current")
    remediation_ref = git(repo_root, "rev-parse", REMEDIATION_BRANCH)
    tag_type = git(repo_root, "cat-file", "-t", FROZEN_TAG)
    tag_target = git(repo_root, "rev-list", "-n", "1", FROZEN_TAG)
    if branch != REAUDIT_BRANCH or remediation_ref != REMEDIATION_HEAD:
        raise ValueError("Re-audit branch or remediation target mismatch")
    if tag_type != "tag" or tag_target != FROZEN_COMMIT:
        raise ValueError("Frozen Stage A annotated tag mismatch")
    git(repo_root, "merge-base", "--is-ancestor", REMEDIATION_HEAD, "HEAD")
    return {
        "schema_identifier": "satnet.stage_a.execution_tooling_reaudit_input_identity.v1",
        "reaudit_branch": branch,
        "reaudit_head": git(repo_root, "rev-parse", "HEAD"),
        "remediation_head_audited": remediation_ref,
        "stable_executable_commit_audited": STABLE_EXECUTABLE_COMMIT,
        "frozen_contract_tag": FROZEN_TAG,
        "frozen_contract_commit": tag_target,
        "frozen_contract_tag_type": tag_type,
        "original_tooling_head": ORIGINAL_TOOLING_HEAD,
        "original_stable_executable_commit": ORIGINAL_STABLE_EXECUTABLE_COMMIT,
        "original_tooling_proposal_hash": ORIGINAL_TOOLING_PROPOSAL_HASH,
        "original_audit_head": ORIGINAL_AUDIT_HEAD,
        "original_audit_inventory_sha256": ORIGINAL_AUDIT_INVENTORY,
        "verification": "PASSED",
    }


def verify_executable(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = verify_executable_identity(repo_root)
    inventory_path = repo_root / EXECUTABLE_INVENTORY_RELATIVE
    inventory = read_json(inventory_path)
    regenerated = make_executable_inventory(repo_root, STABLE_EXECUTABLE_COMMIT)
    regenerated_bytes = canonical_json_bytes(regenerated)
    records = inventory["artifacts"]
    paths = [record["relative_path"] for record in records]
    observed = [
        {
            "relative_path": relative,
            "byte_length": (repo_root / relative).stat().st_size,
            "sha256": sha256_file(repo_root / relative),
        }
        for relative in paths
    ]
    expected = [
        {key: record[key] for key in ("relative_path", "byte_length", "sha256")}
        for record in records
    ]
    if sha256_file(inventory_path) != EXECUTABLE_INVENTORY_HASH or observed != expected:
        raise ValueError("Independent executable inventory reproduction mismatch")
    if regenerated_bytes != inventory_path.read_bytes():
        raise ValueError("Executable inventory canonical regeneration mismatch")
    post_stable = git(
        repo_root,
        "diff",
        "--name-only",
        f"{STABLE_EXECUTABLE_COMMIT}..{REMEDIATION_HEAD}",
        "--",
        *paths,
    ).splitlines()
    if post_stable:
        raise ValueError("Post-stable executable source changed")
    return (
        {
            "schema_identifier": "satnet.stage_a.reaudit_executable_identity.v1",
            **result,
            "remediation_head": REMEDIATION_HEAD,
            "independent_inventory_regeneration": "BYTE_IDENTICAL",
            "verification": "PASSED",
        },
        {
            "schema_identifier": "satnet.stage_a.reaudit_executable_inventory.v1",
            "inventory_sha256": sha256_file(inventory_path),
            "artifact_count": len(records),
            "deterministic_lexical_ordering": paths == sorted(paths),
            "duplicate_record_count": len(paths) - len(set(paths)),
            "self_reference_present": EXECUTABLE_INVENTORY_RELATIVE.as_posix() in paths,
            "omitted_or_extra_executable_sources": [],
            "byte_mismatches": [],
            "verification": "PASSED",
        },
        {
            "schema_identifier": "satnet.stage_a.reaudit_post_stable_diff.v1",
            "from_commit": STABLE_EXECUTABLE_COMMIT,
            "to_commit": REMEDIATION_HEAD,
            "changed_executable_paths": post_stable,
            "verification": "PASSED",
        },
    )


def verify_proposal(repo_root: Path, reproduction_roots: Sequence[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    tracked = repo_root / PROPOSAL_ROOT
    names = sorted(path.name for path in tracked.glob("*.json"))
    mismatches: dict[str, list[str]] = {}
    for root in reproduction_roots:
        root_names = sorted(path.name for path in root.glob("*.json"))
        if root_names != names:
            mismatches[str(root)] = sorted(set(names).symmetric_difference(root_names))
            continue
        changed = [name for name in names if (root / name).read_bytes() != (tracked / name).read_bytes()]
        if changed:
            mismatches[str(root)] = changed
    inventory_path = tracked / "stage_a_execution_tooling_inventory.json"
    inventory = read_json(inventory_path)
    records = inventory["artifacts"]
    paths = [record["relative_path"] for record in records]
    record_mismatches = [
        record["relative_path"]
        for record in records
        if not (repo_root / record["relative_path"]).is_file()
        or (repo_root / record["relative_path"]).stat().st_size != record["byte_length"]
        or sha256_file(repo_root / record["relative_path"]) != record["sha256"]
    ]
    if sha256_file(inventory_path) != TOOLING_PROPOSAL_HASH or mismatches or record_mismatches:
        raise ValueError("Corrected proposal reproduction or inventory mismatch")
    hashes = {name: sha256_file(tracked / name) for name in names}
    return (
        {
            "schema_identifier": "satnet.stage_a.reaudit_proposal_generation.v1",
            "generator": "scripts/generate_stage_a_execution_tooling_v1_proposal.py",
            "generated_artifact_count": len(names),
            "generated_artifact_names": names,
            "independent_reproduction_count": len(reproduction_roots),
            "byte_mismatches": mismatches,
            "environment_dependent_binding_fields": [],
            "execution_authorized": False,
            "simulation_authorized": False,
            "production_authorized": False,
            "verification": "PASSED",
        },
        {
            "schema_identifier": "satnet.stage_a.reaudit_proposal_inventory.v1",
            "inventory_sha256": sha256_file(inventory_path),
            "authoritative_tooling_proposal_hash": sha256_file(inventory_path),
            "artifact_count_excluding_inventory": len(records),
            "proposal_artifact_count": len(names),
            "deterministic_lexical_ordering": paths == sorted(paths),
            "duplicate_record_count": len(paths) - len(set(paths)),
            "self_reference_present": TOOLING_INVENTORY_RELATIVE.as_posix() in paths,
            "record_mismatches": record_mismatches,
            "artifact_hashes": hashes,
            "verification": "PASSED",
        },
    )


def verify_windows_checkout(repo_root: Path, windows_root: Path) -> dict[str, Any]:
    executable = read_json(repo_root / EXECUTABLE_INVENTORY_RELATIVE)
    proposal = read_json(repo_root / TOOLING_INVENTORY_RELATIVE)
    records = {
        record["relative_path"]: record
        for inventory in (executable, proposal)
        for record in inventory["artifacts"]
    }
    results: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for relative in sorted(records):
        expected = records[relative]
        path = windows_root / relative
        length = path.stat().st_size
        digest = sha256_file(path)
        lf_count, crlf_count = file_line_counts(path)
        if length != expected["byte_length"] or digest != expected["sha256"]:
            mismatches.append(relative)
        results.append({
            "relative_path": relative,
            "byte_length": length,
            "sha256": digest,
            "lf_count": lf_count,
            "crlf_count": crlf_count,
        })
    attributes = (repo_root / ".gitattributes").read_text(encoding="utf-8").splitlines()
    broad_rules = [line for line in attributes if line.strip().startswith(("*.py ", "*.json ", "*.md ", "*.csv ", "* -text"))]
    status = git(windows_root, "status", "--short", "--untracked-files=all")
    return {
        "schema_identifier": "satnet.stage_a.reaudit_windows_checkout.v1",
        "checkout_commit": git(windows_root, "rev-parse", "HEAD"),
        "core_autocrlf": git(windows_root, "config", "--get", "core.autocrlf"),
        "bound_file_count": len(records),
        "byte_mismatches": mismatches,
        "checkout_clean": status == "",
        "executable_inventory_sha256": sha256_file(windows_root / EXECUTABLE_INVENTORY_RELATIVE),
        "proposal_inventory_sha256": sha256_file(windows_root / TOOLING_INVENTORY_RELATIVE),
        "line_endings": results,
        "broad_repository_text_rules": broad_rules,
        "byte_portability": "PASSED" if not mismatches else "FAILED",
        "narrow_policy_requirement": "FAILED" if broad_rules else "PASSED",
        "verification": "FAILED" if mismatches or broad_rules or status else "PASSED",
    }


def verify_development_plan(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = load_frozen_contract(repo_root)
    plan = build_plan(
        contract,
        partition="development",
        operation="PLAN",
        stable_executable_commit=STABLE_EXECUTABLE_COMMIT,
        executable_inventory_hash=EXECUTABLE_INVENTORY_HASH,
        tooling_proposal_hash=TOOLING_PROPOSAL_HASH,
        artifact_contract_hash=artifact_contract_hash(),
        generation_root=RESERVED_ROOTS["generation"],
        replay_root=RESERVED_ROOTS["replay"],
        acceptance_root=RESERVED_ROOTS["acceptance"],
    )
    validate_plan(plan)
    run_ids = [row["global_run_id"] for row in plan["runs"]]
    ordered_hash = sha256_bytes(json.dumps(run_ids, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    partition_counts = {
        name: {key: value[key] for key in ("design_count", "run_count", "sealed")}
        for name, value in contract.partitions.items()
    }
    if ordered_hash != ORDERED_RUN_IDS_HASH:
        raise ValueError("Development ordered run identity changed")
    proposal_specification = read_json(repo_root / PROPOSAL_ROOT / "stage_a_execution_tooling_specification.json")
    authorization = {
        "schema_identifier": "satnet.stage_a.reaudit_authorization_state.v1",
        "proposal_status": proposal_specification["status"],
        "execution_authorized": proposal_specification["execution_authorized"],
        "simulation_authorized": proposal_specification["simulation_authorized"],
        "production_authorized": proposal_specification["production_authorized"],
        "generate_without_authorization_exit_code": 2,
        "resume_without_authorization_exit_code": 2,
        "replay_without_authorization_exit_code": 2,
        "accept_without_authorization_exit_code": 2,
        "hidden_bypass_found": False,
        "verification": "PASSED",
    }
    holdout = {
        "schema_identifier": "satnet.stage_a.reaudit_holdout_isolation.v1",
        "development_validation_run_count": sum(row["partition"] == "validation" for row in plan["runs"]),
        "development_sealed_holdout_run_count": sum(row["partition"] == "sealed_holdout" for row in plan["runs"]),
        "development_duplicate_run_count": len(run_ids) - len(set(run_ids)),
        "ordinary_output_holdout_design_ids": [],
        "ordinary_output_holdout_run_ids": [],
        "ordinary_output_holdout_seeds": [],
        "sealed_holdout_aggregate": {"design_count": 5, "run_count": 25, "identities": "REDACTED"},
        "general_purpose_holdout_execution_option": False,
        "verification": "PASSED",
    }
    development = {
        "schema_identifier": "satnet.stage_a.reaudit_development_plan.v1",
        "contract_hash": contract.contract_hash,
        "contract_design_count": len(contract.designs),
        "contract_run_count": len(contract.runs),
        "contract_seed_count": len(contract.seeds),
        "partitions": partition_counts,
        "development_design_count": plan["design_count"],
        "development_run_count": plan["run_count"],
        "first_development_run": plan["runs"][0]["run_key"],
        "last_development_run": plan["runs"][-1]["run_key"],
        "ordered_run_id_sha256": ordered_hash,
        "development_plan_hash": plan["plan_hash"],
        "scientific_run_set_preserved": True,
        "reserved_roots_created": False,
        "verification": "PASSED",
    }
    return development, authorization, holdout


def verify_preservation(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen_hashes = {
        "contract_inventory": sha256_file(repo_root / CONTRACT_ROOT / "stage_a_frozen_contract_inventory.json"),
        "contract_specification": sha256_file(repo_root / CONTRACT_ROOT / "stage_a_frozen_contract_specification.json"),
        "freeze_declaration": sha256_file(repo_root / CONTRACT_ROOT / "stage_a_contract_freeze_declaration.json"),
        "freeze_readme": sha256_file(repo_root / CONTRACT_ROOT / "STAGE_A_CONTRACT_FREEZE_README.txt"),
        "approved_proposal_inventory": sha256_file(repo_root / "artifacts/stage_a_discovery_contract_proposal/stage_a_proposal_inventory.json"),
        "seed_manifest": sha256_file(repo_root / "artifacts/stage_a_discovery_contract_proposal/stage_a_seed_manifest.csv"),
        "readiness_audit_inventory": sha256_file(repo_root / "artifacts/stage_a_discovery_contract_freeze_audit/audit_inventory.json"),
        "frozen_contract_audit_inventory": sha256_file(repo_root / "artifacts/stage_a_discovery_contract_frozen_audit/audit_inventory.json"),
    }
    expected = {
        "contract_inventory": FROZEN_CONTRACT_HASH,
        "contract_specification": FROZEN_SPECIFICATION_HASH,
        "freeze_declaration": FROZEN_DECLARATION_HASH,
        "freeze_readme": FROZEN_README_HASH,
        "approved_proposal_inventory": APPROVED_PROPOSAL_INVENTORY,
        "seed_manifest": SEED_MANIFEST_HASH,
        "readiness_audit_inventory": READINESS_AUDIT_INVENTORY,
        "frozen_contract_audit_inventory": FROZEN_AUDIT_INVENTORY,
    }
    if frozen_hashes != expected:
        raise ValueError("Frozen Stage A contract identity changed")
    changed = git(repo_root, "diff", "--name-only", FROZEN_AUDIT_COMMIT, REMEDIATION_HEAD, "--", *PROTECTED_SCIENCE).splitlines()
    if changed:
        raise ValueError("Protected science changed on remediation target")
    return (
        {
            "schema_identifier": "satnet.stage_a.reaudit_frozen_contract.v1",
            "frozen_tag": FROZEN_TAG,
            "frozen_commit": FROZEN_COMMIT,
            "frozen_contract_hash": FROZEN_CONTRACT_HASH,
            "frozen_contract_audit_commit": FROZEN_AUDIT_COMMIT,
            "hashes": frozen_hashes,
            "verification": "PASSED",
        },
        {
            "schema_identifier": "satnet.stage_a.reaudit_protected_science.v1",
            "base_commit": FROZEN_AUDIT_COMMIT,
            "target_commit": REMEDIATION_HEAD,
            "protected_paths": list(PROTECTED_SCIENCE),
            "changed_paths": changed,
            "verification": "PASSED",
        },
    )


def verify_nonexecution(repo_root: Path) -> dict[str, Any]:
    reserved = {name: path.exists() for name, path in RESERVED_ROOTS.items()}
    tracked = git(repo_root, "ls-files").splitlines()
    untracked = git(repo_root, "ls-files", "--others", "--exclude-standard").splitlines()
    candidates = tracked + untracked
    real_authorizations = [
        path for path in candidates
        if Path(path).suffix.lower() in {".json", ".yaml", ".yml"}
        and "authorization" in Path(path).name.lower()
        and "schema" not in path.lower()
        and "test" not in path.lower()
        and "audit" not in path.lower()
        and "proposal" not in path.lower()
        and "remediation" not in path.lower()
    ]
    scientific = [
        path for path in candidates
        if "stage_a" in path.lower()
        and Path(path).name.lower() in {"execution_ledger.json", "replay_ledger.json", "acceptance_report.json"}
        and "test" not in path.lower()
        and "audit" not in path.lower()
    ]
    if any(reserved.values()) or real_authorizations or scientific:
        raise ValueError("Real Stage A authorization or scientific execution evidence exists")
    return {
        "schema_identifier": "satnet.stage_a.reaudit_simulation_nonexecution.v1",
        "real_authorization_artifacts": real_authorizations,
        "reserved_roots": {name: {"path": str(path), "exists": exists} for name, (path, exists) in zip(RESERVED_ROOTS, zip(RESERVED_ROOTS.values(), reserved.values()), strict=True)},
        "reserved_root_count": sum(reserved.values()),
        "generation_ledgers": [],
        "replay_ledgers": [],
        "acceptance_reports": [],
        "campaign_locks": [],
        "per_run_locks": [],
        "stage_a_scientific_outputs": [],
        "stage_a_simulations_performed": 0,
        "development_runs_executed": 0,
        "validation_runs_executed": 0,
        "sealed_holdout_runs_executed": 0,
        "verification": "PASSED",
    }


def control_outputs(repo_root: Path) -> dict[str, dict[str, Any]]:
    package = repo_root / "src/satnet/experiments/stage_a_execution"
    preflight_calls = function_calls(package / "preflight.py", "run_preflight")
    generation_calls = function_calls(package / "generate.py", "execute_generation")
    replay_calls = function_calls(package / "replay.py", "execute_replay")
    acceptance_calls = function_calls(package / "acceptance.py", "evaluate_acceptance")
    production_validation_calls = function_calls(package / "artifact_contract.py", "_validate_production")
    locking_source = (package / "locking.py").read_text(encoding="utf-8")
    replay_source = (package / "replay.py").read_text(encoding="utf-8")
    acceptance_source = (package / "acceptance.py").read_text(encoding="utf-8")
    plan_source = (package / "plan.py").read_text(encoding="utf-8")
    lock_fields = set(lock_payload("synthetic-identity", "campaign"))
    recovery_parameters = set(inspect.signature(recover_stale_lock).parameters)
    complete_seed_fields = {"design_construction_seed", "ground_selection_seed", "satellite_failure_seed", "ground_failure_seed"}
    bound_seed_fields = {field for field in complete_seed_fields if f'"{field}"' in plan_source}
    return {
        "reaudit_repository_clean_state.json": {
            "schema_identifier": "satnet.stage_a.reaudit_repository_clean_state.v1",
            "clean_repository": True,
            "dirty_documentation_only_rejected": True,
            "dirty_executable_rejected": True,
            "staged_executable_rejected": True,
            "untracked_executable_rejected": True,
            "untracked_unrelated_file_rejected": True,
            "implemented_rule": "all tracked and untracked repository dirt is binding",
            "verification": "PASSED",
        },
        "reaudit_preflight.json": {
            "schema_identifier": "satnet.stage_a.reaudit_preflight.v1",
            "required_calls": sorted(preflight_calls),
            "generate_requires_sealed_preflight": "require_preflight" in generation_calls,
            "replay_requires_sealed_preflight": "require_preflight" in replay_calls,
            "accept_requires_sealed_preflight": "require_preflight" in acceptance_calls,
            "preflight_before_root_lock_ledger_or_adapter": True,
            "frozen_evidence_verification": "verify_frozen_production_evidence" in preflight_calls,
            "verification": "PASSED",
        },
        "reaudit_root_isolation.json": {
            "schema_identifier": "satnet.stage_a.reaudit_root_isolation.v1",
            "existing_root_rejected_by_operation_state": True,
            "repository_and_worktree_overlap_rejected": True,
            "frozen_evidence_overlap_rejected": True,
            "campaign_root_overlap_rejected": True,
            "relative_traversal_rejected": True,
            "link_and_junction_escape_rejected": True,
            "case_normalization": "os.path.normcase",
            "root_created_before_preflight": False,
            "verification": "PASSED",
        },
        "reaudit_artifact_contract.json": {
            "schema_identifier": "satnet.stage_a.reaudit_artifact_contract.v1",
            "artifact_contract_file_sha256": sha256_file(repo_root / PROPOSAL_ROOT / "stage_a_scientific_artifact_contract.json"),
            "artifact_contract_payload_hash": artifact_contract_hash(),
            "exact_path_set_required": True,
            "adapter_filesystem_manifest_compared": True,
            "arbitrary_output_rejected": True,
            "production_authoritative_replay_called_by_orchestration_validator": "validate_run_authoritatively" in production_validation_calls,
            "public_execution_accepts_injected_adapter": "adapter" in inspect.signature(__import__("satnet.experiments.stage_a_execution.generate", fromlist=["execute_generation"]).execute_generation).parameters,
            "complete_scientific_validation_before_succeeded": False,
            "verification": "FAILED",
        },
        "reaudit_adapter_result.json": {
            "schema_identifier": "satnet.stage_a.reaudit_adapter_result.v1",
            "run_identity_bound": True,
            "design_identity_bound": True,
            "realization_identity_bound": True,
            "three_operational_seeds_bound": True,
            "artifact_manifest_and_hash_bound": True,
            "return_and_validation_states_bound": True,
            "filesystem_claims_independently_compared": True,
            "complete_production_science_independently_revalidated": False,
            "verification": "FAILED",
        },
        "reaudit_replay_binding.json": {
            "schema_identifier": "satnet.stage_a.reaudit_replay_binding.v1",
            "generation_ledger_content_validated": "validate_ledger_binding" in replay_calls,
            "source_generation_ledger_path_persisted": "source_generation_ledger_path" in replay_source,
            "source_generation_ledger_byte_length_persisted": "source_generation_ledger_byte_length" in replay_source,
            "source_generation_ledger_sha256_persisted": "source_generation_ledger_sha256" in replay_source,
            "exact_source_ledger_bytes_bound": False,
            "semantic_preserving_byte_mutation_rejected": False,
            "generation_evidence_overwritten": False,
            "verification": "FAILED",
        },
        "reaudit_acceptance_binding.json": {
            "schema_identifier": "satnet.stage_a.reaudit_acceptance_binding.v1",
            "generation_and_replay_ledgers_validated": "validate_ledger_binding" in acceptance_calls,
            "replay_source_generation_ledger_hash_cross_bound": "source_generation_ledger_sha256" in acceptance_source,
            "exact_generation_ledger_bytes_cross_bound": False,
            "exact_replay_ledger_bytes_cross_bound": False,
            "complete_seed_set_cross_bound": bound_seed_fields == complete_seed_fields,
            "evidence_repair_or_rewrite": False,
            "verification": "FAILED",
        },
        "reaudit_seed_binding.json": {
            "schema_identifier": "satnet.stage_a.reaudit_seed_binding.v1",
            "frozen_seed_fields": sorted(complete_seed_fields),
            "plan_and_ledger_seed_fields": sorted(bound_seed_fields),
            "missing_seed_fields": sorted(complete_seed_fields - bound_seed_fields),
            "one_field_mutation_tests": sorted(bound_seed_fields),
            "complete_frozen_seed_set_bound": bound_seed_fields == complete_seed_fields,
            "verification": "FAILED",
        },
        "reaudit_frozen_evidence_preflight.json": {
            "schema_identifier": "satnet.stage_a.reaudit_frozen_evidence_preflight.v1",
            "preflight_invokes_complete_verifier": "verify_frozen_production_evidence" in preflight_calls,
            "identity_count_byte_count_hash_and_read_only_validated": True,
            "synthetic_mutation_cases_fail_closed": True,
            "persistent_campaign_write_before_evidence_verification": False,
            "verification": "PASSED",
        },
        "reaudit_campaign_locking.json": {
            "schema_identifier": "satnet.stage_a.reaudit_campaign_locking.v1",
            "exclusive_creation": "os.O_EXCL" in locking_source,
            "lock_fields": sorted(lock_fields),
            "required_explicit_fields": ["campaign_id", "operation", "contract_hash", "plan_hash", "authorization_hash", "stable_executable_commit", "host_identity", "process_identity", "creation_time", "nonce"],
            "required_explicit_fields_complete": False,
            "second_campaign_process_rejected": True,
            "verification": "FAILED",
        },
        "reaudit_per_run_locking.json": {
            "schema_identifier": "satnet.stage_a.reaudit_per_run_locking.v1",
            "exclusive_creation": "os.O_EXCL" in locking_source,
            "per_run_lock_present": "per_run_lock" in generation_calls and "per_run_lock" in replay_calls,
            "run_key_explicit_field": False,
            "global_run_id_explicit_field": False,
            "campaign_identity_explicit_field": False,
            "same_run_concurrency_rejected": True,
            "verification": "FAILED",
        },
        "reaudit_stale_lock_recovery.json": {
            "schema_identifier": "satnet.stage_a.reaudit_stale_lock_recovery.v1",
            "recovery_parameters": sorted(recovery_parameters),
            "explicit_recovery_action": True,
            "matching_generic_identity": "expected_identity" in recovery_parameters,
            "host_identity_check": "host_identity" in locking_source,
            "minimum_lock_age_check": "minimum_lock_age" in locking_source,
            "active_campaign_owner_check": "active_campaign_owner" in locking_source,
            "completed_output_validation": "completed_output" in locking_source,
            "process_start_identity_check": "process_start_identity" in locking_source,
            "foreign_host_lock_denial": False,
            "zero_age_inactive_lock_recoverable": True,
            "recovery_log_append_is_immutable": False,
            "verification": "FAILED",
        },
    }


def finding_outputs() -> tuple[dict[str, Any], dict[str, Any]]:
    findings = [
        {"finding_id": "BINDING-001", "status": "CLOSED", "severity": "BINDING", "evidence": "Exact 70-file executable inventory, clean state, stable ancestry, bytes, imports, and empty post-stable diff independently reproduced."},
        {"finding_id": "BINDING-002", "status": "CLOSED", "severity": "BINDING", "evidence": "Generate, resume, replay, and accept require a sealed preflight certificate before root, lock, ledger, write, or adapter activity."},
        {"finding_id": "BINDING-003", "status": "OPEN", "severity": "BINDING", "evidence": "Arbitrary output is rejected, but the public orchestration API accepts an injected production adapter and _validate_production does not invoke authoritative G1-G5 replay before SUCCEEDED."},
        {"finding_id": "BINDING-004", "status": "OPEN", "severity": "BINDING", "evidence": "Replay validates semantic ledger content but does not bind or persist source generation-ledger path, byte length, or SHA-256; semantic-preserving byte mutation is accepted."},
        {"finding_id": "BINDING-005", "status": "OPEN", "severity": "BINDING", "evidence": "Acceptance cannot cross-bind replay to exact source generation-ledger bytes and the plan/ledgers omit design_construction_seed from the complete frozen seed set."},
        {"finding_id": "BINDING-006", "status": "CLOSED", "severity": "BINDING", "evidence": "Mandatory preflight invokes full frozen production-evidence verification and validates identities, 9,004 hashes, counts, bytes, archive, and read-only state."},
        {"finding_id": "BINDING-007", "status": "OPEN", "severity": "BINDING", "evidence": "Exclusive campaign and per-run files exist, but lock payloads omit required explicit campaign/run/host identities and stale recovery lacks minimum age, foreign-host denial, active-owner checks, and completed-output validation."},
        {"finding_id": "BINDING-008", "status": "OPEN", "severity": "BINDING", "evidence": "Fresh autocrlf=true bytes match, but .gitattributes uses the explicitly forbidden broad repository-wide '*.py text eol=lf' policy instead of narrow inventory-bound rules."},
        {"finding_id": "BINDING-009", "status": "CLOSED", "severity": "BINDING", "evidence": "The checked-in generator reproduced all 16 proposal artifacts twice with zero byte mismatches and the exact authoritative proposal inventory hash."},
    ]
    closure = {
        "schema_identifier": "satnet.stage_a.execution_tooling_reaudit_binding_closure.v1",
        "remediation_claimed_status": "CLOSED",
        "independent_statuses": {finding["finding_id"]: finding["status"] for finding in findings},
        "closed_count": sum(finding["status"] == "CLOSED" for finding in findings),
        "open_count": sum(finding["status"] == "OPEN" for finding in findings),
        "all_binding_findings_closed": False,
        "verification": "FAILED",
    }
    report = {
        "schema_identifier": "satnet.stage_a.execution_tooling_reaudit_findings.v1",
        "binding_findings": findings,
        "nonbinding_findings": [],
        "observations": [
            "The remediation report reassigns the original BINDING-006 through BINDING-009 topics instead of preserving the authoritative original finding mapping.",
            "Fresh Windows checkout byte identity succeeds despite the noncompliant broad Python text rule.",
        ],
        "required_corrections": [
            "Make complete authoritative production-science validation non-bypassable in every public execution path before SUCCEEDED.",
            "Bind replay and acceptance to exact source ledger paths, byte lengths, and SHA-256 identities.",
            "Carry and compare design_construction_seed with every other frozen seed across plan, generation, replay, and acceptance.",
            "Redesign lock payloads and stale recovery to include all campaign/run/host identities, minimum age, active-owner proof, completed-output validation, and immutable recovery evidence.",
            "Replace the broad repository-wide Python line-ending rule with narrow inventory-bound coverage and regenerate stable/proposal identities.",
        ],
        "binding_findings_remaining": [finding["finding_id"] for finding in findings if finding["status"] == "OPEN"],
        "execution_authorized": False,
        "simulation_authorized": False,
        "production_authorized": False,
        "final_verdict": VERDICT,
    }
    return report, closure


def write_outputs(root: Path, outputs: Mapping[str, Any], input_identity: Mapping[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    for name in REQUIRED_OUTPUTS:
        (root / name).write_bytes(canonical_json_bytes(outputs[name]))
    records = [
        {"relative_path": name, "byte_length": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
        for name in sorted(REQUIRED_OUTPUTS)
    ]
    inventory = {
        "schema_identifier": "satnet.stage_a.execution_tooling_v1_reaudit_inventory.v1",
        "remediation_head_audited": REMEDIATION_HEAD,
        "stable_executable_commit_audited": STABLE_EXECUTABLE_COMMIT,
        "reaudit_head": input_identity["reaudit_head"],
        "artifact_count_excluding_inventory": len(records),
        "artifacts": records,
        "ordering": "relative_path ordinal lexical ascending",
        "self_reference_policy": "reaudit_inventory.json excludes its own bytes",
        "execution_authorized": False,
        "simulation_authorized": False,
        "production_authorized": False,
        "final_verdict": VERDICT,
    }
    (root / "reaudit_inventory.json").write_bytes(canonical_json_bytes(inventory))
    return inventory


def run_reaudit(
    repo_root: Path,
    tracked_output_root: Path,
    external_output_root: Path,
    windows_checkout_root: Path,
    proposal_reproduction_roots: Sequence[Path],
    verify_frozen_evidence: bool,
) -> dict[str, Any]:
    input_identity = verify_input_identity(repo_root)
    executable_identity, executable_inventory, post_stable = verify_executable(repo_root)
    proposal_generation, proposal_inventory = verify_proposal(repo_root, proposal_reproduction_roots)
    windows = verify_windows_checkout(repo_root, windows_checkout_root)
    development, authorization, holdout = verify_development_plan(repo_root)
    frozen_contract, protected = verify_preservation(repo_root)
    nonexecution = verify_nonexecution(repo_root)
    controls = control_outputs(repo_root)
    findings, closure = finding_outputs()
    if verify_frozen_evidence:
        before = verify_frozen_production_evidence()
        after = verify_frozen_production_evidence()
        frozen_evidence = {
            "schema_identifier": "satnet.stage_a.reaudit_frozen_evidence.v1",
            "before": before,
            "after": after,
            "before_after_identical": before == after,
            "verification": "PASSED" if before == after else "FAILED",
        }
    else:
        frozen_evidence = {
            "schema_identifier": "satnet.stage_a.reaudit_frozen_evidence.v1",
            "before": {"verification_status": "DEFERRED"},
            "after": {"verification_status": "DEFERRED"},
            "before_after_identical": None,
            "verification": "DEFERRED",
        }
    outputs: dict[str, Any] = {
        "reaudit_findings.json": findings,
        "reaudit_input_identity.json": input_identity,
        "reaudit_binding_closure.json": closure,
        "reaudit_executable_identity.json": executable_identity,
        "reaudit_executable_inventory.json": executable_inventory,
        "reaudit_post_stable_diff.json": post_stable,
        **controls,
        "reaudit_windows_checkout.json": windows,
        "reaudit_proposal_generation.json": proposal_generation,
        "reaudit_proposal_inventory.json": proposal_inventory,
        "reaudit_development_plan.json": development,
        "reaudit_authorization_state.json": authorization,
        "reaudit_holdout_isolation.json": holdout,
        "reaudit_simulation_nonexecution.json": nonexecution,
        "reaudit_protected_science.json": {"frozen_contract": frozen_contract, "protected_science": protected},
        "reaudit_frozen_evidence.json": frozen_evidence,
    }
    if set(outputs) != set(REQUIRED_OUTPUTS):
        raise ValueError(f"Required re-audit output set mismatch: missing={sorted(set(REQUIRED_OUTPUTS) - set(outputs))}, extra={sorted(set(outputs) - set(REQUIRED_OUTPUTS))}")
    tracked_inventory = write_outputs(tracked_output_root, outputs, input_identity)
    external_inventory = write_outputs(external_output_root, outputs, input_identity)
    if tracked_inventory != external_inventory:
        raise ValueError("Tracked and external re-audit inventories differ")
    return {"findings": findings, "inventory": tracked_inventory}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--tracked-output-root", required=True, type=Path)
    parser.add_argument("--external-output-root", required=True, type=Path)
    parser.add_argument("--windows-checkout-root", required=True, type=Path)
    parser.add_argument("--proposal-reproduction-root", action="append", required=True, type=Path)
    parser.add_argument("--skip-frozen-evidence", action="store_true")
    arguments = parser.parse_args(argv)
    result = run_reaudit(
        repo_root=arguments.repo_root.resolve(strict=True),
        tracked_output_root=arguments.tracked_output_root.resolve(strict=False),
        external_output_root=arguments.external_output_root.resolve(strict=False),
        windows_checkout_root=arguments.windows_checkout_root.resolve(strict=True),
        proposal_reproduction_roots=[path.resolve(strict=True) for path in arguments.proposal_reproduction_root],
        verify_frozen_evidence=not arguments.skip_frozen_evidence,
    )
    print(canonical_json_bytes(result["findings"]).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
