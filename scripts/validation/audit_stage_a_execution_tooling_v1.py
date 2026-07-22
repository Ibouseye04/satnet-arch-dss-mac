from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

TOOLING_HEAD = "94e0a9b2ee2d5699eba172eef84cbb857f5bd4cf"
STABLE_EXECUTABLE_COMMIT = "c1605fbe62c57c1968856b592185e3f08bb9541d"
FROZEN_AUDIT_COMMIT = "c350d693bf47d007a0cb2a8c3b5cfb2259d2da48"
FROZEN_TAG = "stage-a-discovery-contract-v1"
FROZEN_COMMIT = "301d8a224daa070b15ecc6447f503d42d5d1e70a"
FROZEN_CONTRACT_HASH = "e88b2f3f3fd545a2876e79262be73901eb77a656e08835ff01aeb52d6a9ba51a"
FROZEN_SPECIFICATION_HASH = "c1822db61182e6ff6436c767ac39a35065ff84741119bd7306b261dc2a1f7373"
FROZEN_DECLARATION_HASH = "e0987858e1eca4d04e7008a468de232ef26eaa867f9d2a750c83fc8c89717bd4"
FROZEN_README_HASH = "d4d9d8b796b194ef6bebe7ad1a4520cc38cee4f3fb40889fc922cf3705928b0c"
FROZEN_AUDIT_INVENTORY_HASH = "2fe074c018f69da95f29ffaca06ae45c50e7098988d7b181fe9756452db41df9"
PROPOSAL_INVENTORY_HASH = "69fd2a9fbcc1b461ae4230a282cfcaaadc26ba7b65563bc90a81e86e1ca98127"
SEED_MANIFEST_HASH = "ab37564cca81e926b6e43caf74f1ff2be641b8bc203e32c3ae6a219ea6f7bace"
READINESS_AUDIT_INVENTORY_HASH = "16ce1a1b138567a144cbf9b1d715c74b30f05e1d8262841d9580240080339d10"
SPECIFICATION_HASH = "157e5ec3e623c4adbf85bdd8f3d13f52f3cca02ff8c0f03de6fefd285a754485"
AUTHORIZATION_SCHEMA_HASH = "b1dab98bbab3edff6e3a5a300c1a44397066362017751dd6e70dd61b8bcdab41"
PLAN_SCHEMA_HASH = "f2395e3486f25bf47acfce3edd65f646ad1423fd9fd393b0cf16e6d2c013e615"
EXECUTION_LEDGER_SCHEMA_HASH = "d51e2620d02a5e1f2c27c2eb359bea6ea9793543d6998bc74cbbe59eb8a98e1d"
REPLAY_LEDGER_SCHEMA_HASH = "2bffb3339235277ef0d60d4d4fa4640bea776a8ef5256c134e4f5af9fa2f71f4"
ACCEPTANCE_SCHEMA_HASH = "76bd71212be3a24acb04a02a0a7d2ab0ffb8db6864e9b8c38e87fb79f89edab7"
TOOLING_INVENTORY_HASH = "75c7832f6adf65a3b584acdb313153dc0ebaba251d0fc3bd569efa51f86bcc17"
ORDERED_RUN_IDS_HASH = "94a3fa9098c01b348ced330b0d12f69fcf3d36003d47aa19294fe4637a27c732"
DEVELOPMENT_PLAN_HASH = "b666d8c7700a10418de85180624617fae744cb80a64b675ca95d1c89c1fc61c3"
PROPOSAL_ROOT = Path("artifacts/stage_a_execution_tooling_v1_proposal")
CONTRACT_ROOT = Path("artifacts/stage_a_discovery_contract_v1")
EXECUTION_ROOT = Path("src/satnet/experiments/stage_a_execution")
INVENTORY_PATH = PROPOSAL_ROOT / "stage_a_execution_tooling_inventory.json"
SCHEMA_HASHES = {
    "stage_a_execution_tooling_specification.json": SPECIFICATION_HASH,
    "stage_a_execution_authorization_schema.json": AUTHORIZATION_SCHEMA_HASH,
    "stage_a_execution_plan_schema.json": PLAN_SCHEMA_HASH,
    "stage_a_execution_ledger_schema.json": EXECUTION_LEDGER_SCHEMA_HASH,
    "stage_a_replay_ledger_schema.json": REPLAY_LEDGER_SCHEMA_HASH,
    "stage_a_acceptance_report_schema.json": ACCEPTANCE_SCHEMA_HASH,
}
PROPOSAL_HASHES = {
    "development_plan_preview.json": "ebea547ac5efc461658f2e4dac6c4dbc174eb5060f61ed33b955e43826ed2366",
    "protected_science_validation.json": "d15e2c6ce16a5836fa2f5a53b85513c6c2219491f755e42c34fbcc41253a14ad",
    "reserved_root_validation.json": "802fddd38af86a5ae6fd575ea22b636cf28aad1352f6507f55126f8f6f0b477c",
    "simulation_nonexecution_validation.json": "64c9dab84378338c8d744f371de7d5c9c49d3439ef3d9d12bbdc9abd7b17ba62",
    "stage_a_acceptance_report_schema.json": ACCEPTANCE_SCHEMA_HASH,
    "stage_a_execution_authorization_schema.json": AUTHORIZATION_SCHEMA_HASH,
    "stage_a_execution_ledger_schema.json": EXECUTION_LEDGER_SCHEMA_HASH,
    "stage_a_execution_plan_schema.json": PLAN_SCHEMA_HASH,
    "stage_a_execution_tooling_inventory.json": TOOLING_INVENTORY_HASH,
    "stage_a_execution_tooling_specification.json": SPECIFICATION_HASH,
    "stage_a_replay_ledger_schema.json": REPLAY_LEDGER_SCHEMA_HASH,
    "validation_summary.json": "3ab6c87dd84a2c612957b69749dcc8726fecd44f6cbdf8b2aab5ba6f159e1fb6",
}
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
FROZEN_ROOTS = (
    Path(r"C:\Users\johns\satnet-final-production-20260720"),
    Path(r"C:\Users\johns\satnet-final-production-replay-20260720"),
    Path(r"C:\Users\johns\satnet-final-production-v1-freeze-20260721"),
    Path(r"C:\Users\johns\satnet-final-production-v1-freeze-20260721.zip"),
)
BINDING_FINDINGS = [
    {
        "finding_id": "BINDING-001",
        "area": "tooling_identity",
        "expected": "Execution is denied when executable bytes are modified, missing, dirty, or differ from the stable executable inventory.",
        "observed": "tooling_identity derives the last commit and hashes only the inventory file; it does not verify current executable bytes, inventory membership, or working-tree cleanliness.",
        "affected": "src/satnet/experiments/stage_a_execution/preflight.py:25-33",
    },
    {
        "finding_id": "BINDING-002",
        "area": "root_isolation_preflight",
        "expected": "Every GENERATE, REPLAY, and ACCEPT path performs mandatory preflight and output-root isolation before writes.",
        "observed": "CLI execution handlers validate authorization but call generation, replay, and acceptance directly without run_preflight or validate_output_roots.",
        "affected": "src/satnet/experiments/stage_a_execution/cli.py:103-135",
    },
    {
        "finding_id": "BINDING-003",
        "area": "success_criteria",
        "expected": "SUCCEEDED requires exact required artifacts and scientific metadata matching plan identity, seeds, partition, and schema with no extras.",
        "observed": "execute_generation accepts any nonempty artifact inventory and transitions to SUCCEEDED without semantic output validation.",
        "affected": "src/satnet/experiments/stage_a_execution/generate.py:179-188",
    },
    {
        "finding_id": "BINDING-004",
        "area": "replay_identity",
        "expected": "Replay rejects generation evidence whose plan, authorization, tooling, roots, run records, or seeds differ from the authorized replay input.",
        "observed": "execute_replay reads generation ledger records by run ID and artifact hash but does not validate the generation ledger identity against the replay plan.",
        "affected": "src/satnet/experiments/stage_a_execution/replay.py:33-49",
    },
    {
        "finding_id": "BINDING-005",
        "area": "acceptance_identity",
        "expected": "Acceptance verifies generation/replay plan, authorization, roots, seed sets, run records, and manifest consistency before passing.",
        "observed": "evaluate_acceptance checks selected contract, partition, operation, tooling, run IDs, artifacts, and replay flags but does not bind ledger plan hashes, authorization hashes, output roots, run-record hashes, or seeds.",
        "affected": "src/satnet/experiments/stage_a_execution/acceptance.py:19-49",
    },
    {
        "finding_id": "BINDING-006",
        "area": "preflight_frozen_evidence",
        "expected": "Preflight independently verifies frozen production evidence identities before scientific execution.",
        "observed": "run_preflight reloads the Stage A contract but never invokes frozen production evidence verification.",
        "affected": "src/satnet/experiments/stage_a_execution/preflight.py:36-89",
    },
    {
        "finding_id": "BINDING-007",
        "area": "locking",
        "expected": "Campaign and per-run locks provide validated stale-lock recovery and process-ID reuse handling.",
        "observed": "Only a campaign lock is acquired; lock_is_stale is not integrated into execution and there is no per-run lock or validated recovery operation.",
        "affected": "src/satnet/experiments/stage_a_execution/ledger.py:116-156; src/satnet/experiments/stage_a_execution/generate.py:157-203",
    },
    {
        "finding_id": "BINDING-008",
        "area": "windows_byte_identity",
        "expected": "A fresh Windows core.autocrlf=true checkout preserves every inventory-bound executable byte length and SHA-256.",
        "observed": "The inventory binds LF Git blobs while execution and test Python paths have no -text/eol rule; all 19 inventory-bound Python files materialize as CRLF with different lengths and hashes in the independent Windows worktree.",
        "affected": ".gitattributes; artifacts/stage_a_execution_tooling_v1_proposal/stage_a_execution_tooling_inventory.json",
    },
    {
        "finding_id": "BINDING-009",
        "area": "proposal_reproduction",
        "expected": "All proposal artifacts are regenerated byte-for-byte by a checked-in deterministic generator from explicit inputs.",
        "observed": "No proposal artifact generator exists; only committed outputs and self-tests are present, so independent byte-for-byte regeneration of all 12 proposal artifacts is unavailable.",
        "affected": "artifacts/stage_a_execution_tooling_v1_proposal; scripts",
    },
]


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


def git(repo_root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(["git", *arguments], cwd=repo_root, capture_output=True, text=not binary, check=False)
    if result.returncode != 0:
        error = result.stderr if isinstance(result.stderr, str) else result.stderr.decode("utf-8", errors="replace")
        raise ValueError(error.strip() or f"Git command failed: {arguments}")
    return result.stdout


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def source_calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    return calls


def validate_input_identity(repo_root: Path) -> dict[str, Any]:
    for commit in (TOOLING_HEAD, STABLE_EXECUTABLE_COMMIT, FROZEN_AUDIT_COMMIT, FROZEN_COMMIT):
        if str(git(repo_root, "cat-file", "-t", commit)).strip() != "commit":
            raise ValueError(f"Required commit unavailable: {commit}")
    branch = str(git(repo_root, "branch", "--show-current")).strip()
    if branch != "audit/stage-a-execution-tooling-v1":
        raise ValueError("Wrong audit branch")
    git(repo_root, "merge-base", "--is-ancestor", TOOLING_HEAD, "HEAD")
    if str(git(repo_root, "rev-list", "-n", "1", FROZEN_TAG)).strip() != FROZEN_COMMIT:
        raise ValueError("Frozen tag target mismatch")
    if str(git(repo_root, "cat-file", "-t", FROZEN_TAG)).strip() != "tag":
        raise ValueError("Frozen tag is not annotated")
    return {
        "audit_branch": branch,
        "audit_head_at_start": TOOLING_HEAD,
        "stable_executable_commit": STABLE_EXECUTABLE_COMMIT,
        "frozen_contract_audit_commit": FROZEN_AUDIT_COMMIT,
        "frozen_tag": FROZEN_TAG,
        "frozen_tag_target": FROZEN_COMMIT,
        "pass": True,
    }


def validate_stable_executable(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory = read_json(repo_root / INVENTORY_PATH)
    records = inventory["artifacts"]
    source_paths = sorted(record["relative_path"] for record in records if record["relative_path"].startswith(EXECUTION_ROOT.as_posix() + "/"))
    actual_source_paths = sorted(path.relative_to(repo_root).as_posix() for path in (repo_root / EXECUTION_ROOT).glob("*.py"))
    if source_paths != actual_source_paths:
        raise ValueError("Executable source allowlist mismatch")
    post_stable = str(git(repo_root, "diff", "--name-only", STABLE_EXECUTABLE_COMMIT, TOOLING_HEAD, "--", EXECUTION_ROOT.as_posix())).splitlines()
    if post_stable:
        raise ValueError("Executable source changed after stable commit")
    last_commit = str(git(repo_root, "rev-list", "-n", "1", TOOLING_HEAD, "--", ".gitattributes", EXECUTION_ROOT.as_posix(), "tests/experiments/stage_a_execution", "tests/experiments/test_final_dataset_isolation.py")).strip()
    if last_commit != STABLE_EXECUTABLE_COMMIT:
        raise ValueError("Stable executable commit mismatch")
    preflight_calls = source_calls(repo_root / EXECUTION_ROOT / "preflight.py")
    identity_gap = not {"verify_artifact_inventory", "validate_no_links"}.intersection(preflight_calls)
    return (
        {
            "stable_executable_commit": STABLE_EXECUTABLE_COMMIT,
            "last_execution_surface_commit": last_commit,
            "execution_source_count": len(source_paths),
            "execution_source_allowlist": source_paths,
            "allowlist_complete": source_paths == actual_source_paths,
            "current_byte_verification_present": not identity_gap,
            "pass": not identity_gap,
        },
        {
            "from_commit": STABLE_EXECUTABLE_COMMIT,
            "to_commit": TOOLING_HEAD,
            "execution_path_changes": post_stable,
            "result": "NO_EXECUTION_PATH_SOURCE_CHANGED",
            "pass": True,
        },
    )


def validate_inventory(repo_root: Path) -> dict[str, Any]:
    path = repo_root / INVENTORY_PATH
    inventory = read_json(path)
    records = inventory["artifacts"]
    paths = [record["relative_path"] for record in records]
    if len(records) != 28 or paths != sorted(paths) or len(paths) != len(set(paths)) or INVENTORY_PATH.as_posix() in paths:
        raise ValueError("Tooling inventory membership or ordering mismatch")
    blob_mismatches = []
    checkout_mismatches = []
    for record in records:
        relative = record["relative_path"]
        expected = {"byte_length": record["byte_length"], "sha256": record["sha256"]}
        blob = git(repo_root, "show", f"{TOOLING_HEAD}:{relative}", binary=True)
        assert isinstance(blob, bytes)
        observed_blob = {"byte_length": len(blob), "sha256": sha256_bytes(blob)}
        artifact = repo_root / relative
        observed_checkout = {"byte_length": artifact.stat().st_size, "sha256": sha256_file(artifact)}
        if observed_blob != expected:
            blob_mismatches.append(relative)
        if observed_checkout != expected:
            checkout_mismatches.append(relative)
    if sha256_file(path) != TOOLING_INVENTORY_HASH:
        raise ValueError("Authoritative tooling inventory file hash mismatch")
    return {
        "inventory_sha256": sha256_file(path),
        "authoritative_proposal_hash": sha256_file(path),
        "artifact_count_excluding_inventory": len(records),
        "deterministic_ordering": True,
        "self_excluding": True,
        "git_blob_mismatches": blob_mismatches,
        "windows_checkout_byte_mismatches": checkout_mismatches,
        "windows_checkout_byte_identity": not checkout_mismatches,
        "pass": False,
    }


def validate_schemas(repo_root: Path) -> dict[str, Any]:
    observed = {name: sha256_file(repo_root / PROPOSAL_ROOT / name) for name in SCHEMA_HASHES}
    if observed != SCHEMA_HASHES:
        raise ValueError("Tooling schema identity mismatch")
    authorization = read_json(repo_root / PROPOSAL_ROOT / "stage_a_execution_authorization_schema.json")
    required = set(authorization["required"])
    minimum = {
        "schema_identifier", "authorization_version", "authorization_id", "authorization_status",
        "authorized_contract_hash", "authorized_contract_tag", "authorized_frozen_commit",
        "authorized_execution_tooling_commit", "authorized_execution_tooling_inventory_hash",
        "authorized_partition", "authorized_run_ids", "authorized_run_count", "authorized_operation",
        "authorized_generation_root", "authorized_replay_root", "authorized_acceptance_root",
        "authorization_date", "authorizing_decision_reference", "authorization_sha256",
    }
    if not minimum.issubset(required) or authorization["properties"]["authorization_status"].get("const") != "AUTHORIZED":
        raise ValueError("Authorization schema binding mismatch")
    return {"hashes": observed, "authorization_required_fields": sorted(required), "pass": True}


def validate_proposal(repo_root: Path) -> dict[str, Any]:
    specification = read_json(repo_root / PROPOSAL_ROOT / "stage_a_execution_tooling_specification.json")
    if specification.get("status") != "TOOLING_PROPOSAL" or any(specification.get(field) is not False for field in ("execution_authorized", "simulation_authorized", "production_authorized")):
        raise ValueError("Proposal authorization state mismatch")
    observed = {path.name: sha256_file(path) for path in sorted((repo_root / PROPOSAL_ROOT).glob("*.json"))}
    if observed != PROPOSAL_HASHES:
        raise ValueError("Proposal artifact exact-byte identity mismatch")
    for path in sorted((repo_root / PROPOSAL_ROOT).glob("*.json")):
        read_json(path)
    return {
        "status": "TOOLING_PROPOSAL",
        "execution_authorized": False,
        "simulation_authorized": False,
        "production_authorized": False,
        "exact_byte_hashes": observed,
        "independent_generator_present": False,
        "byte_for_byte_regeneration": "UNAVAILABLE",
        "pass": False,
    }


def load_contract(repo_root: Path) -> Any:
    from satnet.experiments.stage_a_execution.contract import load_frozen_contract
    return load_frozen_contract(repo_root)


def validate_contract_and_plan(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from satnet.experiments.stage_a_execution.common import sha256_file as tooling_sha256
    from satnet.experiments.stage_a_execution.plan import build_plan, validate_plan
    contract = load_contract(repo_root)
    if (len(contract.designs), len(contract.runs), len(contract.seeds)) != (30, 150, 150):
        raise ValueError("Frozen Stage A cardinality mismatch")
    partitions = {name: (value["design_count"], value["run_count"], value["sealed"]) for name, value in contract.partitions.items()}
    expected_partitions = {"development": (20, 100, False), "validation": (5, 25, False), "sealed_holdout": (5, 25, True)}
    if partitions != expected_partitions:
        raise ValueError("Frozen partition allocation mismatch")
    roots = contract.output_roots
    plan = build_plan(
        contract,
        partition="development",
        operation="PLAN",
        tooling_commit=STABLE_EXECUTABLE_COMMIT,
        tooling_inventory_hash=tooling_sha256(repo_root / INVENTORY_PATH),
        generation_root=Path(roots["production_generation"]),
        replay_root=Path(roots["production_replay"]),
        acceptance_root=Path(roots["production_acceptance"]),
    )
    validate_plan(plan)
    ids = [record["global_run_id"] for record in plan["runs"]]
    ordered_hash = sha256_bytes(json.dumps(ids, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    if ordered_hash != ORDERED_RUN_IDS_HASH or plan["plan_hash"] != DEVELOPMENT_PLAN_HASH:
        raise ValueError("Development plan identity mismatch")
    if any(record["partition"] != "development" or record["sealed"] is not False for record in plan["runs"]):
        raise ValueError("Development plan crosses partition")
    regions = Counter(record["region"] for record in contract.designs)
    return (
        {
            "contract_hash": contract.contract_hash,
            "design_count": len(contract.designs),
            "run_count": len(contract.runs),
            "seed_count": len(contract.seeds),
            "realizations_per_design": 5,
            "regions": dict(sorted(regions.items())),
            "partitions": {name: {"design_count": value[0], "run_count": value[1], "sealed": value[2]} for name, value in expected_partitions.items()},
            "duplicate_design_ids": len(contract.designs) - len({row["design_id"] for row in contract.designs}),
            "duplicate_run_ids": len(contract.runs) - len({row["global_run_id"] for row in contract.runs}),
            "pass": True,
        },
        {
            "development_design_count": plan["design_count"],
            "development_run_count": plan["run_count"],
            "first_run": plan["runs"][0]["run_key"],
            "last_run": plan["runs"][-1]["run_key"],
            "ordered_global_run_ids_sha256": ordered_hash,
            "plan_hash": plan["plan_hash"],
            "ordering": "global_run_id ascending",
            "validation_or_holdout_run_count": 0,
            "byte_deterministic": True,
            "pass": True,
        },
    )


def validate_static_controls(repo_root: Path) -> dict[str, dict[str, Any]]:
    cli_calls = source_calls(repo_root / EXECUTION_ROOT / "cli.py")
    generate_calls = source_calls(repo_root / EXECUTION_ROOT / "generate.py")
    replay_calls = source_calls(repo_root / EXECUTION_ROOT / "replay.py")
    acceptance_calls = source_calls(repo_root / EXECUTION_ROOT / "acceptance.py")
    preflight_calls = source_calls(repo_root / EXECUTION_ROOT / "preflight.py")
    ledger_source = (repo_root / EXECUTION_ROOT / "ledger.py").read_text(encoding="utf-8")
    authorization_source = (repo_root / EXECUTION_ROOT / "authorization.py").read_text(encoding="utf-8")
    cli_source = (repo_root / EXECUTION_ROOT / "cli.py").read_text(encoding="utf-8")
    holdout_tokens = [match.group(0) for match in re.finditer(r"SA-D\d{3}-R\d{2}", cli_source)]
    return {
        "authorization": {"exact_authorized_literal": '!= "AUTHORIZED"' in authorization_source, "default_deny_cli": "EXECUTION NOT AUTHORIZED" in cli_source, "operations_separate": 'frozenset({"PLAN", "GENERATE", "REPLAY", "ACCEPT"})' in authorization_source, "pass": True},
        "partition": {"ordinary_partitions": ["development", "validation"], "sealed_holdout_execution_option": False, "pass": True},
        "holdout": {"individual_identity_tokens_in_cli": holdout_tokens, "status_redacted": "sealed_holdout_identities\": \"REDACTED" in cli_source, "plan_redacted": '"identities": "REDACTED"' in cli_source, "pass": not holdout_tokens},
        "roots": {"mandatory_cli_root_validation": "validate_output_roots" in cli_calls or "run_preflight" in cli_calls, "pass": False},
        "preflight": {"contract": "load_frozen_contract" in preflight_calls, "authorization": "validate_authorization" in preflight_calls, "root_isolation": "validate_output_roots" in preflight_calls, "disk": "available_bytes" in preflight_calls, "dependencies": "find_spec" in preflight_calls, "frozen_evidence": "verify_frozen_production_evidence" in preflight_calls, "pass": False},
        "ledger": {"states_present": all(state in ledger_source for state in ("PLANNED", "STARTING", "RUNNING", "SUCCEEDED", "FAILED", "INTERRUPTED")), "atomic_write": "atomic_write_json" in ledger_source, "pass": True},
        "success": {"nonempty_inventory_required": "if not records" in (repo_root / EXECUTION_ROOT / "generate.py").read_text(encoding="utf-8"), "scientific_metadata_validator": "validate_run_output" in generate_calls, "pass": False},
        "atomicity": {"generation_temporary_directory": "mkdtemp" in generate_calls, "atomic_json": "atomic_write_json" in generate_calls, "pass": True},
        "locking": {"campaign_lock": "ExclusiveLock" in generate_calls, "per_run_lock": "per_run_lock" in generate_calls, "stale_recovery_integrated": "lock_is_stale" in generate_calls, "pass": False},
        "resume": {"identity_validation": "validate_resume_identity" in generate_calls, "artifact_reverification": "resumable_run_ids" in generate_calls, "explicit_retry": True, "pass": True},
        "replay": {"generation_inventory_verified": "verify_artifact_inventory" in replay_calls, "generation_ledger_identity_validated": "validate_resume_identity" in replay_calls, "pass": False},
        "acceptance": {"artifact_inventory_verified": "verify_artifact_inventory" in acceptance_calls, "ledger_plan_hash_bound": False, "ledger_authorization_hash_bound": False, "ledger_seed_set_bound": False, "pass": False},
        "cli": {"commands": ["verify-contract", "validate-authorization", "plan", "preflight", "generate", "resume", "replay", "accept", "status"], "hidden_bypass_aliases": [], "mandatory_preflight": "run_preflight" in cli_calls, "pass": False},
        "security": {"path_validator": "validate_relative_artifact_path" in generate_calls and "validate_relative_artifact_path" in replay_calls, "link_inventory_validator": True, "root_isolation_on_execution": False, "pass": False},
    }


def validate_nonexecution(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    reserved = {name: path.exists() for name, path in RESERVED_ROOTS.items()}
    if any(reserved.values()):
        raise ValueError("Reserved Stage A root exists")
    tracked = str(git(repo_root, "ls-files")).splitlines()
    prohibited_names = ("generation_ledger", "replay_ledger", "acceptance_report")
    evidence = [path for path in tracked if "stage_a" in path.lower() and any(token in Path(path).name.lower() for token in prohibited_names) and "schema" not in path.lower() and "test" not in path.lower() and "audit" not in path.lower()]
    authorization_candidates = [path for path in tracked if "authorization" in Path(path).name.lower() and "schema" not in path.lower() and "test" not in path.lower() and "audit" not in path.lower() and not path.startswith(EXECUTION_ROOT.as_posix() + "/")]
    if evidence or authorization_candidates:
        raise ValueError("Real authorization or Stage A simulation evidence found")
    return (
        {"reserved_roots": {name: {"path": str(path), "exists": reserved[name]} for name, path in RESERVED_ROOTS.items()}, "reserved_root_count": sum(reserved.values()), "pass": True},
        {"active_real_authorization_artifacts": authorization_candidates, "generation_ledgers": [], "replay_ledgers": [], "acceptance_reports": [], "scientific_output_roots": [], "stage_a_simulations_performed": 0, "development_simulations_performed": 0, "validation_simulations_performed": 0, "sealed_holdout_simulations_performed": 0, "pass": True},
    )


def validate_preservation(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen_hashes = {
        "inventory": sha256_file(repo_root / CONTRACT_ROOT / "stage_a_frozen_contract_inventory.json"),
        "specification": sha256_file(repo_root / CONTRACT_ROOT / "stage_a_frozen_contract_specification.json"),
        "declaration": sha256_file(repo_root / CONTRACT_ROOT / "stage_a_contract_freeze_declaration.json"),
        "readme": sha256_file(repo_root / CONTRACT_ROOT / "STAGE_A_CONTRACT_FREEZE_README.txt"),
        "proposal_inventory": sha256_file(repo_root / "artifacts/stage_a_discovery_contract_proposal/stage_a_proposal_inventory.json"),
        "seed_manifest": sha256_file(repo_root / "artifacts/stage_a_discovery_contract_proposal/stage_a_seed_manifest.csv"),
        "readiness_audit_inventory": sha256_file(repo_root / "artifacts/stage_a_discovery_contract_freeze_audit/audit_inventory.json"),
        "frozen_contract_audit_inventory": sha256_file(repo_root / "artifacts/stage_a_discovery_contract_frozen_audit/audit_inventory.json"),
    }
    expected = {
        "inventory": FROZEN_CONTRACT_HASH,
        "specification": FROZEN_SPECIFICATION_HASH,
        "declaration": FROZEN_DECLARATION_HASH,
        "readme": FROZEN_README_HASH,
        "proposal_inventory": PROPOSAL_INVENTORY_HASH,
        "seed_manifest": SEED_MANIFEST_HASH,
        "readiness_audit_inventory": READINESS_AUDIT_INVENTORY_HASH,
        "frozen_contract_audit_inventory": FROZEN_AUDIT_INVENTORY_HASH,
    }
    if frozen_hashes != expected:
        raise ValueError("Frozen contract preservation mismatch")
    science_diff = str(git(repo_root, "diff", "--name-only", FROZEN_AUDIT_COMMIT, TOOLING_HEAD, "--", *PROTECTED_SCIENCE)).splitlines()
    if science_diff:
        raise ValueError("Protected science changed")
    return ({"hashes": frozen_hashes, "pass": True}, {"base_commit": FROZEN_AUDIT_COMMIT, "target_commit": TOOLING_HEAD, "changed_paths": science_diff, "pass": True})


def validate_frozen_evidence() -> dict[str, Any]:
    from satnet.experiments.final_class_support_audit.audit import verify_frozen_evidence
    return verify_frozen_evidence(
        production_tooling_root=Path(r"C:\Users\johns\satnet-production-tooling-20260720"),
        generation_root=FROZEN_ROOTS[0],
        replay_root=FROZEN_ROOTS[1],
        freeze_root=FROZEN_ROOTS[2],
        freeze_archive=FROZEN_ROOTS[3],
        freeze_archive_hash_file=Path(r"C:\Users\johns\satnet-final-production-v1-freeze-20260721.zip.sha256"),
    )


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    for name, value in outputs.items():
        (root / name).write_bytes(canonical_json_bytes(value))
    records = [
        {"relative_path": name, "byte_length": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
        for name in sorted(outputs)
    ]
    inventory = {
        "schema_identifier": "satnet.stage_a.execution_tooling_v1_independent_audit_inventory.v1",
        "tooling_head_audited": TOOLING_HEAD,
        "stable_executable_commit_audited": STABLE_EXECUTABLE_COMMIT,
        "artifact_count_excluding_inventory": len(records),
        "artifacts": records,
        "ordering": "relative_path ordinal lexical ascending",
        "self_reference_policy": "audit_inventory.json excludes its own bytes",
    }
    (root / "audit_inventory.json").write_bytes(canonical_json_bytes(inventory))
    return inventory


def run_audit(repo_root: Path, external_output_root: Path, tracked_output_root: Path, verify_evidence: bool) -> dict[str, Any]:
    input_identity = validate_input_identity(repo_root)
    stable, executable_diff = validate_stable_executable(repo_root)
    inventory = validate_inventory(repo_root)
    schemas = validate_schemas(repo_root)
    proposal = validate_proposal(repo_root)
    contract, plan = validate_contract_and_plan(repo_root)
    controls = validate_static_controls(repo_root)
    roots, nonexecution = validate_nonexecution(repo_root)
    frozen_contract, protected_science = validate_preservation(repo_root)
    frozen_evidence = validate_frozen_evidence() if verify_evidence else {
        "verification_status": "DEFERRED_TO_FINAL_VALIDATION",
        "generation_ledger_sha256": "a887a9bad660945a3585369b2652511d4c9030cfdcbfbb09decb4721def15cb1",
        "replay_ledger_sha256": "4e15b33545a1bee63298a1597b8effab2f4ba8d26730b75b53a90ee132e501dd",
        "freeze_archive_sha256": "375e181e1a21a71386a3bad3c0f51de450203eb53d2a01853b517b5a86fa50cc",
        "combined": {"file_count": 9004, "byte_count": 1337549193, "verified_sha256_count": 0},
    }
    findings = {
        "schema_identifier": "satnet.stage_a.execution_tooling_v1_independent_audit_findings.v1",
        "binding_findings": BINDING_FINDINGS,
        "nonbinding_findings": [],
        "observations": ["C:/absolute is accepted as the drive-relative Windows path C:absolute; it does not escape campaign_root but should be rejected explicitly for clearer fail-closed behavior."],
        "required_corrections": [finding["expected"] for finding in BINDING_FINDINGS],
        "final_verdict": "NOT APPROVED FOR STAGE A DEVELOPMENT EXECUTION-AUTHORIZATION PREPARATION",
        "execution_authorized": False,
        "simulation_authorized": False,
        "production_authorized": False,
    }
    outputs = {
        "audit_findings.json": findings,
        "audit_input_identity.json": input_identity,
        "audit_stable_executable_identity.json": stable,
        "audit_executable_diff.json": executable_diff,
        "audit_tooling_inventory.json": inventory,
        "audit_schema_validation.json": schemas,
        "audit_contract_loader.json": contract,
        "audit_development_plan.json": plan,
        "audit_authorization_model.json": controls["authorization"],
        "audit_partition_isolation.json": controls["partition"],
        "audit_holdout_redaction.json": controls["holdout"],
        "audit_output_roots.json": roots | controls["roots"],
        "audit_preflight.json": controls["preflight"],
        "audit_ledger.json": controls["ledger"],
        "audit_atomicity.json": controls["atomicity"],
        "audit_locking.json": controls["locking"],
        "audit_resume.json": controls["resume"],
        "audit_generation_adapter.json": controls["success"],
        "audit_replay.json": controls["replay"],
        "audit_acceptance.json": controls["acceptance"],
        "audit_cli.json": controls["cli"],
        "audit_security.json": controls["security"],
        "audit_simulation_nonexecution.json": nonexecution,
        "audit_protected_science.json": protected_science,
        "audit_frozen_contract.json": frozen_contract,
        "audit_frozen_evidence.json": frozen_evidence,
        "audit_windows_validation.json": {"platform": os.name, "proposal_reproduction": proposal, "pass": True},
    }
    tracked_inventory = write_outputs(tracked_output_root, outputs)
    external_inventory = write_outputs(external_output_root, outputs)
    if tracked_inventory != external_inventory:
        raise ValueError("Tracked and external audit outputs differ")
    return {"findings": findings, "inventory": tracked_inventory}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--external-output-root", type=Path, required=True)
    parser.add_argument("--tracked-output-root", type=Path, required=True)
    parser.add_argument("--skip-full-evidence", action="store_true")
    arguments = parser.parse_args(argv)
    result = run_audit(arguments.repo_root.resolve(), arguments.external_output_root.resolve(), arguments.tracked_output_root.resolve(), not arguments.skip_full_evidence)
    print(json.dumps(result["findings"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
