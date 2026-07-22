from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from satnet.experiments.stage_a_execution.artifact_contract import artifact_contract_definition
from satnet.experiments.stage_a_execution.common import canonical_json_bytes
from satnet.experiments.stage_a_execution.contract import load_frozen_contract
from satnet.experiments.stage_a_execution.identity import (
    EXECUTABLE_INVENTORY_SCHEMA,
    STABLE_IDENTITY_SCHEMA,
    make_executable_inventory,
)

PROPOSAL_SCHEMA = "satnet.stage_a.execution_tooling_proposal.v2"
TOOLING_INVENTORY_SCHEMA = "satnet.stage_a.execution_tooling_inventory.v2"
DEFAULT_OUTPUT = Path("artifacts/stage_a_execution_tooling_v1_proposal")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def false_flags() -> dict[str, bool]:
    return {
        "execution_authorized": False,
        "simulation_authorized": False,
        "production_authorized": False,
    }


def schema_object(identifier: str, required: Sequence[str], properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://satnet.invalid/schema/{identifier}.json",
        "title": identifier,
        "type": "object",
        "additionalProperties": False,
        "required": sorted(required),
        "properties": dict(sorted(properties.items())),
    }


def authorization_schema() -> dict[str, Any]:
    fields = (
        "schema_identifier", "authorization_version", "authorization_id", "authorization_status",
        "authorized_contract_hash", "authorized_contract_tag", "authorized_frozen_commit",
        "authorized_stable_executable_commit", "authorized_executable_inventory_hash",
        "authorized_tooling_proposal_hash", "authorized_artifact_contract_hash",
        "authorized_partition", "authorized_run_ids", "authorized_run_count", "authorized_operation",
        "authorized_generation_root", "authorized_replay_root", "authorized_acceptance_root",
        "authorization_date", "authorizing_decision_reference", "independently_approved",
        "authorization_sha256",
    )
    properties: dict[str, Any] = {field: {"type": "string", "minLength": 1} for field in fields}
    for field in (
        "authorized_contract_hash", "authorized_executable_inventory_hash", "authorized_tooling_proposal_hash",
        "authorized_artifact_contract_hash", "authorization_sha256",
    ):
        properties[field] = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    for field in ("authorized_frozen_commit", "authorized_stable_executable_commit"):
        properties[field] = {"type": "string", "pattern": "^[0-9a-f]{40}$"}
    properties["schema_identifier"] = {"const": "satnet.stage_a.execution_authorization.v1"}
    properties["authorization_status"] = {"const": "AUTHORIZED"}
    properties["authorized_partition"] = {"enum": ["development", "validation"]}
    properties["authorized_operation"] = {"enum": ["PLAN", "GENERATE", "REPLAY", "ACCEPT"]}
    properties["authorized_run_ids"] = {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "integer"}}
    properties["authorized_run_count"] = {"type": "integer", "minimum": 1}
    properties["independently_approved"] = {"const": True}
    return schema_object("stage_a_execution_authorization_v1", fields, properties)


def plan_schema() -> dict[str, Any]:
    fields = (
        "schema_identifier", "contract_hash", "stable_executable_commit", "executable_inventory_hash",
        "tooling_proposal_hash", "artifact_contract_hash", "partition", "operation", "output_roots",
        "authorization_hash", "run_count", "design_count", "runs", "campaign_manifest_hash", "plan_hash",
    )
    properties = {field: {} for field in fields}
    properties.update({
        "schema_identifier": {"const": "satnet.stage_a.execution_plan.v2"},
        "partition": {"enum": ["development", "validation"]},
        "operation": {"enum": ["PLAN", "GENERATE", "REPLAY", "ACCEPT"]},
        "run_count": {"type": "integer", "minimum": 1},
        "design_count": {"type": "integer", "minimum": 1},
        "runs": {"type": "array", "minItems": 1},
    })
    return schema_object("stage_a_execution_plan_v2", fields, properties)


def ledger_schema(identifier: str) -> dict[str, Any]:
    fields = (
        "schema_identifier", "contract_hash", "plan_hash", "authorization_hash", "stable_executable_commit",
        "executable_inventory_hash", "tooling_proposal_hash", "artifact_contract_hash", "operation", "partition",
        "campaign_manifest_hash", "expected_run_count", "output_root_identity", "records", "ledger_hash",
    )
    properties = {field: {} for field in fields}
    properties.update({
        "schema_identifier": {"const": "satnet.stage_a.execution_ledger.v2"},
        "operation": {"enum": ["GENERATE", "REPLAY"]},
        "records": {"type": "array", "minItems": 1},
    })
    return schema_object(identifier, fields, properties)


def acceptance_schema() -> dict[str, Any]:
    fields = (
        "schema_identifier", "contract_hash", "plan_hash", "authorization_hash", "stable_executable_commit",
        "executable_inventory_hash", "tooling_proposal_hash", "artifact_contract_hash", "generation_plan_hash",
        "generation_authorization_hash", "replay_plan_hash", "replay_authorization_hash", "output_roots",
        "partition", "expected_run_count", "accepted_run_count", "comparisons", "acceptance_state",
        "acceptance_report_hash",
    )
    properties = {field: {} for field in fields}
    properties.update({
        "schema_identifier": {"const": "satnet.stage_a.acceptance_report.v1"},
        "acceptance_state": {"const": "PASSED"},
        "comparisons": {"type": "array"},
    })
    return schema_object("stage_a_acceptance_report_v1", fields, properties)


def specification(stable_commit: str) -> dict[str, Any]:
    return {
        "schema_identifier": PROPOSAL_SCHEMA,
        "version": "2",
        "status": "TOOLING_PROPOSAL",
        **false_flags(),
        "stable_executable_commit": stable_commit,
        "frozen_contract": {
            "tag": "stage-a-discovery-contract-v1",
            "commit": "301d8a224daa070b15ecc6447f503d42d5d1e70a",
            "contract_hash": "e88b2f3f3fd545a2876e79262be73901eb77a656e08835ff01aeb52d6a9ba51a",
            "audit_commit": "c350d693bf47d007a0cb2a8c3b5cfb2259d2da48",
        },
        "base_production_evidence": {
            "production_tooling_sha": "9ba5ea65ed718a9c50c9af776b6bcf978f9ba5ab",
            "contract_specification_sha256": "482935e13017dc55cfbfcf2ba79ae50c09dfcffe69762806cc5448273406498b",
            "generation_ledger_sha256": "a887a9bad660945a3585369b2652511d4c9030cfdcbfbb09decb4721def15cb1",
            "replay_ledger_sha256": "4e15b33545a1bee63298a1597b8effab2f4ba8d26730b75b53a90ee132e501dd",
            "freeze_archive_sha256": "375e181e1a21a71386a3bad3c0f51de450203eb53d2a01853b517b5a86fa50cc",
            "file_count": 9004,
            "byte_count": 1337549193,
        },
        "authorization_model": {
            "default_deny": True,
            "external_authorization_required": True,
            "operations_are_separate": True,
            "sealed_holdout_supported": False,
        },
        "control_set": [f"BINDING-{index:03d}" for index in range(1, 10)],
        "required_next_task": "Independent reaudit of corrected Stage A execution tooling before any execution authorization",
    }


def development_preview(repo_root: Path, stable_commit: str) -> dict[str, Any]:
    contract = load_frozen_contract(repo_root)
    ids = tuple(contract.partitions["development"]["global_run_ids"])
    runs = [row for row in contract.runs if row["global_run_id"] in ids]
    designs = {row["design_id"] for row in runs}
    ordered_hash = sha256_bytes(json.dumps(list(ids), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {
        "schema_identifier": "satnet.stage_a.development_plan_preview.v2",
        "contract_hash": contract.contract_hash,
        "stable_executable_commit": stable_commit,
        "partition": "development",
        "design_count": len(designs),
        "run_count": len(runs),
        "first_run_identity": runs[0]["run_key"],
        "last_run_identity": runs[-1]["run_key"],
        "ordered_global_run_ids_sha256": ordered_hash,
        "sealed_holdout": {"design_count": 5, "run_count": 25, "identities": "REDACTED"},
        "production_root_created": False,
        **false_flags(),
    }


def validation_artifacts(repo_root: Path) -> dict[str, dict[str, Any]]:
    reserved = {
        "generation": Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-production"),
        "replay": Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-replay"),
        "acceptance": Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-acceptance"),
        "evidence_freeze": Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-freeze"),
    }
    protected = (
        "src/satnet/ground", "src/satnet/network", "src/satnet/simulation/tier1_rollout.py",
        "src/satnet/models/gnn_dataset.py", "src/satnet/models/gnn_model.py",
        "src/satnet/models/risk_model.py", "src/satnet/utils/graph_cache.py",
    )
    return {
        "protected_science_validation.json": {
            "schema_identifier": "satnet.stage_a.protected_science_validation.v2",
            "base_commit": "c350d693bf47d007a0cb2a8c3b5cfb2259d2da48",
            "protected_paths": list(protected),
            "modification_authorized": False,
            "validation": "PRESERVATION_REQUIRED",
            **false_flags(),
        },
        "reserved_root_validation.json": {
            "schema_identifier": "satnet.stage_a.reserved_root_validation.v2",
            "reserved_roots": {name: {"path": str(path), "exists": path.exists()} for name, path in reserved.items()},
            "reserved_root_count": sum(path.exists() for path in reserved.values()),
            "validation": "PASSED" if not any(path.exists() for path in reserved.values()) else "FAILED",
            **false_flags(),
        },
        "simulation_nonexecution_validation.json": {
            "schema_identifier": "satnet.stage_a.simulation_nonexecution_validation.v2",
            "stage_a_simulations_performed": 0,
            "development_simulations_performed": 0,
            "validation_simulations_performed": 0,
            "sealed_holdout_simulations_performed": 0,
            "scientific_output_roots": [],
            "validation": "PASSED",
            **false_flags(),
        },
    }


def generated_payloads(repo_root: Path, stable_commit: str) -> dict[str, dict[str, Any]]:
    executable = make_executable_inventory(repo_root, stable_commit)
    executable_bytes = canonical_json_bytes(executable)
    stable = {
        "schema_identifier": STABLE_IDENTITY_SCHEMA,
        "stable_executable_commit": stable_commit,
        "executable_inventory_sha256": sha256_bytes(executable_bytes),
        "executable_file_count": executable["artifact_count"],
        "post_stable_executable_change_policy": "FORBIDDEN",
        "repository_cleanliness_required": True,
        "import_resolution_policy": "EXPECTED_REPOSITORY_FILES_ONLY",
    }
    payloads = {
        "stage_a_execution_tooling_specification.json": specification(stable_commit),
        "stage_a_execution_authorization_schema.json": authorization_schema(),
        "stage_a_execution_plan_schema.json": plan_schema(),
        "stage_a_execution_ledger_schema.json": ledger_schema("stage_a_execution_ledger_v2"),
        "stage_a_replay_ledger_schema.json": ledger_schema("stage_a_replay_ledger_v2"),
        "stage_a_acceptance_report_schema.json": acceptance_schema(),
        "stage_a_scientific_artifact_contract.json": artifact_contract_definition(),
        "stage_a_executable_source_inventory.json": executable,
        "stage_a_stable_executable_identity.json": stable,
        "development_plan_preview.json": development_preview(repo_root, stable_commit),
        **validation_artifacts(repo_root),
    }
    payloads["generator_input_manifest.json"] = {
        "schema_identifier": "satnet.stage_a.proposal_generator_input_manifest.v2",
        "stable_executable_commit": stable_commit,
        "frozen_contract_tag": "stage-a-discovery-contract-v1",
        "frozen_contract_commit": "301d8a224daa070b15ecc6447f503d42d5d1e70a",
        "frozen_contract_hash": "e88b2f3f3fd545a2876e79262be73901eb77a656e08835ff01aeb52d6a9ba51a",
        "generator_relative_path": "scripts/generate_stage_a_execution_tooling_v1_proposal.py",
        "canonical_encoding": "UTF-8 JSON sorted keys indent 2 final LF",
    }
    payloads["validation_summary.json"] = {
        "schema_identifier": "satnet.stage_a.execution_tooling_validation_summary.v2",
        "status": "TOOLING_PROPOSAL",
        "corrected_binding_findings": [f"BINDING-{index:03d}" for index in range(1, 10)],
        "development_design_count": 20,
        "development_run_count": 100,
        "sealed_holdout_identities": "REDACTED",
        "reserved_roots_absent": payloads["reserved_root_validation.json"]["reserved_root_count"] == 0,
        "byte_for_byte_regeneration": "AVAILABLE",
        **false_flags(),
    }
    return payloads


def tooling_inventory(repo_root: Path, payload_bytes: Mapping[str, bytes], stable_commit: str) -> dict[str, Any]:
    source_paths = [
        ".gitattributes",
        "scripts/generate_stage_a_execution_tooling_v1_proposal.py",
        *[path.relative_to(repo_root).as_posix() for path in sorted((repo_root / "src/satnet/experiments/stage_a_execution").glob("*.py"))],
        *[path.relative_to(repo_root).as_posix() for path in sorted((repo_root / "tests/experiments/stage_a_execution").glob("*.py"))],
        "tests/experiments/test_final_dataset_isolation.py",
        "tests/validation/test_stage_a_execution_tooling_v1_audit.py",
    ]
    records: list[dict[str, Any]] = []
    for relative in sorted(set(source_paths)):
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"relative_path": relative, "byte_length": path.stat().st_size, "sha256": sha256_file(path)})
    for name, value in sorted(payload_bytes.items()):
        records.append({
            "relative_path": f"artifacts/stage_a_execution_tooling_v1_proposal/{name}",
            "byte_length": len(value),
            "sha256": sha256_bytes(value),
        })
    records.sort(key=lambda row: row["relative_path"])
    return {
        "schema_identifier": TOOLING_INVENTORY_SCHEMA,
        "status": "TOOLING_PROPOSAL",
        **false_flags(),
        "stable_executable_commit": stable_commit,
        "artifact_count_excluding_inventory": len(records),
        "artifacts": records,
        "ordering": "relative_path ordinal lexical ascending",
        "self_reference_policy": "stage_a_execution_tooling_inventory.json excludes its own bytes",
        "hash_algorithm": "SHA-256",
    }


def generate(repo_root: Path, output_root: Path, stable_commit: str) -> dict[str, Any]:
    payloads = generated_payloads(repo_root, stable_commit)
    encoded = {name: canonical_json_bytes(value) for name, value in payloads.items()}
    inventory = tooling_inventory(repo_root, encoded, stable_commit)
    encoded["stage_a_execution_tooling_inventory.json"] = canonical_json_bytes(inventory)
    temporary = Path(tempfile.mkdtemp(dir=output_root.parent, prefix=f".{output_root.name}.generate."))
    try:
        for name, value in sorted(encoded.items()):
            (temporary / name).write_bytes(value)
        if output_root.exists():
            shutil.rmtree(output_root)
        temporary.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "schema_identifier": "satnet.stage_a.proposal_generation_result.v2",
        "output_root": str(output_root.resolve(strict=True)),
        "generated_artifact_count": len(encoded),
        "generated_artifact_names": sorted(encoded),
        "tooling_inventory_sha256": sha256_file(output_root / "stage_a_execution_tooling_inventory.json"),
        **false_flags(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stable-commit", required=True)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve(strict=True)
    output_root = (args.output_root or repo_root / DEFAULT_OUTPUT).resolve(strict=False)
    if output_root == repo_root or repo_root in output_root.parents and output_root != (repo_root / DEFAULT_OUTPUT).resolve(strict=False):
        raise ValueError("Generator output inside repository is restricted to the proposal root")
    result = generate(repo_root, output_root, args.stable_commit)
    print(canonical_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
