from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from satnet.experiments.final_generation.artifacts import validate_run_result, validate_scientific_inventory, validate_target_artifact
from satnet.experiments.final_generation.constants import RUN_FILES
from satnet.experiments.final_generation.io import read_canonical_json
from satnet.experiments.final_generation.run_validation import artifact_paths

from .common import payload_hash, read_json_object
from .contract import FROZEN_CONTRACT_HASH
from .integrity import artifact_inventory, inventory_hash

ARTIFACT_CONTRACT_SCHEMA = "satnet.stage_a.scientific_artifact_contract.v2"
ARTIFACT_CONTRACT_DOMAIN = "satnet_stage_a_scientific_artifact_contract_v2"
PRODUCTION_ARTIFACT_CONTRACT = "canonical_tier1_integrated_run_v1"
SYNTHETIC_ARTIFACT_CONTRACT = "synthetic_stage_a_test_run_v1"


@dataclass(frozen=True)
class SimulationAdapterResult:
    run_key: str
    design_id: str
    global_run_id: int
    realization_id: str
    design_construction_seed: int
    ground_selection_seed: int
    satellite_failure_seed: int
    ground_failure_seed: int
    artifact_contract: str
    artifact_manifest: tuple[dict[str, Any], ...]
    artifact_inventory_hash: str
    simulation_return_status: str
    validation_status: str
    result_identity: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_key": self.run_key,
            "design_id": self.design_id,
            "global_run_id": self.global_run_id,
            "realization_id": self.realization_id,
            "design_construction_seed": self.design_construction_seed,
            "ground_selection_seed": self.ground_selection_seed,
            "satellite_failure_seed": self.satellite_failure_seed,
            "ground_failure_seed": self.ground_failure_seed,
            "artifact_contract": self.artifact_contract,
            "artifact_manifest": list(self.artifact_manifest),
            "artifact_inventory_hash": self.artifact_inventory_hash,
            "simulation_return_status": self.simulation_return_status,
            "validation_status": self.validation_status,
            "result_identity": self.result_identity,
        }


@dataclass(frozen=True)
class ScienceCompletionResult:
    validation_status: str
    validation_kind: str
    verified_stages: tuple[str, ...]
    authoritative_result_hash: str
    completion_identity: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "validation_status": self.validation_status,
            "validation_kind": self.validation_kind,
            "verified_stages": list(self.verified_stages),
            "authoritative_result_hash": self.authoritative_result_hash,
            "completion_identity": self.completion_identity,
        }


def science_completion_hash(value: Mapping[str, Any]) -> str:
    return payload_hash(dict(value), domain="satnet_stage_a_science_completion_v1")


def adapter_result_hash(value: Mapping[str, Any]) -> str:
    return payload_hash(dict(value), domain="satnet_stage_a_adapter_result_v2")


def make_adapter_result(plan_run: Mapping[str, Any], output_root: Path, artifact_contract: str) -> SimulationAdapterResult:
    records = artifact_inventory(output_root)
    payload = {
        "run_key": plan_run["run_key"],
        "design_id": plan_run["design_id"],
        "global_run_id": plan_run["global_run_id"],
        "realization_id": plan_run["realization_id"],
        "design_construction_seed": plan_run["design_construction_seed"],
        "ground_selection_seed": plan_run["ground_selection_seed"],
        "satellite_failure_seed": plan_run["satellite_failure_seed"],
        "ground_failure_seed": plan_run["ground_failure_seed"],
        "artifact_contract": artifact_contract,
        "artifact_manifest": records,
        "artifact_inventory_hash": inventory_hash(records),
        "simulation_return_status": "SUCCEEDED",
        "validation_status": "PASSED",
    }
    return SimulationAdapterResult(**payload, result_identity=adapter_result_hash(payload))


def artifact_contract_definition() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_identifier": ARTIFACT_CONTRACT_SCHEMA,
        "production_contract_identifier": PRODUCTION_ARTIFACT_CONTRACT,
        "synthetic_contract_identifier": SYNTHETIC_ARTIFACT_CONTRACT,
        "production_pipeline": "satnet.experiments.final_generation",
        "production_required_run_files": [RUN_FILES[key] for key in sorted(RUN_FILES)],
        "production_required_wrapper_files": [
            "stage_a_binding.json",
            "validated_pipeline/execution_mode.json",
            "validated_pipeline/operational/attempts/run_{pipeline_local_run_id:03d}/attempt_001.json",
            "validated_pipeline/operational/current_state/run_{pipeline_local_run_id:03d}.json",
            "validated_pipeline/run_{pipeline_local_run_id:03d}/input/design_record.json",
            "validated_pipeline/run_{pipeline_local_run_id:03d}/input/run_record.json",
            "validated_pipeline/run_{pipeline_local_run_id:03d}/operational/attempt.json",
        ],
        "synthetic_required_files": ["scientific.json", "stage_a_binding.json"],
        "unexpected_artifacts": "FORBIDDEN",
        "hash_algorithm": "SHA-256",
        "manifest_ordering": "relative_path ordinal lexical ascending",
        "success_requirements": [
            "exact_required_path_set",
            "canonical_parsing",
            "schema_validation",
            "scientific_identity_match",
            "complete_seed_match",
            "adapter_manifest_filesystem_match",
            "authoritative_satellite_and_g1_g5_replay",
            "artifact_and_science_validation_before_succeeded",
        ],
    }
    value["artifact_contract_hash"] = payload_hash(value, domain=ARTIFACT_CONTRACT_DOMAIN)
    return value


def artifact_contract_hash() -> str:
    return artifact_contract_definition()["artifact_contract_hash"]


def bind_plan_run(plan: Mapping[str, Any], plan_run: Mapping[str, Any]) -> dict[str, Any]:
    bound = dict(plan_run)
    bound.update({
        "plan_hash": plan["plan_hash"],
        "stable_executable_commit": plan["stable_executable_commit"],
        "executable_inventory_hash": plan["executable_inventory_hash"],
        "tooling_proposal_hash": plan["tooling_proposal_hash"],
        "artifact_contract_hash": plan["artifact_contract_hash"],
        "pipeline_local_run_id": int(plan_run["global_run_id"]) - 500,
    })
    return bound


def _binding_identity(plan_run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract_hash": plan_run["contract_hash"],
        "plan_hash": plan_run["plan_hash"],
        "stable_executable_commit": plan_run["stable_executable_commit"],
        "executable_inventory_hash": plan_run["executable_inventory_hash"],
        "tooling_proposal_hash": plan_run["tooling_proposal_hash"],
        "artifact_contract_hash": plan_run["artifact_contract_hash"],
        "partition": plan_run["partition"],
        "design_id": plan_run["design_id"],
        "design_record_hash": plan_run["design_record_hash"],
        "run_key": plan_run["run_key"],
        "global_run_id": plan_run["global_run_id"],
        "run_record_hash": plan_run["run_record_hash"],
        "realization_id": plan_run["realization_id"],
        "realization_index": plan_run["realization_index"],
        "design_construction_seed": plan_run["design_construction_seed"],
        "ground_selection_seed": plan_run["ground_selection_seed"],
        "satellite_failure_seed": plan_run["satellite_failure_seed"],
        "ground_failure_seed": plan_run["ground_failure_seed"],
    }


def synthetic_artifacts(plan_run: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    identity = _binding_identity(plan_run)
    scientific_identity = {
        field: identity[field]
        for field in (
            "contract_hash", "partition", "design_id", "design_record_hash", "run_key",
            "global_run_id", "run_record_hash", "realization_id", "realization_index",
            "design_construction_seed", "ground_selection_seed", "satellite_failure_seed",
            "ground_failure_seed",
        )
    }
    return {
        "scientific.json": {
            "schema_identifier": "satnet.stage_a.synthetic_scientific_output.v1",
            **scientific_identity,
            "scientific_validation": "PASSED",
        },
        "stage_a_binding.json": {
            "schema_identifier": "satnet.stage_a.run_binding.v2",
            **identity,
        },
    }


def write_synthetic_artifacts(plan_run: Mapping[str, Any], output_root: Path) -> SimulationAdapterResult:
    from .common import atomic_write_json

    if plan_run["contract_hash"] == FROZEN_CONTRACT_HASH:
        raise PermissionError("Synthetic adapter is prohibited for the real frozen Stage A contract")
    for relative, value in synthetic_artifacts(plan_run).items():
        atomic_write_json(output_root / relative, value)
    return make_adapter_result(plan_run, output_root, SYNTHETIC_ARTIFACT_CONTRACT)


def _validate_adapter_result(plan_run: Mapping[str, Any], output_root: Path, result: SimulationAdapterResult) -> list[dict[str, Any]]:
    expected = {
        "run_key": plan_run["run_key"],
        "design_id": plan_run["design_id"],
        "global_run_id": plan_run["global_run_id"],
        "realization_id": plan_run["realization_id"],
        "design_construction_seed": plan_run["design_construction_seed"],
        "ground_selection_seed": plan_run["ground_selection_seed"],
        "satellite_failure_seed": plan_run["satellite_failure_seed"],
        "ground_failure_seed": plan_run["ground_failure_seed"],
    }
    for field, value in expected.items():
        if getattr(result, field) != value:
            raise ValueError(f"Adapter result identity mismatch: {field}")
    if result.simulation_return_status != "SUCCEEDED" or result.validation_status != "PASSED":
        raise ValueError("Adapter result did not report successful validated execution")
    payload = result.as_dict()
    claimed = payload.pop("result_identity")
    if claimed != adapter_result_hash(payload):
        raise ValueError("Adapter result self-identity mismatch")
    actual = artifact_inventory(output_root)
    if list(result.artifact_manifest) != actual or result.artifact_inventory_hash != inventory_hash(actual):
        raise ValueError("Adapter artifact manifest differs from filesystem")
    return actual


def _validate_binding(path: Path, plan_run: Mapping[str, Any]) -> None:
    binding = read_json_object(path)
    expected = _binding_identity(plan_run)
    if any(binding.get(field) != value for field, value in expected.items()):
        raise ValueError("Stage A artifact binding identity mismatch")
    if binding.get("schema_identifier") != "satnet.stage_a.run_binding.v2":
        raise ValueError("Stage A artifact binding schema mismatch")


def _validate_synthetic(plan_run: Mapping[str, Any], root: Path, actual: list[dict[str, Any]]) -> None:
    if plan_run["contract_hash"] == FROZEN_CONTRACT_HASH:
        raise PermissionError("Synthetic artifact contract cannot validate the real Stage A contract")
    if [record["relative_path"] for record in actual] != ["scientific.json", "stage_a_binding.json"]:
        raise ValueError("Synthetic artifact path set mismatch or unexpected artifact")
    expected = synthetic_artifacts(plan_run)
    for relative, value in expected.items():
        observed = read_json_object(root / relative)
        if observed != value:
            raise ValueError(f"Synthetic artifact identity or schema mismatch: {relative}")


def _production_expected_paths(plan_run: Mapping[str, Any]) -> set[str]:
    local = int(plan_run["pipeline_local_run_id"])
    run_prefix = f"validated_pipeline/run_{local:03d}"
    return {
        "stage_a_binding.json",
        "validated_pipeline/execution_mode.json",
        f"validated_pipeline/operational/attempts/run_{local:03d}/attempt_001.json",
        f"validated_pipeline/operational/current_state/run_{local:03d}.json",
        f"{run_prefix}/input/design_record.json",
        f"{run_prefix}/input/run_record.json",
        f"{run_prefix}/operational/attempt.json",
        *(f"{run_prefix}/{relative}" for relative in RUN_FILES.values()),
    }


def _validate_production(plan_run: Mapping[str, Any], root: Path, actual: list[dict[str, Any]]) -> None:
    paths = {record["relative_path"] for record in actual}
    expected_paths = _production_expected_paths(plan_run)
    if paths != expected_paths:
        raise ValueError(f"Canonical Tier 1 artifact path set mismatch: missing={sorted(expected_paths - paths)}, extra={sorted(paths - expected_paths)}")
    _validate_binding(root / "stage_a_binding.json", plan_run)
    local = int(plan_run["pipeline_local_run_id"])
    run_root = root / "validated_pipeline" / f"run_{local:03d}"
    canonical_paths = artifact_paths(run_root)
    satellite = read_canonical_json(canonical_paths["satellite"])
    target = read_canonical_json(canonical_paths["target"])
    scientific_inventory = read_canonical_json(canonical_paths["inventory"])
    run_result = read_canonical_json(canonical_paths["result"])
    validate_target_artifact(target)
    validate_scientific_inventory(run_root, scientific_inventory)
    validate_run_result(run_result)
    if satellite.get("run_key") != plan_run["run_key"] or satellite.get("run_record_hash") != plan_run["run_record_hash"]:
        raise ValueError("Satellite artifact run identity mismatch")
    if target.get("run_key") != plan_run["run_key"] or target.get("run_record_hash") != plan_run["run_record_hash"]:
        raise ValueError("Target artifact run identity mismatch")
    run_record = read_canonical_json(run_root / "input" / "run_record.json")
    seed_expectations = {
        "design_construction_seed": plan_run["design_construction_seed"],
        "ground_selection_seed": plan_run["ground_selection_seed"],
        "satellite_seed": plan_run["satellite_failure_seed"],
        "ground_failure_seed": plan_run["ground_failure_seed"],
    }
    if any(run_record.get(field) != value for field, value in seed_expectations.items()):
        raise ValueError("Canonical Tier 1 run seed mismatch")
    if run_record.get("global_run_id") != plan_run["global_run_id"] or run_record.get("design_id") != plan_run["design_id"]:
        raise ValueError("Canonical Tier 1 run scientific identity mismatch")
    for relative in RUN_FILES.values():
        path = run_root / relative
        if relative.endswith(".json"):
            json.loads(path.read_bytes())
        elif relative.endswith(".jsonl"):
            source = path.read_text(encoding="utf-8")
            if not source.endswith("\n") or any(not isinstance(json.loads(line), dict) for line in source.splitlines()):
                raise ValueError(f"Malformed canonical JSONL artifact: {relative}")


def validate_science_completion_result(result: ScienceCompletionResult) -> None:
    if result.validation_status != "PASSED":
        raise ValueError("Authoritative science-completion validation did not pass")
    if not result.validation_kind or not result.verified_stages or not result.authoritative_result_hash:
        raise ValueError("Authoritative science-completion result is incomplete")
    payload = result.as_dict()
    claimed = payload.pop("completion_identity")
    if claimed != science_completion_hash(payload):
        raise ValueError("Authoritative science-completion identity mismatch")


def validate_run_output(plan_run: Mapping[str, Any], output_root: Path, result: SimulationAdapterResult) -> list[dict[str, Any]]:
    actual = _validate_adapter_result(plan_run, output_root, result)
    if result.artifact_contract == SYNTHETIC_ARTIFACT_CONTRACT:
        _validate_synthetic(plan_run, output_root, actual)
    elif result.artifact_contract == PRODUCTION_ARTIFACT_CONTRACT:
        _validate_production(plan_run, output_root, actual)
    else:
        raise ValueError("Unsupported scientific artifact contract")
    return actual
