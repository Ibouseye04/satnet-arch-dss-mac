from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifact_contract import (
    SYNTHETIC_ARTIFACT_CONTRACT,
    ScienceCompletionResult,
    SimulationAdapterResult,
    science_completion_hash,
    synthetic_artifacts,
    write_synthetic_artifacts,
)
from .common import read_json_object, sha256_file
from .contract import FROZEN_CONTRACT_HASH, FrozenStageAContract
from .generate import ScienceCompletionValidator, SimulationAdapter, _execute_generation
from .preflight import PreflightCertificate
from .replay import _execute_replay


def validate_synthetic_science_completion(
    plan_run: Mapping[str, Any], output_root: Path, adapter_result: SimulationAdapterResult,
) -> ScienceCompletionResult:
    if plan_run["contract_hash"] == FROZEN_CONTRACT_HASH:
        raise PermissionError("Synthetic science completion is prohibited for the frozen Stage A contract")
    if adapter_result.artifact_contract != SYNTHETIC_ARTIFACT_CONTRACT:
        raise ValueError("Synthetic science completion requires the synthetic artifact contract")
    observed = read_json_object(output_root / "scientific.json")
    expected = synthetic_artifacts(plan_run)["scientific.json"]
    if observed != expected or observed.get("scientific_validation") != "PASSED":
        raise ValueError("Synthetic independent science completion mismatch")
    stages = ("synthetic_science",)
    result_hash = sha256_file(output_root / "scientific.json")
    payload = {
        "validation_status": "PASSED",
        "validation_kind": "synthetic_test_harness_science_completion_v1",
        "verified_stages": list(stages),
        "authoritative_result_hash": result_hash,
    }
    return ScienceCompletionResult(
        validation_status=payload["validation_status"],
        validation_kind=payload["validation_kind"],
        verified_stages=stages,
        authoritative_result_hash=result_hash,
        completion_identity=science_completion_hash(payload),
    )


def execute_synthetic_generation(
    *, repo_root: Path, contract: FrozenStageAContract, plan: dict[str, Any],
    authorization_hash: str, preflight: PreflightCertificate, campaign_root: Path,
    adapter: SimulationAdapter = write_synthetic_artifacts,
    science_validator: ScienceCompletionValidator = validate_synthetic_science_completion,
    resume: bool = False, retry_failed: bool = False,
) -> dict[str, Any]:
    if contract.contract_hash == FROZEN_CONTRACT_HASH:
        raise PermissionError("Synthetic execution is prohibited for the frozen Stage A contract")
    return _execute_generation(
        repo_root=repo_root, contract=contract, plan=plan, authorization_hash=authorization_hash,
        preflight=preflight, campaign_root=campaign_root, adapter=adapter,
        science_validator=science_validator, resume=resume, retry_failed=retry_failed,
    )


def execute_synthetic_replay(
    *, repo_root: Path, contract: FrozenStageAContract, plan: dict[str, Any],
    authorization_hash: str, preflight: PreflightCertificate, generation_root: Path,
    replay_root: Path, adapter: SimulationAdapter = write_synthetic_artifacts,
    science_validator: ScienceCompletionValidator = validate_synthetic_science_completion,
) -> dict[str, Any]:
    if contract.contract_hash == FROZEN_CONTRACT_HASH:
        raise PermissionError("Synthetic replay is prohibited for the frozen Stage A contract")
    return _execute_replay(
        repo_root=repo_root, plan=plan, authorization_hash=authorization_hash,
        preflight=preflight, generation_root=generation_root, replay_root=replay_root,
        adapter=adapter, science_validator=science_validator,
    )
