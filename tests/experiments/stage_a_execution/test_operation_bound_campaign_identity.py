from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from satnet.experiments.stage_a_execution.authorization import (
    Authorization,
    authorization_digest,
)
from satnet.experiments.stage_a_execution.common import (
    payload_hash,
    read_json_object,
    sha256_file,
)
from satnet.experiments.stage_a_execution.contract import load_frozen_contract
from satnet.experiments.stage_a_execution.identity import (
    STABLE_IDENTITY_RELATIVE,
    TOOLING_INVENTORY_RELATIVE,
)
from satnet.experiments.stage_a_execution.ledger import read_bound_ledger
from satnet.experiments.stage_a_execution.plan import (
    LEGACY_CAMPAIGN_MANIFEST_ALGORITHM,
    LEGACY_CAMPAIGN_MANIFEST_VERSION,
    OPERATION_BOUND_CAMPAIGN_MANIFEST_ALGORITHM,
    OPERATION_BOUND_CAMPAIGN_MANIFEST_VERSION,
    PLAN_HASH_DOMAIN,
    build_plan,
    campaign_manifest_hash,
    validate_legacy_campaign_manifest_identity,
    validate_plan,
)
from satnet.experiments.stage_a_execution.preflight import (
    load_source_generation_provenance,
    load_source_replay_provenance,
    run_preflight,
)
from satnet.experiments.stage_a_execution.resume import validate_ledger_binding

ROOT = Path(__file__).parents[3]
GENERATION_ROOT = Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-production")
REPLAY_ROOT = Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-replay")
ACCEPTANCE_ROOT = Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-acceptance")
GENERATION_LEDGER_SHA256 = "9df1448612d4719d164cf014a2ab60d932d82ab07a90e05c50a0c1650b097328"
REPLAY_LEDGER_SHA256 = "140bda39b3f519d7baeae09d222cd3f6c80791ff03215c11c48dd66fb6e8b9bb"
GENERATION_CAMPAIGN_HASH = "af76d64bbabdb055d413a1c3e2f28809749710d67f7819465d45c0ef0769e164"
REPLAY_CAMPAIGN_HASH = "3a8a62a5e73c81503e5b75d0c44a820eb545596968d837aa72e7e1719a7e3a50"
ACCEPTANCE_PROPOSAL = ROOT / "artifacts/stage_a_development_acceptance_authorization_v1_proposal/stage_a_development_acceptance_authorization_proposal.json"


def _current_authorization(contract: Any, operation: str) -> Authorization:
    stable = read_json_object(ROOT / STABLE_IDENTITY_RELATIVE)
    proposal = read_json_object(ACCEPTANCE_PROPOSAL)
    source_generation = operation in {"REPLAY", "ACCEPT"}
    source_replay = operation == "ACCEPT"
    document: dict[str, Any] = {
        "schema_identifier": "satnet.stage_a.execution_authorization.v1",
        "authorization_version": "1",
        "authorization_id": f"TEST-ONLY-OPERATION-BOUND-{operation}",
        "authorization_status": "AUTHORIZED",
        "authorized_contract_hash": contract.contract_hash,
        "authorized_contract_tag": contract.frozen_tag,
        "authorized_frozen_commit": contract.frozen_commit,
        "authorized_stable_executable_commit": stable["stable_executable_commit"],
        "authorized_executable_inventory_hash": stable["executable_inventory_sha256"],
        "authorized_tooling_proposal_hash": sha256_file(ROOT / TOOLING_INVENTORY_RELATIVE),
        "authorized_artifact_contract_hash": proposal["artifact_contract_sha256"],
        "authorized_partition": "development",
        "authorized_run_ids": list(contract.partitions["development"]["global_run_ids"]),
        "authorized_run_count": contract.partitions["development"]["run_count"],
        "authorized_operation": operation,
        "authorized_generation_root": str(GENERATION_ROOT.resolve()),
        "authorized_replay_root": str(REPLAY_ROOT.resolve()),
        "authorized_acceptance_root": str(ACCEPTANCE_ROOT.resolve()),
        "source_generation_ledger_relative_path": "execution_ledger.json" if source_generation else None,
        "source_generation_ledger_byte_length": 1_060_132 if source_generation else None,
        "source_generation_ledger_sha256": GENERATION_LEDGER_SHA256 if source_generation else None,
        "source_replay_ledger_relative_path": "replay_ledger.json" if source_replay else None,
        "source_replay_ledger_byte_length": 1_079_014 if source_replay else None,
        "source_replay_ledger_sha256": REPLAY_LEDGER_SHA256 if source_replay else None,
        "authorization_date": "2099-01-01",
        "authorizing_decision_reference": "TEST-ONLY-NONPERSISTED-PREFLIGHT",
        "independently_approved": True,
    }
    document["authorization_sha256"] = authorization_digest(document)
    return Authorization(document=document, sha256=document["authorization_sha256"])


def _current_plan(contract: Any, operation: str) -> tuple[Authorization, dict[str, Any]]:
    authorization = _current_authorization(contract, operation)
    document = authorization.document
    plan = build_plan(
        contract,
        partition="development",
        operation=operation,
        stable_executable_commit=document["authorized_stable_executable_commit"],
        executable_inventory_hash=document["authorized_executable_inventory_hash"],
        tooling_proposal_hash=document["authorized_tooling_proposal_hash"],
        artifact_contract_hash=document["authorized_artifact_contract_hash"],
        generation_root=GENERATION_ROOT,
        replay_root=REPLAY_ROOT,
        acceptance_root=ACCEPTANCE_ROOT,
        authorization=authorization,
    )
    return authorization, plan


def _ledgers(plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    generation, _ = read_bound_ledger(
        GENERATION_ROOT,
        relative_path="execution_ledger.json",
        byte_length=1_060_132,
        sha256=GENERATION_LEDGER_SHA256,
    )
    replay, _ = read_bound_ledger(
        REPLAY_ROOT,
        relative_path="replay_ledger.json",
        byte_length=1_079_014,
        sha256=REPLAY_LEDGER_SHA256,
    )
    return generation, replay


def test_operation_bound_campaign_identity_is_distinct_and_deterministic() -> None:
    contract = load_frozen_contract(ROOT, operation="ACCEPT")
    plans = {operation: _current_plan(contract, operation)[1] for operation in ("GENERATE", "REPLAY", "ACCEPT")}
    repeated = _current_plan(contract, "ACCEPT")[1]

    assert len({plan["campaign_manifest_hash"] for plan in plans.values()}) == 3
    assert repeated["campaign_manifest_hash"] == plans["ACCEPT"]["campaign_manifest_hash"]
    assert repeated == plans["ACCEPT"]
    assert all(plan["campaign_manifest_version"] == OPERATION_BOUND_CAMPAIGN_MANIFEST_VERSION for plan in plans.values())
    assert all(plan["campaign_manifest_algorithm"] == OPERATION_BOUND_CAMPAIGN_MANIFEST_ALGORITHM for plan in plans.values())
    assert all(plan["campaign_manifest_operation"] == operation for operation, plan in plans.items())
    assert plans["ACCEPT"]["campaign_manifest_hash"] != REPLAY_CAMPAIGN_HASH


def test_completed_legacy_campaign_identities_reconstruct_exactly() -> None:
    contract = load_frozen_contract(ROOT, operation="ACCEPT")
    _, plan = _current_plan(contract, "ACCEPT")
    generation, replay = _ledgers(plan)

    validate_legacy_campaign_manifest_identity(generation, plan, operation="GENERATE")
    validate_legacy_campaign_manifest_identity(replay, plan, operation="REPLAY")

    assert generation["campaign_manifest_hash"] == GENERATION_CAMPAIGN_HASH
    assert replay["campaign_manifest_hash"] == REPLAY_CAMPAIGN_HASH
    assert generation["stable_executable_commit"] == "fafe3fe36eac4429c860bd6d281923fed2980ea7"
    assert generation["executable_inventory_hash"] == "1066d17645d4975a40a93a80c1ca9797b1b28cf37531337355ccb3297fd13b2d"
    assert replay["stable_executable_commit"] == "8bde92da762269998632eb6c3e3cb2565a6ca971"
    assert replay["executable_inventory_hash"] == "d52ffc2afa5f94268d89187afe58d4c7695bdf261ec20113644c9cb7f5840215"


def test_historical_generation_and_replay_provenance_remains_fail_closed() -> None:
    contract = load_frozen_contract(ROOT, operation="ACCEPT")
    _, plan = _current_plan(contract, "ACCEPT")
    generation, replay = _ledgers(plan)
    generation_provenance = load_source_generation_provenance(ROOT, contract=contract, plan=plan)
    replay_provenance = load_source_replay_provenance(ROOT, contract=contract, plan=plan, ledger=replay)

    validate_ledger_binding(generation, plan, operation="GENERATE", provenance=generation_provenance)
    validate_ledger_binding(replay, plan, operation="REPLAY", provenance=replay_provenance)

    changed_generation = deepcopy(generation)
    changed_generation["campaign_manifest_hash"] = "0" * 64
    with pytest.raises(ValueError, match="campaign_manifest_hash"):
        validate_ledger_binding(changed_generation, plan, operation="GENERATE", provenance=generation_provenance)

    changed_replay = deepcopy(replay)
    changed_replay["stable_executable_commit"] = "0" * 40
    with pytest.raises(ValueError, match="stable_executable_commit"):
        validate_ledger_binding(changed_replay, plan, operation="REPLAY", provenance=replay_provenance)


@pytest.mark.parametrize(
    ("field", "wrong", "message"),
    [
        ("campaign_manifest_version", LEGACY_CAMPAIGN_MANIFEST_VERSION, "version"),
        ("campaign_manifest_algorithm", LEGACY_CAMPAIGN_MANIFEST_ALGORITHM, "algorithm"),
        ("campaign_manifest_operation", "REPLAY", "operation"),
        ("campaign_manifest_hash", "0" * 64, "hash"),
    ],
)
def test_current_campaign_identity_rejects_incorrect_metadata(
    field: str,
    wrong: str,
    message: str,
) -> None:
    contract = load_frozen_contract(ROOT, operation="ACCEPT")
    _, plan = _current_plan(contract, "ACCEPT")
    changed = deepcopy(plan)
    changed[field] = wrong
    changed["plan_hash"] = payload_hash(
        {key: value for key, value in changed.items() if key != "plan_hash"},
        domain=PLAN_HASH_DOMAIN,
    )

    with pytest.raises(ValueError, match=message):
        validate_plan(changed)


def test_campaign_hash_rejects_crossed_version_and_algorithm() -> None:
    contract = load_frozen_contract(ROOT, operation="ACCEPT")
    _, plan = _current_plan(contract, "ACCEPT")

    with pytest.raises(ValueError, match="Legacy campaign manifest algorithm"):
        campaign_manifest_hash(
            plan,
            version=LEGACY_CAMPAIGN_MANIFEST_VERSION,
            algorithm=OPERATION_BOUND_CAMPAIGN_MANIFEST_ALGORITHM,
            operation="ACCEPT",
        )
    with pytest.raises(ValueError, match="Operation-bound campaign manifest algorithm"):
        campaign_manifest_hash(
            plan,
            version=OPERATION_BOUND_CAMPAIGN_MANIFEST_VERSION,
            algorithm=LEGACY_CAMPAIGN_MANIFEST_ALGORITHM,
            operation="ACCEPT",
        )
    with pytest.raises(ValueError, match="operation"):
        campaign_manifest_hash(
            plan,
            version=OPERATION_BOUND_CAMPAIGN_MANIFEST_VERSION,
            algorithm=OPERATION_BOUND_CAMPAIGN_MANIFEST_ALGORITHM,
            operation="INVALID",
        )


def test_real_accept_preflight_validates_both_historical_campaigns_without_writes() -> None:
    contract = load_frozen_contract(ROOT, operation="ACCEPT")
    authorization, plan = _current_plan(contract, "ACCEPT")
    generation_ledger = GENERATION_ROOT / "execution_ledger.json"
    replay_ledger = REPLAY_ROOT / "replay_ledger.json"
    before = (
        generation_ledger.stat().st_size,
        sha256_file(generation_ledger),
        replay_ledger.stat().st_size,
        sha256_file(replay_ledger),
        ACCEPTANCE_ROOT.exists(),
    )

    certificate = run_preflight(
        repo_root=ROOT,
        contract=contract,
        plan=plan,
        authorization=authorization,
        generation_root=GENERATION_ROOT,
        replay_root=REPLAY_ROOT,
        acceptance_root=ACCEPTANCE_ROOT,
        check_write_probe=False,
        minimum_free_bytes=0,
    )

    after = (
        generation_ledger.stat().st_size,
        sha256_file(generation_ledger),
        replay_ledger.stat().st_size,
        sha256_file(replay_ledger),
        ACCEPTANCE_ROOT.exists(),
    )
    assert certificate.report["preflight"] == "PASSED"
    assert certificate.report["simulation_executed"] is False
    assert certificate.report["source_provenance"]["generation"]["campaign_manifest_hash"] == GENERATION_CAMPAIGN_HASH
    assert certificate.report["source_provenance"]["replay"]["campaign_manifest_hash"] == REPLAY_CAMPAIGN_HASH
    assert certificate.report["current_operation_campaign_manifest_hash"] == plan["campaign_manifest_hash"]
    assert "comparisons" not in certificate.report
    assert before == after
    assert not ACCEPTANCE_ROOT.exists()
