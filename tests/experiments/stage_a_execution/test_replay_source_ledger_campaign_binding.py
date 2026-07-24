from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from satnet.experiments.stage_a_execution.authorization import load_authorization, validate_authorization
from satnet.experiments.stage_a_execution.common import payload_hash, read_json_object
from satnet.experiments.stage_a_execution.contract import (
    FROZEN_COMMIT,
    FROZEN_CONTRACT_HASH,
    FROZEN_TAG,
    FrozenStageAContract,
    _INTEGER_DESIGN_FIELDS,
    _INTEGER_RUN_FIELDS,
    _INTEGER_SEED_FIELDS,
    _csv,
)
from satnet.experiments.stage_a_execution.integrity import verify_artifact_inventory
from satnet.experiments.stage_a_execution.ledger import read_bound_ledger
from satnet.experiments.stage_a_execution.plan import (
    OPERATION_BOUND_CAMPAIGN_MANIFEST_ALGORITHM,
    OPERATION_BOUND_CAMPAIGN_MANIFEST_VERSION,
    PLAN_HASH_DOMAIN,
    build_plan,
    validate_legacy_campaign_manifest_identity,
    validate_plan,
)
from satnet.experiments.stage_a_execution.preflight import load_source_generation_provenance
from satnet.experiments.stage_a_execution.resume import ledger_provenance_from_mapping, validate_ledger_binding

ROOT = Path(__file__).parents[3]
GENERATION_ROOT = Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-production")
REPLAY_ROOT = Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-replay")
ACCEPTANCE_ROOT = Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-acceptance")
GENERATION_LEDGER = GENERATION_ROOT / "execution_ledger.json"
PROVENANCE_PATH = ROOT / "artifacts/stage_a_replay_source_ledger_binding_v1_evidence/source_generation_provenance.json"
GENERATION_AUTHORIZATION_PATH = ROOT / "artifacts/stage_a_development_authorization_v1_active/stage_a_development_execution_authorization.json"
REPLAY_AUTHORIZATION_PATH = ROOT / "artifacts/stage_a_development_replay_authorization_v2_active/stage_a_development_replay_execution_authorization.json"
SOURCE_CAMPAIGN_HASH = "af76d64bbabdb055d413a1c3e2f28809749710d67f7819465d45c0ef0769e164"
SOURCE_PLAN_HASH = "cd32f773fe9ce2853b3a0bae699bf4a5199cfe67dc5baae767858387b0a36d2e"
SOURCE_LEDGER_SHA256 = "9df1448612d4719d164cf014a2ab60d932d82ab07a90e05c50a0c1650b097328"


def _contract() -> FrozenStageAContract:
    source = ROOT / "artifacts/stage_a_discovery_contract_v1/source_bundle"
    partitions = read_json_object(source / "stage_a_partition_manifest.json")["partitions"]
    roots = read_json_object(source / "stage_a_output_root_manifest.json")["proposed_resolved_paths"]
    return FrozenStageAContract(
        repo_root=ROOT,
        contract_root=ROOT / "artifacts/stage_a_discovery_contract_v1",
        specification={},
        declaration={},
        designs=tuple(_csv(source / "stage_a_design_manifest.csv", _INTEGER_DESIGN_FIELDS)),
        runs=tuple(_csv(source / "stage_a_run_manifest.csv", _INTEGER_RUN_FIELDS)),
        seeds=tuple(_csv(source / "stage_a_seed_manifest.csv", _INTEGER_SEED_FIELDS)),
        partitions=partitions,
        output_roots=roots,
        contract_identity=FROZEN_CONTRACT_HASH,
        frozen_tag=FROZEN_TAG,
        frozen_commit=FROZEN_COMMIT,
    )


def _plan(operation: str) -> tuple[Any, dict[str, Any]]:
    path = GENERATION_AUTHORIZATION_PATH if operation == "GENERATE" else REPLAY_AUTHORIZATION_PATH
    authorization = load_authorization(path)
    document = authorization.document
    plan = build_plan(
        _contract(),
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


def _source() -> tuple[dict[str, Any], dict[str, Any], Any]:
    _, replay_plan = _plan("REPLAY")
    provenance = load_source_generation_provenance(ROOT, contract=_contract(), plan=replay_plan)
    ledger, identity = read_bound_ledger(
        GENERATION_ROOT,
        relative_path=replay_plan["source_generation_ledger_relative_path"],
        byte_length=replay_plan["source_generation_ledger_byte_length"],
        sha256=replay_plan["source_generation_ledger_sha256"],
    )
    return replay_plan, ledger, provenance


def test_replay_source_binding_accepts_distinct_historical_and_current_campaigns() -> None:
    replay_plan, ledger, provenance = _source()
    validate_plan(replay_plan)
    validate_ledger_binding(ledger, replay_plan, operation="GENERATE", provenance=provenance)
    assert provenance.campaign_manifest_hash == SOURCE_CAMPAIGN_HASH
    assert replay_plan["campaign_manifest_version"] == OPERATION_BOUND_CAMPAIGN_MANIFEST_VERSION
    assert replay_plan["campaign_manifest_algorithm"] == OPERATION_BOUND_CAMPAIGN_MANIFEST_ALGORITHM
    assert replay_plan["campaign_manifest_operation"] == "REPLAY"
    assert provenance.campaign_manifest_hash != replay_plan["campaign_manifest_hash"]
    assert provenance.plan_hash == SOURCE_PLAN_HASH


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("campaign_manifest_hash", "0" * 64),
        ("stable_executable_commit", "0" * 40),
        ("executable_inventory_hash", "0" * 64),
    ],
)
def test_replay_rejects_wrong_source_generation_identity(field: str, wrong: str) -> None:
    replay_plan, ledger, provenance = _source()
    mapping = provenance.as_dict()
    mapping[field] = wrong
    wrong_provenance = ledger_provenance_from_mapping(mapping)
    with pytest.raises(ValueError, match=f"Ledger identity mismatch: {field}"):
        validate_ledger_binding(ledger, replay_plan, operation="GENERATE", provenance=wrong_provenance)


def test_replay_rejects_wrong_current_campaign_manifest_hash() -> None:
    _, replay_plan = _plan("REPLAY")
    replay_plan["campaign_manifest_hash"] = "0" * 64
    payload = {key: value for key, value in replay_plan.items() if key != "plan_hash"}
    replay_plan["plan_hash"] = payload_hash(payload, domain=PLAN_HASH_DOMAIN)
    with pytest.raises(ValueError, match="Campaign manifest hash mismatch"):
        validate_plan(replay_plan)


@pytest.mark.parametrize("field", ["authorized_stable_executable_commit", "authorized_executable_inventory_hash"])
def test_replay_rejects_wrong_current_executable_identity(field: str) -> None:
    authorization, replay_plan = _plan("REPLAY")
    arguments = {
        "stable_executable_commit": replay_plan["stable_executable_commit"],
        "executable_inventory_hash": replay_plan["executable_inventory_hash"],
    }
    arguments["stable_executable_commit" if field.endswith("commit") else "executable_inventory_hash"] = "0" * (40 if field.endswith("commit") else 64)
    with pytest.raises(PermissionError, match=f"Authorization identity mismatch: {field}"):
        validate_authorization(
            authorization,
            contract=_contract(),
            tooling_proposal_hash=replay_plan["tooling_proposal_hash"],
            artifact_contract_hash=replay_plan["artifact_contract_hash"],
            operation="REPLAY",
            partition="development",
            run_ids=[row["global_run_id"] for row in replay_plan["runs"]],
            generation_root=GENERATION_ROOT,
            replay_root=REPLAY_ROOT,
            acceptance_root=ACCEPTANCE_ROOT,
            **arguments,
        )


@pytest.mark.parametrize(
    ("relative_path", "byte_length", "sha256", "message"),
    [
        ("replay_ledger.json", 1_060_132, SOURCE_LEDGER_SHA256, "cannot find the file"),
        ("execution_ledger.json", 1_060_131, SOURCE_LEDGER_SHA256, "byte length"),
        ("execution_ledger.json", 1_060_132, "0" * 64, "SHA-256"),
    ],
)
def test_source_ledger_path_length_and_sha256_remain_fail_closed(
    relative_path: str,
    byte_length: int,
    sha256: str,
    message: str,
) -> None:
    with pytest.raises((FileNotFoundError, ValueError), match=message):
        read_bound_ledger(
            GENERATION_ROOT,
            relative_path=relative_path,
            byte_length=byte_length,
            sha256=sha256,
        )


@pytest.mark.parametrize("mutation", ["membership", "seed"])
def test_source_membership_and_seeds_remain_fail_closed(mutation: str) -> None:
    replay_plan, ledger, provenance = _source()
    changed = deepcopy(replay_plan)
    if mutation == "membership":
        changed["runs"] = changed["runs"][:-1]
    else:
        changed["runs"][0]["satellite_failure_seed"] += 1
    with pytest.raises(ValueError, match="run-record or seed identity"):
        validate_ledger_binding(ledger, changed, operation="GENERATE", provenance=provenance)


def test_source_artifact_inventory_remains_fail_closed() -> None:
    _, ledger, _ = _source()
    record = ledger["records"][0]
    output = GENERATION_ROOT / record["output_relative_path"]
    verify_artifact_inventory(output, record["artifacts"])
    changed = deepcopy(record["artifacts"])
    changed[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact inventory changed"):
        verify_artifact_inventory(output, changed)


def test_generate_binding_behavior_remains_unchanged() -> None:
    authorization, generation_plan = _plan("GENERATE")
    validate_plan(generation_plan)
    ledger, _ = read_bound_ledger(
        GENERATION_ROOT,
        relative_path="execution_ledger.json",
        byte_length=1_060_132,
        sha256=SOURCE_LEDGER_SHA256,
    )
    provenance = load_source_generation_provenance(
        ROOT,
        contract=_contract(),
        plan=generation_plan,
    )
    validate_legacy_campaign_manifest_identity(ledger, generation_plan, operation="GENERATE")
    validate_ledger_binding(
        ledger,
        generation_plan,
        operation="GENERATE",
        provenance=provenance,
    )
    assert ledger["campaign_manifest_hash"] == SOURCE_CAMPAIGN_HASH
    assert ledger["plan_hash"] == SOURCE_PLAN_HASH
    assert generation_plan["campaign_manifest_hash"] != SOURCE_CAMPAIGN_HASH
    assert ledger["authorization_hash"] == authorization.sha256


def test_binding_validation_creates_no_roots_and_executes_no_simulation() -> None:
    before = (GENERATION_LEDGER.read_bytes(), REPLAY_ROOT.exists(), ACCEPTANCE_ROOT.exists())
    replay_plan, ledger, provenance = _source()
    validate_ledger_binding(ledger, replay_plan, operation="GENERATE", provenance=provenance)
    after = (GENERATION_LEDGER.read_bytes(), REPLAY_ROOT.exists(), ACCEPTANCE_ROOT.exists())
    assert after == before
    assert REPLAY_ROOT.exists()
    assert not ACCEPTANCE_ROOT.exists()


def test_source_provenance_document_and_authorizations_are_exact() -> None:
    document = json.loads(PROVENANCE_PATH.read_bytes())
    generation = load_authorization(GENERATION_AUTHORIZATION_PATH)
    replay = load_authorization(REPLAY_AUTHORIZATION_PATH)
    assert document["source_generation_ledger_sha256"] == SOURCE_LEDGER_SHA256
    assert document["ledger_provenance"]["authorization_hash"] == generation.sha256
    assert document["ledger_provenance"]["stable_executable_commit"] == "fafe3fe36eac4429c860bd6d281923fed2980ea7"
    assert document["ledger_provenance"]["executable_inventory_hash"] == "1066d17645d4975a40a93a80c1ca9797b1b28cf37531337355ccb3297fd13b2d"
    assert replay.document["authorized_stable_executable_commit"] == "957c565954443f814ae324d3cf29412dcee72319"
    assert replay.document["authorized_executable_inventory_hash"] == "9f62a9c94feaaea9ce7cf315864d99a69c5127d8323e2a77c6881e0b36b1f596"
