from __future__ import annotations

from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from typing import Any

import pytest

from satnet.experiments.stage_a_execution import generate as generate_module
from satnet.experiments.stage_a_execution import ledger as ledger_module
from satnet.experiments.stage_a_execution import locking as locking_module
from satnet.experiments.stage_a_execution import replay as replay_module
from satnet.experiments.stage_a_execution._synthetic_harness import (
    execute_synthetic_generation,
    execute_synthetic_replay,
)
from satnet.experiments.stage_a_execution.acceptance import evaluate_acceptance
from satnet.experiments.stage_a_execution.artifact_contract import (
    PRODUCTION_ARTIFACT_CONTRACT,
    ScienceCompletionResult,
    SimulationAdapterResult,
    bind_plan_run,
    science_completion_hash,
    validate_run_output,
    write_synthetic_artifacts,
)
from satnet.experiments.stage_a_execution.authorization import (
    Authorization,
    authorization_digest,
    validate_authorization,
)
from satnet.experiments.stage_a_execution.cli import build_parser
from satnet.experiments.stage_a_execution.common import (
    canonical_json_bytes,
    payload_hash,
    sha256_bytes,
    sha256_file,
)
from satnet.experiments.stage_a_execution.contract import FrozenStageAContract
from satnet.experiments.stage_a_execution.generate import execute_generation
from satnet.experiments.stage_a_execution.ledger import (
    build_ledger,
    read_bound_ledger,
    read_ledger,
    transition,
    write_ledger,
)
from satnet.experiments.stage_a_execution.locking import (
    LOCK_DOMAIN,
    ExclusiveLock,
    campaign_identity,
    host_identity,
    lock_is_stale,
    lock_payload,
    process_start_identity,
    recover_stale_lock,
    run_identity,
)
from satnet.experiments.stage_a_execution.plan import (
    build_plan,
    validate_plan_contract_binding,
)
import satnet.experiments.stage_a_execution.preflight as preflight_module
from satnet.experiments.stage_a_execution.preflight import run_preflight
from satnet.experiments.stage_a_execution.replay import execute_replay
from satnet.experiments.stage_a_execution.resume import validate_ledger_binding

ROOT = Path(__file__).parents[2]
STABLE_EXECUTABLE_COMMIT = "1" * 40
EXECUTABLE_INVENTORY = "2" * 64
TOOLING_PROPOSAL = "3" * 64
ARTIFACT_CONTRACT = "4" * 64
EXPECTED_EXECUTABLE_INVENTORY = "1066d17645d4975a40a93a80c1ca9797b1b28cf37531337355ccb3297fd13b2d"
EXPECTED_PROPOSAL_INVENTORY = "7b21af1bad6faa10696d1417b28b75300727033446c84424beb94d263510778b"


@pytest.fixture
def synthetic_contract(tmp_path: Path) -> FrozenStageAContract:
    designs = (
        {"design_id": "SYN-D000", "design_index": 0, "region": "synthetic", "partition": "development", "sealed": False, "design_record_hash": "3" * 64},
        {"design_id": "SYN-D001", "design_index": 1, "region": "synthetic", "partition": "validation", "sealed": False, "design_record_hash": "4" * 64},
        {"design_id": "SYN-D002", "design_index": 2, "region": "synthetic", "partition": "sealed_holdout", "sealed": True, "design_record_hash": "5" * 64},
    )
    runs = (
        {"global_run_id": 1, "run_key": "SYN-D000-R00", "design_id": "SYN-D000", "design_index": 0, "realization_id": "R00", "realization_index": 0, "region": "synthetic", "partition": "development", "sealed": False, "design_record_hash": "3" * 64, "run_record_hash": "6" * 64},
        {"global_run_id": 2, "run_key": "SYN-D000-R01", "design_id": "SYN-D000", "design_index": 0, "realization_id": "R01", "realization_index": 1, "region": "synthetic", "partition": "development", "sealed": False, "design_record_hash": "3" * 64, "run_record_hash": "7" * 64},
        {"global_run_id": 3, "run_key": "SYN-D001-R00", "design_id": "SYN-D001", "design_index": 1, "realization_id": "R00", "realization_index": 0, "region": "synthetic", "partition": "validation", "sealed": False, "design_record_hash": "4" * 64, "run_record_hash": "8" * 64},
        {"global_run_id": 4, "run_key": "SYN-D002-R00", "design_id": "SYN-D002", "design_index": 2, "realization_id": "R00", "realization_index": 0, "region": "synthetic", "partition": "sealed_holdout", "sealed": True, "design_record_hash": "5" * 64, "run_record_hash": "9" * 64},
    )
    seeds = tuple(
        {
            "global_run_id": row["global_run_id"],
            "run_key": row["run_key"],
            "design_id": row["design_id"],
            "realization_id": row["realization_id"],
            "design_construction_seed": 1000 + row["design_index"],
            "ground_selection_seed": row["global_run_id"] * 10 + 1,
            "satellite_failure_seed": row["global_run_id"] * 10 + 2,
            "ground_failure_seed": row["global_run_id"] * 10 + 3,
        }
        for row in runs
    )
    return FrozenStageAContract(
        repo_root=tmp_path / "repo",
        contract_root=tmp_path / "contract",
        specification={},
        declaration={},
        designs=designs,
        runs=runs,
        seeds=seeds,
        partitions={
            "development": {"global_run_ids": [1, 2], "design_count": 1, "run_count": 2, "sealed": False},
            "validation": {"global_run_ids": [3], "design_count": 1, "run_count": 1, "sealed": False},
            "sealed_holdout": {"global_run_ids": [4], "design_count": 1, "run_count": 1, "sealed": True},
        },
        output_roots={},
        contract_identity="a" * 64,
        frozen_tag="synthetic-contract-v1",
        frozen_commit="b" * 40,
    )


def _roots(tmp_path: Path, prefix: str = "") -> tuple[Path, Path, Path]:
    return tuple(tmp_path / f"{prefix}{name}" for name in ("generation", "replay", "acceptance"))


def _authorization(
    contract: FrozenStageAContract,
    operation: str,
    roots: tuple[Path, Path, Path],
    generation_identity: tuple[str, int, str] | None = None,
    replay_identity: tuple[str, int, str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> Authorization:
    document: dict[str, Any] = {
        "schema_identifier": "satnet.stage_a.execution_authorization.v1",
        "authorization_version": "1",
        "authorization_id": f"SYNTHETIC-{operation}",
        "authorization_status": "AUTHORIZED",
        "authorized_contract_hash": contract.contract_hash,
        "authorized_contract_tag": contract.frozen_tag,
        "authorized_frozen_commit": contract.frozen_commit,
        "authorized_stable_executable_commit": STABLE_EXECUTABLE_COMMIT,
        "authorized_executable_inventory_hash": EXECUTABLE_INVENTORY,
        "authorized_tooling_proposal_hash": TOOLING_PROPOSAL,
        "authorized_artifact_contract_hash": ARTIFACT_CONTRACT,
        "authorized_partition": "development",
        "authorized_run_ids": [1, 2],
        "authorized_run_count": 2,
        "authorized_operation": operation,
        "authorized_generation_root": str(roots[0].resolve()),
        "authorized_replay_root": str(roots[1].resolve()),
        "authorized_acceptance_root": str(roots[2].resolve()),
        "source_generation_ledger_relative_path": None if generation_identity is None else generation_identity[0],
        "source_generation_ledger_byte_length": None if generation_identity is None else generation_identity[1],
        "source_generation_ledger_sha256": None if generation_identity is None else generation_identity[2],
        "source_replay_ledger_relative_path": None if replay_identity is None else replay_identity[0],
        "source_replay_ledger_byte_length": None if replay_identity is None else replay_identity[1],
        "source_replay_ledger_sha256": None if replay_identity is None else replay_identity[2],
        "authorization_date": "2099-01-01",
        "authorizing_decision_reference": "SYNTHETIC-TEST-ONLY",
        "independently_approved": True,
    }
    if overrides:
        document.update(overrides)
    document["authorization_sha256"] = authorization_digest(document)
    return Authorization(document=document, sha256=authorization_digest(document))


def _plan(
    contract: FrozenStageAContract,
    operation: str,
    roots: tuple[Path, Path, Path],
    authorization: Authorization,
) -> dict[str, Any]:
    return build_plan(
        contract,
        partition="development",
        operation=operation,
        stable_executable_commit=STABLE_EXECUTABLE_COMMIT,
        executable_inventory_hash=EXECUTABLE_INVENTORY,
        tooling_proposal_hash=TOOLING_PROPOSAL,
        artifact_contract_hash=ARTIFACT_CONTRACT,
        generation_root=roots[0],
        replay_root=roots[1],
        acceptance_root=roots[2],
        authorization=authorization,
    )


def _certificate(
    monkeypatch: pytest.MonkeyPatch,
    contract: FrozenStageAContract,
    plan: dict[str, Any],
    roots: tuple[Path, Path, Path],
    authorization: Authorization,
) -> Any:
    monkeypatch.setattr(preflight_module, "verify_executable_identity", lambda _: {
        "stable_executable_commit": STABLE_EXECUTABLE_COMMIT,
        "executable_inventory_sha256": EXECUTABLE_INVENTORY,
    })
    monkeypatch.setattr(preflight_module, "load_frozen_contract", lambda *_: contract)
    monkeypatch.setattr(preflight_module, "verify_frozen_production_evidence", lambda *_: {
        "combined": {"file_count": 9004, "byte_count": 1337549193, "verified_sha256_count": 9004},
    })
    return run_preflight(
        repo_root=ROOT,
        contract=contract,
        plan=plan,
        authorization=authorization,
        generation_root=roots[0],
        replay_root=roots[1],
        acceptance_root=roots[2],
        minimum_free_bytes=0,
    )


def _ledger_identity(path: Path) -> tuple[str, int, str]:
    return path.name, path.stat().st_size, sha256_file(path)


def _validate_authorization(
    authorization: Authorization,
    contract: FrozenStageAContract,
    operation: str,
    roots: tuple[Path, Path, Path],
) -> None:
    validate_authorization(
        authorization,
        contract=contract,
        stable_executable_commit=STABLE_EXECUTABLE_COMMIT,
        executable_inventory_hash=EXECUTABLE_INVENTORY,
        tooling_proposal_hash=TOOLING_PROPOSAL,
        artifact_contract_hash=ARTIFACT_CONTRACT,
        operation=operation,
        partition="development",
        run_ids=[1, 2],
        generation_root=roots[0],
        replay_root=roots[1],
        acceptance_root=roots[2],
    )


def _rehash_lock(value: dict[str, Any]) -> dict[str, Any]:
    payload = {field: item for field, item in value.items() if field != "lock_hash"}
    value["lock_hash"] = payload_hash(payload, domain=LOCK_DOMAIN)
    return value


def _aged_inactive_lock(identity: dict[str, Any], scope: str = "campaign") -> dict[str, Any]:
    value = lock_payload(identity, scope)
    value["process_id"] = 2_147_483_647
    value["process_start_identity"] = "inactive"
    value["creation_unix_ns"] = time.time_ns() - 120_000_000_000
    value["creation_time"] = "2000-01-01T00:00:00+00:00"
    return _rehash_lock(value)


def test_binding_003_authoritative_scientific_completion_is_non_bypassable(
    synthetic_contract: FrozenStageAContract,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for entry_point in (execute_generation, execute_replay):
        parameters = inspect.signature(entry_point).parameters
        assert "adapter" not in parameters
        assert "science_validator" not in parameters
        with pytest.raises(TypeError):
            entry_point(adapter=object())
    parser = build_parser()
    command = [
        "generate",
        "--partition", "development",
        "--generation-root", str(tmp_path / "g"),
        "--replay-root", str(tmp_path / "r"),
        "--acceptance-root", str(tmp_path / "a"),
        "--authorization", str(tmp_path / "authorization.json"),
    ]
    for option in ("--adapter", "--science-validator", "--validator"):
        with pytest.raises(SystemExit):
            parser.parse_args([*command, option, "injected"])
    for name in ("SATNET_STAGE_A_ADAPTER", "SATNET_STAGE_A_VALIDATOR", "SATNET_EXECUTION_BYPASS"):
        monkeypatch.setenv(name, "injected")
    for module in (generate_module, replay_module):
        source = inspect.getsource(module)
        assert "os.getenv" not in source
        assert "os.environ" not in source
    roots = _roots(tmp_path, "science-")
    authorization = _authorization(synthetic_contract, "GENERATE", roots)
    plan = _plan(synthetic_contract, "GENERATE", roots, authorization)
    bound = bind_plan_run(plan, plan["runs"][0])
    valid_root = tmp_path / "valid-looking"
    valid_root.mkdir()
    adapter_result = write_synthetic_artifacts(bound, valid_root)
    (valid_root / "scientific.json").unlink()
    with pytest.raises(ValueError, match="path set|artifact"):
        validate_run_output(bound, valid_root, adapter_result)
    certificate = _certificate(monkeypatch, synthetic_contract, plan, roots, authorization)
    payload = {
        "validation_status": "FAILED",
        "validation_kind": "independent_test",
        "verified_stages": ["satellite"],
        "authoritative_result_hash": "8" * 64,
    }
    incomplete = ScienceCompletionResult(
        validation_status="FAILED",
        validation_kind="independent_test",
        verified_stages=("satellite",),
        authoritative_result_hash="8" * 64,
        completion_identity=science_completion_hash(payload),
    )
    with pytest.raises(ValueError, match="science-completion"):
        execute_synthetic_generation(
            repo_root=ROOT,
            contract=synthetic_contract,
            plan=plan,
            authorization_hash=authorization.sha256,
            preflight=certificate,
            campaign_root=roots[0],
            adapter=write_synthetic_artifacts,
            science_validator=lambda *_: incomplete,
        )
    assert read_ledger(roots[0] / "execution_ledger.json")["records"][0]["state"] == "FAILED"
    authoritative = {
        "stages": [
            {"stage": stage, "state": "matched"}
            for stage in ("satellite", "g1", "g2", "g3", "g4", "g5", "target", "inventory", "result")
        ],
        "result": {"run_result_hash": "9" * 64},
    }
    monkeypatch.setattr(generate_module, "_build_production_mapping", lambda *_: SimpleNamespace(run_id=7))
    monkeypatch.setattr(generate_module, "read_json_object", lambda *_: {"validated_pipeline_result_hash": "9" * 64})
    monkeypatch.setattr(generate_module, "validate_run_authoritatively", lambda **_: authoritative)
    validator = generate_module._make_production_science_validator(synthetic_contract, object())
    production_result = SimulationAdapterResult(
        run_key="run",
        design_id="design",
        global_run_id=1,
        realization_id="R00",
        design_construction_seed=1,
        ground_selection_seed=2,
        satellite_failure_seed=3,
        ground_failure_seed=4,
        artifact_contract=PRODUCTION_ARTIFACT_CONTRACT,
        artifact_manifest=(),
        artifact_inventory_hash="7" * 64,
        simulation_return_status="SUCCEEDED",
        validation_status="PASSED",
        result_identity="8" * 64,
    )
    assert validator({}, tmp_path, production_result).validation_status == "PASSED"
    for omitted in ("satellite", "g1", "g3", "g5"):
        full = authoritative["stages"]
        authoritative["stages"] = [record for record in full if record["stage"] != omitted]
        with pytest.raises(ValueError, match="G1-G5 completion"):
            validator({}, tmp_path, production_result)
        authoritative["stages"] = full
    authoritative["stages"][3] = {"stage": "g3", "state": "mismatched"}
    with pytest.raises(ValueError, match="G1-G5 completion"):
        validator({}, tmp_path, production_result)
    authoritative["stages"][3] = {"stage": "g3", "state": "matched"}
    authoritative["result"]["run_result_hash"] = "0" * 64
    with pytest.raises(ValueError, match="inconsistent"):
        validator({}, tmp_path, production_result)


def test_binding_004_replay_binds_exact_generation_ledger_raw_bytes(
    synthetic_contract: FrozenStageAContract,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _roots(tmp_path)
    authorization = _authorization(synthetic_contract, "GENERATE", roots)
    plan = _plan(synthetic_contract, "GENERATE", roots, authorization)
    roots[0].mkdir()
    ledger_path = roots[0] / "execution_ledger.json"
    write_ledger(ledger_path, build_ledger(plan, authorization.sha256), overwrite=False)
    original = ledger_path.read_bytes()
    original_identity = (len(original), sha256_bytes(original))
    observed, identity = read_bound_ledger(
        roots[0],
        relative_path="execution_ledger.json",
        byte_length=original_identity[0],
        sha256=original_identity[1],
    )
    assert observed["plan_hash"] == plan["plan_hash"]
    assert identity == {
        "relative_path": "execution_ledger.json",
        "byte_length": original_identity[0],
        "sha256": original_identity[1],
    }
    parsed = json.loads(original)
    reversed_top_level = {key: parsed[key] for key in reversed(tuple(parsed))}
    semantic_mutations = (
        original + b" ",
        original.rstrip(b"\n"),
        json.dumps(reversed_top_level, separators=(",", ":")).encode("utf-8") + b"\n",
        original.replace(b"\n", b"\r\n"),
    )
    for mutated in semantic_mutations:
        assert json.loads(mutated) == parsed
        ledger_path.write_bytes(mutated)
        with pytest.raises(ValueError, match="byte length|SHA-256"):
            read_bound_ledger(
                roots[0],
                relative_path="execution_ledger.json",
                byte_length=original_identity[0],
                sha256=original_identity[1],
            )
    one_byte = bytearray(original)
    one_byte[-2] = ord(" ") if one_byte[-2] != ord(" ") else ord("x")
    ledger_path.write_bytes(bytes(one_byte))
    parse_called = False

    def forbidden_parse(_: bytes) -> dict[str, Any]:
        nonlocal parse_called
        parse_called = True
        raise AssertionError("ledger parsing occurred before byte verification")

    monkeypatch.setattr(ledger_module, "parse_ledger_bytes", forbidden_parse)
    with pytest.raises(ValueError, match="SHA-256"):
        read_bound_ledger(
            roots[0],
            relative_path="execution_ledger.json",
            byte_length=len(one_byte),
            sha256=original_identity[1],
        )
    assert parse_called is False
    ledger_path.write_bytes(original)
    copied = roots[0] / "copied_ledger.json"
    copied.write_bytes(original)
    with pytest.raises(ValueError, match="canonical"):
        read_bound_ledger(
            roots[0],
            relative_path="copied_ledger.json",
            byte_length=len(original),
            sha256=original_identity[1],
        )
    alternate = roots[0] / "replay_ledger.json"
    alternate.write_bytes(original)
    wrong_path_authorization = _authorization(
        synthetic_contract,
        "REPLAY",
        roots,
        ("replay_ledger.json", len(original), original_identity[1]),
    )
    with pytest.raises(PermissionError, match="ledger path"):
        _validate_authorization(wrong_path_authorization, synthetic_contract, "REPLAY", roots)


def test_binding_005_acceptance_cross_binds_ledgers_and_all_four_seeds(
    synthetic_contract: FrozenStageAContract,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _roots(tmp_path)
    generation_authorization = _authorization(synthetic_contract, "GENERATE", roots)
    generation_plan = _plan(synthetic_contract, "GENERATE", roots, generation_authorization)
    generation_certificate = _certificate(
        monkeypatch, synthetic_contract, generation_plan, roots, generation_authorization,
    )
    execute_synthetic_generation(
        repo_root=ROOT,
        contract=synthetic_contract,
        plan=generation_plan,
        authorization_hash=generation_authorization.sha256,
        preflight=generation_certificate,
        campaign_root=roots[0],
    )
    generation_path = roots[0] / "execution_ledger.json"
    generation_bytes = generation_path.read_bytes()
    generation_identity = _ledger_identity(generation_path)
    replay_authorization = _authorization(
        synthetic_contract, "REPLAY", roots, generation_identity=generation_identity,
    )
    replay_plan = _plan(synthetic_contract, "REPLAY", roots, replay_authorization)
    replay_certificate = _certificate(
        monkeypatch, synthetic_contract, replay_plan, roots, replay_authorization,
    )
    execute_synthetic_replay(
        repo_root=ROOT,
        contract=synthetic_contract,
        plan=replay_plan,
        authorization_hash=replay_authorization.sha256,
        preflight=replay_certificate,
        generation_root=roots[0],
        replay_root=roots[1],
    )
    replay_path = roots[1] / "replay_ledger.json"
    replay_bytes = replay_path.read_bytes()
    replay_identity_value = _ledger_identity(replay_path)
    acceptance_authorization = _authorization(
        synthetic_contract,
        "ACCEPT",
        roots,
        generation_identity=generation_identity,
        replay_identity=replay_identity_value,
    )
    acceptance_plan = _plan(synthetic_contract, "ACCEPT", roots, acceptance_authorization)
    acceptance_certificate = _certificate(
        monkeypatch, synthetic_contract, acceptance_plan, roots, acceptance_authorization,
    )
    report = evaluate_acceptance(
        repo_root=ROOT,
        plan=acceptance_plan,
        authorization_hash=acceptance_authorization.sha256,
        preflight=acceptance_certificate,
        generation_root=roots[0],
        replay_root=roots[1],
        acceptance_root=roots[2],
    )
    assert report["acceptance_state"] == "PASSED"
    assert report["generation_ledger_sha256"] == generation_identity[2]
    assert report["replay_ledger_sha256"] == replay_identity_value[2]
    assert report["replay_recorded_source_generation_ledger_sha256"] == generation_identity[2]
    seed_fields = (
        "design_construction_seed",
        "ground_selection_seed",
        "satellite_failure_seed",
        "ground_failure_seed",
    )
    assert all(set(seed_fields).issubset(comparison) for comparison in report["comparisons"])
    for field in seed_fields:
        changed_plan = deepcopy(generation_plan)
        changed_plan["runs"][0][field] += 1
        with pytest.raises(ValueError, match="seed identity"):
            validate_plan_contract_binding(changed_plan, synthetic_contract)
    generation_path.write_bytes(generation_bytes + b" ")
    with pytest.raises(ValueError, match="byte length|SHA-256"):
        evaluate_acceptance(
            repo_root=ROOT,
            plan=acceptance_plan,
            authorization_hash=acceptance_authorization.sha256,
            preflight=acceptance_certificate,
            generation_root=roots[0],
            replay_root=roots[1],
            acceptance_root=roots[2],
        )
    generation_path.write_bytes(generation_bytes)
    replay_path.write_bytes(replay_bytes + b" ")
    with pytest.raises(ValueError, match="byte length|SHA-256"):
        evaluate_acceptance(
            repo_root=ROOT,
            plan=acceptance_plan,
            authorization_hash=acceptance_authorization.sha256,
            preflight=acceptance_certificate,
            generation_root=roots[0],
            replay_root=roots[1],
            acceptance_root=roots[2],
        )
    replay_path.write_bytes(replay_bytes)
    wrong_generation_path = _authorization(
        synthetic_contract,
        "ACCEPT",
        roots,
        generation_identity=("replay_ledger.json", generation_identity[1], generation_identity[2]),
        replay_identity=replay_identity_value,
    )
    with pytest.raises(PermissionError, match="ledger path"):
        _validate_authorization(wrong_generation_path, synthetic_contract, "ACCEPT", roots)
    wrong_replay_path = _authorization(
        synthetic_contract,
        "ACCEPT",
        roots,
        generation_identity=generation_identity,
        replay_identity=("execution_ledger.json", replay_identity_value[1], replay_identity_value[2]),
    )
    with pytest.raises(PermissionError, match="ledger path"):
        _validate_authorization(wrong_replay_path, synthetic_contract, "ACCEPT", roots)
    generation_ledger = read_ledger(generation_path)
    replay_ledger = read_ledger(replay_path)
    mutations: list[tuple[dict[str, Any], str]] = []
    reordered = deepcopy(generation_ledger)
    reordered["records"].reverse()
    mutations.append((reordered, "run ordering"))
    different_set = deepcopy(generation_ledger)
    different_set["records"][0]["global_run_id"] = 999
    mutations.append((different_set, "run set"))
    different_partition = deepcopy(generation_ledger)
    different_partition["partition"] = "validation"
    mutations.append((different_partition, "partition"))
    different_contract = deepcopy(generation_ledger)
    different_contract["contract_hash"] = "f" * 64
    mutations.append((different_contract, "contract"))
    different_stable = deepcopy(generation_ledger)
    different_stable["stable_executable_commit"] = "f" * 40
    mutations.append((different_stable, "stable executable"))
    for field in seed_fields:
        changed_seed = deepcopy(generation_ledger)
        changed_seed["records"][0][field] += 1
        mutations.append((changed_seed, field))
    for changed, _ in mutations:
        with pytest.raises(ValueError):
            validate_ledger_binding(changed, acceptance_plan, operation="GENERATE")
    replay_ledger["source_generation_ledger_sha256"] = "f" * 64
    write_ledger(replay_path, replay_ledger, overwrite=True)
    changed_replay_identity = _ledger_identity(replay_path)
    cross_authorization = _authorization(
        synthetic_contract,
        "ACCEPT",
        roots,
        generation_identity=generation_identity,
        replay_identity=changed_replay_identity,
    )
    cross_plan = _plan(synthetic_contract, "ACCEPT", roots, cross_authorization)
    with pytest.raises(ValueError, match="Ledger identity|cross-binding|accepted generation"):
        validate_ledger_binding(read_ledger(replay_path), cross_plan, operation="REPLAY")
    replay_path.write_bytes(replay_bytes)


def test_binding_007_lock_identity_and_stale_recovery_fail_closed(
    synthetic_contract: FrozenStageAContract,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _roots(tmp_path)
    authorization = _authorization(synthetic_contract, "GENERATE", roots)
    plan = _plan(synthetic_contract, "GENERATE", roots, authorization)
    campaign = campaign_identity(plan, authorization.sha256)
    run = run_identity(plan, authorization.sha256, plan["runs"][0])
    campaign_payload = lock_payload(campaign, "campaign")
    run_payload = lock_payload(run, "run")
    campaign_fields = {
        "campaign_id", "operation", "partition", "contract_hash", "plan_hash",
        "authorization_hash", "stable_executable_commit", "tooling_proposal_hash",
        "host_identity", "process_id", "process_start_identity", "creation_time", "lock_nonce",
    }
    assert campaign_fields.issubset(campaign_payload)
    assert {"run_key", "global_run_id", "design_id", "realization_id"}.issubset(run_payload)
    atomic_lock = tmp_path / "atomic.lock"
    with ExclusiveLock(atomic_lock, campaign):
        with pytest.raises(RuntimeError, match="already exists"):
            ExclusiveLock(atomic_lock, campaign).acquire()
    active_lock = tmp_path / "active.lock"
    with ExclusiveLock(active_lock, campaign):
        time.sleep(0.002)
        with pytest.raises(RuntimeError, match="Active"):
            recover_stale_lock(
                active_lock,
                expected_identity=campaign,
                minimum_age_seconds=0.000001,
                campaign_root=tmp_path / "active-campaign",
                recovery_event_root=tmp_path / "active-events",
            )
    stale_lock = tmp_path / "stale.lock"
    stale_lock.write_bytes(canonical_json_bytes(_aged_inactive_lock(campaign)))
    record = recover_stale_lock(
        stale_lock,
        expected_identity=campaign,
        minimum_age_seconds=60,
        campaign_root=tmp_path / "stale-campaign",
        recovery_event_root=tmp_path / "stale-events",
    )
    event_path = Path(record["recovery_event_path"])
    assert event_path.is_file()
    assert not stale_lock.exists()
    with pytest.raises(FileExistsError):
        locking_module._write_recovery_event(event_path.parent, json.loads(event_path.read_bytes()))
    foreign_lock = tmp_path / "foreign.lock"
    foreign_value = _aged_inactive_lock(campaign)
    foreign_value["host_identity"] = "foreign-host:000000000000"
    foreign_lock.write_bytes(canonical_json_bytes(_rehash_lock(foreign_value)))
    with pytest.raises(PermissionError, match="Foreign-host"):
        recover_stale_lock(
            foreign_lock,
            expected_identity=campaign,
            minimum_age_seconds=60,
            campaign_root=tmp_path / "foreign-campaign",
            recovery_event_root=tmp_path / "foreign-events",
        )
    reused_pid_lock = tmp_path / "reused-pid.lock"
    reused_value = _aged_inactive_lock(campaign)
    reused_value["process_id"] = os.getpid()
    reused_value["process_start_identity"] = "different-process-start"
    reused_pid_lock.write_bytes(canonical_json_bytes(_rehash_lock(reused_value)))
    assert lock_is_stale(reused_pid_lock) is True
    recover_stale_lock(
        reused_pid_lock,
        expected_identity=campaign,
        minimum_age_seconds=60,
        campaign_root=tmp_path / "reused-campaign",
        recovery_event_root=tmp_path / "reused-events",
    )
    malformed = tmp_path / "malformed.lock"
    malformed.write_bytes(b"{}")
    with pytest.raises(ValueError, match="identity|field|hash"):
        recover_stale_lock(
            malformed,
            expected_identity=campaign,
            minimum_age_seconds=60,
            campaign_root=tmp_path / "malformed-campaign",
            recovery_event_root=tmp_path / "malformed-events",
        )
    wrong_campaign_lock = tmp_path / "wrong-campaign.lock"
    wrong_campaign_lock.write_bytes(canonical_json_bytes(_aged_inactive_lock(campaign)))
    with pytest.raises(PermissionError, match="identity"):
        recover_stale_lock(
            wrong_campaign_lock,
            expected_identity={**campaign, "contract_hash": "f" * 64},
            minimum_age_seconds=60,
            campaign_root=tmp_path / "wrong-campaign",
            recovery_event_root=tmp_path / "wrong-campaign-events",
        )
    wrong_run_lock = tmp_path / "wrong-run.lock"
    wrong_run_lock.write_bytes(canonical_json_bytes(_aged_inactive_lock(run, "run")))
    with pytest.raises(PermissionError, match="identity"):
        recover_stale_lock(
            wrong_run_lock,
            expected_identity={**run, "run_key": "SYN-WRONG-R00"},
            minimum_age_seconds=60,
            campaign_root=tmp_path / "wrong-run-campaign",
            recovery_event_root=tmp_path / "wrong-run-events",
        )
    completed_lock = tmp_path / "completed.lock"
    completed_lock.write_bytes(canonical_json_bytes(_aged_inactive_lock(campaign)))
    monkeypatch.setattr(locking_module, "_valid_completed_output", lambda *_: True)
    with pytest.raises(RuntimeError, match="completed output"):
        recover_stale_lock(
            completed_lock,
            expected_identity=campaign,
            minimum_age_seconds=60,
            campaign_root=tmp_path / "completed-campaign",
            recovery_event_root=tmp_path / "completed-events",
        )
    monkeypatch.undo()
    concurrent_lock = tmp_path / "concurrent.lock"
    concurrent_lock.write_bytes(canonical_json_bytes(_aged_inactive_lock(campaign)))
    guard = concurrent_lock.with_name(concurrent_lock.name + ".recovery.lock")
    guard.write_bytes(b"")
    with pytest.raises(FileExistsError):
        recover_stale_lock(
            concurrent_lock,
            expected_identity=campaign,
            minimum_age_seconds=60,
            campaign_root=tmp_path / "concurrent-campaign",
            recovery_event_root=tmp_path / "concurrent-events",
        )
    guard.unlink()
    age_only_lock = tmp_path / "age-only.lock"
    age_only_value = _aged_inactive_lock(campaign)
    age_only_value["process_id"] = os.getpid()
    current_start = process_start_identity(os.getpid())
    assert current_start is not None
    age_only_value["process_start_identity"] = current_start
    age_only_lock.write_bytes(canonical_json_bytes(_rehash_lock(age_only_value)))
    with pytest.raises(RuntimeError, match="Active"):
        recover_stale_lock(
            age_only_lock,
            expected_identity=campaign,
            minimum_age_seconds=60,
            campaign_root=tmp_path / "age-only-campaign",
            recovery_event_root=tmp_path / "age-only-events",
        )
    owner_root = tmp_path / "owner-campaign"
    owner_lock = owner_root.with_name(owner_root.name + ".lock")
    target_run_lock = tmp_path / "owner-run.lock"
    target_run_lock.write_bytes(canonical_json_bytes(_aged_inactive_lock(run, "run")))
    with ExclusiveLock(owner_lock, campaign):
        with pytest.raises(RuntimeError, match="active campaign owner"):
            recover_stale_lock(
                target_run_lock,
                expected_identity=run,
                minimum_age_seconds=60,
                campaign_root=owner_root,
                recovery_event_root=tmp_path / "owner-events",
            )
    assert host_identity() == campaign_payload["host_identity"]


def test_binding_008_windows_byte_policy_is_narrow_and_inventories_are_exact() -> None:
    lines = [line.strip() for line in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()]
    prohibited = {"*.py text eol=lf", "* -text", "*.py -text", "*.json -text", "*.md -text"}
    assert prohibited.isdisjoint(lines)
    active_rules = [line for line in lines if line and not line.startswith("#") and " -text" in line]
    assert active_rules
    assert all(not line.startswith("*") for line in active_rules)
    proposal_root = ROOT / "artifacts/stage_a_execution_tooling_v1_proposal"
    executable_path = proposal_root / "stage_a_executable_source_inventory.json"
    proposal_path = proposal_root / "stage_a_execution_tooling_inventory.json"
    assert sha256_file(executable_path) == EXPECTED_EXECUTABLE_INVENTORY
    assert sha256_file(proposal_path) == EXPECTED_PROPOSAL_INVENTORY
    executable = json.loads(executable_path.read_bytes())
    proposal = json.loads(proposal_path.read_bytes())
    assert executable["artifact_count"] == 71 == len(executable["artifacts"])
    assert proposal["artifact_count_excluding_inventory"] == 48 == len(proposal["artifacts"])
    records = {
        record["relative_path"]: record
        for record in (*executable["artifacts"], *proposal["artifacts"])
    }
    records[proposal_path.relative_to(ROOT).as_posix()] = {
        "relative_path": proposal_path.relative_to(ROOT).as_posix(),
        "byte_length": proposal_path.stat().st_size,
        "sha256": EXPECTED_PROPOSAL_INVENTORY,
    }
    assert len(records) == 100
    for record in records.values():
        path = ROOT / record["relative_path"]
        assert path.stat().st_size == record["byte_length"]
        assert sha256_file(path) == record["sha256"]
    result = subprocess.run(
        ["git", "check-attr", "text", "--stdin"],
        cwd=ROOT,
        input="".join(f"{relative}\n" for relative in sorted(records)).encode("utf-8"),
        capture_output=True,
        check=True,
    )
    observed = {
        line.split(": ", 2)[0]: line.rsplit(": ", 1)[1]
        for line in result.stdout.decode("utf-8").splitlines()
    }
    assert set(observed) == set(records)
    assert set(observed.values()) == {"unset"}
