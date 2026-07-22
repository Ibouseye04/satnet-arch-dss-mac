from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from satnet.experiments.stage_a_execution._synthetic_harness import execute_synthetic_generation, execute_synthetic_replay
from satnet.experiments.stage_a_execution.artifact_contract import SimulationAdapterResult, make_adapter_result, write_synthetic_artifacts
from satnet.experiments.stage_a_execution.common import atomic_write_json, canonical_json_bytes, payload_hash, sha256_file
import satnet.experiments.stage_a_execution.identity as identity_module
from satnet.experiments.stage_a_execution.identity import STABLE_IDENTITY_SCHEMA, make_executable_inventory, verify_executable_identity
from satnet.experiments.stage_a_execution.ledger import read_ledger, write_ledger
from satnet.experiments.stage_a_execution.locking import LOCK_DOMAIN, ExclusiveLock, _write_recovery_event, lock_is_stale, lock_payload, recover_stale_lock
from satnet.experiments.stage_a_execution.plan import build_plan
import satnet.experiments.stage_a_execution.preflight as preflight_module
from satnet.experiments.stage_a_execution.preflight import run_preflight
from satnet.experiments.stage_a_execution.acceptance import evaluate_acceptance
from scripts.generate_stage_a_execution_tooling_v1_proposal import generate

from .conftest import ARTIFACT_CONTRACT, EXECUTABLE_INVENTORY, STABLE_EXECUTABLE_COMMIT, TOOLING_PROPOSAL, make_authorization

ROOT = Path(__file__).parents[3]


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _ledger_identity(path: Path) -> tuple[str, int, str]:
    return path.name, path.stat().st_size, sha256_file(path)


def _authorization(contract, operation: str, roots, generation_identity=None, replay_identity=None):
    return make_authorization(
        contract, operation=operation, partition="development", run_ids=[1, 2],
        generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
        source_generation_ledger=generation_identity, source_replay_ledger=replay_identity,
    )


def _plan(contract, operation: str, roots: tuple[Path, Path, Path], authorization) -> dict:
    return build_plan(
        contract, partition="development", operation=operation,
        stable_executable_commit=STABLE_EXECUTABLE_COMMIT,
        executable_inventory_hash=EXECUTABLE_INVENTORY,
        tooling_proposal_hash=TOOLING_PROPOSAL,
        artifact_contract_hash=ARTIFACT_CONTRACT,
        generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
        authorization=authorization,
    )


def _lock_identity() -> dict[str, str]:
    return {
        "campaign_id": "1" * 64, "operation": "GENERATE", "partition": "development",
        "contract_hash": "2" * 64, "plan_hash": "3" * 64, "authorization_hash": "4" * 64,
        "stable_executable_commit": "5" * 40, "tooling_proposal_hash": "6" * 64,
    }


def _certificate(monkeypatch, contract, plan: dict, roots: tuple[Path, Path, Path], authorization):
    monkeypatch.setattr(preflight_module, "verify_executable_identity", lambda _: {
        "stable_executable_commit": STABLE_EXECUTABLE_COMMIT,
        "executable_inventory_sha256": EXECUTABLE_INVENTORY,
    })
    monkeypatch.setattr(preflight_module, "load_frozen_contract", lambda *_: contract)
    monkeypatch.setattr(preflight_module, "verify_frozen_production_evidence", lambda *_: {
        "combined": {"file_count": 9004, "byte_count": 1337549193, "verified_sha256_count": 9004},
    })
    certificate = run_preflight(
        repo_root=ROOT, contract=contract, plan=plan, authorization=authorization,
        generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2], minimum_free_bytes=0,
    )
    return certificate


def test_executable_identity_rejects_dirty_missing_extra_and_changed_bytes(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "tool.py"
    source.write_text("VALUE = 1\n", encoding="utf-8", newline="\n")
    _git(repo, "init")
    _git(repo, "config", "user.name", "Synthetic Audit")
    _git(repo, "config", "user.email", "audit@example.invalid")
    _git(repo, "add", "tool.py")
    _git(repo, "commit", "-m", "stable executable")
    stable_commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(identity_module, "executable_source_paths", lambda _: ("tool.py",))
    inventory = make_executable_inventory(repo, stable_commit)
    inventory_path = repo / "inventory.json"
    inventory_path.write_bytes(canonical_json_bytes(inventory))
    stable_path = repo / "stable.json"
    stable_path.write_bytes(canonical_json_bytes({
        "schema_identifier": STABLE_IDENTITY_SCHEMA,
        "stable_executable_commit": stable_commit,
        "executable_inventory_sha256": sha256_file(inventory_path),
        "executable_file_count": inventory["artifact_count"],
    }))
    _git(repo, "add", "inventory.json", "stable.json")
    _git(repo, "commit", "-m", "bind executable")
    assert verify_executable_identity(repo, inventory_path, stable_path)["verification"] == "PASSED"
    source.write_text("VALUE = 2\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="clean|dirty|byte"):
        verify_executable_identity(repo, inventory_path, stable_path)
    source.write_text("VALUE = 1\n", encoding="utf-8", newline="\n")
    source.unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        verify_executable_identity(repo, inventory_path, stable_path)
    source.write_text("VALUE = 1\n", encoding="utf-8", newline="\n")
    extra = repo / "extra.py"
    extra.write_text("EXTRA = True\n", encoding="utf-8", newline="\n")
    monkeypatch.setattr(identity_module, "executable_source_paths", lambda _: ("extra.py", "tool.py"))
    with pytest.raises(ValueError, match="clean|membership"):
        verify_executable_identity(repo, inventory_path, stable_path)


def test_preflight_root_rejection_precedes_evidence_and_writes(synthetic_contract, tmp_path: Path, monkeypatch) -> None:
    repo = ROOT
    roots = (repo / ".synthetic-forbidden-generation", tmp_path / "replay", tmp_path / "acceptance")
    authorization = _authorization(synthetic_contract, "GENERATE", roots)
    plan = _plan(synthetic_contract, "GENERATE", roots, authorization)
    monkeypatch.setattr(preflight_module, "verify_executable_identity", lambda _: {
        "stable_executable_commit": STABLE_EXECUTABLE_COMMIT,
        "executable_inventory_sha256": EXECUTABLE_INVENTORY,
    })
    monkeypatch.setattr(preflight_module, "load_frozen_contract", lambda *_: synthetic_contract)
    evidence_called = False
    def evidence(*_):
        nonlocal evidence_called
        evidence_called = True
        raise AssertionError("root isolation must precede evidence scan")
    monkeypatch.setattr(preflight_module, "verify_frozen_production_evidence", evidence)
    with pytest.raises(ValueError, match="protected"):
        run_preflight(
            repo_root=repo, contract=synthetic_contract, plan=plan, authorization=authorization,
            generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2], minimum_free_bytes=0,
        )
    assert evidence_called is False
    assert not any(path.exists() for path in roots)
    assert not roots[0].with_name(roots[0].name + ".lock").exists()


def test_malformed_nonempty_output_never_succeeds(synthetic_contract, tmp_path: Path, monkeypatch) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    authorization = _authorization(synthetic_contract, "GENERATE", roots)
    plan = _plan(synthetic_contract, "GENERATE", roots, authorization)
    certificate = _certificate(monkeypatch, synthetic_contract, plan, roots, authorization)
    def malformed(plan_run, output_root: Path):
        (output_root / "junk.bin").write_bytes(b"junk")
        return make_adapter_result(plan_run, output_root, "synthetic_stage_a_test_run_v1")
    with pytest.raises(ValueError, match="path set|artifact"):
        execute_synthetic_generation(
            repo_root=ROOT, contract=synthetic_contract, plan=plan,
            authorization_hash=authorization.sha256, preflight=certificate,
            campaign_root=roots[0], adapter=malformed,
        )
    ledger = read_ledger(roots[0] / "execution_ledger.json")
    assert ledger["records"][0]["state"] == "FAILED"
    assert ledger["records"][0]["adapter_result"] is None


def test_replay_and_acceptance_reject_bound_identity_and_seed_mutations(synthetic_contract, tmp_path: Path, monkeypatch) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    generation_authorization = _authorization(synthetic_contract, "GENERATE", roots)
    generation_plan = _plan(synthetic_contract, "GENERATE", roots, generation_authorization)
    generation_certificate = _certificate(monkeypatch, synthetic_contract, generation_plan, roots, generation_authorization)
    execute_synthetic_generation(
        repo_root=ROOT, contract=synthetic_contract, plan=generation_plan,
        authorization_hash=generation_authorization.sha256, preflight=generation_certificate,
        campaign_root=roots[0], adapter=write_synthetic_artifacts,
    )
    generation_ledger_path = roots[0] / "execution_ledger.json"
    authorized_generation_bytes = generation_ledger_path.read_bytes()
    generation_identity = _ledger_identity(generation_ledger_path)
    replay_authorization = _authorization(synthetic_contract, "REPLAY", roots, generation_identity)
    replay_plan = _plan(synthetic_contract, "REPLAY", roots, replay_authorization)
    replay_certificate = _certificate(monkeypatch, synthetic_contract, replay_plan, roots, replay_authorization)
    generation_ledger = read_ledger(generation_ledger_path)
    generation_ledger["stable_executable_commit"] = "f" * 40
    write_ledger(generation_ledger_path, generation_ledger, overwrite=True)
    with pytest.raises(ValueError, match="SHA-256|byte length"):
        execute_synthetic_replay(
            repo_root=ROOT, contract=synthetic_contract, plan=replay_plan,
            authorization_hash=replay_authorization.sha256, preflight=replay_certificate,
            generation_root=roots[0], replay_root=roots[1], adapter=write_synthetic_artifacts,
        )
    generation_ledger_path.write_bytes(authorized_generation_bytes)
    replay_certificate = _certificate(monkeypatch, synthetic_contract, replay_plan, roots, replay_authorization)
    execute_synthetic_replay(
        repo_root=ROOT, contract=synthetic_contract, plan=replay_plan,
        authorization_hash=replay_authorization.sha256, preflight=replay_certificate,
        generation_root=roots[0], replay_root=roots[1], adapter=write_synthetic_artifacts,
    )
    replay_identity = _ledger_identity(roots[1] / "replay_ledger.json")
    acceptance_authorization = _authorization(
        synthetic_contract, "ACCEPT", roots, generation_identity, replay_identity,
    )
    acceptance_plan = _plan(synthetic_contract, "ACCEPT", roots, acceptance_authorization)
    acceptance_certificate = _certificate(monkeypatch, synthetic_contract, acceptance_plan, roots, acceptance_authorization)
    generation_ledger = read_ledger(generation_ledger_path)
    generation_ledger["records"][0]["satellite_failure_seed"] += 1
    write_ledger(generation_ledger_path, generation_ledger, overwrite=True)
    with pytest.raises(ValueError, match="SHA-256|byte length"):
        evaluate_acceptance(
            repo_root=ROOT, plan=acceptance_plan, authorization_hash=acceptance_authorization.sha256,
            preflight=acceptance_certificate, generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
        )


def test_stale_lock_recovery_validates_identity_and_records_immutable_evidence(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    lock = campaign_root.with_name(campaign_root.name + ".lock")
    identity = _lock_identity()
    value = lock_payload(identity, "campaign")
    value["process_id"] = 2_147_483_647
    value["process_start_identity"] = "inactive"
    value["creation_unix_ns"] = time.time_ns() - 120_000_000_000
    value["creation_time"] = "2000-01-01T00:00:00+00:00"
    payload = {field: item for field, item in value.items() if field != "lock_hash"}
    value["lock_hash"] = payload_hash(payload, domain=LOCK_DOMAIN)
    lock.write_bytes(canonical_json_bytes(value))
    assert lock_is_stale(lock) is True
    wrong = {**identity, "contract_hash": "f" * 64}
    with pytest.raises(PermissionError, match="identity"):
        recover_stale_lock(
            lock, expected_identity=wrong, minimum_age_seconds=60,
            campaign_root=campaign_root, recovery_event_root=tmp_path / "events",
        )
    record = recover_stale_lock(
        lock, expected_identity=identity, minimum_age_seconds=60,
        campaign_root=campaign_root, recovery_event_root=tmp_path / "events",
    )
    assert record["recovered_lock"] == value
    assert not lock.exists()
    event_path = Path(record["recovery_event_path"])
    event = json.loads(event_path.read_bytes())
    assert event == {field: item for field, item in record.items() if field != "recovery_event_path"}
    with pytest.raises(FileExistsError):
        _write_recovery_event(tmp_path / "events", event)


def test_proposal_generator_reproduces_all_artifacts_byte_for_byte(tmp_path: Path) -> None:
    stable_commit = _git(ROOT, "rev-parse", "HEAD")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = generate(ROOT, first, stable_commit)
    second_result = generate(ROOT, second, stable_commit)
    assert first_result["generated_artifact_names"] == second_result["generated_artifact_names"]
    assert first_result["generated_artifact_count"] == second_result["generated_artifact_count"]
    for name in first_result["generated_artifact_names"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert first_result["execution_authorized"] is False
    assert first_result["simulation_authorized"] is False
    assert first_result["production_authorized"] is False
    specification = json.loads((first / "stage_a_execution_tooling_specification.json").read_bytes())
    assert specification["status"] == "TOOLING_PROPOSAL"
    assert specification["execution_authorized"] is False


def test_live_lock_cannot_be_recovered(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    lock = campaign_root.with_name(campaign_root.name + ".lock")
    identity = _lock_identity()
    with ExclusiveLock(lock, identity):
        assert lock_is_stale(lock) is False
        time.sleep(0.01)
        with pytest.raises(RuntimeError, match="Active"):
            recover_stale_lock(
                lock, expected_identity=identity, minimum_age_seconds=0.001,
                campaign_root=campaign_root, recovery_event_root=tmp_path / "events",
            )
