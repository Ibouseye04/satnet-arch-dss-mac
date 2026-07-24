from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pytest

import satnet.experiments.stage_a_execution.acceptance as acceptance_module
from satnet.experiments.stage_a_execution._synthetic_harness import (
    execute_synthetic_generation,
    execute_synthetic_replay,
)
from satnet.experiments.stage_a_execution.acceptance import evaluate_acceptance
from satnet.experiments.stage_a_execution.artifact_contract import write_synthetic_artifacts
from satnet.experiments.stage_a_execution.authorization import Authorization
from satnet.experiments.stage_a_execution.common import canonical_json_bytes, sha256_file
from satnet.experiments.stage_a_execution.plan import build_plan
import satnet.experiments.stage_a_execution.preflight as preflight_module
from satnet.experiments.stage_a_execution.preflight import PreflightCertificate, run_preflight

from .conftest import (
    ARTIFACT_CONTRACT,
    TOOLING_PROPOSAL,
    make_authorization,
    source_provenance_from_ledger,
)

ROOT = Path(__file__).parents[3]
GENERATION_COMMIT = "fafe3fe36eac4429c860bd6d281923fed2980ea7"
GENERATION_INVENTORY = "1066d17645d4975a40a93a80c1ca9797b1b28cf37531337355ccb3297fd13b2d"
REPLAY_COMMIT = "8bde92da762269998632eb6c3e3cb2565a6ca971"
REPLAY_INVENTORY = "d52ffc2afa5f94268d89187afe58d4c7695bdf261ec20113644c9cb7f5840215"
ACCEPT_COMMIT = "b1da92c3184ee86997079f1eb458fae95d994c22"
ACCEPT_INVENTORY = "2ebac3bb615ed3204edc1359a78f78e4354495364fa1f76dfd1648b417df7b56"


@dataclass(frozen=True)
class DomainCampaign:
    roots: tuple[Path, Path, Path]
    generation_plan: dict[str, Any]
    replay_plan: dict[str, Any]
    acceptance_plan: dict[str, Any]
    acceptance_authorization: Authorization
    acceptance_certificate: PreflightCertificate


def _ledger_identity(path: Path) -> tuple[str, int, str]:
    return path.name, path.stat().st_size, sha256_file(path)


def _authorization(
    contract: Any,
    operation: str,
    roots: tuple[Path, Path, Path],
    *,
    stable_commit: str,
    executable_inventory: str,
    generation_identity: tuple[str, int, str] | None = None,
    replay_identity: tuple[str, int, str] | None = None,
) -> Authorization:
    return make_authorization(
        contract,
        operation=operation,
        partition="development",
        run_ids=[1, 2],
        generation_root=roots[0],
        replay_root=roots[1],
        acceptance_root=roots[2],
        source_generation_ledger=generation_identity,
        source_replay_ledger=replay_identity,
        overrides={
            "authorized_stable_executable_commit": stable_commit,
            "authorized_executable_inventory_hash": executable_inventory,
        },
    )


def _plan(
    contract: Any,
    operation: str,
    roots: tuple[Path, Path, Path],
    authorization: Authorization,
    *,
    stable_commit: str,
    executable_inventory: str,
) -> dict[str, Any]:
    return build_plan(
        contract,
        partition="development",
        operation=operation,
        stable_executable_commit=stable_commit,
        executable_inventory_hash=executable_inventory,
        tooling_proposal_hash=TOOLING_PROPOSAL,
        artifact_contract_hash=ARTIFACT_CONTRACT,
        generation_root=roots[0],
        replay_root=roots[1],
        acceptance_root=roots[2],
        authorization=authorization,
    )


def _certificate(
    monkeypatch: pytest.MonkeyPatch,
    contract: Any,
    plan: dict[str, Any],
    roots: tuple[Path, Path, Path],
    authorization: Authorization,
) -> PreflightCertificate:
    monkeypatch.setattr(
        preflight_module,
        "verify_executable_identity",
        lambda _: {
            "stable_executable_commit": plan["stable_executable_commit"],
            "executable_inventory_sha256": plan["executable_inventory_hash"],
        },
    )
    monkeypatch.setattr(preflight_module, "load_frozen_contract", lambda *_: contract)
    monkeypatch.setattr(
        preflight_module,
        "verify_frozen_production_evidence",
        lambda *_: {
            "combined": {
                "file_count": 9004,
                "byte_count": 1337549193,
                "verified_sha256_count": 9004,
            },
        },
    )
    monkeypatch.setattr(
        preflight_module,
        "load_source_generation_provenance",
        lambda *_args, **_kwargs: source_provenance_from_ledger(
            roots[0] / "execution_ledger.json"
        ),
    )
    monkeypatch.setattr(
        preflight_module,
        "load_source_replay_provenance",
        lambda *_args, **_kwargs: source_provenance_from_ledger(
            roots[1] / "replay_ledger.json"
        ),
    )
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


def _prepare(
    monkeypatch: pytest.MonkeyPatch,
    contract: Any,
    tmp_path: Path,
) -> DomainCampaign:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    generation_authorization = _authorization(
        contract,
        "GENERATE",
        roots,
        stable_commit=GENERATION_COMMIT,
        executable_inventory=GENERATION_INVENTORY,
    )
    generation_plan = _plan(
        contract,
        "GENERATE",
        roots,
        generation_authorization,
        stable_commit=GENERATION_COMMIT,
        executable_inventory=GENERATION_INVENTORY,
    )
    generation_certificate = _certificate(
        monkeypatch, contract, generation_plan, roots, generation_authorization
    )
    execute_synthetic_generation(
        repo_root=ROOT,
        contract=contract,
        plan=generation_plan,
        authorization_hash=generation_authorization.sha256,
        preflight=generation_certificate,
        campaign_root=roots[0],
        adapter=write_synthetic_artifacts,
    )
    generation_identity = _ledger_identity(roots[0] / "execution_ledger.json")
    replay_authorization = _authorization(
        contract,
        "REPLAY",
        roots,
        stable_commit=REPLAY_COMMIT,
        executable_inventory=REPLAY_INVENTORY,
        generation_identity=generation_identity,
    )
    replay_plan = _plan(
        contract,
        "REPLAY",
        roots,
        replay_authorization,
        stable_commit=REPLAY_COMMIT,
        executable_inventory=REPLAY_INVENTORY,
    )
    replay_certificate = _certificate(
        monkeypatch, contract, replay_plan, roots, replay_authorization
    )
    execute_synthetic_replay(
        repo_root=ROOT,
        contract=contract,
        plan=replay_plan,
        authorization_hash=replay_authorization.sha256,
        preflight=replay_certificate,
        generation_root=roots[0],
        replay_root=roots[1],
        adapter=write_synthetic_artifacts,
    )
    replay_identity = _ledger_identity(roots[1] / "replay_ledger.json")
    acceptance_authorization = _authorization(
        contract,
        "ACCEPT",
        roots,
        stable_commit=ACCEPT_COMMIT,
        executable_inventory=ACCEPT_INVENTORY,
        generation_identity=generation_identity,
        replay_identity=replay_identity,
    )
    acceptance_plan = _plan(
        contract,
        "ACCEPT",
        roots,
        acceptance_authorization,
        stable_commit=ACCEPT_COMMIT,
        executable_inventory=ACCEPT_INVENTORY,
    )
    acceptance_certificate = _certificate(
        monkeypatch, contract, acceptance_plan, roots, acceptance_authorization
    )
    return DomainCampaign(
        roots=roots,
        generation_plan=generation_plan,
        replay_plan=replay_plan,
        acceptance_plan=acceptance_plan,
        acceptance_authorization=acceptance_authorization,
        acceptance_certificate=acceptance_certificate,
    )


def _evaluate(campaign: DomainCampaign) -> dict[str, Any]:
    return evaluate_acceptance(
        repo_root=ROOT,
        plan=campaign.acceptance_plan,
        authorization_hash=campaign.acceptance_authorization.sha256,
        preflight=campaign.acceptance_certificate,
        generation_root=campaign.roots[0],
        replay_root=campaign.roots[1],
        acceptance_root=campaign.roots[2],
    )


def _first_replay_report(campaign: DomainCampaign) -> Path:
    relative = campaign.replay_plan["runs"][0]["expected_output_relative_path"]
    return campaign.roots[1] / relative / "replay_report.json"


def _replace_report_field(campaign: DomainCampaign, field: str, value: Any) -> None:
    path = _first_replay_report(campaign)
    report = json.loads(path.read_bytes())
    report[field] = value
    path.write_bytes(canonical_json_bytes(report))


def _disable_artifact_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acceptance_module, "verify_artifact_inventory", lambda *_: None)


def test_historical_replay_executable_is_accepted_by_distinct_accept_executable(
    synthetic_contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _prepare(monkeypatch, synthetic_contract, tmp_path)
    replay_report = json.loads(_first_replay_report(campaign).read_bytes())
    result = _evaluate(campaign)
    assert replay_report["stable_executable_commit"] == REPLAY_COMMIT
    assert result["stable_executable_commit"] == ACCEPT_COMMIT
    assert result["acceptance_state"] == "PASSED"
    assert result["accepted_run_count"] == 2


@pytest.mark.parametrize("replacement", ["0" * 40, ACCEPT_COMMIT])
def test_replay_report_executable_must_match_replay_ledger_provenance(
    replacement: str,
    synthetic_contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _prepare(monkeypatch, synthetic_contract, tmp_path)
    _replace_report_field(campaign, "stable_executable_commit", replacement)
    _disable_artifact_verification(monkeypatch)
    with pytest.raises(ValueError, match="stable_executable_commit"):
        _evaluate(campaign)
    assert not campaign.roots[2].exists()
    assert not campaign.roots[2].with_name(campaign.roots[2].name + ".lock").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_identifier", "satnet.stage_a.replay_report.invalid"),
        ("plan_hash", "0" * 64),
        ("authorization_hash", "0" * 64),
        ("global_run_id", 999),
        ("run_key", "SYN-D999-R99"),
        ("run_record_hash", "0" * 64),
        ("design_construction_seed", -1),
        ("ground_selection_seed", -1),
        ("satellite_failure_seed", -1),
        ("ground_failure_seed", -1),
        ("source_generation_ledger_relative_path", "wrong.json"),
        ("source_generation_ledger_byte_length", 1),
        ("source_generation_ledger_sha256", "0" * 64),
    ],
)
def test_replay_report_identity_seed_and_generation_binding_fail_closed(
    field: str,
    value: Any,
    synthetic_contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _prepare(monkeypatch, synthetic_contract, tmp_path)
    _replace_report_field(campaign, field, value)
    _disable_artifact_verification(monkeypatch)
    with pytest.raises(ValueError, match=field):
        _evaluate(campaign)
    assert not campaign.roots[2].exists()


def test_generation_executable_provenance_is_independently_validated(
    synthetic_contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _prepare(monkeypatch, synthetic_contract, tmp_path)
    campaign.acceptance_certificate.report["source_provenance"]["generation"][
        "stable_executable_commit"
    ] = REPLAY_COMMIT
    with pytest.raises(ValueError, match="Ledger identity mismatch: stable_executable_commit"):
        _evaluate(campaign)
    assert not campaign.roots[2].exists()


def test_replay_ledger_generation_ledger_cross_binding_fails_closed(
    synthetic_contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _prepare(monkeypatch, synthetic_contract, tmp_path)
    wrong_sha256 = "0" * 64
    campaign.acceptance_certificate.report["source_provenance"]["replay"][
        "source_generation_ledger_sha256"
    ] = wrong_sha256
    original = acceptance_module.read_bound_ledger
    call_count = 0

    def changed_replay(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal call_count
        ledger, identity = original(*args, **kwargs)
        call_count += 1
        if call_count == 2:
            ledger = deepcopy(ledger)
            ledger["source_generation_ledger_sha256"] = wrong_sha256
        return ledger, identity

    monkeypatch.setattr(acceptance_module, "read_bound_ledger", changed_replay)
    with pytest.raises(ValueError, match="not bound to the accepted generation ledger bytes"):
        _evaluate(campaign)
    assert not campaign.roots[2].exists()


def test_current_accept_authorization_plan_campaign_and_executable_govern_result(
    synthetic_contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _prepare(monkeypatch, synthetic_contract, tmp_path)
    result = _evaluate(campaign)
    assert result["authorization_hash"] == campaign.acceptance_authorization.sha256
    assert result["plan_hash"] == campaign.acceptance_plan["plan_hash"]
    assert result["stable_executable_commit"] == ACCEPT_COMMIT
    assert result["executable_inventory_hash"] == ACCEPT_INVENTORY
    assert campaign.acceptance_certificate.report[
        "current_operation_campaign_manifest_hash"
    ] == campaign.acceptance_plan["campaign_manifest_hash"]


def test_current_accept_executable_validation_remains_fail_closed(
    synthetic_contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _prepare(monkeypatch, synthetic_contract, tmp_path)
    monkeypatch.setattr(
        preflight_module,
        "verify_executable_identity",
        lambda _: {
            "stable_executable_commit": REPLAY_COMMIT,
            "executable_inventory_sha256": ACCEPT_INVENTORY,
        },
    )
    with pytest.raises(PermissionError, match="Plan stable executable identity mismatch"):
        run_preflight(
            repo_root=ROOT,
            contract=synthetic_contract,
            plan=campaign.acceptance_plan,
            authorization=campaign.acceptance_authorization,
            generation_root=campaign.roots[0],
            replay_root=campaign.roots[1],
            acceptance_root=campaign.roots[2],
            minimum_free_bytes=0,
        )
    assert not campaign.roots[2].exists()


def test_operation_bound_campaign_identities_are_distinct(
    synthetic_contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _prepare(monkeypatch, synthetic_contract, tmp_path)
    identities = {
        campaign.generation_plan["campaign_manifest_hash"],
        campaign.replay_plan["campaign_manifest_hash"],
        campaign.acceptance_plan["campaign_manifest_hash"],
    }
    assert len(identities) == 3
    assert campaign.generation_plan["campaign_manifest_operation"] == "GENERATE"
    assert campaign.replay_plan["campaign_manifest_operation"] == "REPLAY"
    assert campaign.acceptance_plan["campaign_manifest_operation"] == "ACCEPT"


def test_artifact_hash_mismatch_prevents_acceptance_result(
    synthetic_contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _prepare(monkeypatch, synthetic_contract, tmp_path)
    _replace_report_field(campaign, "stable_executable_commit", "0" * 40)
    with pytest.raises(ValueError, match="inventory changed"):
        _evaluate(campaign)
    assert not campaign.roots[2].exists()


def test_accept_preflight_is_nonexecuting_and_creates_no_acceptance_state(
    synthetic_contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _prepare(monkeypatch, synthetic_contract, tmp_path)
    assert campaign.acceptance_certificate.report["preflight"] == "PASSED"
    assert campaign.acceptance_certificate.report["simulation_executed"] is False
    assert not campaign.roots[2].exists()
    assert not campaign.roots[2].with_name(campaign.roots[2].name + ".lock").exists()
