from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import satnet.experiments.stage_a_contract.freeze as freeze
import satnet.experiments.stage_a_contract.proposal as proposal
from satnet.experiments.stage_a_execution.authorization import (
    Authorization,
    authorization_digest,
    load_authorization,
)
from satnet.experiments.stage_a_execution.common import sha256_file
from satnet.experiments.stage_a_execution.contract import (
    FROZEN_CONTRACT_HASH,
    frozen_contract_output_root_states,
    load_frozen_contract,
)
from satnet.experiments.stage_a_execution.paths import validate_output_roots
from satnet.experiments.stage_a_execution.plan import build_plan
import satnet.experiments.stage_a_execution.preflight as preflight_module
from satnet.experiments.stage_a_execution.preflight import run_preflight

ROOT = Path(__file__).parents[3]
AUTHORIZATION_PATH = (
    ROOT
    / "artifacts/stage_a_development_replay_authorization_v1_active"
    / "stage_a_development_replay_execution_authorization.json"
)


def _authorization(**overrides: Any) -> Authorization:
    document = json.loads(AUTHORIZATION_PATH.read_bytes())
    document.update(overrides)
    document["authorization_sha256"] = authorization_digest(document)
    return Authorization(document=document, sha256=document["authorization_sha256"])


def _roots(authorization: Authorization) -> tuple[Path, Path, Path]:
    return (
        Path(authorization.document["authorized_generation_root"]),
        Path(authorization.document["authorized_replay_root"]),
        Path(authorization.document["authorized_acceptance_root"]),
    )


def _plan(contract: Any, authorization: Authorization, operation: str) -> dict[str, Any]:
    generation, replay, acceptance = _roots(authorization)
    document = authorization.document
    return build_plan(
        contract,
        partition="development",
        operation=operation,
        stable_executable_commit=document["authorized_stable_executable_commit"],
        executable_inventory_hash=document["authorized_executable_inventory_hash"],
        tooling_proposal_hash=document["authorized_tooling_proposal_hash"],
        artifact_contract_hash=document["authorized_artifact_contract_hash"],
        generation_root=generation,
        replay_root=replay,
        acceptance_root=acceptance,
        authorization=authorization,
    )


def _patch_noncontract_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    authorization: Authorization,
) -> None:
    document = authorization.document
    monkeypatch.setattr(
        preflight_module,
        "verify_executable_identity",
        lambda _: {
            "stable_executable_commit": document["authorized_stable_executable_commit"],
            "executable_inventory_sha256": document["authorized_executable_inventory_hash"],
        },
    )
    monkeypatch.setattr(
        preflight_module,
        "verify_frozen_production_evidence",
        lambda *_: {
            "combined": {
                "file_count": 9004,
                "byte_count": 1_337_549_193,
                "verified_sha256_count": 9004,
            }
        },
    )


def _run(
    contract: Any,
    authorization: Authorization,
    plan: dict[str, Any],
) -> Any:
    generation, replay, acceptance = _roots(authorization)
    return run_preflight(
        repo_root=ROOT,
        contract=contract,
        plan=plan,
        authorization=authorization,
        generation_root=generation,
        replay_root=replay,
        acceptance_root=acceptance,
        check_write_probe=False,
        minimum_free_bytes=0,
    )


def test_real_replay_contract_load_reconstructs_exact_hash_with_generation_present() -> None:
    generation = Path(freeze.OUTPUT_ROOTS["production_generation"])
    replay = Path(freeze.OUTPUT_ROOTS["production_replay"])
    acceptance = Path(freeze.OUTPUT_ROOTS["production_acceptance"])
    assert generation.is_dir()
    assert not replay.exists()
    assert not acceptance.exists()

    contract = load_frozen_contract(ROOT, operation="REPLAY")

    assert contract.contract_hash == FROZEN_CONTRACT_HASH
    assert contract.contract_hash == "e88b2f3f3fd545a2876e79262be73901eb77a656e08835ff01aeb52d6a9ba51a"
    assert freeze.verify_output_roots(
        ROOT,
        expected_states=frozen_contract_output_root_states("REPLAY"),
    ) == frozen_contract_output_root_states("REPLAY")


def test_proposal_reconstruction_is_byte_exact_while_generation_exists() -> None:
    generation = Path(freeze.OUTPUT_ROOTS["production_generation"])
    assert generation.is_dir()

    reproduced = freeze.reproduce_approved_proposal(ROOT)

    source = ROOT / freeze.SOURCE_ROOT_RELATIVE
    assert tuple(sorted(reproduced)) == freeze.SOURCE_ARTIFACTS
    assert all(reproduced[name] == source.joinpath(name).read_bytes() for name in reproduced)
    assert sha256_file(source / "stage_a_proposal_inventory.json") == freeze.APPROVED_PROPOSAL_INVENTORY


def test_real_replay_preflight_passes_and_binds_authoritative_generation_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = load_authorization(AUTHORIZATION_PATH)
    contract = load_frozen_contract(ROOT, operation="REPLAY")
    plan = _plan(contract, authorization, "REPLAY")
    generation, replay, acceptance = _roots(authorization)
    ledger = generation / "execution_ledger.json"
    before = (ledger.stat().st_size, sha256_file(ledger), replay.exists(), acceptance.exists())
    _patch_noncontract_dependencies(monkeypatch, authorization)

    certificate = _run(contract, authorization, plan)

    assert certificate.report["preflight"] == "PASSED"
    assert certificate.report["contract_hash"] == FROZEN_CONTRACT_HASH
    assert certificate.report["source_ledgers"]["generation"] == {
        "relative_path": "execution_ledger.json",
        "byte_length": 1_060_132,
        "sha256": "9df1448612d4719d164cf014a2ab60d932d82ab07a90e05c50a0c1650b097328",
    }
    assert certificate.report["simulation_executed"] is False
    assert before == (ledger.stat().st_size, sha256_file(ledger), replay.exists(), acceptance.exists())
    assert not replay.exists()
    assert not acceptance.exists()


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        (
            {"source_generation_ledger_relative_path": "replay_ledger.json"},
            PermissionError,
            "Authorization ledger path mismatch",
        ),
        (
            {"source_generation_ledger_byte_length": 1_060_131},
            PermissionError,
            "Current authorization source-generation ledger identity mismatch",
        ),
        (
            {"source_generation_ledger_sha256": "0" * 64},
            PermissionError,
            "Current authorization source-generation ledger identity mismatch",
        ),
    ],
)
def test_real_replay_preflight_enforces_source_ledger_path_length_and_sha256(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    authorization = _authorization(**overrides)
    contract = load_frozen_contract(ROOT, operation="REPLAY")
    plan = _plan(contract, authorization, "REPLAY")
    _, replay, acceptance = _roots(authorization)
    _patch_noncontract_dependencies(monkeypatch, authorization)

    with pytest.raises(error, match=message):
        _run(contract, authorization, plan)
    assert not replay.exists()
    assert not acceptance.exists()


def test_real_path_replay_rejects_missing_generation_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    authorization = _authorization(
        authorized_generation_root=str(roots[0].resolve()),
        authorized_replay_root=str(roots[1].resolve()),
        authorized_acceptance_root=str(roots[2].resolve()),
    )
    contract = load_frozen_contract(ROOT, operation="REPLAY")
    plan = _plan(contract, authorization, "REPLAY")
    _patch_noncontract_dependencies(monkeypatch, authorization)

    with pytest.raises(FileExistsError, match="Output-root state mismatch for REPLAY"):
        _run(contract, authorization, plan)
    assert not any(root.exists() for root in roots)


@pytest.mark.parametrize("mutation", ["missing", "changed", "mismatched"])
def test_real_path_replay_rejects_missing_changed_or_mismatched_generation_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    roots[0].mkdir()
    source = Path(freeze.OUTPUT_ROOTS["production_generation"]) / "execution_ledger.json"
    target = roots[0] / "execution_ledger.json"
    if mutation != "missing":
        target.write_bytes(source.read_bytes() + (b"\n" if mutation == "changed" else b""))
    overrides: dict[str, Any] = {
        "authorized_generation_root": str(roots[0].resolve()),
        "authorized_replay_root": str(roots[1].resolve()),
        "authorized_acceptance_root": str(roots[2].resolve()),
    }
    if mutation == "mismatched":
        overrides["source_generation_ledger_sha256"] = "0" * 64
    authorization = _authorization(**overrides)
    contract = load_frozen_contract(ROOT, operation="REPLAY")
    plan = _plan(contract, authorization, "REPLAY")
    _patch_noncontract_dependencies(monkeypatch, authorization)

    with pytest.raises((FileNotFoundError, PermissionError, ValueError)):
        _run(contract, authorization, plan)
    assert roots[0].is_dir()
    assert not roots[1].exists()
    assert not roots[2].exists()


def test_real_path_replay_rejects_existing_replay_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    roots[0].mkdir()
    roots[1].mkdir()
    authorization = _authorization(
        authorized_generation_root=str(roots[0].resolve()),
        authorized_replay_root=str(roots[1].resolve()),
        authorized_acceptance_root=str(roots[2].resolve()),
    )
    contract = load_frozen_contract(ROOT, operation="REPLAY")
    plan = _plan(contract, authorization, "REPLAY")
    _patch_noncontract_dependencies(monkeypatch, authorization)

    with pytest.raises(FileExistsError, match="Output-root state mismatch for REPLAY"):
        _run(contract, authorization, plan)
    assert roots[0].is_dir()
    assert roots[1].is_dir()
    assert not roots[2].exists()


def test_real_path_generate_still_rejects_existing_generation_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    roots[0].mkdir()
    authorization = _authorization(
        authorized_operation="GENERATE",
        authorized_generation_root=str(roots[0].resolve()),
        authorized_replay_root=str(roots[1].resolve()),
        authorized_acceptance_root=str(roots[2].resolve()),
        source_generation_ledger_relative_path=None,
        source_generation_ledger_byte_length=None,
        source_generation_ledger_sha256=None,
    )
    contract = load_frozen_contract(ROOT, operation="REPLAY")
    plan = _plan(contract, authorization, "GENERATE")
    _patch_noncontract_dependencies(monkeypatch, authorization)

    with pytest.raises(ValueError, match="Reserved Stage A output root exists: production_generation"):
        _run(contract, authorization, plan)
    assert roots[0].is_dir()
    assert not roots[1].exists()
    assert not roots[2].exists()


def test_real_path_accept_requires_generation_and_replay_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    authorization = _authorization(
        authorized_operation="ACCEPT",
        authorized_generation_root=str(roots[0].resolve()),
        authorized_replay_root=str(roots[1].resolve()),
        authorized_acceptance_root=str(roots[2].resolve()),
        source_replay_ledger_relative_path="replay_ledger.json",
        source_replay_ledger_byte_length=1,
        source_replay_ledger_sha256="0" * 64,
    )
    contract = load_frozen_contract(ROOT, operation="REPLAY")
    plan = _plan(contract, authorization, "ACCEPT")
    _patch_noncontract_dependencies(monkeypatch, authorization)

    assert frozen_contract_output_root_states("ACCEPT")["production_generation"] is True
    assert frozen_contract_output_root_states("ACCEPT")["production_replay"] is True
    with pytest.raises(ValueError, match="Required Stage A source root does not exist: production_replay"):
        _run(contract, authorization, plan)
    assert not any(root.exists() for root in roots)


def test_root_overlap_and_protected_path_rejection_remain_unconditional(tmp_path: Path) -> None:
    generation = tmp_path / "generation"
    with pytest.raises(ValueError, match="Execution roots overlap"):
        validate_output_roots(
            repo_root=ROOT,
            generation_root=generation,
            replay_root=generation / "replay",
            acceptance_root=tmp_path / "acceptance",
            require_absent=False,
        )
    with pytest.raises(ValueError, match="protected path"):
        validate_output_roots(
            repo_root=ROOT,
            generation_root=ROOT / "forbidden",
            replay_root=tmp_path / "replay",
            acceptance_root=tmp_path / "acceptance",
            require_absent=False,
        )
    overlapping = dict(proposal.OUTPUT_ROOTS)
    overlapping["production_replay"] = str(
        Path(overlapping["production_generation"]) / "replay"
    )
    with pytest.raises(ValueError, match="output roots overlap"):
        proposal.validate_output_roots(overlapping)
    protected = dict(proposal.OUTPUT_ROOTS)
    protected["production_generation"] = proposal.FROZEN_ROOTS[0]
    with pytest.raises(ValueError, match="frozen evidence"):
        proposal.validate_output_roots(protected)
    assert not generation.exists()
