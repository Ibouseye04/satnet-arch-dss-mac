from __future__ import annotations

from pathlib import Path
import stat
from typing import Any

import pytest

import satnet.experiments.stage_a_contract.freeze as freeze
from satnet.experiments.stage_a_execution.common import sha256_file
from satnet.experiments.stage_a_execution.contract import frozen_contract_output_root_states
from satnet.experiments.stage_a_execution.ledger import build_ledger, write_ledger
from satnet.experiments.stage_a_execution.plan import build_plan
import satnet.experiments.stage_a_execution.preflight as preflight_module
from satnet.experiments.stage_a_execution.preflight import run_preflight

from .conftest import (
    ARTIFACT_CONTRACT,
    EXECUTABLE_INVENTORY,
    STABLE_EXECUTABLE_COMMIT,
    TOOLING_PROPOSAL,
    make_authorization,
    source_provenance_from_ledger,
)

ROOT = Path(__file__).parents[3]


def _plan(contract: Any, operation: str, roots: tuple[Path, Path, Path], authorization: Any) -> dict[str, Any]:
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


def _generation_ledger(contract: Any, roots: tuple[Path, Path, Path]) -> tuple[str, int, str]:
    authorization = make_authorization(
        contract,
        operation="GENERATE",
        partition="development",
        run_ids=[1, 2],
        generation_root=roots[0],
        replay_root=roots[1],
        acceptance_root=roots[2],
    )
    plan = _plan(contract, "GENERATE", roots, authorization)
    roots[0].mkdir()
    path = roots[0] / "execution_ledger.json"
    write_ledger(path, build_ledger(plan, authorization.sha256), overwrite=False)
    return path.name, path.stat().st_size, sha256_file(path)


def _replay_context(contract: Any, roots: tuple[Path, Path, Path], identity: tuple[str, int, str]):
    authorization = make_authorization(
        contract,
        operation="REPLAY",
        partition="development",
        run_ids=[1, 2],
        generation_root=roots[0],
        replay_root=roots[1],
        acceptance_root=roots[2],
        source_generation_ledger=identity,
    )
    return authorization, _plan(contract, "REPLAY", roots, authorization)


def _patch_preflight(
    monkeypatch: pytest.MonkeyPatch,
    contract: Any,
    roots: tuple[Path, Path, Path],
) -> list[str]:
    operations: list[str] = []
    monkeypatch.setattr(
        preflight_module,
        "verify_executable_identity",
        lambda _: {
            "stable_executable_commit": STABLE_EXECUTABLE_COMMIT,
            "executable_inventory_sha256": EXECUTABLE_INVENTORY,
        },
    )

    def load_contract(_: Path, __: Path, operation: str):
        operations.append(operation)
        return contract

    monkeypatch.setattr(preflight_module, "load_frozen_contract", load_contract)
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
    monkeypatch.setattr(
        preflight_module,
        "load_source_generation_provenance",
        lambda *_args, **_kwargs: source_provenance_from_ledger(
            roots[0] / "execution_ledger.json"
        ),
    )
    return operations


def _run(contract: Any, roots: tuple[Path, Path, Path], authorization: Any, plan: dict[str, Any]):
    return run_preflight(
        repo_root=ROOT,
        contract=contract,
        plan=plan,
        authorization=authorization,
        generation_root=roots[0],
        replay_root=roots[1],
        acceptance_root=roots[2],
        check_write_probe=False,
        minimum_free_bytes=0,
    )


def test_replay_preflight_accepts_existing_bound_generation_source_only(
    synthetic_contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    identity = _generation_ledger(synthetic_contract, roots)
    source_artifact = roots[0] / "source-evidence.bin"
    source_artifact.write_bytes(b"immutable-source-evidence")
    source_artifact.chmod(stat.S_IREAD)
    source_before = (source_artifact.read_bytes(), source_artifact.stat().st_mode)
    authorization, plan = _replay_context(synthetic_contract, roots, identity)
    operations = _patch_preflight(monkeypatch, synthetic_contract, roots)

    certificate = _run(synthetic_contract, roots, authorization, plan)

    assert certificate.report["preflight"] == "PASSED"
    assert certificate.report["source_ledgers"]["generation"] == {
        "relative_path": identity[0],
        "byte_length": identity[1],
        "sha256": identity[2],
    }
    assert certificate.report["simulation_executed"] is False
    assert operations == ["REPLAY"]
    assert source_before == (source_artifact.read_bytes(), source_artifact.stat().st_mode)
    assert roots[0].is_dir()
    assert not roots[1].exists()
    assert not roots[2].exists()


def test_replay_preflight_rejects_missing_generation_root(
    synthetic_contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    authorization, plan = _replay_context(
        synthetic_contract,
        roots,
        ("execution_ledger.json", 1, "a" * 64),
    )
    _patch_preflight(monkeypatch, synthetic_contract, roots)

    with pytest.raises(FileExistsError, match="Output-root state mismatch for REPLAY"):
        _run(synthetic_contract, roots, authorization, plan)
    assert not any(root.exists() for root in roots)


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_replay_preflight_rejects_missing_or_changed_bound_generation_ledger(
    synthetic_contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    identity = _generation_ledger(synthetic_contract, roots)
    authorization, plan = _replay_context(synthetic_contract, roots, identity)
    ledger_path = roots[0] / identity[0]
    if mutation == "missing":
        ledger_path.unlink()
    else:
        ledger_path.write_bytes(ledger_path.read_bytes() + b"\n")
    _patch_preflight(monkeypatch, synthetic_contract, roots)

    with pytest.raises((FileNotFoundError, ValueError)):
        _run(synthetic_contract, roots, authorization, plan)
    assert roots[0].is_dir()
    assert not roots[1].exists()
    assert not roots[2].exists()


def test_replay_preflight_rejects_existing_replay_root(
    synthetic_contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    identity = _generation_ledger(synthetic_contract, roots)
    authorization, plan = _replay_context(synthetic_contract, roots, identity)
    roots[1].mkdir()
    _patch_preflight(monkeypatch, synthetic_contract, roots)

    with pytest.raises(FileExistsError, match="Output-root state mismatch for REPLAY"):
        _run(synthetic_contract, roots, authorization, plan)
    assert roots[1].is_dir()
    assert not roots[2].exists()


def test_replay_preflight_rejects_overlapping_roots(
    synthetic_contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = tmp_path / "generation"
    roots = (generation, generation / "replay", tmp_path / "acceptance")
    identity = _generation_ledger(synthetic_contract, roots)
    authorization, plan = _replay_context(synthetic_contract, roots, identity)
    _patch_preflight(monkeypatch, synthetic_contract, roots)

    with pytest.raises(ValueError, match="Execution roots overlap"):
        _run(synthetic_contract, roots, authorization, plan)
    assert roots[0].is_dir()
    assert not roots[1].exists()
    assert not roots[2].exists()


def test_generate_preflight_still_rejects_existing_generation_root(
    synthetic_contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    roots[0].mkdir()
    authorization = make_authorization(
        synthetic_contract,
        operation="GENERATE",
        partition="development",
        run_ids=[1, 2],
        generation_root=roots[0],
        replay_root=roots[1],
        acceptance_root=roots[2],
    )
    plan = _plan(synthetic_contract, "GENERATE", roots, authorization)
    operations = _patch_preflight(monkeypatch, synthetic_contract, roots)

    with pytest.raises(FileExistsError, match="Output-root state mismatch for GENERATE"):
        _run(synthetic_contract, roots, authorization, plan)
    assert operations == ["GENERATE"]
    assert roots[0].is_dir()
    assert not roots[1].exists()
    assert not roots[2].exists()


def test_frozen_contract_root_policy_is_operation_aware_without_weakening_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = tmp_path / "generation"
    replay = tmp_path / "replay"
    acceptance = tmp_path / "acceptance"
    evidence = tmp_path / "freeze"
    generation.mkdir()
    monkeypatch.setattr(
        freeze,
        "OUTPUT_ROOTS",
        {
            "production_generation": str(generation),
            "production_replay": str(replay),
            "production_acceptance": str(acceptance),
            "evidence_freeze": str(evidence),
        },
    )

    states = freeze.verify_output_roots(
        ROOT,
        expected_states=frozen_contract_output_root_states("REPLAY"),
    )
    assert states == frozen_contract_output_root_states("REPLAY")
    with pytest.raises(ValueError, match="output root exists: production_generation"):
        freeze.verify_output_roots(
            ROOT,
            expected_states=frozen_contract_output_root_states("GENERATE"),
        )

    monkeypatch.setitem(freeze.OUTPUT_ROOTS, "production_replay", str(generation / "replay"))
    with pytest.raises(ValueError, match="overlaps protected path"):
        freeze.verify_output_roots(
            ROOT,
            expected_states=frozen_contract_output_root_states("REPLAY"),
        )
