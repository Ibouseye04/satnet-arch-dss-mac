from __future__ import annotations

import json
from pathlib import Path

import pytest

from satnet.experiments.stage_a_execution import cli
from satnet.experiments.stage_a_execution.authorization import validate_authorization
from satnet.experiments.stage_a_execution.common import atomic_write_bytes, sha256_file
from satnet.experiments.stage_a_execution.contract import load_frozen_contract
from satnet.experiments.stage_a_execution.ledger import build_ledger, write_ledger
from satnet.experiments.stage_a_execution.paths import validate_output_roots, validate_relative_artifact_path
from satnet.experiments.stage_a_execution.plan import build_plan
import satnet.experiments.stage_a_execution.preflight as preflight_module
from satnet.experiments.stage_a_execution.preflight import run_preflight, tooling_identity

from .conftest import ARTIFACT_CONTRACT, EXECUTABLE_INVENTORY, STABLE_EXECUTABLE_COMMIT, TOOLING_PROPOSAL, make_authorization

ROOT = Path(__file__).parents[3]
PROPOSAL = ROOT / "artifacts/stage_a_execution_tooling_v1_proposal"


def test_path_traversal_absolute_overlap_existing_and_repository_denied(tmp_path: Path) -> None:
    for value in ("../escape", "/absolute", "a\\b", ""):
        with pytest.raises(ValueError):
            validate_relative_artifact_path(value)
    generation = tmp_path / "generation"
    generation.mkdir()
    with pytest.raises(FileExistsError):
        validate_output_roots(repo_root=ROOT, generation_root=generation, replay_root=tmp_path / "replay", acceptance_root=tmp_path / "acceptance", require_absent=True)
    with pytest.raises(ValueError, match="overlap"):
        validate_output_roots(repo_root=ROOT, generation_root=tmp_path / "x", replay_root=tmp_path / "x/replay", acceptance_root=tmp_path / "acceptance", require_absent=False)
    with pytest.raises(ValueError, match="protected"):
        validate_output_roots(repo_root=ROOT, generation_root=ROOT / "output", replay_root=tmp_path / "replay", acceptance_root=tmp_path / "acceptance", require_absent=False)


def test_preflight_checks_authorization_roots_dependencies_and_lock(synthetic_contract, tmp_path: Path, monkeypatch) -> None:
    roots = [tmp_path / name for name in ("generation", "replay", "acceptance")]
    monkeypatch.setattr(preflight_module, "verify_executable_identity", lambda _: {
        "stable_executable_commit": STABLE_EXECUTABLE_COMMIT,
        "executable_inventory_sha256": EXECUTABLE_INVENTORY,
    })
    monkeypatch.setattr(preflight_module, "load_frozen_contract", lambda *_: synthetic_contract)
    monkeypatch.setattr(preflight_module, "verify_frozen_production_evidence", lambda *_: {
        "combined": {"file_count": 9004, "byte_count": 1337549193, "verified_sha256_count": 9004},
    })
    plan = build_plan(
        synthetic_contract, partition="development", operation="GENERATE",
        stable_executable_commit=STABLE_EXECUTABLE_COMMIT, executable_inventory_hash=EXECUTABLE_INVENTORY,
        tooling_proposal_hash=TOOLING_PROPOSAL, artifact_contract_hash=ARTIFACT_CONTRACT,
        generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
    )
    authorization = make_authorization(
        synthetic_contract, operation="GENERATE", partition="development", run_ids=[1, 2],
        generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
    )
    certificate = run_preflight(
        repo_root=ROOT, contract=synthetic_contract, plan=plan, authorization=authorization,
        generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2], minimum_free_bytes=0,
    )
    assert certificate.report["preflight"] == "PASSED"
    assert certificate.report["simulation_executed"] is False
    assert not any(root.exists() for root in roots)
    roots[0].with_name(roots[0].name + ".lock").write_text("locked", encoding="utf-8")
    with pytest.raises(RuntimeError, match="lock"):
        run_preflight(
            repo_root=ROOT, contract=synthetic_contract, plan=plan, authorization=authorization,
            generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2], minimum_free_bytes=0,
        )
    roots[0].with_name(roots[0].name + ".lock").unlink()


def test_atomic_write_failure_preserves_existing_target(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"original")
    def fail_replace(source, destination) -> None:
        raise OSError("synthetic interruption")
    monkeypatch.setattr("satnet.experiments.stage_a_execution.common.os.replace", fail_replace)
    with pytest.raises(OSError, match="interruption"):
        atomic_write_bytes(target, b"changed", overwrite=True)
    assert target.read_bytes() == b"original"
    assert list(tmp_path.glob("*.tmp")) == []


def test_cli_plan_default_deny_and_holdout_redaction(tmp_path: Path, capsys) -> None:
    roots = [tmp_path / name for name in ("generation", "replay", "acceptance")]
    code = cli.main(["plan", "--partition", "development", "--generation-root", str(roots[0]), "--replay-root", str(roots[1]), "--acceptance-root", str(roots[2])])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["authorization_status"] == "EXECUTION NOT AUTHORIZED"
    assert result["execution_authorized"] is False
    assert result["sealed_holdout"]["identities"] == "REDACTED"
    with pytest.raises(SystemExit) as error:
        cli.main(["generate", "--partition", "development", "--generation-root", str(roots[0]), "--replay-root", str(roots[1]), "--acceptance-root", str(roots[2])])
    assert error.value.code == 2
    assert not any(root.exists() for root in roots)


def test_cli_status_never_lists_holdout_identities(synthetic_contract, tmp_path: Path, capsys) -> None:
    roots = [tmp_path / name for name in ("generation", "replay", "acceptance")]
    plan = build_plan(synthetic_contract, partition="development", operation="GENERATE", stable_executable_commit=STABLE_EXECUTABLE_COMMIT, executable_inventory_hash=EXECUTABLE_INVENTORY, tooling_proposal_hash=TOOLING_PROPOSAL, artifact_contract_hash=ARTIFACT_CONTRACT, generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2])
    ledger = build_ledger(plan, "a" * 64)
    path = tmp_path / "ledger.json"
    write_ledger(path, ledger, overwrite=False)
    assert cli.main(["status", "--ledger", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["sealed_holdout_identities"] == "REDACTED"
    assert "SYN-D002" not in json.dumps(result)


def test_tooling_identity_uses_last_executable_change() -> None:
    commit, executable_inventory_hash, proposal_hash = tooling_identity(ROOT)
    assert len(commit) == 40
    assert executable_inventory_hash == sha256_file(PROPOSAL / "stage_a_executable_source_inventory.json")
    assert proposal_hash == sha256_file(PROPOSAL / "stage_a_execution_tooling_inventory.json")


def test_tooling_proposal_has_no_authorization_and_false_flags() -> None:
    specification = json.loads((PROPOSAL / "stage_a_execution_tooling_specification.json").read_text(encoding="utf-8"))
    assert specification["status"] == "TOOLING_PROPOSAL"
    assert specification["execution_authorized"] is False
    assert specification["simulation_authorized"] is False
    assert specification["production_authorized"] is False
    assert not list(PROPOSAL.glob("*authorization*.json")) or [path.name for path in PROPOSAL.glob("*authorization*.json")] == ["stage_a_execution_authorization_schema.json"]


def test_inventory_is_canonical_ordered_and_self_excluding() -> None:
    inventory_path = PROPOSAL / "stage_a_execution_tooling_inventory.json"
    value = json.loads(inventory_path.read_text(encoding="utf-8"))
    paths = [record["relative_path"] for record in value["artifacts"]]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    assert inventory_path.relative_to(ROOT).as_posix() not in paths
    for record in value["artifacts"]:
        path = ROOT / record["relative_path"]
        assert path.stat().st_size == record["byte_length"]
        assert sha256_file(path) == record["sha256"]
