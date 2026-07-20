from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from satnet.experiments.final_generation import contract as contract_module
from satnet.experiments.final_generation.cli import build_parser
from satnet.experiments.final_generation.constants import FROZEN_ARTIFACTS
from satnet.experiments.final_generation.contract import (
    ensure_mode_root,
    validate_frozen_contract,
    validate_initial_untracked_entries,
)
from satnet.experiments.final_generation.io import read_canonical_json, tree_inventory
from satnet.experiments.final_generation.mapping import map_all_runs
from satnet.experiments.final_generation.orchestrator import generate_run, run_directory
from satnet.experiments.final_generation.repeat import compare_repeat
from satnet.experiments.final_generation.replay import replay_run_read_only


def test_extra_and_missing_initial_untracked_entries_rejected() -> None:
    expected = {
        "data/example.bin",
        "docs/refactor_plans/2026-07-15_tier1_validity_remediation_atomic_gameplan.md",
        "docs/validation/tier1_defect_verification.md",
    }
    validate_initial_untracked_entries(expected)
    with pytest.raises(ValueError, match="extra"):
        validate_initial_untracked_entries((*expected, "unexpected.txt"))
    with pytest.raises(ValueError, match="missing"):
        validate_initial_untracked_entries(expected - {next(iter(expected - {"data/example.bin"}))})


def test_modified_and_missing_frozen_artifact_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = contract_module.contract_root()
    copied = tmp_path / "contract"
    shutil.copytree(source, copied)
    monkeypatch.setattr(contract_module, "contract_root", lambda: copied)
    target = copied / "contract_specification.json"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError):
        validate_frozen_contract(compare_tag_blobs=True)
    shutil.copy2(source / "contract_specification.json", target)
    (copied / "golden_vectors.json").unlink()
    with pytest.raises(FileNotFoundError):
        validate_frozen_contract(compare_tag_blobs=True)


def test_all_frozen_artifacts_are_compared_as_raw_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    original = contract_module._git

    def recording_git(*arguments: str, **kwargs):
        if arguments and arguments[0] == "show":
            calls.append(arguments[1])
        return original(*arguments, **kwargs)

    monkeypatch.setattr(contract_module, "_git", recording_git)
    validate_frozen_contract(compare_tag_blobs=True)
    assert len(calls) == 11
    assert {value.rsplit("/", 1)[-1] for value in calls} == set(FROZEN_ARTIFACTS)


def test_stage_failure_preserves_immutable_attempt_and_stops_downstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = validate_frozen_contract(compare_tag_blobs=False)
    mapping = map_all_runs(contract)[0]
    root = tmp_path / "qualification"

    def fail_rollout(_config):
        raise RuntimeError("sanitized satellite failure")

    monkeypatch.setattr(
        "satnet.experiments.final_generation.orchestrator.run_tier1_rollout",
        fail_rollout,
    )
    with pytest.raises(RuntimeError, match="satellite failure"):
        generate_run(
            mapping=mapping,
            catalog=contract_module.validate_catalog(),
            output_root=root,
            mode="qualification",
        )
    assert not run_directory(root, 0).exists()
    attempt = read_canonical_json(
        root / "operational" / "attempts" / "run_000" / "attempt_001.json"
    )
    assert attempt["state"] == "failed"
    assert attempt["failed_stage"] == "satellite"
    assert attempt["run_record_hash"] == mapping.run["run_record_hash"]
    assert "failure_evidence_hash" in attempt
    preserved = tree_inventory(root / "operational" / "failed_attempts" / attempt["attempt_id"])
    with pytest.raises(ValueError, match="retry"):
        generate_run(
            mapping=mapping,
            catalog=contract_module.validate_catalog(),
            output_root=root,
            mode="qualification",
        )
    assert tree_inventory(root / "operational" / "failed_attempts" / attempt["attempt_id"]) == preserved


def test_repeat_comparison_detects_single_byte_mutation(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    repeat = tmp_path / "repeat"
    ensure_mode_root(primary, "qualification", create=True)
    ensure_mode_root(repeat, "qualification_repeat", create=True)
    source_run = run_directory(primary, 200)
    repeat_run = run_directory(repeat, 200)
    from satnet.experiments.final_generation.constants import RUN_FILES, SCIENTIFIC_FILE_KEYS

    for key in (*SCIENTIFIC_FILE_KEYS, "inventory", "result"):
        for root in (source_run, repeat_run):
            path = root / RUN_FILES[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{key}\n".encode())
    assert compare_repeat(primary_root=primary, repeat_root=repeat)["deterministic_match"]
    (repeat_run / RUN_FILES["target"]).write_bytes(b"mutation\n")
    with pytest.raises(ValueError, match="target"):
        compare_repeat(primary_root=primary, repeat_root=repeat)


def test_replay_failure_keeps_input_tree_unchanged_and_rejects_nested_root(tmp_path: Path) -> None:
    contract = validate_frozen_contract(compare_tag_blobs=False)
    mapping = map_all_runs(contract)[0]
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    ensure_mode_root(primary, "qualification", create=True)
    source_run = run_directory(primary, 0)
    source_run.mkdir()
    (source_run / "sentinel.bin").write_bytes(b"immutable")
    before = tree_inventory(source_run)
    with pytest.raises(ValueError, match="Read-only replay failed"):
        replay_run_read_only(
            mapping=mapping,
            catalog=contract_module.validate_catalog(),
            input_root=primary,
            replay_output_root=replay,
        )
    assert tree_inventory(source_run) == before
    report = read_canonical_json(run_directory(replay, 0) / "replay_report.json")
    assert report["input_tree_unchanged"] is True
    with pytest.raises(ValueError, match="intersect"):
        replay_run_read_only(
            mapping=mapping,
            catalog=contract_module.validate_catalog(),
            input_root=primary,
            replay_output_root=source_run / "nested_replay",
        )


def test_production_cli_requires_all_confirmations() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["production"])
    parsed = parser.parse_args(
        [
            "production", "--output-root", "X:/production", "--confirm-production",
            "--confirm-contract-spec-hash", "0" * 64, "--confirm-run-count", "500",
            "--expected-tooling-sha", "1" * 40,
        ]
    )
    assert parsed.confirm_production is True
    assert parsed.confirm_run_count == 500
