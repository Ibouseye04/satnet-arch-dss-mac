from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from satnet.experiments.final_generation import cli, contract as contract_module
from satnet.experiments.final_generation.constants import (
    CONTRACT_SPEC_HASH,
    FROZEN_ARTIFACTS,
    SUPPORTED_MODES,
)
from satnet.experiments.final_generation.contract import (
    git_common_directory,
    mode_marker,
    repository_worktree_roots,
    validate_frozen_contract,
    validate_output_root,
)
from satnet.experiments.final_generation.evidence import (
    validate_generation_evidence,
    validate_replay_evidence,
)
from satnet.experiments.final_generation.io import atomic_write_json
from satnet.experiments.final_generation.mapping import FinalRunMapping, map_all_runs
from satnet.experiments.final_generation.orchestrator import (
    _validate_resume_certificate,
    _validate_retry_identities,
    attempt_input_identity,
    generate_run,
    validate_completed_run,
)

ROOT = Path(__file__).parents[2]


def _mapping() -> FinalRunMapping:
    return map_all_runs(validate_frozen_contract(compare_tag_blobs=False))[0]


def _generation_fixture(tmp_path: Path, mapping: FinalRunMapping) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "generation"
    run_root = root / "run_000"
    (run_root / "targets").mkdir(parents=True)
    target_hash = "1" * 64
    inventory_hash = "2" * 64
    result_hash = "3" * 64
    atomic_write_json(run_root / "targets" / "target.json", {"target_artifact_hash": target_hash})
    atomic_write_json(run_root / "scientific_inventory.json", {"scientific_inventory_hash": inventory_hash})
    identity = attempt_input_identity(mapping)
    from satnet.ground.canonical import canonical_hash

    attempt = {
        "attempt_input_identity": identity,
        "attempt_input_identity_hash": canonical_hash(identity),
        "published_result_hash": result_hash,
        "state": "succeeded",
    }
    atomic_write_json(root / "operational" / "attempts" / "run_000" / "attempt_001.json", attempt)
    atomic_write_json(
        root / "operational" / "current_state" / "run_000.json",
        {
            "published_result_hash": result_hash,
            "run_record_hash": mapping.run["run_record_hash"],
            "state": "succeeded",
        },
    )
    record = {
        "attempt_count": 1,
        **identity,
        "failure_evidence_hashes": [],
        "published_result_hash": result_hash,
        "scientific_inventory_hash": inventory_hash,
        "state": "succeeded",
        "target_artifact_hash": target_hash,
    }
    ledger = {
        "contract_spec_hash": CONTRACT_SPEC_HASH,
        "distinct_frozen_run_submission_count": 1,
        "operational_attempt_event_count": 1,
        "records": [record],
        "successful_generation_count": 1,
    }
    return root, ledger


def _production_namespace(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "acceptance_report": Path("C:/audit/acceptance.json"),
        "confirm_contract_spec_hash": CONTRACT_SPEC_HASH,
        "confirm_production": True,
        "confirm_run_count": 500,
        "expected_tooling_sha": "a" * 40,
        "input_root": Path("C:/audit/generation"),
        "output_root": Path("C:/audit/generation"),
        "replay_output_root": Path("C:/audit/replay"),
        "replay_root": Path("C:/audit/replay"),
        "resume_replay_root": None,
        "retry": False,
        "verified_resume": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_frozen_paths_are_binary_checkout_inputs() -> None:
    text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "artifacts/final_integrated_dataset_contract/** -text" in text
    assert "artifacts/integrated_ground_pilot_25/inputs/pilot_catalog.csv -text" in text
    paths = [
        f"artifacts/final_integrated_dataset_contract/{name}" for name in FROZEN_ARTIFACTS
    ]
    paths.append("artifacts/integrated_ground_pilot_25/inputs/pilot_catalog.csv")
    output = subprocess.run(
        ["git", "check-attr", "text", "--", *paths],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert len(output.splitlines()) == 12
    assert all(line.endswith("text: unset") for line in output.splitlines())


def test_mode_contract_includes_exact_production_replay_marker() -> None:
    assert SUPPORTED_MODES == {
        "production",
        "production_replay",
        "qualification",
        "qualification_repeat",
        "qualification_replay",
    }
    assert mode_marker("production_replay") == {
        "contract_spec_hash": CONTRACT_SPEC_HASH,
        "execution_mode": "production_replay",
        "mode_marker_schema_version": "1",
    }


def test_repository_family_discovery_includes_current_root_and_git_common_dir() -> None:
    assert ROOT.resolve() in repository_worktree_roots()
    assert git_common_directory().is_dir()


def test_repository_family_paths_and_git_common_dir_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    common = tmp_path / "git-common"
    for path in (first, second, common):
        path.mkdir()
    monkeypatch.setattr(contract_module, "repository_worktree_roots", lambda: (first, second))
    monkeypatch.setattr(contract_module, "git_common_directory", lambda: common)
    prohibited = (
        first,
        first / "artifacts" / "final_integrated_dataset_contract",
        first / "data",
        first / "docs" / "refactor_plans" / "2026-07-15_tier1_validity_remediation_atomic_gameplan.md",
        first / "docs" / "validation" / "tier1_defect_verification.md",
        first / "src" / "satnet" / "ground",
        second,
        common,
    )
    for path in prohibited:
        with pytest.raises(ValueError, match="repository-family"):
            validate_output_root(path)


@pytest.mark.parametrize(
    ("container", "field", "value"),
    [
        ("run", "run_record_hash", "a" * 64),
        ("design", "design_record_hash", "b" * 64),
        ("run", "satellite_seed", 999),
        ("run", "ground_failure_seed", 999),
        ("run", "ground_selection_seed", 999),
        ("run", "split_assignment", "wrong"),
        ("design", "ground_design_hash", "c" * 64),
    ],
)
def test_retry_rejects_changed_frozen_identity(
    tmp_path: Path, container: str, field: str, value: object
) -> None:
    mapping = _mapping()
    identity = attempt_input_identity(mapping)
    from satnet.ground.canonical import canonical_hash

    atomic_write_json(
        tmp_path / "operational" / "attempts" / "run_000" / "attempt_001.json",
        {
            "attempt_input_identity": identity,
            "attempt_input_identity_hash": canonical_hash(identity),
            "state": "failed",
        },
    )
    if container == "run":
        source = dict(mapping.run)
        source[field] = value
        changed = replace(mapping, run=source)
    else:
        source = dict(mapping.design)
        source[field] = value
        changed = replace(mapping, design=source)
    with pytest.raises(ValueError, match="Attempt input identity"):
        _validate_retry_identities(tmp_path, changed)


def test_retry_accepts_exact_frozen_identity(tmp_path: Path) -> None:
    mapping = _mapping()
    identity = attempt_input_identity(mapping)
    from satnet.ground.canonical import canonical_hash

    atomic_write_json(
        tmp_path / "operational" / "attempts" / "run_000" / "attempt_001.json",
        {
            "attempt_input_identity": identity,
            "attempt_input_identity_hash": canonical_hash(identity),
            "state": "failed",
        },
    )
    _validate_retry_identities(tmp_path, mapping)


def test_generation_evidence_rejects_empty_counts_and_identity_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = _mapping()
    root, valid = _generation_fixture(tmp_path, mapping)
    monkeypatch.setattr(
        "satnet.experiments.final_generation.evidence.validate_completed_run",
        lambda **kwargs: {
            "run_result_hash": "3" * 64,
            "scientific_inventory_hash": "2" * 64,
        },
    )
    forged = dict(valid)
    forged["records"] = []
    forged["distinct_frozen_run_submission_count"] = 500
    forged["successful_generation_count"] = 500
    with pytest.raises(ValueError, match="generation run set mismatch"):
        validate_generation_evidence(
            mappings=(mapping,), generation_root=root, catalog=object(), ledger=forged
        )
    changed = {**valid, "records": [{**valid["records"][0], "satellite_seed": 999}]}
    with pytest.raises(ValueError, match="satellite_seed"):
        validate_generation_evidence(
            mappings=(mapping,), generation_root=root, catalog=object(), ledger=changed
        )


def test_generation_evidence_derives_valid_record_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = _mapping()
    root, ledger = _generation_fixture(tmp_path, mapping)
    monkeypatch.setattr(
        "satnet.experiments.final_generation.evidence.validate_completed_run",
        lambda **kwargs: {
            "run_result_hash": "3" * 64,
            "scientific_inventory_hash": "2" * 64,
        },
    )
    targets, results = validate_generation_evidence(
        mappings=(mapping,), generation_root=root, catalog=object(), ledger=ledger
    )
    assert set(targets) == {0}
    assert results[0]["run_result_hash"] == "3" * 64


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("run_record_hash", "9" * 64, "run_record_hash"),
        ("published_result_hash", "9" * 64, "result hash"),
        ("target_artifact_hash", "9" * 64, "target hash"),
        ("ground_failure_seed", 999, "ground_failure_seed"),
        ("split", "wrong", "split"),
    ],
)
def test_generation_evidence_rejects_wrong_record_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    match: str,
) -> None:
    mapping = _mapping()
    root, ledger = _generation_fixture(tmp_path, mapping)
    monkeypatch.setattr(
        "satnet.experiments.final_generation.evidence.validate_completed_run",
        lambda **kwargs: {
            "run_result_hash": "3" * 64,
            "scientific_inventory_hash": "2" * 64,
        },
    )
    ledger["records"] = [{**ledger["records"][0], field: value}]
    with pytest.raises(ValueError, match=match):
        validate_generation_evidence(
            mappings=(mapping,), generation_root=root, catalog=object(), ledger=ledger
        )


def test_generation_evidence_rejects_duplicate_extra_and_aggregate_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = _mapping()
    root, ledger = _generation_fixture(tmp_path, mapping)
    monkeypatch.setattr(
        "satnet.experiments.final_generation.evidence.validate_completed_run",
        lambda **kwargs: {
            "run_result_hash": "3" * 64,
            "scientific_inventory_hash": "2" * 64,
        },
    )
    record = ledger["records"][0]
    for records, match in (
        ([record, record], "Duplicate generation run ID"),
        ([record, {**record, "run_id": 499}], "extra"),
    ):
        changed = {**ledger, "records": records}
        with pytest.raises(ValueError, match=match):
            validate_generation_evidence(
                mappings=(mapping,), generation_root=root, catalog=object(), ledger=changed
            )
    changed = {**ledger, "successful_generation_count": 0}
    with pytest.raises(ValueError, match="aggregate disagrees"):
        validate_generation_evidence(
            mappings=(mapping,), generation_root=root, catalog=object(), ledger=changed
        )


def _replay_fixture(tmp_path: Path, mapping: FinalRunMapping) -> tuple[Path, dict[str, object]]:
    stages = [
        {"stage": name, "state": "matched"}
        for name in ("satellite", "g1", "g2", "g3", "g4", "g5", "target", "inventory", "result")
    ]
    record = {
        "after_input_tree_inventory_hash": "4" * 64,
        "before_input_tree_inventory_hash": "4" * 64,
        "contract_spec_hash": CONTRACT_SPEC_HASH,
        "design_record_hash": mapping.design["design_record_hash"],
        "expected_result_hash": "3" * 64,
        "first_mismatch": None,
        "input_result_hash": "3" * 64,
        "input_tree_unchanged": True,
        "per_stage_comparison": stages,
        "recomputed_result_hash": "3" * 64,
        "replay_report_schema_version": "2",
        "replay_state": "succeeded",
        "run_id": mapping.run_id,
        "run_key": mapping.run_key,
        "run_record_hash": mapping.run["run_record_hash"],
        "scientific_inventory_hash": "2" * 64,
        "target_artifact_hash": "1" * 64,
    }
    root = tmp_path / "replay"
    atomic_write_json(root / "run_000" / "replay_report.json", record)
    return root, {
        "contract_spec_hash": CONTRACT_SPEC_HASH,
        "records": [record],
        "replay_submission_count": 1,
        "successful_replay_count": 1,
    }


def test_replay_evidence_rejects_missing_and_mismatched_records(tmp_path: Path) -> None:
    mapping = _mapping()
    results = {0: {"run_result_hash": "3" * 64, "scientific_inventory_hash": "2" * 64}}
    targets = {0: {"target_artifact_hash": "1" * 64}}
    empty = {
        "contract_spec_hash": CONTRACT_SPEC_HASH,
        "records": [],
        "replay_submission_count": 500,
        "successful_replay_count": 500,
    }
    with pytest.raises(ValueError, match="replay run set mismatch"):
        validate_replay_evidence(
            mappings=(mapping,),
            replay_root=tmp_path,
            generation_results=results,
            generation_targets=targets,
            ledger=empty,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"run_record_hash": "9" * 64}, "Replay evidence mismatch"),
        ({"input_result_hash": "9" * 64}, "Replay evidence mismatch"),
        ({"input_tree_unchanged": False}, "Replay evidence mismatch"),
        ({"after_input_tree_inventory_hash": "9" * 64}, "Replay evidence mismatch"),
        ({"per_stage_comparison": []}, "Replay evidence mismatch"),
        ({"successful_replay_count": 0}, "aggregate disagrees"),
    ],
)
def test_replay_evidence_rejects_mismatch_and_aggregate_disagreement(
    tmp_path: Path, mutation: dict[str, object], match: str
) -> None:
    mapping = _mapping()
    root, ledger = _replay_fixture(tmp_path, mapping)
    if "successful_replay_count" in mutation:
        ledger.update(mutation)
    else:
        record = {**ledger["records"][0], **mutation}
        ledger["records"] = [record]
        atomic_write_json(root / "run_000" / "replay_report.json", record, overwrite=True)
    with pytest.raises(ValueError, match=match):
        validate_replay_evidence(
            mappings=(mapping,),
            replay_root=root,
            generation_results={0: {"run_result_hash": "3" * 64, "scientific_inventory_hash": "2" * 64}},
            generation_targets={0: {"target_artifact_hash": "1" * 64}},
            ledger=ledger,
        )


def test_replay_evidence_accepts_complete_record(tmp_path: Path) -> None:
    mapping = _mapping()
    root, ledger = _replay_fixture(tmp_path, mapping)
    result = validate_replay_evidence(
        mappings=(mapping,),
        replay_root=root,
        generation_results={0: {"run_result_hash": "3" * 64, "scientific_inventory_hash": "2" * 64}},
        generation_targets={0: {"target_artifact_hash": "1" * 64}},
        ledger=ledger,
    )
    assert set(result) == {0}


def test_production_commands_exist_and_require_confirmations() -> None:
    parser = cli.build_parser()
    choices = next(action.choices for action in parser._actions if action.choices)
    assert {"generate-production", "replay-production", "validate-production"}.issubset(choices)
    for handler in (
        cli.command_generate_production,
        cli.command_replay_production,
        cli.command_validate_production,
    ):
        with pytest.raises(ValueError, match="confirm-production"):
            handler(_production_namespace(confirm_production=False))


def test_production_replay_uses_production_mode_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping()
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "_load", lambda **kwargs: ({}, (mapping,)))
    monkeypatch.setattr(cli, "validate_catalog", lambda: object())

    def replay(**kwargs: object) -> tuple[dict[str, object], ...]:
        captured.update(kwargs)
        return ({"run_id": 0},)

    monkeypatch.setattr(cli, "replay_runs_read_only", replay)
    monkeypatch.setattr(cli, "materialize_replay_ledger", lambda **kwargs: {"records": []})
    monkeypatch.setattr(cli, "_print", lambda value: None)
    cli.command_replay_production(_production_namespace())
    assert captured["input_mode"] == "production"
    assert captured["output_mode"] == "production_replay"


def test_completed_run_requires_authoritative_stage_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = _mapping()
    monkeypatch.setattr(
        "satnet.experiments.final_generation.orchestrator.validate_run_authoritatively",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("G3 semantic corruption")),
    )
    with pytest.raises(ValueError, match="G3 semantic corruption"):
        validate_completed_run(mapping=mapping, catalog=object(), run_root=tmp_path)


def test_verified_resume_requires_matching_replay_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = _mapping()
    root = tmp_path / "qualification"
    from satnet.experiments.final_generation.contract import ensure_mode_root

    ensure_mode_root(root, "qualification", create=True)
    (root / "run_000").mkdir()
    monkeypatch.setattr(
        "satnet.experiments.final_generation.orchestrator.validate_completed_run",
        lambda **kwargs: {"run_result_hash": "3" * 64},
    )
    monkeypatch.setattr(
        "satnet.experiments.final_generation.orchestrator._validate_published_attempt",
        lambda **kwargs: None,
    )
    with pytest.raises(ValueError, match="replay certificate"):
        generate_run(
            mapping=mapping,
            catalog=object(),
            output_root=root,
            mode="qualification",
            verified_resume=True,
        )


def test_resume_certificate_rejects_missing_stage(tmp_path: Path) -> None:
    mapping = _mapping()
    root, ledger = _replay_fixture(tmp_path, mapping)
    report = ledger["records"][0]
    report["per_stage_comparison"] = report["per_stage_comparison"][:-1]
    atomic_write_json(root / "execution_mode.json", mode_marker("qualification_replay"))
    atomic_write_json(root / "run_000" / "replay_report.json", report, overwrite=True)
    with pytest.raises(ValueError, match="certificate mismatch"):
        _validate_resume_certificate(
            mapping=mapping,
            replay_root=root,
            result_hash="3" * 64,
            generation_mode="qualification",
        )
