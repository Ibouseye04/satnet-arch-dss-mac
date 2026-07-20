from __future__ import annotations

from pathlib import Path

import pytest

from satnet.experiments.final_generation import acceptance as acceptance_module
from satnet.experiments.final_generation.acceptance import (
    validate_production_acceptance,
    validate_qualification,
)
from satnet.experiments.final_generation.constants import QUALIFICATION_RUN_IDS
from satnet.experiments.final_generation.contract import ensure_mode_root, validate_frozen_contract
from satnet.experiments.final_generation.io import atomic_write_json
from satnet.experiments.final_generation.mapping import map_all_runs


def _mappings():
    return map_all_runs(validate_frozen_contract(compare_tag_blobs=False))


def _target(run_id: int) -> dict[str, bool | float]:
    value = (run_id % 97) / 100.0
    return {
        "overall_threshold_breach_any": bool(run_id % 2),
        "ground_threshold_breach_any": bool(run_id % 3),
        "space_threshold_breach_any": bool(run_id % 5),
        "failure_adjusted_overall_service_fraction_mean": value,
        "failure_adjusted_overall_service_fraction_min": value,
        "failure_adjusted_ground_service_fraction_min": value,
        "space_gcc_fraction_original_min": value,
        "ground_service_loss_due_to_failures_max": value,
    }


def test_qualification_subset_passes_without_claiming_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tuple(_mappings()[run_id] for run_id in QUALIFICATION_RUN_IDS)
    monkeypatch.setattr(
        acceptance_module,
        "ensure_mode_root",
        lambda root, mode, create: Path(root),
    )
    monkeypatch.setattr(
        acceptance_module,
        "read_canonical_json",
        lambda path: {"records": []},
    )
    monkeypatch.setattr(acceptance_module, "validate_catalog", lambda: object())
    results = {
        mapping.run_id: {"run_result_hash": f"{mapping.run_id:064x}"}
        for mapping in selected
    }
    targets = {mapping.run_id: _target(mapping.run_id) for mapping in selected}
    monkeypatch.setattr(
        acceptance_module,
        "validate_generation_evidence",
        lambda **kwargs: (targets, results),
    )
    monkeypatch.setattr(
        acceptance_module,
        "validate_replay_evidence",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(acceptance_module, "_validate_ground_consistency", lambda *args: None)
    result = validate_qualification(
        mappings=selected,
        generation_root=tmp_path / "generation",
        replay_root=tmp_path / "replay",
    )
    assert result["generation_count"] == 15
    assert result["replay_count"] == 15
    assert result["production_acceptance_claimed"] is False


def test_qualification_rejects_wrong_subset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="15-run set"):
        validate_qualification(
            mappings=_mappings()[:15],
            generation_root=tmp_path / "generation",
            replay_root=tmp_path / "replay",
        )


def test_production_acceptance_rejects_forged_counts_without_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mappings = _mappings()
    original_contract = validate_frozen_contract(compare_tag_blobs=False)
    monkeypatch.setattr(
        acceptance_module,
        "validate_frozen_contract",
        lambda **kwargs: original_contract,
    )
    generation_root = tmp_path / "generation"
    replay_root = tmp_path / "replay"
    ensure_mode_root(generation_root, "production", create=True)
    ensure_mode_root(replay_root, "production_replay", create=True)
    atomic_write_json(
        generation_root / "operational" / "generation_ledger.json",
        {
            "contract_spec_hash": original_contract["contract_spec_hash"],
            "distinct_frozen_run_submission_count": 500,
            "operational_attempt_event_count": 500,
            "records": [],
            "successful_generation_count": 500,
        },
    )
    atomic_write_json(
        replay_root / "replay_ledger.json",
        {
            "contract_spec_hash": original_contract["contract_spec_hash"],
            "records": [],
            "replay_submission_count": 500,
            "successful_replay_count": 500,
        },
    )
    with pytest.raises(ValueError, match="generation run set mismatch"):
        validate_production_acceptance(
            mappings=mappings,
            generation_root=generation_root,
            replay_root=replay_root,
            protected_science_diff_empty=True,
        )


def test_production_acceptance_rejects_protected_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = validate_frozen_contract(compare_tag_blobs=False)
    monkeypatch.setattr(acceptance_module, "validate_frozen_contract", lambda **kwargs: contract)
    with pytest.raises(ValueError, match="Protected-science"):
        validate_production_acceptance(
            mappings=_mappings(),
            generation_root=tmp_path / "generation",
            replay_root=tmp_path / "replay",
            protected_science_diff_empty=False,
        )
