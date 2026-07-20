from __future__ import annotations

from pathlib import Path

import pytest

from satnet.experiments.final_generation import acceptance as acceptance_module
from satnet.experiments.final_generation.acceptance import (
    validate_production_acceptance,
    validate_qualification,
)
from satnet.experiments.final_generation.constants import QUALIFICATION_RUN_IDS
from satnet.experiments.final_generation.contract import validate_frozen_contract
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
        "validate_completed_run",
        lambda **kwargs: {"run_result_hash": f"{kwargs['mapping'].run_id:064x}"},
    )
    monkeypatch.setattr(
        acceptance_module,
        "ensure_mode_root",
        lambda root, mode, create: Path(root),
    )

    def read_evidence(path: Path):
        if path.name == "generation_ledger.json":
            return {
                "distinct_frozen_run_submission_count": 15,
                "successful_generation_count": 15,
            }
        if path.name == "replay_ledger.json":
            return {"replay_submission_count": 15, "successful_replay_count": 15}
        return {
            "expected_result_hash": f"{int(path.parent.name[-3:]):064x}",
            "input_tree_unchanged": True,
            "replay_state": "succeeded",
        }

    monkeypatch.setattr(acceptance_module, "read_canonical_json", read_evidence)
    monkeypatch.setattr(acceptance_module, "_target_values", lambda path: _target(int(path.parents[1].name[-3:])))
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


def test_production_acceptance_uses_frozen_gates_and_rejects_missing_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mappings = _mappings()
    original_contract = validate_frozen_contract(compare_tag_blobs=False)
    monkeypatch.setattr(
        acceptance_module,
        "validate_frozen_contract",
        lambda **kwargs: original_contract,
    )
    generation = {
        "distinct_frozen_run_submission_count": 499,
        "successful_generation_count": 500,
    }
    replay = {"replay_submission_count": 500, "successful_replay_count": 500}
    with pytest.raises(ValueError, match="submission"):
        validate_production_acceptance(
            mappings=mappings,
            generation_ledger=generation,
            replay_ledger=replay,
            generation_root=tmp_path,
            protected_science_diff_empty=True,
        )


def test_production_acceptance_rejects_protected_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = validate_frozen_contract(compare_tag_blobs=False)
    monkeypatch.setattr(acceptance_module, "validate_frozen_contract", lambda **kwargs: contract)
    generation = {
        "distinct_frozen_run_submission_count": 500,
        "successful_generation_count": 500,
    }
    replay = {"replay_submission_count": 500, "successful_replay_count": 500}
    with pytest.raises(ValueError, match="Protected-science"):
        validate_production_acceptance(
            mappings=_mappings(),
            generation_ledger=generation,
            replay_ledger=replay,
            generation_root=tmp_path,
            protected_science_diff_empty=False,
        )
