from __future__ import annotations

from collections import Counter
import hashlib
import os
from pathlib import Path

import pytest

from satnet.experiments.final_class_support.analysis import (
    augmentation_contract,
    proposed_augmentation_designs,
)
from satnet.experiments.final_class_support.constants import (
    EXPECTED_CLASS_COUNTS,
    EXPECTED_NON_BREACH_RUN_IDS,
    PROPOSAL_LABEL,
)
from satnet.experiments.final_class_support.extraction import extract_corpus
from satnet.experiments.final_class_support.io import validate_output_root
from satnet.experiments.final_class_support.pipeline import run_analysis

ROOT = Path(__file__).parents[2]
GENERATION_ROOT = Path(
    os.environ.get("SATNET_FINAL_GENERATION_ROOT", "C:/Users/johns/satnet-final-production-20260720")
)
REPLAY_ROOT = Path(
    os.environ.get(
        "SATNET_FINAL_REPLAY_ROOT", "C:/Users/johns/satnet-final-production-replay-20260720"
    )
)
FREEZE_ROOT = Path(
    os.environ.get(
        "SATNET_FINAL_FREEZE_ROOT", "C:/Users/johns/satnet-final-production-v1-freeze-20260721"
    )
)


@pytest.fixture(scope="module")
def corpus() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not GENERATION_ROOT.is_dir():
        pytest.skip("Frozen production evidence is not available")
    return extract_corpus(GENERATION_ROOT)


def test_extracts_exact_frozen_run_and_design_sets(
    corpus: tuple[list[dict[str, object]], list[dict[str, object]]]
) -> None:
    runs, designs = corpus
    assert len(runs) == 500
    assert len(designs) == 100
    assert [row["run_id"] for row in runs] == list(range(500))
    assert [row["design_id"] for row in designs] == [f"D{index:03d}" for index in range(100)]
    assert len({row["run_key"] for row in runs}) == 500
    assert len({(row["design_id"], row["realization_id"]) for row in runs}) == 500


def test_reproduces_exact_splits_classes_and_non_breach_runs(
    corpus: tuple[list[dict[str, object]], list[dict[str, object]]]
) -> None:
    runs, designs = corpus
    assert Counter(row["split"] for row in runs) == {"train": 350, "validation": 75, "test": 75}
    assert Counter(row["split"] for row in designs) == {"train": 70, "validation": 15, "test": 15}
    for split, expected in EXPECTED_CLASS_COUNTS.items():
        observed = Counter(bool(row["overall_threshold_breach_any"]) for row in runs if row["split"] == split)
        assert {False: observed[False], True: observed[True]} == expected
    assert tuple(row["run_id"] for row in runs if not row["overall_threshold_breach_any"]) == EXPECTED_NON_BREACH_RUN_IDS


def test_boundary_polarity_and_authoritative_temporal_consistency(
    corpus: tuple[list[dict[str, object]], list[dict[str, object]]]
) -> None:
    runs, _ = corpus
    for row in runs:
        margin = float(row["overall_boundary_margin"])
        assert bool(row["overall_threshold_breach_any"]) is (margin < 0.0)
        assert int(row["sampled_state_count"]) == 11
        assert int(row["temporal_breach_count"]) == sum(
            float(value) < 0.80 for value in row["overall_service_sequence"]
        )
        assert float(row["temporal_breach_fraction"]) == int(row["temporal_breach_count"]) / 11


def test_every_design_has_five_colocated_realizations_and_original_split(
    corpus: tuple[list[dict[str, object]], list[dict[str, object]]]
) -> None:
    runs, designs = corpus
    for design in designs:
        group = [row for row in runs if row["design_id"] == design["design_id"]]
        assert len(group) == 5
        assert {row["realization_index"] for row in group} == set(range(5))
        assert {row["split"] for row in group} == {design["split"]}
        for row in group:
            source = GENERATION_ROOT / f"run_{int(row['run_id']):03d}" / "input" / "run_record.json"
            assert f'"split_assignment":"{row["split"]}"' in source.read_text(encoding="utf-8")


def test_augmentation_proposal_has_new_grouped_ids_regions_and_label() -> None:
    rows = proposed_augmentation_designs()
    assert len(rows) == 90
    assert all(row["proposal_label"] == PROPOSAL_LABEL for row in rows)
    assert all(row["proposal_status"] == "NOT_FROZEN" for row in rows)
    assert all(row["simulation_authorized"] is False for row in rows)
    assert all(str(row["augmentation_design_id"]).startswith("AUGV1-D") for row in rows)
    assert not ({row["augmentation_design_id"] for row in rows} & {f"D{index:03d}" for index in range(100)})
    assert len({row["augmentation_design_id"] for row in rows}) == 90
    for region in ("resilient_core", "boundary", "global_control"):
        assert {row["intended_split"] for row in rows if row["intended_region"] == region} == {
            "train",
            "validation",
            "test",
        }
    contract = augmentation_contract(rows)
    assert contract["proposal_status"] == "NOT_FROZEN"
    assert contract["simulation_authorized"] is False
    assert contract["recommended_strategy"] == "STAGED"
    assert contract["stage_a"]["region_split_allocation"] == {
        "resilient_core": {"train": 8, "validation": 2, "test": 2},
        "boundary": {"train": 8, "validation": 2, "test": 2},
        "global_control": {"train": 4, "validation": 1, "test": 1},
    }
    assert contract["stage_b"]["region_split_allocation"] == {
        "resilient_core": {"train": 24, "validation": 6, "test": 6},
        "boundary": {"train": 24, "validation": 6, "test": 6},
        "global_control": {"train": 12, "validation": 3, "test": 3},
    }
    neighbor_assessment = contract["stage_b"]["preliminary_near_neighbor_assessment"]
    assert neighbor_assessment["minimum_cross_split_distance"] > 0.10
    assert neighbor_assessment["cross_split_pairs_within_radius"] == 0


def test_output_root_refuses_all_frozen_evidence_descendants(tmp_path: Path) -> None:
    roots = [tmp_path / "generation", tmp_path / "replay", tmp_path / "freeze"]
    for root in roots:
        root.mkdir()
        with pytest.raises(ValueError, match="inside frozen evidence"):
            validate_output_root(root / "analysis", roots)


def test_analysis_has_no_simulation_entrypoint_dependency() -> None:
    package = ROOT / "src" / "satnet" / "experiments" / "final_class_support"
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(package.glob("*.py")))
    prohibited = (
        "run_tier1_rollout(",
        "generate_run(",
        "generate_runs(",
        "replay_runs_read_only(",
        "train_rf_model(",
        "SatelliteGNN(",
    )
    assert not any(value in source for value in prohibited)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_outputs_are_deterministic_and_ordered(tmp_path: Path) -> None:
    if not GENERATION_ROOT.is_dir():
        pytest.skip("Frozen production evidence is not available")
    first = tmp_path / "first"
    second = tmp_path / "second"
    kwargs = {
        "production_tooling_root": tmp_path / "unused-tooling",
        "generation_root": GENERATION_ROOT,
        "replay_root": REPLAY_ROOT,
        "freeze_root": FREEZE_ROOT,
        "freeze_archive": tmp_path / "unused.zip",
        "freeze_archive_hash_file": tmp_path / "unused.zip.sha256",
        "verify_all_hashes": False,
    }
    run_analysis(output_root=first, **kwargs)
    run_analysis(output_root=second, **kwargs)
    assert _tree_hashes(first) == _tree_hashes(second)
    run_lines = (first / "run_level_class_support.csv").read_text(encoding="utf-8").splitlines()
    design_lines = (first / "design_level_class_support.csv").read_text(encoding="utf-8").splitlines()
    assert run_lines[1].startswith("0,D000-R00,D000,")
    assert run_lines[-1].startswith("499,D099-R04,D099,")
    assert design_lines[1].startswith("D000,0,")
    assert design_lines[-1].startswith("D099,99,")
    assert _tree_hashes(first)["analysis_inventory.json"] == _tree_hashes(second)["analysis_inventory.json"]
