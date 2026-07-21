from __future__ import annotations

from collections import Counter
import os
from pathlib import Path

import pytest

from satnet.experiments.final_class_support_audit.audit import (
    ANALYSIS_COMMIT,
    ANALYSIS_INVENTORY_SHA256,
    FREEZE_ARCHIVE_SHA256,
    boundary_audit,
    gate_feasibility_rows,
    nearest_neighbor_audit,
    normalized_distance,
    parameter_audit,
    proposal_audit,
    regression_audit,
    reproduce_corpus,
    scientific_signature,
    temporal_audit,
    validate_output_root,
    verify_analysis_outputs,
    verify_frozen_evidence,
)

ROOT = Path(__file__).parents[2]
GENERATION_ROOT = Path(os.environ.get("SATNET_FINAL_GENERATION_ROOT", "C:/Users/johns/satnet-final-production-20260720"))
REPLAY_ROOT = Path(os.environ.get("SATNET_FINAL_REPLAY_ROOT", "C:/Users/johns/satnet-final-production-replay-20260720"))
FREEZE_ROOT = Path(os.environ.get("SATNET_FINAL_FREEZE_ROOT", "C:/Users/johns/satnet-final-production-v1-freeze-20260721"))
FREEZE_ARCHIVE = Path(os.environ.get("SATNET_FINAL_FREEZE_ARCHIVE", "C:/Users/johns/satnet-final-production-v1-freeze-20260721.zip"))
PRODUCTION_TOOLING_ROOT = Path(os.environ.get("SATNET_PRODUCTION_TOOLING_ROOT", "C:/Users/johns/satnet-production-tooling-20260720"))
ANALYSIS_OUTPUT_ROOT = Path(os.environ.get("SATNET_CLASS_SUPPORT_ANALYSIS_ROOT", "C:/Users/johns/satnet-class-support-analysis-20260721"))
TRACKED_ANALYSIS_ROOT = ROOT / "artifacts/final_integrated_dataset_class_support_analysis"


@pytest.fixture(scope="module")
def corpus() -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    if not GENERATION_ROOT.is_dir():
        pytest.skip("Frozen production evidence unavailable")
    return reproduce_corpus(GENERATION_ROOT)


def test_independently_verifies_all_frozen_hashes_and_analysis_inventory() -> None:
    if not all(path.exists() for path in (GENERATION_ROOT, REPLAY_ROOT, FREEZE_ROOT, FREEZE_ARCHIVE, PRODUCTION_TOOLING_ROOT, ANALYSIS_OUTPUT_ROOT)):
        pytest.skip("External frozen evidence unavailable")
    evidence = verify_frozen_evidence(
        production_tooling_root=PRODUCTION_TOOLING_ROOT,
        generation_root=GENERATION_ROOT,
        replay_root=REPLAY_ROOT,
        freeze_root=FREEZE_ROOT,
        freeze_archive=FREEZE_ARCHIVE,
        freeze_archive_hash_file=Path(f"{FREEZE_ARCHIVE}.sha256"),
    )
    assert evidence["verification_status"] == "passed"
    assert evidence["freeze_archive_sha256"] == FREEZE_ARCHIVE_SHA256
    assert evidence["combined"] == {
        "file_count": 9004,
        "byte_count": 1_337_549_193,
        "verified_sha256_count": 9004,
    }
    assert evidence["generation"]["all_files_read_only"] is True
    assert evidence["replay"]["all_files_read_only"] is True
    analysis = verify_analysis_outputs(ANALYSIS_OUTPUT_ROOT, TRACKED_ANALYSIS_ROOT)
    assert analysis["analysis_inventory_sha256"] == ANALYSIS_INVENTORY_SHA256
    assert analysis["inventory_output_count"] == 24


def test_reproduces_exact_corpus_classes_and_minority_identities(
    corpus: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
) -> None:
    summary, runs, designs = corpus
    assert summary["run_count"] == 500
    assert summary["design_count"] == 100
    assert summary["classification_counts"] == {
        "train": {"false": 2, "true": 348},
        "validation": {"false": 0, "true": 75},
        "test": {"false": 5, "true": 70},
    }
    assert summary["non_breach_run_ids"] == [0, 1, 2, 3, 4, 5, 7]
    assert summary["non_breach_design_ids"] == ["D000", "D001"]
    assert summary["zero_of_five_non_breach_design_count"] == 98
    assert Counter(row["split"] for row in runs) == {"train": 350, "validation": 75, "test": 75}
    assert Counter(row["split"] for row in designs) == {"train": 70, "validation": 15, "test": 15}


def test_reproduces_boundary_polarity_exact_threshold_and_temporal_metrics(
    corpus: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
) -> None:
    _, runs, designs = corpus
    boundary, rows = boundary_audit(runs, designs)
    assert boundary["all_500_polarities_match"] is True
    assert boundary["exact_threshold_run_ids"] == [5, 7]
    assert boundary["runs_below_negative_0_20_exclusive"] == 480
    assert boundary["design_means_below_negative_0_20_exclusive"] == 96
    assert boundary["nearest_breach"] == {"run_id": 6, "design_id": "D001", "margin": pytest.approx(-0.05)}
    assert all(row["polarity_match"] for row in rows)
    temporal = temporal_audit(runs)
    assert temporal["total_sampled_state_count"] == 5500
    assert temporal["mean_temporal_breach_fraction_by_stratum"]["transition"] == pytest.approx(0.9677922077922079)
    assert temporal["mean_temporal_breach_fraction_by_stratum"]["global"] == pytest.approx(0.9893939393939394)


def test_stage_arithmetic_and_final_gate_feasibility_are_scope_specific() -> None:
    rows = gate_feasibility_rows()
    assert len(rows) == 108
    final_rows = [row for row in rows if row["scope"] in {"original_plus_stage_b", "original_plus_stage_a_plus_stage_b"}]
    assert final_rows
    assert all(row["feasible"] for row in final_rows)
    stage_a_rows = [row for row in rows if row["scope"] == "stage_a_augmentation_only"]
    assert all(row["applicability"] == "not_applicable_discovery_only" for row in stage_a_rows)
    assert any(not row["feasible"] for row in stage_a_rows)
    validation_joint = next(
        row
        for row in rows
        if row["scope"] == "stage_b_augmentation_only"
        and row["split"] == "validation"
        and row["gate"] == "joint_minimum_class_run_counts"
    )
    assert validation_joint["required"] == 74
    assert validation_joint["maximum_physically_available"] == 75
    assert validation_joint["feasible"] is True


def test_proposal_identity_uniqueness_authorization_and_stage_separation(
    corpus: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
) -> None:
    _, runs, designs = corpus
    stage_a, stage_b, collisions, _ = proposal_audit(
        analysis_output_root=ANALYSIS_OUTPUT_ROOT,
        runs=runs,
        designs=designs,
    )
    assert stage_a["arithmetic_passed"] is True
    assert stage_a["exact_stage_a_design_table_present"] is False
    assert stage_a["stage_a_namespace_present"] is False
    assert stage_b["arithmetic_passed"] is True
    assert stage_b["proposal_status"] == "NOT_FROZEN"
    assert stage_b["simulation_authorized"] is False
    assert stage_b["exact_duplicate_new_count"] == 0
    assert stage_b["exact_duplicate_original_count"] == 0
    assert next(row for row in collisions if row["check"] == "stage_a_namespace_defined")["status"] == "failed"


def test_duplicate_signatures_and_cross_split_near_neighbors_are_detectable() -> None:
    base = {
        "num_planes": 6,
        "sats_per_plane": 8,
        "altitude_km": 1000.0,
        "inclination_deg": 80.0,
        "phasing_factor": 1,
        "satellite_node_failure_probability": 0.02,
        "satellite_edge_failure_probability": 0.03,
        "civilian_count": 8,
        "government_count": 6,
        "military_count": 6,
        "total_ground_station_count": 20,
        "ground_station_failure_probability": 0.02,
        "duration_minutes": 10,
        "step_seconds": 60,
        "epoch_iso": "2000-01-01T12:00:00+00:00",
        "orbital_engine": "sgp4",
        "max_isl_distance_km": 10000.0,
        "isl_policy": "grid_fixed",
        "adjacent_search_k": 1,
        "max_inter_plane_links_per_sat": 1,
        "satellite_failure_model": "persistent_temporal_union_edges_v1",
        "minimum_elevation_deg": 10.0,
        "space_gcc_threshold": 0.8,
        "ground_service_threshold": 0.8,
    }
    duplicate = dict(base)
    assert scientific_signature(base) == scientific_signature(duplicate)
    assert normalized_distance(base, duplicate) == 0.0
    nearby = dict(base, altitude_km=1001.0)
    assert 0.0 < normalized_distance(base, nearby) <= 0.10


def test_output_policy_rejects_frozen_descendants_and_source_has_no_execution_entrypoints(tmp_path: Path) -> None:
    protected = [tmp_path / name for name in ("generation", "replay", "freeze")]
    for root in protected:
        root.mkdir()
        with pytest.raises(ValueError, match="inside protected evidence"):
            validate_output_root(root / "audit", protected)
    source = (ROOT / "src/satnet/experiments/final_class_support_audit/audit.py").read_text(encoding="utf-8")
    prohibited = ("run_tier1_rollout(", "generate_run(", "train_rf_model(", "SatelliteGNN(")
    assert not any(value in source for value in prohibited)
    assert ANALYSIS_COMMIT in source


def test_parameter_neighbor_and_regression_results_reproduce_independently(
    corpus: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
) -> None:
    _, runs, designs = corpus
    parameters = parameter_audit(runs, designs)
    assert parameters["correlations"]["altitude_km"] == pytest.approx(0.6627247027078719)
    assert parameters["correlations"]["sats_per_plane"] == pytest.approx(0.5338963974657238)
    assert parameters["correlations"]["configured_satellite_count"] == pytest.approx(0.39215337541491824)
    assert parameters["correlations"]["inclination_deg"] == pytest.approx(0.26015529260699544)
    neighbors = nearest_neighbor_audit(designs)
    assert neighbors["neighbors"]["D000"][0]["design_id"] == "D001"
    assert neighbors["neighbors"]["D000"][0]["distance"] == pytest.approx(0.8294212447374086)
    assert neighbors["neighbors"]["D001"][0]["design_id"] == "D022"
    assert neighbors["neighbors"]["D001"][0]["distance"] == pytest.approx(0.3932311802307627)
    regression = regression_audit(runs)
    primary = "failure_adjusted_overall_service_fraction_mean"
    assert regression["by_split"]["train"][primary]["mean"] == pytest.approx(0.2339977272727273)
    assert regression["by_split"]["validation"][primary]["population_standard_deviation"] == pytest.approx(0.2447719355573115)
    assert regression["all_four_targets_have_nonzero_spread_in_every_split"] is True
