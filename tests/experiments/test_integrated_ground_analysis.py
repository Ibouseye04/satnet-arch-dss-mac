from __future__ import annotations

import math

import pytest

from satnet.experiments.integrated_ground_runner import artifact_paths, run_directory

from satnet.experiments.integrated_ground_analysis import (
    artifact_size_diagnostics,
    bottleneck_diagnostics,
    classification_diagnostics,
    numeric_statistics,
    regression_diagnostics,
    runtime_diagnostics,
)


def row(run_id: int, design_id: str, breach: bool, value: float) -> dict[str, object]:
    return {
        "run_id": run_id,
        "design_id": design_id,
        "realization_id": f"R{run_id % 5 + 1:02d}",
        "space_threshold_breach_any": breach,
        "ground_threshold_breach_any": not breach,
        "overall_threshold_breach_any": breach,
        "space_gcc_fraction_original_min": value,
        "failure_adjusted_ground_service_fraction_min": 1.0 - value,
        "failure_adjusted_overall_service_fraction_min": min(value, 1.0 - value),
        "failure_adjusted_overall_service_fraction_mean": min(value, 1.0 - value) + 0.1,
        "ground_service_loss_due_to_failures_max": value / 2.0,
        "space_bottleneck_timestep_count": run_id,
        "ground_bottleneck_timestep_count": 10 - run_id,
        "tie_bottleneck_timestep_count": 1,
        "dominant_bottleneck": "ground" if run_id < 5 else "space",
        "civilian_count": 2,
        "government_count": 1,
        "military_count": 1,
        "satellite_rollout_seconds": 1.0 + run_id,
        "g1_seconds": 0.1,
        "g2_seconds": 0.2,
        "g3_seconds": 0.3,
        "g4_seconds": 0.4,
        "g5_seconds": 0.5,
        "persistence_seconds": 0.6,
        "generation_runtime_seconds": 3.0 + run_id,
        "replay_runtime_seconds": 2.0 + run_id,
        "total_run_seconds": 5.0 + 2 * run_id,
    }


def test_numeric_statistics_use_population_spread_and_quantiles() -> None:
    result = numeric_statistics([0.0, 0.25, 0.5, 0.75, 1.0])
    assert result["count"] == 5
    assert result["mean"] == 0.5
    assert result["standard_deviation"] == math.sqrt(0.125)
    assert result["unique_value_count"] == 5
    assert result["quantiles"] == {
        "p00": 0.0,
        "p25": 0.25,
        "p50": 0.5,
        "p75": 0.75,
        "p100": 1.0,
    }
    with pytest.raises(ValueError):
        numeric_statistics([float("nan")])


def test_classification_counts_candidates_and_inverses() -> None:
    rows = tuple(row(index, "P01" if index < 5 else "P02", index < 6, index / 10) for index in range(10))
    result = classification_diagnostics(rows)
    space = result["space_threshold_breach_any"]
    assert space["positive_count"] == 6
    assert space["negative_count"] == 4
    assert space["both_classes_present"] is True
    assert space["minority_count"] == 4
    inverse = result["space_threshold_met_for_entire_run"]
    assert inverse["positive_count"] == 4
    assert inverse["negative_count"] == 6


def test_regression_diagnostics_report_useful_spread() -> None:
    rows = tuple(row(index, "P01" if index < 5 else "P02", False, index / 10) for index in range(10))
    result = regression_diagnostics(rows)
    assert result["space_gcc_fraction_original_min"]["unique_value_count"] == 10
    assert result["space_gcc_fraction_original_min"]["standard_deviation"] > 0.0
    assert result["space_gcc_fraction_original_min"]["useful_spread"] is True
    assert set(result["space_gcc_fraction_original_min"]["values_by_design"]) == {"P01", "P02"}


def test_bottleneck_analysis_aggregates_runs_designs_and_compositions() -> None:
    rows = tuple(row(index, "P01" if index < 5 else "P02", False, index / 10) for index in range(10))
    result = bottleneck_diagnostics(rows)
    assert result["all_runs"]["total_timestep_count"] == sum(
        value["space_bottleneck_timestep_count"]
        + value["ground_bottleneck_timestep_count"]
        + value["tie_bottleneck_timestep_count"]
        for value in rows
    )
    assert set(result["by_design"]) == {"P01", "P02"}
    assert result["space_and_ground_metrics_distinct"] is True


def test_runtime_structure_reports_ordered_means_and_projection() -> None:
    rows = tuple(row(index, "P01", False, 0.5) for index in range(5))
    result = runtime_diagnostics(rows)
    assert result["generation_runtime_seconds"]["mean"] == 5.0
    assert result["generation_runtime_seconds"]["median"] == 5.0
    assert result["generation_runtime_seconds"]["total"] == 25.0
    assert result["projected_500_run_generation_seconds"] == 2500.0


def test_artifact_accounting_reports_stage_run_design_and_projection(tmp_path) -> None:
    rows = []
    for run_id, design_id in ((0, "P01"), (1, "P01"), (5, "P02")):
        paths = artifact_paths(run_directory(tmp_path, run_id))
        paths["inventory"].parent.mkdir(parents=True, exist_ok=True)
        inventory = {
            "artifacts": [
                {"artifact_key": "g1", "bytes": 10, "filename": "g1", "sha256": "a" * 64},
                {"artifact_key": "g2", "bytes": 20, "filename": "g2", "sha256": "b" * 64},
            ],
            "canonical_artifact_bytes": 30,
            "pilot_artifact_inventory_schema_version": "1",
        }
        paths["inventory"].write_text(
            __import__("json").dumps(inventory, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        rows.append({"run_id": run_id, "design_id": design_id, "artifact_bytes": 30})
    result = artifact_size_diagnostics(rows, tmp_path)
    assert result["total_pilot_bytes"] == 90
    assert result["by_stage_bytes"] == {"g1": 30, "g2": 60}
    assert result["by_design_bytes"] == {"P01": 60, "P02": 30}
    assert result["estimated_500_run_bytes"] == 1800
