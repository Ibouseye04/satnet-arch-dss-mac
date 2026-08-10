from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from satnet.experiments.integrated_ground_manifest import (
    PILOT_EXPECTED_RUN_COUNT,
    IntegratedPilotDesign,
    IntegratedPilotRun,
    pilot_run_manifest_hash,
    read_pilot_design_manifest,
    read_pilot_run_manifest,
)
from satnet.experiments.integrated_ground_runner import (
    _atomic_write,
    _read_single_json,
    artifact_paths,
    pilot_json,
    run_directory,
)
from satnet.ground.catalog import load_ground_station_catalog
from satnet.experiments.final_generation.constants import FINAL_RUN_COUNT

CLASSIFICATION_FIELDS = (
    "space_threshold_breach_any",
    "ground_threshold_breach_any",
    "overall_threshold_breach_any",
)
REGRESSION_FIELDS = (
    "space_gcc_fraction_original_min",
    "failure_adjusted_ground_service_fraction_min",
    "failure_adjusted_overall_service_fraction_min",
    "failure_adjusted_overall_service_fraction_mean",
    "ground_service_loss_due_to_failures_max",
)
RUNTIME_FIELDS = (
    "satellite_rollout_seconds",
    "g1_seconds",
    "g2_seconds",
    "g3_seconds",
    "g4_seconds",
    "g5_seconds",
    "persistence_seconds",
    "generation_runtime_seconds",
    "replay_runtime_seconds",
    "total_run_seconds",
)
CLASS_NAMES = ("civilian", "government", "military")


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Quantile requires values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Quantile probability must be within [0, 1]")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_statistics(values: Sequence[float]) -> dict[str, object]:
    normalized = tuple(values)
    if not normalized:
        raise ValueError("Numeric diagnostics require at least one value")
    if any(type(value) is not float or not math.isfinite(value) for value in normalized):
        raise ValueError("Numeric diagnostics require finite floats")
    mean = math.fsum(normalized) / len(normalized)
    variance = math.fsum((value - mean) ** 2 for value in normalized) / len(normalized)
    return {
        "count": len(normalized),
        "maximum": max(normalized),
        "mean": mean,
        "minimum": min(normalized),
        "quantiles": {
            "p00": _quantile(normalized, 0.0),
            "p25": _quantile(normalized, 0.25),
            "p50": _quantile(normalized, 0.5),
            "p75": _quantile(normalized, 0.75),
            "p100": _quantile(normalized, 1.0),
        },
        "standard_deviation": math.sqrt(variance),
        "unique_value_count": len(set(normalized)),
    }


def classification_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    normalized = tuple(rows)
    result: dict[str, object] = {}
    for field_name in CLASSIFICATION_FIELDS:
        values = [row[field_name] for row in normalized]
        if any(type(value) is not bool for value in values):
            raise TypeError(f"{field_name} must contain Booleans")
        positive = sum(values)
        negative = len(values) - positive
        by_design = {
            design_id: {
                "negative_count": sum(
                    not row[field_name]
                    for row in normalized
                    if row["design_id"] == design_id
                ),
                "positive_count": sum(
                    row[field_name]
                    for row in normalized
                    if row["design_id"] == design_id
                ),
            }
            for design_id in sorted({row["design_id"] for row in normalized})
        }
        by_realization = {
            realization_id: {
                "negative_count": sum(
                    not row[field_name]
                    for row in normalized
                    if row["realization_id"] == realization_id
                ),
                "positive_count": sum(
                    row[field_name]
                    for row in normalized
                    if row["realization_id"] == realization_id
                ),
            }
            for realization_id in sorted(
                {row["realization_id"] for row in normalized}
            )
        }
        result[field_name] = {
            "both_classes_present": positive > 0 and negative > 0,
            "counts_by_design": by_design,
            "counts_by_realization": by_realization,
            "minority_count": min(positive, negative),
            "minority_count_at_least_five": min(positive, negative) >= 5,
            "negative_count": negative,
            "negative_fraction": negative / len(values),
            "positive_count": positive,
            "positive_fraction": positive / len(values),
        }
        inverse_name = field_name.replace("_breach_any", "_met_for_entire_run")
        result[inverse_name] = {
            "both_classes_present": positive > 0 and negative > 0,
            "negative_count": positive,
            "negative_fraction": positive / len(values),
            "positive_count": negative,
            "positive_fraction": negative / len(values),
        }
    return result


def regression_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    normalized = tuple(rows)
    result: dict[str, object] = {}
    design_ids = sorted({row["design_id"] for row in normalized})
    for field_name in REGRESSION_FIELDS:
        values = [row[field_name] for row in normalized]
        if any(type(value) is not float for value in values):
            raise TypeError(f"{field_name} must contain floats")
        stats = numeric_statistics(values)
        stats["not_entirely_endpoint_concentrated"] = not set(values) <= {0.0, 1.0}
        stats["useful_spread"] = (
            stats["unique_value_count"] >= 5
            and stats["standard_deviation"] > 0.0
            and stats["not_entirely_endpoint_concentrated"]
        )
        stats["values_by_design"] = {
            design_id: [
                row[field_name]
                for row in normalized
                if row["design_id"] == design_id
            ]
            for design_id in design_ids
        }
        result[field_name] = stats
    return result


def bottleneck_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    normalized = tuple(rows)

    def aggregate(subset: Sequence[Mapping[str, object]]) -> dict[str, object]:
        counts = {
            "space": sum(row["space_bottleneck_timestep_count"] for row in subset),
            "ground": sum(row["ground_bottleneck_timestep_count"] for row in subset),
            "tie": sum(row["tie_bottleneck_timestep_count"] for row in subset),
        }
        maximum = max(counts.values())
        leaders = [name for name, value in counts.items() if value == maximum]
        return {
            "dominant_bottleneck": leaders[0] if len(leaders) == 1 else "tie",
            "ground_bottleneck_timestep_count": counts["ground"],
            "space_bottleneck_timestep_count": counts["space"],
            "tie_bottleneck_timestep_count": counts["tie"],
            "total_timestep_count": sum(counts.values()),
        }

    by_design = {
        design_id: aggregate(
            [row for row in normalized if row["design_id"] == design_id]
        )
        for design_id in sorted({row["design_id"] for row in normalized})
    }
    by_class_composition = {
        f"c{row['civilian_count']}_g{row['government_count']}_m{row['military_count']}": aggregate(
            [candidate for candidate in normalized if candidate["design_id"] == row["design_id"]]
        )
        for row in normalized[::5]
    }
    run_values = {
        str(row["run_id"]): {
            "dominant_bottleneck": row["dominant_bottleneck"],
            "ground_bottleneck_timestep_count": row[
                "ground_bottleneck_timestep_count"
            ],
            "space_bottleneck_timestep_count": row[
                "space_bottleneck_timestep_count"
            ],
            "tie_bottleneck_timestep_count": row["tie_bottleneck_timestep_count"],
        }
        for row in normalized
    }
    return {
        "all_runs": aggregate(normalized),
        "by_class_composition": by_class_composition,
        "by_design": by_design,
        "per_run": run_values,
        "space_and_ground_metrics_distinct": any(
            row["space_gcc_fraction_original_min"]
            != row["failure_adjusted_ground_service_fraction_min"]
            for row in normalized
        ),
    }


def class_service_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    normalized = tuple(rows)
    result: dict[str, object] = {}
    for class_name in CLASS_NAMES:
        minimum_field = f"failure_adjusted_{class_name}_service_fraction_min"
        mean_field = f"failure_adjusted_{class_name}_service_fraction_mean"
        selected_field = f"{class_name}_count"
        failed_field = f"failed_{class_name}_count"
        values_min = [row[minimum_field] for row in normalized if row[minimum_field] is not None]
        values_mean = [row[mean_field] for row in normalized if row[mean_field] is not None]
        result[class_name] = {
            "failed_station_count_total": sum(row[failed_field] for row in normalized),
            "global_threshold_breach_run_count": sum(
                row[minimum_field] is not None and row[minimum_field] < 0.8
                for row in normalized
            ),
            "mean_adjusted_service_across_runs": math.fsum(values_mean) / len(values_mean),
            "minimum_adjusted_service_across_runs": min(values_min),
            "operational_station_count_total": sum(
                row[selected_field] - row[failed_field] for row in normalized
            ),
            "selected_station_count_total": sum(row[selected_field] for row in normalized),
        }
    return {
        "caution": (
            "Class counts vary by design; these descriptive values do not establish causal "
            "or intrinsic class resilience."
        ),
        "classes": result,
    }


def runtime_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for field_name in RUNTIME_FIELDS:
        values = [row[field_name] for row in rows]
        if any(type(value) is not float or not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(f"Runtime field is invalid: {field_name}")
        result[field_name] = {
            "maximum": max(values),
            "mean": math.fsum(values) / len(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "total": math.fsum(values),
        }
    result["projected_10000_run_generation_seconds"] = (
        result["generation_runtime_seconds"]["mean"] * FINAL_RUN_COUNT
    )
    result["projected_10000_run_replay_seconds"] = (
        result["replay_runtime_seconds"]["mean"] * FINAL_RUN_COUNT
    )
    result["projection_note"] = (
        "Linear extrapolation from this 25-run pilot; available compute and parallelism are "
        "not modeled."
    )
    return result


def artifact_size_diagnostics(
    rows: Sequence[Mapping[str, object]], output_root: str | Path
) -> dict[str, object]:
    root = Path(output_root)
    per_run: dict[str, object] = {}
    stage_totals: dict[str, int] = {}
    design_totals: dict[str, int] = {}
    totals: list[float] = []
    for row in rows:
        run_id = row["run_id"]
        inventory = _read_single_json(
            artifact_paths(run_directory(root, run_id))["inventory"]
        )
        artifacts = inventory["artifacts"]
        stage_values = {item["artifact_key"]: item["bytes"] for item in artifacts}
        total = inventory["canonical_artifact_bytes"]
        if total != row["artifact_bytes"]:
            raise ValueError(f"Run {run_id} artifact bytes do not match summary")
        per_run[str(run_id)] = {
            "artifact_bytes": total,
            "by_stage": stage_values,
            "design_id": row["design_id"],
        }
        totals.append(float(total))
        design_totals[row["design_id"]] = design_totals.get(row["design_id"], 0) + total
        for stage, size in stage_values.items():
            stage_totals[stage] = stage_totals.get(stage, 0) + size
    stats = numeric_statistics(totals)
    total_pilot_bytes = int(math.fsum(totals))
    return {
        "by_design_bytes": design_totals,
        "by_stage_bytes": stage_totals,
        "estimated_10000_run_bytes": total_pilot_bytes * (FINAL_RUN_COUNT // 25),
        "estimate_note": (
            "Linear extrapolation from canonical pilot artifacts only; future ML datasets are excluded."
        ),
        "per_run": per_run,
        "per_run_statistics": stats,
        "total_pilot_bytes": total_pilot_bytes,
    }


def _read_verified_rows(
    *,
    output_root: Path,
    runs: Sequence[IntegratedPilotRun],
    catalog_hash: str,
) -> tuple[dict[str, object], ...]:
    replay_summary = _read_single_json(output_root / "summaries" / "replay_summary.json")
    if not isinstance(replay_summary, dict):
        raise TypeError("Replay summary must be an object")
    if replay_summary.get("successful_replay_run_count") != PILOT_EXPECTED_RUN_COUNT:
        raise ValueError("Analysis requires 25 successful replays")
    if replay_summary.get("failed_replay_run_count") != 0:
        raise ValueError("Analysis refuses failed replay evidence")
    replay_by_id = {value["run_id"]: value for value in replay_summary["results"]}
    if set(replay_by_id) != {run.run_id for run in runs}:
        raise ValueError("Replay summary does not cover exact pilot run IDs")
    rows: list[dict[str, object]] = []
    for run in runs:
        path = artifact_paths(run_directory(output_root, run.run_id))["summary"]
        value = _read_single_json(path)
        if not isinstance(value, dict):
            raise TypeError("Run summary must be an object")
        replay = replay_by_id[run.run_id]
        if replay["g4_run_summary_hash"] != value["g4_run_summary_hash"]:
            raise ValueError("Replay and summary G4 identities differ")
        if replay["g5_run_summary_hash"] != value["g5_run_summary_hash"]:
            raise ValueError("Replay and summary G5 identities differ")
        if replay["satellite_artifact_hash"] != value["satellite_artifact_hash"]:
            raise ValueError("Replay and summary satellite identities differ")
        row = dict(value)
        row["catalog_hash"] = catalog_hash
        row["satellite_rollout_seed"] = run.satellite_rollout_seed
        row["ground_station_selection_seed"] = run.ground_station_selection_seed
        row["ground_failure_seed"] = run.ground_failure_seed
        row["replay_status"] = "success"
        row["replay_runtime_seconds"] = replay["replay_runtime_seconds"]
        row["total_run_seconds"] = (
            row["generation_runtime_seconds"] + row["replay_runtime_seconds"]
        )
        rows.append(row)
    return tuple(rows)


def _write_json(path: Path, value: object) -> None:
    _atomic_write(path, pilot_json(value) + "\n")


def _write_jsonl(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    _atomic_write(
        path,
        "\n".join(pilot_json(value) for value in values) + "\n",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("CSV requires rows")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("CSV rows must use identical field order")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(path, stream.getvalue())


def _design_summary(rows: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    result: list[dict[str, object]] = []
    for design_id in sorted({row["design_id"] for row in rows}):
        subset = [row for row in rows if row["design_id"] == design_id]
        result.append(
            {
                "design_id": design_id,
                "run_count": len(subset),
                "space_breach_run_count": sum(
                    row["space_threshold_breach_any"] for row in subset
                ),
                "ground_breach_run_count": sum(
                    row["ground_threshold_breach_any"] for row in subset
                ),
                "overall_breach_run_count": sum(
                    row["overall_threshold_breach_any"] for row in subset
                ),
                "adjusted_overall_minimum": min(
                    row["failure_adjusted_overall_service_fraction_min"]
                    for row in subset
                ),
                "adjusted_overall_mean": math.fsum(
                    row["failure_adjusted_overall_service_fraction_mean"]
                    for row in subset
                )
                / len(subset),
                "generation_runtime_seconds_mean": math.fsum(
                    row["generation_runtime_seconds"] for row in subset
                )
                / len(subset),
                "replay_runtime_seconds_mean": math.fsum(
                    row["replay_runtime_seconds"] for row in subset
                )
                / len(subset),
                "artifact_bytes_total": sum(row["artifact_bytes"] for row in subset),
            }
        )
    return tuple(result)


def _feature_inventory() -> str:
    return """# Candidate Feature Inventory

## Future random-forest inputs

### Satellite design variables

- Number of planes.
- Satellites per plane.
- Configured satellite count.
- Altitude.
- Inclination.
- Walker phasing factor.
- ISL policy and fixed ISL search parameters.
- Fixed temporal profile.

### Ground design variables

- Selected civilian count.
- Selected government count.
- Selected military count.
- Total selected ground count.
- Synthetic catalog identity when comparing catalog designs.

### Failure probabilities

- Persistent satellite-node failure probability.
- Persistent accepted-edge failure probability.
- Persistent ground-station failure probability.
- Versioned satellite and ground failure-model identities.

### Visibility-policy variables

- Minimum elevation threshold.
- Visibility-model and frame-contract versions.

### Service-policy variables

- Original-denominator space GCC threshold.
- Ground-service threshold.

### Aggregate pre-outcome features

- Deterministic design-only constellation size and class-composition ratios.
- Policy identities or explicit policy values.
- No post-rollout connectivity or service result belongs in this group.

## Prohibited leakage fields

- Final classification or regression target values.
- G4 or G5 service summaries when predicting those same outcomes.
- Failure-adjusted labels or breach states.
- Realized failed-node, failed-edge, or failed-station counts when the task is pre-run design prediction.
- Run, scientific, sequence, record, or replay hashes.
- Replay status, generation status, runtime, or artifact size.
- Future timestep values in a forecasting task.
- Full-sequence statistics when predicting a future suffix from a prefix.

## Future TGNN concepts

Potential satellite-node concepts include orbital-plane identity, normalized orbital position, operational eligibility, and prior-prefix graph state. Potential ground-node concepts include class, deterministic coordinates, selected eligibility, and visibility-policy context. Potential ISL and satellite-ground edge concepts include physical link attributes already present in canonical graphs and explicit edge kind.

These are conceptual candidates only. No TGNN implementation or feature schema is changed by this pilot.

## Assessment mode distinction

The current TGNN consumes complete temporal sequences and is therefore an ex-post run-level assessment. A future-prefix forecasting experiment must explicitly truncate inputs at a declared cutoff and prohibit all suffix information. Results from these two tasks are not interchangeable.
"""


def analyze_pilot(output_root: str | Path) -> dict[str, object]:
    root = Path(output_root)
    input_root = root / "inputs"
    summary_root = root / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    designs = read_pilot_design_manifest(input_root / "pilot_designs.json")
    runs = read_pilot_run_manifest(input_root / "pilot_runs.jsonl", designs)
    catalog = load_ground_station_catalog(input_root / "pilot_catalog.csv")
    if len(runs) != PILOT_EXPECTED_RUN_COUNT:
        raise ValueError("Analysis requires the complete 25-run manifest")
    rows = _read_verified_rows(
        output_root=root,
        runs=runs,
        catalog_hash=catalog.catalog_hash,
    )
    classification = classification_diagnostics(rows)
    regression = regression_diagnostics(rows)
    bottleneck = bottleneck_diagnostics(rows)
    class_service = class_service_diagnostics(rows)
    runtime = runtime_diagnostics(rows)
    artifact_size = artifact_size_diagnostics(rows, root)
    design_summary = _design_summary(rows)
    classification_variance = any(
        classification[field]["minority_count_at_least_five"]
        for field in CLASSIFICATION_FIELDS
    )
    regression_variance = any(
        regression[field]["useful_spread"] for field in REGRESSION_FIELDS
    )
    verdict = (
        "READY FOR FINAL INTEGRATED DATASET DESIGN"
        if classification_variance and regression_variance
        else "READY FOR REVISED 50-RUN PILOT"
    )
    _write_jsonl(summary_root / "pilot_run_summary.jsonl", rows)
    _write_csv(summary_root / "pilot_run_summary.csv", rows)
    _write_csv(summary_root / "design_summary.csv", design_summary)
    _write_json(summary_root / "label_diagnostics.json", classification)
    _write_json(summary_root / "regression_diagnostics.json", regression)
    _write_json(summary_root / "bottleneck_diagnostics.json", bottleneck)
    _write_json(summary_root / "class_service_diagnostics.json", class_service)
    _write_json(summary_root / "runtime_summary.json", runtime)
    runtime_rows = tuple(
        {
            "runtime_field": field_name,
            "minimum": runtime[field_name]["minimum"],
            "median": runtime[field_name]["median"],
            "mean": runtime[field_name]["mean"],
            "maximum": runtime[field_name]["maximum"],
            "total": runtime[field_name]["total"],
        }
        for field_name in RUNTIME_FIELDS
    )
    _write_csv(summary_root / "runtime_summary.csv", runtime_rows)
    _write_json(summary_root / "artifact_size_summary.json", artifact_size)
    artifact_rows = tuple(
        {
            "run_id": int(run_id),
            "design_id": value["design_id"],
            "artifact_bytes": value["artifact_bytes"],
            **{
                f"{stage}_bytes": value["by_stage"].get(stage, 0)
                for stage in sorted(artifact_size["by_stage_bytes"])
            },
        }
        for run_id, value in sorted(
            artifact_size["per_run"].items(), key=lambda item: int(item[0])
        )
    )
    _write_csv(summary_root / "artifact_size_summary.csv", artifact_rows)
    _atomic_write(summary_root / "candidate_feature_inventory.md", _feature_inventory())
    data_dictionary = {
        "derived_pilot_fields": [
            "space_bottleneck_timestep_count",
            "ground_bottleneck_timestep_count",
            "tie_bottleneck_timestep_count",
            "dominant_bottleneck",
            "generation_status",
            "replay_status",
            "runtime fields",
            "artifact_bytes",
        ],
        "row_field_order": list(rows[0]),
        "scientific_sources": {
            "G1": "ground design record",
            "G4": "ground service run record",
            "G5": "failure-adjusted run and step records",
            "satellite": "production Tier1 rollout artifact",
        },
    }
    _write_json(summary_root / "pilot_run_summary_data_dictionary.json", data_dictionary)
    hash_inventory = {
        "catalog_hash": catalog.catalog_hash,
        "design_hashes": {design.design_id: design.design_hash for design in designs},
        "pilot_run_manifest_hash": pilot_run_manifest_hash(runs),
        "run_hashes": {
            str(row["run_id"]): {
                "g4_run_summary_hash": row["g4_run_summary_hash"],
                "g5_run_summary_hash": row["g5_run_summary_hash"],
                "ground_design_hash": row["ground_design_hash"],
                "ground_failure_realization_hash": row[
                    "ground_failure_realization_hash"
                ],
                "satellite_artifact_hash": row["satellite_artifact_hash"],
                "satellite_config_hash": row["satellite_config_hash"],
            }
            for row in rows
        },
    }
    _write_json(summary_root / "hash_inventory.json", hash_inventory)
    result = {
        "artifact_size": artifact_size,
        "bottleneck": bottleneck,
        "catalog_hash": catalog.catalog_hash,
        "classification": classification,
        "classification_variance_gate": classification_variance,
        "design_count": len(designs),
        "pilot_run_manifest_hash": pilot_run_manifest_hash(runs),
        "regression": regression,
        "regression_variance_gate": regression_variance,
        "run_count": len(rows),
        "runtime": runtime,
        "verdict": verdict,
    }
    _write_json(summary_root / "analysis_summary.json", result)
    return result
