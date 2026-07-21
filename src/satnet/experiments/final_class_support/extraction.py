from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

from satnet.experiments.final_class_support.constants import (
    DESIGN_COUNT,
    EXPECTED_CLASS_COUNTS,
    EXPECTED_NON_BREACH_RUN_IDS,
    EXPECTED_SPLIT_DESIGN_COUNTS,
    EXPECTED_SPLIT_RUN_COUNTS,
    REALIZATIONS_PER_DESIGN,
    RUN_COUNT,
    SERVICE_THRESHOLD,
    TARGET_FIELDS,
)
from satnet.experiments.final_class_support.io import read_json, read_jsonl
from satnet.experiments.final_generation.io import parse_canonical_float


def _canonical_float(value: object, field_name: str) -> float:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a canonical float string")
    return parse_canonical_float(value, field_name)


def _longest_true_streak(values: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _recovery_count(breaches: list[bool]) -> int:
    return sum(previous and not current for previous, current in zip(breaches, breaches[1:]))


def _run_paths(root: Path, run_id: int) -> dict[str, Path]:
    run_root = root / f"run_{run_id:03d}"
    return {
        "design": run_root / "input" / "design_record.json",
        "run": run_root / "input" / "run_record.json",
        "target": run_root / "targets" / "target.json",
        "g5_steps": run_root / "g5" / "ground_failure_service_steps.jsonl",
        "g5_run": run_root / "g5" / "ground_failure_service_run.jsonl",
        "g5_realization": run_root / "g5" / "ground_failure_realization.jsonl",
        "satellite": run_root / "satellite" / "satellite_rollout.json",
    }


def extract_run(root: str | Path, run_id: int) -> dict[str, Any]:
    paths = _run_paths(Path(root), run_id)
    if any(not path.is_file() for path in paths.values()):
        missing = sorted(str(path) for path in paths.values() if not path.is_file())
        raise FileNotFoundError(f"Authoritative run artifacts missing for run {run_id}: {missing}")
    design = read_json(paths["design"])
    run = read_json(paths["run"])
    target = read_json(paths["target"])
    g5_steps = read_jsonl(paths["g5_steps"])
    g5_runs = read_jsonl(paths["g5_run"])
    realizations = read_jsonl(paths["g5_realization"])
    satellite = read_json(paths["satellite"])
    if len(g5_runs) != 1 or len(realizations) != 1:
        raise ValueError(f"Run {run_id} requires one G5 run and one G5 realization record")
    if not (
        design.get("design_index") == run.get("design_index")
        and run.get("run_id") == target.get("run_id") == satellite.get("run_id") == run_id
        and run.get("run_key") == target.get("run_key") == satellite.get("run_key")
        and run.get("design_id") == design.get("design_id")
        and run_id == int(run["design_index"]) * REALIZATIONS_PER_DESIGN + int(run["realization_index"])
    ):
        raise ValueError(f"Authoritative identity mismatch for run {run_id}")
    if len(g5_steps) != 11:
        raise ValueError(f"Run {run_id} must contain 11 authoritative G5 sampled states")
    ordered_steps = sorted(g5_steps, key=lambda value: int(value["metrics"]["timestep_index"]))
    indices = [int(value["metrics"]["timestep_index"]) for value in ordered_steps]
    if indices != list(range(11)):
        raise ValueError(f"Run {run_id} G5 timesteps are not ordered 0 through 10")
    overall_values = [
        _canonical_float(value["metrics"]["failure_adjusted_overall_service_fraction"], "failure_adjusted_overall_service_fraction")
        for value in ordered_steps
    ]
    ground_values = [
        _canonical_float(value["metrics"]["failure_adjusted_ground_service_fraction"], "failure_adjusted_ground_service_fraction")
        for value in ordered_steps
    ]
    space_values = [
        _canonical_float(value["metrics"]["space_gcc_fraction_original"], "space_gcc_fraction_original")
        for value in ordered_steps
    ]
    threshold = float(SERVICE_THRESHOLD)
    breaches = [value < threshold for value in overall_values]
    recorded_met = [bool(value["metrics"]["overall_threshold_met"]) for value in ordered_steps]
    if recorded_met != [not value for value in breaches]:
        raise ValueError(f"Run {run_id} authoritative G5 threshold semantics disagree")
    overall_min = _canonical_float(
        target["failure_adjusted_overall_service_fraction_min"],
        "failure_adjusted_overall_service_fraction_min",
    )
    margin = overall_min - threshold
    recorded_breach = bool(target["overall_threshold_breach_any"])
    if recorded_breach != (margin < 0.0) or recorded_breach != any(breaches):
        raise ValueError(f"Run {run_id} target and 0.80 boundary disagree")
    summary = g5_runs[0]["summary"]
    if (
        bool(summary["overall_threshold_breach_any"]) != recorded_breach
        or int(summary["overall_threshold_breach_timestep_count"]) != sum(breaches)
        or _canonical_float(
            summary["failure_adjusted_overall_service_fraction_min"],
            "failure_adjusted_overall_service_fraction_min",
        )
        != overall_min
    ):
        raise ValueError(f"Run {run_id} G5 run summary disagrees with target or steps")
    minimum_timestep = min(range(len(overall_values)), key=lambda index: (overall_values[index], index))
    first_breach = next((index for index, value in enumerate(breaches) if value), None)
    last_breach = next((index for index in range(len(breaches) - 1, -1, -1) if breaches[index]), None)
    target_values: dict[str, Any] = {}
    for field in TARGET_FIELDS:
        target_values[field] = target[field] if isinstance(target[field], bool) else _canonical_float(target[field], field)
    realization = realizations[0]["realization"]
    row: dict[str, Any] = {
        "run_id": run_id,
        "run_key": run["run_key"],
        "design_id": design["design_id"],
        "realization_id": run["realization_id"],
        "design_index": int(design["design_index"]),
        "realization_index": int(run["realization_index"]),
        "split": run["split_assignment"],
        "doe_stratum": design["doe_stratum"],
        "satellite_seed": int(run["satellite_seed"]),
        "ground_failure_seed": int(run["ground_failure_seed"]),
        "ground_selection_seed": int(run["ground_selection_seed"]),
        **target_values,
        "overall_boundary_margin": margin,
        "absolute_boundary_distance": abs(margin),
        "breach_severity_below_0_80": max(0.0, -margin),
        "temporal_breach_count": sum(breaches),
        "temporal_breach_fraction": sum(breaches) / len(breaches),
        "first_overall_breach_timestep": first_breach,
        "last_overall_breach_timestep": last_breach,
        "longest_overall_breach_streak": _longest_true_streak(breaches),
        "minimum_overall_service_timestep": minimum_timestep,
        "number_of_recoveries_above_threshold": _recovery_count(breaches),
        "sampled_state_count": len(overall_values),
        "overall_service_sequence": overall_values,
        "ground_service_sequence": ground_values,
        "space_gcc_sequence": space_values,
        "failed_ground_station_count": int(realization["failed_ground_station_count"]),
        "failed_ground_station_ids": realization["failed_station_ids"],
        "failed_satellite_node_count": len(satellite["failed_nodes"]),
        "failed_satellite_nodes": satellite["failed_nodes"],
        "failed_satellite_edge_count": len(satellite["failed_edges"]),
        "failed_satellite_edges": satellite["failed_edges"],
    }
    for field, value in design.items():
        if field not in row:
            if field in {
                "altitude_km",
                "inclination_deg",
                "satellite_node_failure_probability",
                "satellite_edge_failure_probability",
                "ground_station_failure_probability",
                "minimum_elevation_deg",
                "space_gcc_threshold",
                "ground_service_threshold",
                "max_isl_distance_km",
            }:
                row[field] = _canonical_float(value, field)
            else:
                row[field] = value
    return row


def extract_corpus(root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows = [extract_run(root, run_id) for run_id in range(RUN_COUNT)]
    run_ids = [row["run_id"] for row in run_rows]
    run_keys = [row["run_key"] for row in run_rows]
    pairs = [(row["design_id"], row["realization_id"]) for row in run_rows]
    if run_ids != list(range(RUN_COUNT)) or len(set(run_keys)) != RUN_COUNT or len(set(pairs)) != RUN_COUNT:
        raise ValueError("Frozen run identities are missing, extra, duplicated, or unordered")
    design_groups: dict[str, list[dict[str, Any]]] = {}
    for row in run_rows:
        design_groups.setdefault(row["design_id"], []).append(row)
    if len(design_groups) != DESIGN_COUNT:
        raise ValueError("Frozen corpus does not contain exactly 100 designs")
    design_field_names = tuple(read_json(_run_paths(Path(root), 0)["design"]).keys())
    design_rows: list[dict[str, Any]] = []
    for design_id in sorted(design_groups, key=lambda value: int(value[1:])):
        group = sorted(design_groups[design_id], key=lambda value: value["realization_index"])
        if len(group) != REALIZATIONS_PER_DESIGN or len({row["split"] for row in group}) != 1:
            raise ValueError(f"Design {design_id} lacks five colocated realizations")
        margins = [float(row["overall_boundary_margin"]) for row in group]
        overall_minima = [float(row["failure_adjusted_overall_service_fraction_min"]) for row in group]
        non_breach_count = sum(not row["overall_threshold_breach_any"] for row in group)
        labels = {
            5: "consistently resilient",
            4: "mostly resilient",
            3: "mostly resilient",
            2: "mixed",
            1: "mixed",
            0: "consistently breached",
        }
        first = group[0]
        design = {key: first[key] for key in design_field_names}
        design["split"] = first["split"]
        design.update(
            {
                "non_breach_realization_count": non_breach_count,
                "breach_realization_count": REALIZATIONS_PER_DESIGN - non_breach_count,
                "non_breach_realization_fraction": non_breach_count / REALIZATIONS_PER_DESIGN,
                "minimum_boundary_margin": min(margins),
                "maximum_boundary_margin": max(margins),
                "mean_boundary_margin": fmean(margins),
                "boundary_margin_population_stddev": pstdev(margins),
                "minimum_overall_service": min(overall_minima),
                "mean_overall_service": fmean(overall_minima),
                "maximum_overall_service": max(overall_minima),
                "mean_temporal_breach_fraction": fmean(
                    float(row["temporal_breach_fraction"]) for row in group
                ),
                "maximum_temporal_breach_fraction": max(
                    float(row["temporal_breach_fraction"]) for row in group
                ),
                "descriptive_resilience_class": labels[non_breach_count],
                "run_ids": [row["run_id"] for row in group],
            }
        )
        design_rows.append(design)
    _validate_expected_outcome(run_rows, design_rows)
    return run_rows, design_rows


def _validate_expected_outcome(
    run_rows: list[dict[str, Any]], design_rows: list[dict[str, Any]]
) -> None:
    split_counts = Counter(row["split"] for row in run_rows)
    if dict(split_counts) != EXPECTED_SPLIT_RUN_COUNTS:
        raise ValueError(f"Frozen run split cardinality changed: {dict(split_counts)}")
    design_split_counts = Counter(row["split"] for row in design_rows)
    if dict(design_split_counts) != EXPECTED_SPLIT_DESIGN_COUNTS:
        raise ValueError(f"Frozen design split cardinality changed: {dict(design_split_counts)}")
    observed_counters = {
        split: Counter(bool(row["overall_threshold_breach_any"]) for row in run_rows if row["split"] == split)
        for split in EXPECTED_CLASS_COUNTS
    }
    observed = {
        split: {False: observed_counters[split][False], True: observed_counters[split][True]}
        for split in EXPECTED_CLASS_COUNTS
    }
    if any(observed[split] != EXPECTED_CLASS_COUNTS[split] for split in EXPECTED_CLASS_COUNTS):
        raise ValueError(f"Frozen class counts changed: {observed}")
    non_breach_ids = tuple(row["run_id"] for row in run_rows if not row["overall_threshold_breach_any"])
    if non_breach_ids != EXPECTED_NON_BREACH_RUN_IDS:
        raise ValueError(f"Frozen non-breach run set changed: {non_breach_ids}")
