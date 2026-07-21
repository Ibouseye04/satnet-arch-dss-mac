from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from statistics import fmean, median, pstdev
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from satnet.experiments.final_class_support.constants import (
    ANALYSIS_SCHEMA_VERSION,
    AUGMENTATION_CONTRACT_VERSION,
    BOUNDARY_BANDS,
    CONTRACT_SPECIFICATION_HASH,
    DOE_RANGES,
    FREEZE_ARCHIVE_SHA256,
    GENERATION_LEDGER_SHA256,
    NEIGHBOR_FEATURES,
    NUMERIC_DESIGN_PARAMETERS,
    PRODUCTION_TOOLING_SHA,
    PROPOSAL_LABEL,
    REGRESSION_TARGETS,
    REPLAY_LEDGER_SHA256,
)


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    series = pd.Series(values, dtype="float64")
    return {
        "minimum": float(series.min()),
        "q01": float(series.quantile(0.01)),
        "q05": float(series.quantile(0.05)),
        "q10": float(series.quantile(0.10)),
        "q25": float(series.quantile(0.25)),
        "median": float(series.quantile(0.50)),
        "q75": float(series.quantile(0.75)),
        "q90": float(series.quantile(0.90)),
        "q95": float(series.quantile(0.95)),
        "q99": float(series.quantile(0.99)),
        "maximum": float(series.max()),
    }


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    value = pd.Series(x, dtype="float64").corr(pd.Series(y, dtype="float64"), method="spearman")
    return None if pd.isna(value) else float(value)


def _iqr(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    series = pd.Series(values, dtype="float64")
    return float(series.median()), float(series.quantile(0.25)), float(series.quantile(0.75))


def _in_band(value: float, band: tuple[str, Any, Any, bool, bool]) -> bool:
    _, lower, upper, lower_inclusive, upper_inclusive = band
    lower_value = None if lower is None else float(lower)
    upper_value = None if upper is None else float(upper)
    if lower_value is not None and (value < lower_value or (value == lower_value and not lower_inclusive)):
        return False
    if upper_value is not None and (value > upper_value or (value == upper_value and not upper_inclusive)):
        return False
    return True


def boundary_region_summary(
    run_rows: Sequence[dict[str, Any]], design_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    run_margins = [float(row["overall_boundary_margin"]) for row in run_rows]
    design_margins = [float(row["mean_boundary_margin"]) for row in design_rows]
    bands = [
        {
            "band": band[0],
            "run_count": sum(_in_band(value, band) for value in run_margins),
            "design_mean_margin_count": sum(_in_band(value, band) for value in design_margins),
        }
        for band in BOUNDARY_BANDS
    ]
    windows = {
        format(width, ".3f"): {
            "run_count": sum(abs(value) <= width for value in run_margins),
            "design_mean_margin_count": sum(abs(value) <= width for value in design_margins),
            "design_best_realization_count": sum(
                abs(float(row["maximum_boundary_margin"])) <= width for row in design_rows
            ),
        }
        for width in (0.01, 0.025, 0.05, 0.10)
    }
    return {
        "schema_identifier": "satnet.class_support.boundary_summary.v1",
        "boundary": 0.80,
        "band_counts": bands,
        "symmetric_window_counts": windows,
        "run_margin_quantiles": _quantiles(run_margins),
        "design_mean_margin_quantiles": _quantiles(design_margins),
    }


def boundary_rankings(
    run_rows: Sequence[dict[str, Any]], design_rows: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nearest_breach = sorted(
        (row for row in run_rows if row["overall_threshold_breach_any"]),
        key=lambda row: (abs(float(row["overall_boundary_margin"])), row["run_id"]),
    )[:25]
    non_breach = sorted(
        (row for row in run_rows if not row["overall_threshold_breach_any"]),
        key=lambda row: (-float(row["overall_boundary_margin"]), row["run_id"]),
    )
    strongest = sorted(
        (row for row in run_rows if row["overall_threshold_breach_any"]),
        key=lambda row: (float(row["overall_boundary_margin"]), row["run_id"]),
    )[:25]
    run_ranking: list[dict[str, Any]] = []
    for ranking_type, values in (
        ("nearest_breach", nearest_breach),
        ("all_non_breach", non_breach),
        ("strongest_breach", strongest),
    ):
        for rank, row in enumerate(values, start=1):
            run_ranking.append(
                {
                    "ranking_type": ranking_type,
                    "rank": rank,
                    "run_id": row["run_id"],
                    "run_key": row["run_key"],
                    "design_id": row["design_id"],
                    "split": row["split"],
                    "doe_stratum": row["doe_stratum"],
                    "overall_threshold_breach_any": row["overall_threshold_breach_any"],
                    "failure_adjusted_overall_service_fraction_min": row[
                        "failure_adjusted_overall_service_fraction_min"
                    ],
                    "overall_boundary_margin": row["overall_boundary_margin"],
                    "absolute_boundary_distance": row["absolute_boundary_distance"],
                    "temporal_breach_fraction": row["temporal_breach_fraction"],
                }
            )
    design_ranking: list[dict[str, Any]] = []
    for ranking_type, key in (
        ("nearest_by_mean_margin", "mean_boundary_margin"),
        ("nearest_by_best_realization", "maximum_boundary_margin"),
    ):
        ordered = sorted(
            design_rows,
            key=lambda row: (abs(float(row[key])), int(row["design_index"])),
        )[:25]
        for rank, row in enumerate(ordered, start=1):
            design_ranking.append(
                {
                    "ranking_type": ranking_type,
                    "rank": rank,
                    "design_id": row["design_id"],
                    "design_index": row["design_index"],
                    "split": row["split"],
                    "doe_stratum": row["doe_stratum"],
                    "non_breach_realization_count": row["non_breach_realization_count"],
                    "minimum_boundary_margin": row["minimum_boundary_margin"],
                    "mean_boundary_margin": row["mean_boundary_margin"],
                    "maximum_boundary_margin": row["maximum_boundary_margin"],
                    "ranking_margin": row[key],
                }
            )
    return run_ranking, design_ranking


def split_summary(
    run_rows: Sequence[dict[str, Any]], design_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        runs = [row for row in run_rows if row["split"] == split]
        designs = [row for row in design_rows if row["split"] == split]
        non_breach_runs = [row for row in runs if not row["overall_threshold_breach_any"]]
        non_breach_designs = [row for row in designs if row["non_breach_realization_count"] > 0]
        splits[split] = {
            "run_count": len(runs),
            "design_count": len(designs),
            "breach_run_count": len(runs) - len(non_breach_runs),
            "non_breach_run_count": len(non_breach_runs),
            "non_breach_run_ids": [row["run_id"] for row in non_breach_runs],
            "non_breach_design_count": len(non_breach_designs),
            "non_breach_design_ids": [row["design_id"] for row in non_breach_designs],
        }
    return {
        "schema_identifier": "satnet.class_support.split_summary.v1",
        "production_verdict": "PRODUCTION DATASET NOT ACCEPTED",
        "technical_split_failure": "Validation contains zero non-breach runs.",
        "underlying_scientific_support_failure": "Only seven of 500 runs and two of 100 designs have any non-breach support.",
        "outcome_driven_reshuffling_prohibited": True,
        "splits": splits,
        "overall": {
            "run_count": len(run_rows),
            "design_count": len(design_rows),
            "non_breach_run_count": sum(not row["overall_threshold_breach_any"] for row in run_rows),
            "non_breach_design_count": sum(row["non_breach_realization_count"] > 0 for row in design_rows),
        },
    }


def regression_summary(run_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        result[split] = {}
        for target in REGRESSION_TARGETS:
            values = [float(row[target]) for row in run_rows if row["split"] == split]
            result[split][target] = {
                "count": len(values),
                "mean": fmean(values),
                "population_standard_deviation": pstdev(values),
                "unique_value_count": len(set(values)),
                **_quantiles(values),
            }
    return {
        "schema_identifier": "satnet.class_support.regression_summary.v1",
        "assessment": "Evidence supports review of a separate regression-only acceptance contract; it does not constitute regression approval.",
        "production_regression_gates_passed": True,
        "by_split": result,
    }


def parameter_support(
    run_rows: Sequence[dict[str, Any]], design_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_margin = [float(row["overall_boundary_margin"]) for row in run_rows]
    run_mean = [float(row["failure_adjusted_overall_service_fraction_mean"]) for row in run_rows]
    run_min = [float(row["failure_adjusted_overall_service_fraction_min"]) for row in run_rows]
    design_fraction = [float(row["non_breach_realization_fraction"]) for row in design_rows]
    non_breach_runs = [row for row in run_rows if not row["overall_threshold_breach_any"]]
    breach_runs = [row for row in run_rows if row["overall_threshold_breach_any"]]
    near_runs = [row for row in run_rows if abs(float(row["overall_boundary_margin"])) <= 0.10]
    for parameter in NUMERIC_DESIGN_PARAMETERS:
        run_values = [float(row[parameter]) for row in run_rows]
        design_values = [float(row[parameter]) for row in design_rows]
        non_values = [float(row[parameter]) for row in non_breach_runs]
        breach_values = [float(row[parameter]) for row in breach_runs]
        near_values = [float(row[parameter]) for row in near_runs]
        non_median, non_q1, non_q3 = _iqr(non_values)
        breach_median, breach_q1, breach_q3 = _iqr(breach_values)
        quartile_groups: list[dict[str, Any]] = []
        if len(set(design_values)) > 1:
            ranked = pd.qcut(pd.Series(design_values), q=4, duplicates="drop")
            frame = pd.DataFrame(
                {
                    "group": ranked.astype(str),
                    "margin": [float(row["mean_boundary_margin"]) for row in design_rows],
                    "non_breach_fraction": design_fraction,
                }
            )
            for group, values in frame.groupby("group", observed=True, sort=True):
                quartile_groups.append(
                    {
                        "group": group,
                        "design_count": int(len(values)),
                        "mean_boundary_margin": float(values["margin"].mean()),
                        "mean_non_breach_fraction": float(values["non_breach_fraction"].mean()),
                    }
                )
        rows.append(
            {
                "parameter": parameter,
                "parameter_type": "numeric",
                "unique_design_values": len(set(design_values)),
                "design_minimum": min(design_values),
                "design_maximum": max(design_values),
                "spearman_run_boundary_margin": _spearman(run_values, run_margin),
                "spearman_design_non_breach_fraction": _spearman(design_values, design_fraction),
                "spearman_run_overall_service_mean": _spearman(run_values, run_mean),
                "spearman_run_overall_service_minimum": _spearman(run_values, run_min),
                "non_breach_minimum": min(non_values) if non_values else None,
                "non_breach_maximum": max(non_values) if non_values else None,
                "non_breach_median": non_median,
                "non_breach_q25": non_q1,
                "non_breach_q75": non_q3,
                "breach_median": breach_median,
                "breach_q25": breach_q1,
                "breach_q75": breach_q3,
                "near_boundary_minimum": min(near_values) if near_values else None,
                "near_boundary_maximum": max(near_values) if near_values else None,
                "quartile_group_summary": quartile_groups,
                "small_sample_limitation": "Seven non-breach runs from two designs; descriptive, non-causal evidence only.",
            }
        )
    for parameter in (
        "doe_stratum",
        "isl_policy",
        "satellite_failure_model",
        "orbital_engine",
        "duration_minutes",
        "step_seconds",
        "minimum_elevation_deg",
        "space_gcc_threshold",
        "ground_service_threshold",
    ):
        groups: list[dict[str, Any]] = []
        values = sorted({str(row[parameter]) for row in design_rows})
        for value in values:
            designs = [row for row in design_rows if str(row[parameter]) == value]
            groups.append(
                {
                    "value": value,
                    "design_count": len(designs),
                    "mean_boundary_margin": fmean(float(row["mean_boundary_margin"]) for row in designs),
                    "mean_non_breach_fraction": fmean(
                        float(row["non_breach_realization_fraction"]) for row in designs
                    ),
                }
            )
        rows.append(
            {
                "parameter": parameter,
                "parameter_type": "categorical_or_fixed",
                "unique_design_values": len(values),
                "grouped_summary": groups,
                "small_sample_limitation": "Descriptive only; fixed parameters cannot explain outcome variation.",
            }
        )
    return rows


def _neighbor_vector(row: dict[str, Any]) -> np.ndarray:
    total = float(row["total_ground_station_count"])
    values = {
        **{key: float(row[key]) for key in DOE_RANGES if key not in {"civilian_fraction", "government_fraction", "military_fraction"}},
        "civilian_fraction": float(row["civilian_count"]) / total,
        "government_fraction": float(row["government_count"]) / total,
        "military_fraction": float(row["military_count"]) / total,
    }
    return np.asarray(
        [
            (values[key] - DOE_RANGES[key][0]) / (DOE_RANGES[key][1] - DOE_RANGES[key][0])
            for key in NEIGHBOR_FEATURES
        ],
        dtype=float,
    )


def nearest_neighbors(design_rows: Sequence[dict[str, Any]], count: int = 8) -> list[dict[str, Any]]:
    non_breach_designs = [row for row in design_rows if row["non_breach_realization_count"] > 0]
    vectors = {row["design_id"]: _neighbor_vector(row) for row in design_rows}
    result: list[dict[str, Any]] = []
    for source in non_breach_designs:
        distances = sorted(
            (
                (float(np.linalg.norm(vectors[source["design_id"]] - vectors[candidate["design_id"]])), candidate)
                for candidate in design_rows
                if candidate["design_id"] != source["design_id"]
            ),
            key=lambda value: (value[0], int(value[1]["design_index"])),
        )[:count]
        for rank, (distance, neighbor) in enumerate(distances, start=1):
            result.append(
                {
                    "source_design_id": source["design_id"],
                    "source_split": source["split"],
                    "source_non_breach_realization_count": source["non_breach_realization_count"],
                    "source_mean_boundary_margin": source["mean_boundary_margin"],
                    "neighbor_rank": rank,
                    "normalized_euclidean_distance": distance,
                    "neighbor_design_id": neighbor["design_id"],
                    "neighbor_split": neighbor["split"],
                    "neighbor_doe_stratum": neighbor["doe_stratum"],
                    "neighbor_non_breach_realization_count": neighbor["non_breach_realization_count"],
                    "neighbor_mean_boundary_margin": neighbor["mean_boundary_margin"],
                    "neighbor_maximum_boundary_margin": neighbor["maximum_boundary_margin"],
                    "neighbor_mean_temporal_breach_fraction": neighbor["mean_temporal_breach_fraction"],
                    "satellite_node_failure_probability_difference": float(
                        neighbor["satellite_node_failure_probability"]
                    )
                    - float(source["satellite_node_failure_probability"]),
                    "satellite_edge_failure_probability_difference": float(
                        neighbor["satellite_edge_failure_probability"]
                    )
                    - float(source["satellite_edge_failure_probability"]),
                    "ground_station_failure_probability_difference": float(
                        neighbor["ground_station_failure_probability"]
                    )
                    - float(source["ground_station_failure_probability"]),
                    "distance_metric": "Euclidean distance over 11 DOE-range-normalized architecture, failure, station-total, and composition-fraction variables.",
                }
            )
    return result


def augmentation_options() -> dict[str, Any]:
    options = []
    for designs, allocation, split in (
        (60, {"resilient_core": 24, "boundary": 24, "global_control": 12}, {"train": 40, "validation": 10, "test": 10}),
        (90, {"resilient_core": 36, "boundary": 36, "global_control": 18}, {"train": 60, "validation": 15, "test": 15}),
        (120, {"resilient_core": 48, "boundary": 48, "global_control": 24}, {"train": 80, "validation": 20, "test": 20}),
    ):
        runs = designs * 5
        lower = round(allocation["resilient_core"] * 5 * 0.20 + allocation["boundary"] * 5 * 0.05)
        upper = round(
            allocation["resilient_core"] * 5 * 0.70
            + allocation["boundary"] * 5 * 0.40
            + allocation["global_control"] * 5 * 0.05
        )
        options.append(
            {
                "new_design_count": designs,
                "new_run_count": runs,
                "region_design_allocation": allocation,
                "split_design_allocation": split,
                "scenario_non_breach_run_range": [lower, upper],
                "scenario_range_status": "Exploratory scenario, not a probability interval or guarantee.",
                "estimated_generation_bytes": round(1336139056 / 500 * runs),
                "estimated_replay_bytes": round(1410137 / 500 * runs),
                "estimated_generation_seconds": 1438.5169468 / 500 * runs,
                "estimated_replay_seconds": 875.8138745 / 500 * runs,
                "scientific_advantages": {
                    60: "Lowest cost; useful as a discovery-scale augmentation but weak independent split support.",
                    90: "Balanced region/split allocation with materially stronger independent validation and test design support.",
                    120: "Strongest coverage and class-support opportunity; highest cost and greatest targeted-population shift.",
                }[designs],
                "scientific_risks": {
                    60: "May still miss stable non-breach support in validation or test.",
                    90: "Targeted regions may remain realization-sensitive; requires control-region comparison.",
                    120: "May overinvest before the resilient region is scientifically localized.",
                }[designs],
            }
        )
    return {
        "schema_identifier": "satnet.class_support.augmentation_options.v1",
        "realizations_per_design": 5,
        "options": options,
        "recommended_new_design_count": 90,
        "recommendation": "Use 90 designs as the separately frozen Stage B target only after a 30-design discovery Stage A.",
    }


def _fractional(index: int, multiplier: int, modulus: int = 997) -> float:
    return ((index + 1) * multiplier % modulus) / modulus


def _counts(total: int, index: int) -> tuple[int, int, int]:
    patterns = ((4, 3, 3), (3, 4, 3), (3, 3, 4), (5, 3, 2), (2, 4, 4), (4, 2, 4))
    weights = patterns[index % len(patterns)]
    raw = [total * value / sum(weights) for value in weights]
    values = [max(1, int(math.floor(value))) for value in raw]
    while sum(values) < total:
        values[(index + sum(values)) % 3] += 1
    while sum(values) > total:
        position = max(range(3), key=lambda item: values[item])
        values[position] -= 1
    return values[0], values[1], values[2]


def proposed_augmentation_designs() -> list[dict[str, Any]]:
    allocation = {
        "resilient_core": {"train": 24, "validation": 6, "test": 6},
        "boundary": {"train": 24, "validation": 6, "test": 6},
        "global_control": {"train": 12, "validation": 3, "test": 3},
    }
    rows: list[dict[str, Any]] = []
    index = 0
    for region in ("resilient_core", "boundary", "global_control"):
        region_index = 0
        for split in ("train", "validation", "test"):
            for _ in range(allocation[region][split]):
                u = _fractional(index, 613)
                v = _fractional(index, 421)
                w = _fractional(index, 281)
                if region == "resilient_core":
                    num_planes = 6
                    sats_per_plane = 8
                    altitude = 850.0 + 350.0 * u
                    inclination = 65.0 + 33.0 * v
                    node_probability = 0.04 * w
                    edge_probability = 0.05 * u
                    ground_probability = 0.05 * v
                    total_ground = 20 + 5 * (region_index % 4)
                elif region == "boundary":
                    num_planes = 5 + (region_index % 2)
                    sats_per_plane = 7 + ((region_index // 2) % 2)
                    altitude = 650.0 + 350.0 * u
                    inclination = 50.0 + 35.0 * v
                    node_probability = 0.03 + 0.07 * w
                    edge_probability = 0.03 + 0.09 * u
                    ground_probability = 0.03 + 0.12 * v
                    total_ground = 15 + 5 * (region_index % 5)
                else:
                    num_planes = 4 + region_index % 3
                    sats_per_plane = 5 + (region_index // 3) % 4
                    altitude = 300.0 + 900.0 * u
                    inclination = 30.0 + 68.0 * v
                    node_probability = 0.2 * w
                    edge_probability = 0.25 * u
                    ground_probability = 0.4 * v
                    totals = (6, 10, 15, 20, 25, 30, 35, 40, 45, 50)
                    total_ground = totals[region_index % len(totals)]
                civilian, government, military = _counts(total_ground, index)
                rows.append(
                    {
                        "proposal_label": PROPOSAL_LABEL,
                        "proposal_status": "NOT_FROZEN",
                        "simulation_authorized": False,
                        "augmentation_design_id": f"AUGV1-D{index:03d}",
                        "augmentation_design_index": index,
                        "run_identity_template": f"AUGV1-D{index:03d}-R00..R04",
                        "intended_region": region,
                        "intended_split": split,
                        "realizations_per_design": 5,
                        "num_planes": num_planes,
                        "sats_per_plane": sats_per_plane,
                        "configured_satellite_count": num_planes * sats_per_plane,
                        "altitude_km": altitude,
                        "inclination_deg": inclination,
                        "phasing_factor": 1,
                        "satellite_node_failure_probability": node_probability,
                        "satellite_edge_failure_probability": edge_probability,
                        "civilian_count": civilian,
                        "government_count": government,
                        "military_count": military,
                        "total_ground_station_count": total_ground,
                        "ground_station_failure_probability": ground_probability,
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
                        "space_gcc_threshold": 0.80,
                        "ground_service_threshold": 0.80,
                        "seed_status": "NOT_ISSUED",
                        "near_neighbor_control_status": "REQUIRES_PRE_FREEZE_CLUSTER_AUDIT",
                    }
                )
                index += 1
                region_index += 1
    if len(rows) != 90 or len({row["augmentation_design_id"] for row in rows}) != 90:
        raise ValueError("Augmentation proposal must contain 90 unique design identities")
    parameter_keys = (
        "num_planes",
        "sats_per_plane",
        "altitude_km",
        "inclination_deg",
        "satellite_node_failure_probability",
        "satellite_edge_failure_probability",
        "civilian_count",
        "government_count",
        "military_count",
        "ground_station_failure_probability",
    )
    signatures = {tuple(row[key] for key in parameter_keys) for row in rows}
    if len(signatures) != len(rows):
        raise ValueError("Augmentation proposal contains duplicate scientific designs")
    return rows


def augmentation_contract(proposal_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    region_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in proposal_rows:
        region_split[row["intended_region"]][row["intended_split"]] += 1
    cross_split_distances = [
        float(np.linalg.norm(_neighbor_vector(first) - _neighbor_vector(second)))
        for index, first in enumerate(proposal_rows)
        for second in proposal_rows[index + 1 :]
        if first["intended_split"] != second["intended_split"]
    ]
    near_neighbor_radius = 0.10
    return {
        "contract_version": AUGMENTATION_CONTRACT_VERSION,
        "proposal_status": "NOT_FROZEN",
        "simulation_authorized": False,
        "proposal_label": PROPOSAL_LABEL,
        "base_corpus_identity": {
            "name": "SATNET Final Integrated Production Corpus v1",
            "verdict": "PRODUCTION DATASET NOT ACCEPTED",
            "tooling_sha": PRODUCTION_TOOLING_SHA,
            "contract_specification_hash": CONTRACT_SPECIFICATION_HASH,
            "generation_ledger_sha256": GENERATION_LEDGER_SHA256,
            "replay_ledger_sha256": REPLAY_LEDGER_SHA256,
            "freeze_archive_sha256": FREEZE_ARCHIVE_SHA256,
        },
        "augmentation_purpose": "Add independent non-breach and threshold-transition design support without changing any original run, design, target, or split.",
        "recommended_strategy": "STAGED",
        "stage_a": {
            "status": "FUTURE_SEPARATE_DISCOVERY_CONTRACT_REQUIRED",
            "design_count": 30,
            "realization_count_per_design": 5,
            "run_count": 150,
            "region_allocation": {"resilient_core": 12, "boundary": 12, "global_control": 6},
            "split_allocation": {"train": 20, "validation": 5, "test": 5},
            "region_split_allocation": {
                "resilient_core": {"train": 8, "validation": 2, "test": 2},
                "boundary": {"train": 8, "validation": 2, "test": 2},
                "global_control": {"train": 4, "validation": 1, "test": 1},
            },
            "acceptance_criteria": [
                "All 150 runs generate and replay exactly with no substitutions or mutations.",
                "At least two resilient-core designs produce at least three of five non-breach realizations.",
                "At least four boundary designs collectively sample both margin signs; otherwise revise the boundary model under a new proposal.",
                "The discovery test partition remains sealed and cannot influence Stage B DOE selection.",
            ],
            "permitted_influence_on_stage_b": [
                "Stage A train and validation direct boundary-margin and neighborhood evidence.",
                "Observed storage and runtime for planning.",
            ],
            "prohibited_adaptation": "No silent design, seed, split, gate, or target change within Stage A. Stage B requires a separately reviewed and frozen contract.",
        },
        "stage_b": {
            "status": "PROPOSAL_ONLY",
            "design_count": len(proposal_rows),
            "realizations_per_design": 5,
            "run_count": len(proposal_rows) * 5,
            "doe_regions": {
                "resilient_core": "Low-failure, high-capacity region anchored by D000 and bounded toward D001.",
                "boundary": "Transition region around D001 and nearest breached designs, spanning architecture and failure gradients.",
                "global_control": "Broad deterministic coverage of the original admissible DOE range.",
            },
            "region_split_allocation": {region: dict(values) for region, values in region_split.items()},
            "preliminary_near_neighbor_assessment": {
                "distance_metric": "Euclidean distance over 11 DOE-range-normalized architecture, failure, station-total, and composition-fraction variables.",
                "predeclared_candidate_radius": near_neighbor_radius,
                "minimum_cross_split_distance": min(cross_split_distances),
                "cross_split_pairs_within_radius": sum(
                    value <= near_neighbor_radius for value in cross_split_distances
                ),
                "status": "PRELIMINARY_PASS_REQUIRES_PRE_FREEZE_REAUDIT",
            },
        },
        "new_design_identity_namespace": "AUGV1-D000 through AUGV1-D089",
        "new_run_identity_namespace": "AUGV1-Dxxx-R00 through R04; separate from frozen integer run IDs 0 through 499",
        "seed_policy": {
            "status": "NOT_ISSUED",
            "requirement": "Derive and freeze new domain-separated design, satellite, ground-selection, and ground-failure seeds before simulation; no seed substitution.",
        },
        "output_root_policy": "Use new immutable Stage A and Stage B generation/replay roots outside all production and freeze roots.",
        "target_schema_reference": "Frozen eight-target schema and unchanged 0.80 thresholds; any change requires a new reviewed scientific contract.",
        "class_support_gates": {
            "minimum_non_breach_designs": {"train": 12, "validation": 4, "test": 4},
            "minimum_non_breach_runs": {"train": 50, "validation": 14, "test": 14},
            "minimum_breach_designs": {"train": 40, "validation": 10, "test": 10},
            "minimum_breach_runs": {"train": 200, "validation": 60, "test": 60},
            "maximum_majority_to_minority_run_ratio": {"train": 12.0, "validation": 10.0, "test": 10.0},
            "minimum_boundary_region_designs": {"train": 10, "validation": 3, "test": 3},
            "minimum_distinct_run_margin_values": {"train": 30, "validation": 12, "test": 12},
            "priority": "Independent design support is binding; repeated realizations from one design cannot satisfy design-level gates.",
        },
        "regression_gates": {
            "status": "REQUIRES_SEPARATE_PREDECLARATION",
            "principle": "Preserve finite [0,1] targets, nonzero split spread, and minimum unique-value support at least as strong as the frozen base contract.",
        },
        "leakage_controls": [
            "Group all five realizations by design.",
            "No original or augmentation design overlap across splits.",
            "No outcome-driven reshuffling, seed substitution, replacement run, or movement of original designs.",
            "Deduplicate exact scientific parameter vectors before freezing.",
            "Build a normalized DOE-distance graph at a predeclared 0.10 radius; assign each connected component wholly to one split or document a stricter justified radius before freeze.",
            "Do not tune against either Stage A or Stage B final test split.",
        ],
        "acceptance_gates": [
            "All generation and replay identities and inventories match.",
            "All class-support gates pass in each combined fixed-original-plus-augmentation split.",
            "All target/boundary and temporal-summary consistency checks pass.",
            "Targeted-region and global-control results are reported separately to expose population shift.",
        ],
        "required_future_audit_steps": [
            "Review Stage A proposal and freeze a separate discovery contract.",
            "Execute Stage A only after independent preflight approval.",
            "Analyze Stage A train/validation while keeping its test partition sealed.",
            "Create, review, and freeze a distinct Stage B contract and seed/run manifests.",
            "Perform clean generation, authoritative replay, protected-science audit, and fail-closed acceptance.",
        ],
    }


def analysis_summary(
    run_rows: Sequence[dict[str, Any]],
    design_rows: Sequence[dict[str, Any]],
    neighbors: Sequence[dict[str, Any]],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    non_breach = [row for row in run_rows if not row["overall_threshold_breach_any"]]
    non_breach_designs = [row for row in design_rows if row["non_breach_realization_count"] > 0]
    nearest_breach = min(
        (row for row in run_rows if row["overall_threshold_breach_any"]),
        key=lambda row: abs(float(row["overall_boundary_margin"])),
    )
    return {
        "schema_identifier": ANALYSIS_SCHEMA_VERSION,
        "non_breach_run_ids": [row["run_id"] for row in non_breach],
        "non_breach_design_ids": [row["design_id"] for row in non_breach_designs],
        "non_breach_design_count": len(non_breach_designs),
        "consistently_resilient_design_ids": [
            row["design_id"] for row in non_breach_designs if row["non_breach_realization_count"] == 5
        ],
        "realization_sensitive_design_ids": [
            row["design_id"] for row in non_breach_designs if row["non_breach_realization_count"] < 5
        ],
        "nearest_breach_run": {
            "run_id": nearest_breach["run_id"],
            "design_id": nearest_breach["design_id"],
            "margin": nearest_breach["overall_boundary_margin"],
        },
        "boundary_windows": boundary["symmetric_window_counts"],
        "neighbor_non_breach_counts": Counter(
            int(row["neighbor_non_breach_realization_count"]) for row in neighbors
        ),
        "interpretation": "D000 is a well-inside resilient anchor; D001 is an exact-boundary, realization-sensitive transition point. No non-breach support occurs in transition or global strata.",
        "small_sample_limitation": "Seven non-breach runs arise from only two pilot-anchor designs; no causal or validated predictive claim is supported.",
    }
