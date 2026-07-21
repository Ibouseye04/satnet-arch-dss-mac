from __future__ import annotations

from collections import Counter
import html
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Sequence

import numpy as np

from satnet.experiments.final_class_support.constants import DOE_RANGES, REGRESSION_TARGETS

_COLORS = {"train": "#2563eb", "validation": "#d97706", "test": "#059669", "breach": "#dc2626", "non_breach": "#16a34a"}


def _write_svg(path: Path, title: str, body: str, width: int = 1000, height: int = 620) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        f'<text x="40" y="42" font-family="Segoe UI,Arial" font-size="24" font-weight="700" fill="#111827">{html.escape(title)}</text>'
        f'{body}'
        '<text x="40" y="600" font-family="Segoe UI,Arial" font-size="12" fill="#6b7280">SATNET frozen corpus read-only analysis · descriptive, non-causal</text>'
        '</svg>\n'
    )
    path.write_text(payload, encoding="utf-8", newline="\n")


def _axes(x_label: str, y_label: str) -> str:
    return (
        '<line x1="90" y1="540" x2="950" y2="540" stroke="#374151" stroke-width="1.5"/>'
        '<line x1="90" y1="80" x2="90" y2="540" stroke="#374151" stroke-width="1.5"/>'
        f'<text x="520" y="580" text-anchor="middle" font-family="Segoe UI,Arial" font-size="14">{html.escape(x_label)}</text>'
        f'<text x="22" y="310" transform="rotate(-90 22 310)" text-anchor="middle" font-family="Segoe UI,Arial" font-size="14">{html.escape(y_label)}</text>'
    )


def _scale(value: float, minimum: float, maximum: float, low: float, high: float) -> float:
    if maximum == minimum:
        return (low + high) / 2
    return low + (value - minimum) / (maximum - minimum) * (high - low)


def plot_classification_counts(path: Path, run_rows: Sequence[dict[str, Any]]) -> None:
    body = _axes("Frozen split", "Run count")
    maximum = max(sum(row["split"] == split for row in run_rows) for split in _COLORS if split in {"train", "validation", "test"})
    for split_index, split in enumerate(("train", "validation", "test")):
        breach = sum(row["split"] == split and row["overall_threshold_breach_any"] for row in run_rows)
        non_breach = sum(row["split"] == split and not row["overall_threshold_breach_any"] for row in run_rows)
        base_x = 205 + split_index * 270
        for offset, value, color, label in ((0, breach, _COLORS["breach"], "breach"), (75, non_breach, _COLORS["non_breach"], "non-breach")):
            height = 420 * value / maximum
            body += f'<rect x="{base_x + offset}" y="{540-height:.2f}" width="60" height="{height:.2f}" fill="{color}" rx="4"/>'
            body += f'<text x="{base_x + offset + 30}" y="{530-height:.2f}" text-anchor="middle" font-family="Segoe UI,Arial" font-size="13">{value}</text>'
            body += f'<text x="{base_x + offset + 30}" y="558" text-anchor="middle" font-family="Segoe UI,Arial" font-size="11">{label}</text>'
        body += f'<text x="{base_x + 67}" y="85" text-anchor="middle" font-family="Segoe UI,Arial" font-size="14" font-weight="600">{split}</text>'
    _write_svg(path, "Primary classification support by frozen split", body)


def plot_margin_histogram(path: Path, run_rows: Sequence[dict[str, Any]]) -> None:
    margins = [float(row["overall_boundary_margin"]) for row in run_rows]
    bins = np.linspace(min(margins), max(margins), 21)
    body = _axes("Overall boundary margin bin", "Run count")
    histograms = {split: np.histogram([float(row["overall_boundary_margin"]) for row in run_rows if row["split"] == split], bins=bins)[0] for split in ("train", "validation", "test")}
    maximum = max(int(value) for counts in histograms.values() for value in counts)
    for split_index, split in enumerate(("train", "validation", "test")):
        points = []
        for index, value in enumerate(histograms[split]):
            x = 90 + (index + 0.5) / len(histograms[split]) * 860
            y = 540 - int(value) / maximum * 420
            points.append(f"{x:.2f},{y:.2f}")
        body += f'<polyline points="{" ".join(points)}" fill="none" stroke="{_COLORS[split]}" stroke-width="3"/>'
        body += f'<text x="{760 + split_index * 65}" y="85" font-family="Segoe UI,Arial" font-size="12" fill="{_COLORS[split]}">{split}</text>'
    zero_x = _scale(0.0, float(bins[0]), float(bins[-1]), 90, 950)
    body += f'<line x1="{zero_x:.2f}" y1="80" x2="{zero_x:.2f}" y2="540" stroke="#111827" stroke-dasharray="6 5"/>'
    _write_svg(path, "Boundary-margin histogram by split", body)


def plot_margin_ecdf(path: Path, run_rows: Sequence[dict[str, Any]]) -> None:
    margins = [float(row["overall_boundary_margin"]) for row in run_rows]
    minimum, maximum = min(margins), max(margins)
    body = _axes("Overall boundary margin", "Empirical cumulative fraction")
    for split in ("train", "validation", "test"):
        values = sorted(float(row["overall_boundary_margin"]) for row in run_rows if row["split"] == split)
        points = [f'{_scale(value, minimum, maximum, 90, 950):.2f},{540-(index+1)/len(values)*420:.2f}' for index, value in enumerate(values)]
        body += f'<polyline points="{" ".join(points)}" fill="none" stroke="{_COLORS[split]}" stroke-width="2.5"/>'
    zero_x = _scale(0.0, minimum, maximum, 90, 950)
    body += f'<line x1="{zero_x:.2f}" y1="80" x2="{zero_x:.2f}" y2="540" stroke="#111827" stroke-dasharray="6 5"/>'
    _write_svg(path, "Boundary-margin empirical distributions", body)


def plot_design_counts(path: Path, design_rows: Sequence[dict[str, Any]]) -> None:
    counts = Counter(int(row["non_breach_realization_count"]) for row in design_rows)
    body = _axes("Non-breach realizations per design", "Design count")
    maximum = max(counts.values())
    for value in range(6):
        count = counts.get(value, 0)
        x = 145 + value * 135
        height = 420 * count / maximum
        color = _COLORS["non_breach"] if value else _COLORS["breach"]
        body += f'<rect x="{x}" y="{540-height:.2f}" width="80" height="{height:.2f}" fill="{color}" rx="4"/>'
        body += f'<text x="{x+40}" y="{530-height:.2f}" text-anchor="middle" font-family="Segoe UI,Arial" font-size="14">{count}</text>'
        body += f'<text x="{x+40}" y="560" text-anchor="middle" font-family="Segoe UI,Arial" font-size="14">{value}</text>'
    _write_svg(path, "Design-level non-breach realization support", body)


def plot_parameter_margin(path: Path, design_rows: Sequence[dict[str, Any]]) -> None:
    body = _axes("Satellite node failure probability", "Mean boundary margin")
    x_values = [float(row["satellite_node_failure_probability"]) for row in design_rows]
    y_values = [float(row["mean_boundary_margin"]) for row in design_rows]
    for row, x_value, y_value in zip(design_rows, x_values, y_values):
        color = _COLORS["non_breach"] if row["non_breach_realization_count"] else _COLORS[row["split"]]
        x = _scale(x_value, min(x_values), max(x_values), 90, 950)
        y = _scale(y_value, min(y_values), max(y_values), 540, 80)
        radius = 7 if row["non_breach_realization_count"] else 3
        body += f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}" fill-opacity="0.72"/>'
    zero_y = _scale(0.0, min(y_values), max(y_values), 540, 80)
    body += f'<line x1="90" y1="{zero_y:.2f}" x2="950" y2="{zero_y:.2f}" stroke="#111827" stroke-dasharray="6 5"/>'
    _write_svg(path, "Design parameter versus boundary margin", body)


def plot_profiles(path: Path, run_rows: Sequence[dict[str, Any]]) -> None:
    parameters = ("num_planes", "sats_per_plane", "altitude_km", "inclination_deg", "satellite_node_failure_probability", "satellite_edge_failure_probability", "total_ground_station_count", "ground_station_failure_probability")
    groups = {
        "non-breach": [row for row in run_rows if not row["overall_threshold_breach_any"]],
        "near-boundary": [row for row in run_rows if abs(float(row["overall_boundary_margin"])) <= 0.10],
        "all": list(run_rows),
    }
    body = _axes("Normalized design parameter profile", "Mean normalized value")
    colors = {"non-breach": "#16a34a", "near-boundary": "#d97706", "all": "#6b7280"}
    for group, rows in groups.items():
        points = []
        for index, parameter in enumerate(parameters):
            minimum, maximum = DOE_RANGES[parameter]
            value = fmean((float(row[parameter]) - minimum) / (maximum - minimum) for row in rows)
            x = 110 + index / (len(parameters) - 1) * 820
            y = 540 - value * 420
            points.append(f"{x:.2f},{y:.2f}")
            if group == "all":
                body += f'<text x="{x:.2f}" y="558" transform="rotate(25 {x:.2f} 558)" font-family="Segoe UI,Arial" font-size="10">{html.escape(parameter)}</text>'
        body += f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[group]}" stroke-width="3"/>'
    _write_svg(path, "Non-breach and near-boundary parameter profiles", body)


def plot_neighbors(path: Path, neighbors: Sequence[dict[str, Any]]) -> None:
    body = _axes("Normalized Euclidean distance", "Neighbor mean boundary margin")
    x_values = [float(row["normalized_euclidean_distance"]) for row in neighbors]
    y_values = [float(row["neighbor_mean_boundary_margin"]) for row in neighbors]
    for row, x_value, y_value in zip(neighbors, x_values, y_values):
        color = _COLORS["non_breach"] if row["neighbor_non_breach_realization_count"] else _COLORS["breach"]
        x = _scale(x_value, min(x_values), max(x_values), 90, 950)
        y = _scale(y_value, min(y_values), max(y_values), 540, 80)
        body += f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}" fill-opacity="0.75"/>'
    zero_y = _scale(0.0, min(y_values), max(y_values), 540, 80)
    body += f'<line x1="90" y1="{zero_y:.2f}" x2="950" y2="{zero_y:.2f}" stroke="#111827" stroke-dasharray="6 5"/>'
    _write_svg(path, "Nearest-neighbor outcome comparison", body)


def plot_regression(path: Path, run_rows: Sequence[dict[str, Any]]) -> None:
    body = _axes("Regression target and split", "Mean target value")
    short = {target: target.replace("failure_adjusted_", "").replace("_fraction", "") for target in REGRESSION_TARGETS}
    index = 0
    for target in REGRESSION_TARGETS:
        for split in ("train", "validation", "test"):
            values = [float(row[target]) for row in run_rows if row["split"] == split]
            value = fmean(values)
            x = 115 + index * 65
            height = value * 420
            body += f'<rect x="{x}" y="{540-height:.2f}" width="48" height="{height:.2f}" fill="{_COLORS[split]}" rx="3"/>'
            body += f'<text x="{x+24}" y="560" transform="rotate(35 {x+24} 560)" font-family="Segoe UI,Arial" font-size="9">{html.escape(short[target][:18])}:{split[0]}</text>'
            index += 1
    _write_svg(path, "Regression target means by frozen split", body)


def plot_augmentation(path: Path, proposal_rows: Sequence[dict[str, Any]]) -> None:
    body = _axes("Region and split", "Proposed design count")
    combinations = [(region, split) for region in ("resilient_core", "boundary", "global_control") for split in ("train", "validation", "test")]
    maximum = max(sum(row["intended_region"] == region and row["intended_split"] == split for row in proposal_rows) for region, split in combinations)
    for index, (region, split) in enumerate(combinations):
        count = sum(row["intended_region"] == region and row["intended_split"] == split for row in proposal_rows)
        x = 110 + index * 90
        height = count / maximum * 420
        body += f'<rect x="{x}" y="{540-height:.2f}" width="62" height="{height:.2f}" fill="{_COLORS[split]}" rx="3"/>'
        body += f'<text x="{x+31}" y="{530-height:.2f}" text-anchor="middle" font-family="Segoe UI,Arial" font-size="12">{count}</text>'
        body += f'<text x="{x+31}" y="560" transform="rotate(30 {x+31} 560)" font-family="Segoe UI,Arial" font-size="9">{region[:9]}:{split[0]}</text>'
    _write_svg(path, "Proposed Stage B augmentation allocation", body)


def create_plots(
    output_root: Path,
    run_rows: Sequence[dict[str, Any]],
    design_rows: Sequence[dict[str, Any]],
    neighbors: Sequence[dict[str, Any]],
    proposal_rows: Sequence[dict[str, Any]],
) -> None:
    plots: tuple[tuple[str, Callable[..., None], tuple[Any, ...]], ...] = (
        ("classification_counts_by_split.svg", plot_classification_counts, (run_rows,)),
        ("boundary_margin_histogram_by_split.svg", plot_margin_histogram, (run_rows,)),
        ("boundary_margin_empirical_distribution.svg", plot_margin_ecdf, (run_rows,)),
        ("design_non_breach_realization_counts.svg", plot_design_counts, (design_rows,)),
        ("parameter_vs_boundary_margin.svg", plot_parameter_margin, (design_rows,)),
        ("non_breach_near_boundary_profiles.svg", plot_profiles, (run_rows,)),
        ("nearest_neighbor_outcome_comparison.svg", plot_neighbors, (neighbors,)),
        ("regression_target_distributions_by_split.svg", plot_regression, (run_rows,)),
        ("proposed_augmentation_allocation.svg", plot_augmentation, (proposal_rows,)),
    )
    for name, function, arguments in plots:
        function(output_root / "plots" / name, *arguments)
