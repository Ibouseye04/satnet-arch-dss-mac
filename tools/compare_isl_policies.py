from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import networkx as nx

from satnet.metrics.labels import compute_gcc_size, compute_partitioned
from satnet.network.hypatia_adapter import HypatiaAdapter
from satnet.simulation.tier1_rollout import DEFAULT_EPOCH_ISO


BASELINE_RUNS = Path("artifacts/controlled_validation/baseline_no_failures/tier1_design_runs.csv")
OUT_DIR = Path("artifacts/controlled_validation/isl_policy_comparison")
GCC_THRESHOLD = 0.8


@dataclass(frozen=True)
class PolicySpec:
    name: str
    isl_policy: str
    adjacent_search_k: int
    max_inter_plane_links_per_sat: int = 1


@dataclass
class RunPolicyMetrics:
    policy: str
    run_id: int
    num_planes: int
    sats_per_plane: int
    total_satellites: int
    inclination_deg: float
    altitude_km: float
    duration_minutes: int
    step_seconds: int
    num_steps: int
    partition_any: int
    partition_fraction: float
    gcc_frac_min: float
    median_node_degree: float
    mean_degree_median: float
    median_edge_count: float
    median_isolated_nodes: float
    median_components: float
    accepted_intra_plane_links_per_step: float
    accepted_inter_plane_links_per_step: float
    total_candidate_links_per_step: float
    links_rejected_los_per_step: float
    links_rejected_budget_per_step: float
    runtime_seconds: float


def _load_designs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _median(values: Iterable[float]) -> float:
    items = list(values)
    return float(statistics.median(items)) if items else 0.0


def _graph_metrics(G: nx.Graph, total_satellites: int) -> dict[str, float]:
    degrees = [degree for _, degree in G.degree()]
    gcc_size = compute_gcc_size(G)
    gcc_frac_original = gcc_size / total_satellites if total_satellites > 0 else 0.0
    return {
        "num_edges": float(G.number_of_edges()),
        "median_node_degree": _median(float(degree) for degree in degrees),
        "mean_degree": (2.0 * G.number_of_edges() / G.number_of_nodes()) if G.number_of_nodes() else 0.0,
        "isolated_nodes": float(sum(1 for degree in degrees if degree == 0)),
        "num_components": float(nx.number_connected_components(G)) if G.number_of_nodes() else 0.0,
        "gcc_frac_original": gcc_frac_original,
        "partitioned": float(compute_partitioned(gcc_frac_original, GCC_THRESHOLD)),
    }


def _run_policy_for_design(design: dict[str, str], policy: PolicySpec) -> tuple[RunPolicyMetrics, list[dict[str, object]]]:
    run_id = int(design["run_id"])
    num_planes = int(design["num_planes"])
    sats_per_plane = int(design["sats_per_plane"])
    total_satellites = int(design["total_satellites"])
    inclination_deg = float(design["inclination_deg"])
    altitude_km = float(design["altitude_km"])
    duration_minutes = int(design["duration_minutes"])
    step_seconds = int(design["step_seconds"])
    epoch_iso = design.get("epoch_iso", DEFAULT_EPOCH_ISO)
    collect_examples = 200 if policy.isl_policy == "grid_adaptive" and run_id < 10 else 0

    start = time.perf_counter()
    adapter = HypatiaAdapter(
        num_planes=num_planes,
        sats_per_plane=sats_per_plane,
        inclination_deg=inclination_deg,
        altitude_km=altitude_km,
        epoch=__import__("datetime").datetime.fromisoformat(epoch_iso),
    )
    try:
        adapter.generate_tles()
        _, stats = adapter.calculate_isls(
            duration_minutes=duration_minutes,
            step_seconds=step_seconds,
            isl_policy=policy.isl_policy,
            adjacent_search_k=policy.adjacent_search_k,
            max_inter_plane_links_per_sat=policy.max_inter_plane_links_per_sat,
            collect_adaptive_examples=collect_examples,
        )
        step_metrics = [_graph_metrics(G, total_satellites) for _, G in adapter.iter_graphs()]
    finally:
        adapter.cleanup()
    runtime_seconds = time.perf_counter() - start

    num_steps = len(step_metrics)
    partition_flags = [int(metric["partitioned"]) for metric in step_metrics]
    metrics = RunPolicyMetrics(
        policy=policy.name,
        run_id=run_id,
        num_planes=num_planes,
        sats_per_plane=sats_per_plane,
        total_satellites=total_satellites,
        inclination_deg=inclination_deg,
        altitude_km=altitude_km,
        duration_minutes=duration_minutes,
        step_seconds=step_seconds,
        num_steps=num_steps,
        partition_any=1 if any(partition_flags) else 0,
        partition_fraction=(sum(partition_flags) / num_steps) if num_steps else 0.0,
        gcc_frac_min=min((metric["gcc_frac_original"] for metric in step_metrics), default=0.0),
        median_node_degree=_median(metric["median_node_degree"] for metric in step_metrics),
        mean_degree_median=_median(metric["mean_degree"] for metric in step_metrics),
        median_edge_count=_median(metric["num_edges"] for metric in step_metrics),
        median_isolated_nodes=_median(metric["isolated_nodes"] for metric in step_metrics),
        median_components=_median(metric["num_components"] for metric in step_metrics),
        accepted_intra_plane_links_per_step=stats.accepted_intra_plane_links / num_steps if num_steps else 0.0,
        accepted_inter_plane_links_per_step=stats.accepted_inter_plane_links / num_steps if num_steps else 0.0,
        total_candidate_links_per_step=stats.total_candidate_links / num_steps if num_steps else 0.0,
        links_rejected_los_per_step=stats.links_rejected_los / num_steps if num_steps else 0.0,
        links_rejected_budget_per_step=stats.links_rejected_budget / num_steps if num_steps else 0.0,
        runtime_seconds=runtime_seconds,
    )
    examples = []
    for example in stats.adaptive_selection_examples:
        original = example["original_adjacent_candidate"]
        candidates = example["adaptive_candidates"]
        original_outcomes = [candidate for candidate in candidates if candidate["sat_id"] == original["sat_id"]]
        selected = example["selected"]
        if original_outcomes and not bool(original_outcomes[0]["los"]) and selected:
            selected_ids = {item["sat_id"] for item in selected}
            if original["sat_id"] not in selected_ids:
                enriched = dict(example)
                enriched["policy"] = policy.name
                enriched["run_id"] = run_id
                examples.append(enriched)
    return metrics, examples[:10]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[RunPolicyMetrics]) -> list[dict[str, object]]:
    result = []
    for policy in sorted({row.policy for row in rows}):
        group = [row for row in rows if row.policy == policy]
        result.append({
            "policy": policy,
            "num_runs": len(group),
            "partition_rate": sum(row.partition_any for row in group) / len(group),
            "median_gcc_frac_min": _median(row.gcc_frac_min for row in group),
            "median_degree": _median(row.median_node_degree for row in group),
            "median_mean_degree": _median(row.mean_degree_median for row in group),
            "median_isolated_nodes": _median(row.median_isolated_nodes for row in group),
            "median_components": _median(row.median_components for row in group),
            "median_edge_count": _median(row.median_edge_count for row in group),
            "median_accepted_inter_plane_links": _median(row.accepted_inter_plane_links_per_step for row in group),
            "median_accepted_intra_plane_links": _median(row.accepted_intra_plane_links_per_step for row in group),
            "median_los_rejected_links": _median(row.links_rejected_los_per_step for row in group),
            "median_budget_rejected_links": _median(row.links_rejected_budget_per_step for row in group),
            "total_runtime_seconds": sum(row.runtime_seconds for row in group),
            "median_runtime_seconds": _median(row.runtime_seconds for row in group),
        })
    return result


def _write_report(path: Path, summary: list[dict[str, object]]) -> None:
    headers = [
        "policy",
        "partition_rate",
        "median_gcc_frac_min",
        "median_degree",
        "median_isolated_nodes",
        "median_components",
        "median_edge_count",
        "median_accepted_inter_plane_links",
        "total_runtime_seconds",
    ]
    lines = ["# Tier 1 ISL Policy Comparison", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in summary:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n")


def _write_plots(summary: list[dict[str, object]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    policies = [row["policy"] for row in summary]
    plot_specs = [
        ("partition_rate", "Partition rate", "partition_rate_by_policy.png"),
        ("median_gcc_frac_min", "Median gcc_frac_min", "gcc_frac_min_by_policy.png"),
        ("median_isolated_nodes", "Median isolated nodes", "isolated_nodes_by_policy.png"),
        ("median_edge_count", "Median edge count", "edge_count_by_policy.png"),
        ("total_runtime_seconds", "Total runtime seconds", "runtime_by_policy.png"),
    ]
    manifest = []
    for key, title, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(policies, [float(row[key]) for row in summary])
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        path = plots_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        manifest.append(str(path))
    (out_dir / "plot_manifest.txt").write_text("\n".join(manifest) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    designs = _load_designs(BASELINE_RUNS)
    policies = [
        PolicySpec("grid_fixed", "grid_fixed", 1),
        PolicySpec("grid_adaptive_k1", "grid_adaptive", 1),
        PolicySpec("grid_adaptive_k2", "grid_adaptive", 2),
        PolicySpec("grid_adaptive_k3", "grid_adaptive", 3),
    ]
    rows: list[RunPolicyMetrics] = []
    examples: list[dict[str, object]] = []
    for policy in policies:
        print(f"Running {policy.name}...")
        for design in designs:
            metrics, run_examples = _run_policy_for_design(design, policy)
            rows.append(metrics)
            examples.extend(run_examples)
    row_dicts = [asdict(row) for row in rows]
    summary = _summarize(rows)
    _write_csv(OUT_DIR / "isl_policy_run_metrics.csv", row_dicts)
    _write_csv(OUT_DIR / "isl_policy_summary.csv", summary)
    (OUT_DIR / "adaptive_selection_examples.json").write_text(json.dumps(examples[:40], indent=2))
    _write_report(OUT_DIR / "isl_policy_comparison_report.md", summary)
    _write_plots(summary, OUT_DIR)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
