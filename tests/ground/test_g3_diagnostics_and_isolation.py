from __future__ import annotations

import ast
from pathlib import Path

import networkx as nx

from scripts.validation.validate_integrated_ground_graph import (
    build_diagnostic_snapshots,
    build_diagnostics,
)
from satnet.ground.integrated_graph import operational_snapshot_from_networkx
from satnet.ground.integrated_persistence import (
    make_integrated_graph_record,
    read_integrated_graph_manifest,
    write_integrated_graph_manifest,
)

ROOT = Path(__file__).parents[2]
GROUND = ROOT / "src" / "satnet" / "ground"
PURE_MODULES = (
    GROUND / "graph_attributes.py",
    GROUND / "integrated_graph.py",
    GROUND / "integrated_builder.py",
)
PROHIBITED = (
    "satnet.models",
    "satnet.metrics",
    "satnet.utils.graph_cache",
)


def imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend((alias.name, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append((node.module or "", tuple(alias.name for alias in node.names)))
    return result


def test_diagnostic_harness_covers_required_structural_cases() -> None:
    diagnostics = build_diagnostics()
    names = {item["case"] for item in diagnostics}
    assert names == {
        "one_station_one_satellite_visible",
        "one_station_one_satellite_nonvisible",
        "multiple_stations_multiple_satellites",
        "station_with_no_access",
        "all_satellites_failed",
        "complete_satellite_graph_zero_ground_links",
    }
    by_name = {item["case"]: item for item in diagnostics}
    assert by_name["one_station_one_satellite_visible"]["satellite_ground_edges"] == 1
    assert by_name["one_station_one_satellite_nonvisible"]["satellite_ground_edges"] == 0
    assert by_name["multiple_stations_multiple_satellites"]["satellite_nodes"] == 2
    assert by_name["multiple_stations_multiple_satellites"]["ground_nodes"] == 2
    assert by_name["multiple_stations_multiple_satellites"]["isl_edges"] == 1
    assert by_name["all_satellites_failed"]["satellite_nodes"] == 0
    assert by_name["all_satellites_failed"]["ground_nodes"] == 2
    assert all(item["satellite_projection_valid"] for item in diagnostics)
    assert all(item["visible_edge_validation_valid"] for item in diagnostics)


def test_pure_g3_modules_have_no_prohibited_imports() -> None:
    violations = []
    for path in PURE_MODULES:
        for module, names in imported_modules(path):
            if module.startswith(PROHIBITED):
                violations.append((path.name, module, names))
    assert violations == []


def test_satellite_adapter_import_boundary_is_minimal() -> None:
    imports = imported_modules(GROUND / "satellite_graph_adapter.py")
    satellite_imports = [
        (module, names)
        for module, names in imports
        if module.startswith(("satnet.network", "satnet.simulation"))
    ]
    assert satellite_imports == [
        ("satnet.network.hypatia_adapter", ("HypatiaAdapter",)),
        (
            "satnet.simulation.tier1_rollout",
            ("Tier1FailureRealization", "Tier1RolloutConfig"),
        ),
    ]


def test_protected_upstream_modules_do_not_import_g3() -> None:
    protected = [
        *sorted((ROOT / "src" / "satnet" / "network").rglob("*.py")),
        ROOT / "src" / "satnet" / "simulation" / "tier1_rollout.py",
        ROOT / "src" / "satnet" / "models" / "gnn_dataset.py",
        ROOT / "src" / "satnet" / "utils" / "graph_cache.py",
    ]
    assert all(
        not module.startswith("satnet.ground")
        for path in protected
        for module, _ in imported_modules(path)
    )


def test_source_graph_insertion_order_does_not_change_identity() -> None:
    first = nx.Graph()
    first.graph.update({"b": 2, "a": 1})
    first.add_node(10, label="ten")
    first.add_node(2, label="two")
    first.add_edge(10, 2, distance_km=500.0)
    second = nx.Graph()
    second.graph.update({"a": 1, "b": 2})
    second.add_node(2, label="two")
    second.add_node(10, label="ten")
    second.add_edge(2, 10, distance_km=500.0)
    timestamp = build_diagnostic_snapshots()[0][1].timestamp_utc
    kwargs = {
        "timestep_index": 0,
        "timestamp_utc": timestamp,
        "satellite_config_hash": "1" * 64,
    }
    assert operational_snapshot_from_networkx(graph=first, **kwargs) == operational_snapshot_from_networkx(
        graph=second, **kwargs
    )


def test_integrated_artifact_path_does_not_change_identity(tmp_path: Path) -> None:
    _, snapshot, _, _, ground_design = build_diagnostic_snapshots()[0]
    record = make_integrated_graph_record(
        ground_design=ground_design,
        snapshot=snapshot,
    )
    first = tmp_path / "first" / "integrated.jsonl"
    second = tmp_path / "second" / "integrated.jsonl"
    write_integrated_graph_manifest((record,), first)
    write_integrated_graph_manifest((record,), second)
    assert first.read_bytes() == second.read_bytes()
    assert read_integrated_graph_manifest(first) == read_integrated_graph_manifest(second)
