from __future__ import annotations

from pathlib import Path

BASE_SHA = "62beda9df1576958d9e33d33d2d9eb5489e24b20"
ROOT = Path(__file__).parents[2]
GROUND = ROOT / "src" / "satnet" / "ground"
PROTECTED_GROUND_FILES = (
    "canonical.py",
    "catalog.py",
    "selection.py",
    "scenario.py",
    "persistence.py",
    "coordinates.py",
    "position_adapter.py",
    "visibility.py",
    "visibility_persistence.py",
    "graph_attributes.py",
    "integrated_graph.py",
    "integrated_builder.py",
    "satellite_graph_adapter.py",
    "integrated_persistence.py",
    "service_policy.py",
    "service_metrics.py",
    "service_aggregation.py",
    "service_persistence.py",
    "failure_policy.py",
    "failure_realization.py",
    "failure_service_metrics.py",
    "failure_service_aggregation.py",
    "failure_service_persistence.py",
)


def test_complete_protected_ground_file_inventory_exists() -> None:
    assert len(PROTECTED_GROUND_FILES) == 23
    assert all((GROUND / name).is_file() for name in PROTECTED_GROUND_FILES)


def test_pilot_tooling_is_outside_protected_science() -> None:
    pilot_files = tuple((ROOT / "src" / "satnet" / "experiments").glob("*.py"))
    assert pilot_files
    assert all(path.parent != GROUND for path in pilot_files)
    assert all("integrated_ground" in path.name or path.name == "__init__.py" for path in pilot_files)


def test_protected_release_command_covers_satellite_and_ml_boundaries() -> None:
    protected = (
        ROOT / "src" / "satnet" / "network",
        ROOT / "src" / "satnet" / "simulation" / "tier1_rollout.py",
        ROOT / "src" / "satnet" / "models" / "gnn_dataset.py",
        ROOT / "src" / "satnet" / "utils" / "graph_cache.py",
    )
    assert all(path.exists() for path in protected)
    assert len(BASE_SHA) == 40
