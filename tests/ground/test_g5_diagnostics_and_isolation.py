from __future__ import annotations

import ast
from pathlib import Path

from scripts.validation.validate_ground_failure_model import build_diagnostics

ROOT = Path(__file__).parents[2]
GROUND = ROOT / "src" / "satnet" / "ground"
PURE_G5_MODULES = (
    GROUND / "failure_policy.py",
    GROUND / "failure_realization.py",
    GROUND / "failure_service_metrics.py",
    GROUND / "failure_service_aggregation.py",
)
PROHIBITED = (
    "satnet.models",
    "satnet.metrics",
    "satnet.utils.graph_cache",
    "satnet.network",
    "satnet.simulation",
)
PROTECTED_G1_G4 = (
    GROUND / "canonical.py",
    GROUND / "catalog.py",
    GROUND / "selection.py",
    GROUND / "scenario.py",
    GROUND / "persistence.py",
    GROUND / "coordinates.py",
    GROUND / "position_adapter.py",
    GROUND / "visibility.py",
    GROUND / "visibility_persistence.py",
    GROUND / "graph_attributes.py",
    GROUND / "integrated_graph.py",
    GROUND / "integrated_builder.py",
    GROUND / "satellite_graph_adapter.py",
    GROUND / "integrated_persistence.py",
    GROUND / "service_policy.py",
    GROUND / "service_metrics.py",
    GROUND / "service_aggregation.py",
    GROUND / "service_persistence.py",
)


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append(node.module or "")
    return result


def test_diagnostic_harness_covers_persistent_failure_semantics() -> None:
    diagnostics = build_diagnostics()
    assert diagnostics["status"] == "passed"
    zero = diagnostics["zero"]
    full = diagnostics["full"]
    partial = diagnostics["partial"]
    assert zero["failed_station_ids"] == ()
    assert full["failed_station_ids"] == full["selected_station_ids"]
    assert full["adjusted_ground_fraction"] == 0.0
    assert partial["failed_station_ids"]
    assert partial["operational_station_ids"]
    assert len(partial["realization_hash"]) == 64
    assert len(partial["step_sequence_hash"]) == 64
    assert len(partial["run_summary_hash"]) == 64
    assert partial["run_breach_counts"]["ground"] >= 1


def test_pure_g5_modules_have_no_prohibited_direct_imports() -> None:
    violations = [
        (path.name, module)
        for path in PURE_G5_MODULES
        for module in imported_modules(path)
        if module.startswith(PROHIBITED)
    ]
    assert violations == []


def test_g5_orchestration_uses_authoritative_g4_replay_without_model_dependencies() -> None:
    modules = imported_modules(GROUND / "failure_service_persistence.py")
    assert "satnet.ground.service_persistence" in modules
    assert not any(
        module.startswith(("satnet.models", "satnet.metrics", "satnet.utils.graph_cache"))
        for module in modules
    )


def test_protected_satellite_modules_do_not_import_g5() -> None:
    protected = [
        *sorted((ROOT / "src" / "satnet" / "network").rglob("*.py")),
        ROOT / "src" / "satnet" / "simulation" / "tier1_rollout.py",
        ROOT / "src" / "satnet" / "models" / "gnn_dataset.py",
        ROOT / "src" / "satnet" / "utils" / "graph_cache.py",
    ]
    assert all(
        not module.startswith("satnet.ground.failure_")
        for path in protected
        for module in imported_modules(path)
    )


def test_complete_protected_g1_g4_module_list_exists_and_excludes_g5() -> None:
    assert len(PROTECTED_G1_G4) == 18
    assert all(path.is_file() for path in PROTECTED_G1_G4)
    assert all("failure_" not in path.name for path in PROTECTED_G1_G4)
    assert GROUND / "service_policy.py" in PROTECTED_G1_G4
