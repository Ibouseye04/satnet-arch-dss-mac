from __future__ import annotations

import ast
from pathlib import Path

from scripts.validation.validate_ground_service_metrics import build_diagnostics

ROOT = Path(__file__).parents[2]
GROUND = ROOT / "src" / "satnet" / "ground"
PURE_G4_MODULES = (
    GROUND / "service_policy.py",
    GROUND / "service_metrics.py",
    GROUND / "service_aggregation.py",
)
PROHIBITED = (
    "satnet.models",
    "satnet.metrics",
    "satnet.utils.graph_cache",
    "satnet.network",
    "satnet.simulation",
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


def test_diagnostic_harness_covers_required_service_and_temporal_cases() -> None:
    diagnostics = build_diagnostics()
    steps = {value["case"]: value for value in diagnostics["steps"]}
    assert set(steps) == {
        "fully_serviced",
        "station_with_no_access",
        "station_connected_outside_gcc",
        "split_satellite_network",
        "equal_size_gcc_tie_different_attachments",
        "catastrophic_satellite_attrition",
        "mixed_class_service",
        "zero_operational_satellites",
        "threshold_boundary",
        "zero_threshold_semantics",
    }
    assert steps["fully_serviced"]["overall_service_fraction"] == 1.0
    assert steps["station_with_no_access"]["ground_service_fraction"] == 0.75
    assert steps["station_connected_outside_gcc"]["ground_service_fraction"] == 0.0
    assert steps["equal_size_gcc_tie_different_attachments"]["satellite_gcc_ids"] == [0, 1]
    assert steps["catastrophic_satellite_attrition"]["space_gcc_fraction_surviving"] == 1.0
    assert steps["catastrophic_satellite_attrition"]["space_gcc_fraction_original"] == 0.01
    assert steps["zero_operational_satellites"]["satellite_component_count"] == 0
    assert steps["zero_threshold_semantics"]["overall_threshold_met"] is True
    run = diagnostics["run"]
    assert run["case"] == "multi_timestep_disconnect_and_recovery"
    assert run["ground_threshold_breach_any"] is True
    assert run["ground_threshold_breach_timestep_count"] == 1
    assert run["first_ground_threshold_breach_timestep"] == 1
    assert len(run["step_sequence_hash"]) == 64
    assert len(run["run_summary_hash"]) == 64


def test_pure_g4_modules_have_no_prohibited_direct_imports() -> None:
    violations = [
        (path.name, module)
        for path in PURE_G4_MODULES
        for module in imported_modules(path)
        if module.startswith(PROHIBITED)
    ]
    assert violations == []


def test_g4_orchestration_imports_only_verified_production_boundaries() -> None:
    modules = imported_modules(GROUND / "service_persistence.py")
    forbidden = [
        module
        for module in modules
        if module.startswith(("satnet.models", "satnet.metrics", "satnet.utils.graph_cache"))
    ]
    assert forbidden == []
    assert "satnet.ground.integrated_persistence" in modules
    assert "satnet.simulation.tier1_rollout" in modules


def test_protected_satellite_modules_do_not_import_g4() -> None:
    protected = [
        *sorted((ROOT / "src" / "satnet" / "network").rglob("*.py")),
        ROOT / "src" / "satnet" / "simulation" / "tier1_rollout.py",
        ROOT / "src" / "satnet" / "models" / "gnn_dataset.py",
        ROOT / "src" / "satnet" / "utils" / "graph_cache.py",
    ]
    assert all(
        not module.startswith(
            (
                "satnet.ground.service_policy",
                "satnet.ground.service_metrics",
                "satnet.ground.service_aggregation",
                "satnet.ground.service_persistence",
            )
        )
        for path in protected
        for module in imported_modules(path)
    )
