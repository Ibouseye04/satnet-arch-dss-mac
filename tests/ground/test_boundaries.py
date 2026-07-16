from __future__ import annotations

import ast
from pathlib import Path

GROUND_SOURCE = Path(__file__).parents[2] / "src" / "satnet" / "ground"
PROJECT_SOURCE = Path(__file__).parents[2] / "src" / "satnet"
DOMAIN_MODULES = (
    GROUND_SOURCE / "canonical.py",
    GROUND_SOURCE / "catalog.py",
    GROUND_SOURCE / "selection.py",
    GROUND_SOURCE / "persistence.py",
)
PROHIBITED_PREFIXES = (
    "satnet.network",
    "satnet.simulation",
    "satnet.models",
    "satnet.metrics",
    "satnet.utils.graph_cache",
)

def imported_modules(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result.append((alias.name, ()))
        elif isinstance(node, ast.ImportFrom):
            result.append((node.module or "", tuple(alias.name for alias in node.names)))
    return result


def test_ground_domain_modules_have_no_prohibited_static_imports() -> None:
    violations = []
    for path in DOMAIN_MODULES:
        for module, names in imported_modules(path):
            if module.startswith(PROHIBITED_PREFIXES):
                violations.append((path.name, module, names))
    assert violations == []


def test_scenario_module_imports_only_satellite_configuration_contract() -> None:
    scenario_path = GROUND_SOURCE / "scenario.py"
    satellite_imports = [
        (module, names)
        for module, names in imported_modules(scenario_path)
        if module.startswith("satnet.simulation")
    ]
    assert satellite_imports == [
        ("satnet.simulation.tier1_rollout", ("Tier1RolloutConfig",))
    ]
    prohibited_names = {
        "run_tier1_rollout",
        "HypatiaAdapter",
        "SatNetTemporalDataset",
        "make_sample_cache_key",
        "compute_gcc_size",
    }
    assert not prohibited_names.intersection(scenario_path.read_text(encoding="utf-8").split())


def test_ground_domain_has_no_dynamic_import_or_registration_calls() -> None:
    prohibited_calls = {"__import__", "import_module", "register", "setattr"}
    violations = []
    for path in DOMAIN_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in prohibited_calls:
                violations.append((path.name, node.func.id))
            if isinstance(node.func, ast.Attribute) and node.func.attr in prohibited_calls:
                violations.append((path.name, node.func.attr))
    assert violations == []


def test_protected_satellite_modules_do_not_import_ground_subsystem() -> None:
    protected_paths = [
        *sorted((PROJECT_SOURCE / "network").rglob("*.py")),
        *sorted((PROJECT_SOURCE / "simulation").rglob("*.py")),
        PROJECT_SOURCE / "models" / "gnn_dataset.py",
        PROJECT_SOURCE / "utils" / "graph_cache.py",
    ]
    violations = []
    for path in protected_paths:
        for module, names in imported_modules(path):
            if module.startswith("satnet.ground"):
                violations.append((str(path), module, names))
    assert violations == []
