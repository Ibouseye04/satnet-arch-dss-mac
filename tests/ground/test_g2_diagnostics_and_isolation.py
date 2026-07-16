from __future__ import annotations

import ast
from pathlib import Path

from scripts.validation.validate_ground_visibility_geometry import build_diagnostics
from satnet.ground.catalog import load_ground_station_catalog
from satnet.ground.persistence import make_enabled_ground_design_record
from satnet.ground.position_adapter import reconstruct_operational_satellite_position_sequence
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from satnet.ground.visibility import (
    GroundVisibilityPolicy,
    evaluate_ground_design_visibility_sequence,
)
from satnet.ground.visibility_persistence import (
    make_ground_visibility_record,
    read_ground_visibility_manifest,
    write_ground_visibility_manifest,
)
from satnet.simulation.tier1_rollout import Tier1RolloutConfig, run_tier1_rollout

ROOT = Path(__file__).parents[2]
GROUND_SOURCE = ROOT / "src" / "satnet" / "ground"
FIXTURE = ROOT / "tests" / "fixtures" / "ground_segment" / "synthetic_ground_station_catalog.csv"
PURE_G2_MODULES = (
    GROUND_SOURCE / "coordinates.py",
    GROUND_SOURCE / "visibility.py",
    GROUND_SOURCE / "visibility_persistence.py",
)
PROHIBITED_PURE_PREFIXES = (
    "satnet.network",
    "satnet.simulation",
    "satnet.models",
    "satnet.metrics",
    "satnet.utils.graph_cache",
)


def imported_modules(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.module or "", tuple(alias.name for alias in node.names)))
    return imports


def satellite_config() -> Tier1RolloutConfig:
    return Tier1RolloutConfig(
        num_planes=2,
        sats_per_plane=3,
        duration_minutes=1,
        step_seconds=60,
        max_isl_distance_km=10000.0,
        isl_policy="grid_adaptive",
        adjacent_search_k=1,
        max_inter_plane_links_per_sat=1,
        node_failure_prob=0.2,
        edge_failure_prob=0.2,
        seed=2026,
    )


def test_diagnostic_harness_covers_required_cases() -> None:
    diagnostics = build_diagnostics()
    names = {record["case"] for record in diagnostics}
    assert names == {
        "overhead",
        "high_elevation",
        "threshold_boundary",
        "near_horizon",
        "below_horizon",
        "multiple_stations_satellites_second_timestamp",
    }
    by_name = {record["case"]: record for record in diagnostics}
    assert by_name["overhead"]["elevation_deg"] == 90.0
    assert by_name["threshold_boundary"]["is_visible"] is True
    assert by_name["near_horizon"]["is_visible"] is False
    assert by_name["below_horizon"]["is_visible"] is False
    assert by_name["multiple_stations_satellites_second_timestamp"]["observation_count"] == 4
    assert all(record["coordinate_frame"] == "ecef" for record in diagnostics)


def test_pure_g2_modules_have_no_prohibited_imports() -> None:
    violations = []
    for path in PURE_G2_MODULES:
        for module, names in imported_modules(path):
            if module.startswith(PROHIBITED_PURE_PREFIXES):
                violations.append((path.name, module, names))
    assert violations == []


def test_position_adapter_has_only_approved_satellite_imports() -> None:
    imports = imported_modules(GROUND_SOURCE / "position_adapter.py")
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


def test_protected_satellite_modules_do_not_import_g2() -> None:
    protected = [
        *sorted((ROOT / "src" / "satnet" / "network").rglob("*.py")),
        ROOT / "src" / "satnet" / "simulation" / "tier1_rollout.py",
        ROOT / "src" / "satnet" / "models" / "gnn_dataset.py",
        ROOT / "src" / "satnet" / "utils" / "graph_cache.py",
    ]
    violations = []
    for path in protected:
        for module, names in imported_modules(path):
            if module.startswith("satnet.ground"):
                violations.append((str(path), module, names))
    assert violations == []


def test_visibility_evaluation_is_observational_only() -> None:
    config = satellite_config()
    original_hash = config.config_hash()
    baseline_steps, baseline_summary, failures = run_tier1_rollout(config)
    sources_before = reconstruct_operational_satellite_position_sequence(
        satellite_config=config,
        failure_realization=failures,
    )
    catalog = load_ground_station_catalog(FIXTURE)
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(3, 2, 1, 42),
    )
    ground_design = make_enabled_ground_design_record(
        run_id=0,
        satellite_config_hash=original_hash,
        selection=selection,
    )
    visibility = evaluate_ground_design_visibility_sequence(
        ground_design=ground_design,
        catalog=catalog,
        satellite_sequence=sources_before,
        policy=GroundVisibilityPolicy(10.0),
    )
    after_steps, after_summary, after_failures = run_tier1_rollout(config)
    sources_after = reconstruct_operational_satellite_position_sequence(
        satellite_config=config,
        failure_realization=failures,
    )
    assert len(visibility) == config.num_steps
    assert config.config_hash() == original_hash
    assert after_steps == baseline_steps
    assert after_summary == baseline_summary
    assert after_failures == failures
    assert sources_after == sources_before


def test_visibility_artifact_path_does_not_affect_identity(tmp_path: Path) -> None:
    config = satellite_config()
    _, _, failures = run_tier1_rollout(config)
    sources = reconstruct_operational_satellite_position_sequence(
        satellite_config=config,
        failure_realization=failures,
    )
    catalog = load_ground_station_catalog(FIXTURE)
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(1, 0, 0, 42),
    )
    ground_design = make_enabled_ground_design_record(
        run_id=0,
        satellite_config_hash=config.config_hash(),
        selection=selection,
    )
    snapshots = evaluate_ground_design_visibility_sequence(
        ground_design=ground_design,
        catalog=catalog,
        satellite_sequence=sources,
        policy=GroundVisibilityPolicy(10.0),
    )
    records = tuple(
        make_ground_visibility_record(run_id=0, snapshot=snapshot)
        for snapshot in snapshots
    )
    first_path = tmp_path / "first" / "visibility.jsonl"
    second_path = tmp_path / "second" / "visibility.jsonl"
    write_ground_visibility_manifest(records, first_path)
    write_ground_visibility_manifest(records, second_path)
    assert read_ground_visibility_manifest(first_path) == records
    assert read_ground_visibility_manifest(second_path) == records
    assert first_path.read_bytes() == second_path.read_bytes()
