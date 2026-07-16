from __future__ import annotations

from pathlib import Path

import pytest

from satnet.ground.catalog import load_ground_station_catalog
from satnet.ground.persistence import (
    make_disabled_ground_design_record,
    make_enabled_ground_design_record,
)
from satnet.ground.scenario import ScenarioDesign
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from satnet.network.hypatia_adapter import PHYSICS_MODEL_VERSION, HypatiaAdapter, LinkBudgetEngine
from satnet.simulation.monte_carlo import (
    Tier1MonteCarloConfig,
    generate_tier1_temporal_dataset,
    write_tier1_dataset_csv,
)
from satnet.simulation.tier1_rollout import Tier1RolloutConfig, run_tier1_rollout
from satnet.utils.graph_cache import extract_cache_key_config, make_sample_cache_key

FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "ground_segment"
    / "synthetic_ground_station_catalog.csv"
)


def satellite_config() -> Tier1RolloutConfig:
    return Tier1RolloutConfig(
        num_planes=3,
        sats_per_plane=4,
        inclination_deg=53.0,
        altitude_km=550.0,
        phasing_factor=1,
        duration_minutes=2,
        step_seconds=60,
        max_isl_distance_km=10000.0,
        isl_policy="grid_adaptive",
        adjacent_search_k=1,
        max_inter_plane_links_per_sat=1,
        node_failure_prob=0.15,
        edge_failure_prob=0.2,
        seed=818,
    )


def scenarios(config: Tier1RolloutConfig) -> tuple[ScenarioDesign, ScenarioDesign]:
    satellite_hash = config.config_hash()
    disabled = make_disabled_ground_design_record(
        run_id=0,
        satellite_config_hash=satellite_hash,
    )
    catalog = load_ground_station_catalog(FIXTURE)
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(6, 3, 3, 42),
    )
    enabled = make_enabled_ground_design_record(
        run_id=0,
        satellite_config_hash=satellite_hash,
        selection=selection,
    )
    return ScenarioDesign(config, disabled), ScenarioDesign(config, enabled)


def graph_and_position_snapshot(config: Tier1RolloutConfig):
    snapshots = []
    with HypatiaAdapter(
        num_planes=config.num_planes,
        sats_per_plane=config.sats_per_plane,
        inclination_deg=config.inclination_deg,
        altitude_km=config.altitude_km,
        phasing_factor=config.phasing_factor,
        epoch=config.epoch,
        orbital_engine=config.orbital_engine,
    ) as adapter:
        adapter.generate_tles()
        adapter.calculate_isls(
            duration_minutes=config.duration_minutes,
            step_seconds=config.step_seconds,
            max_isl_distance_km=config.max_isl_distance_km,
            isl_policy=config.isl_policy,
            adjacent_search_k=config.adjacent_search_k,
            max_inter_plane_links_per_sat=config.max_inter_plane_links_per_sat,
        )
        for time_step, graph in adapter.iter_graphs():
            positions = tuple(
                (position.sat_id, position.x_km, position.y_km, position.z_km, position.alt_km)
                for position in adapter.get_positions_at_step(time_step)
            )
            nodes = tuple(sorted(int(node) for node in graph.nodes))
            edges = tuple(
                sorted(
                    (
                        min(int(u), int(v)),
                        max(int(u), int(v)),
                        tuple(sorted(attributes.items())),
                    )
                    for u, v, attributes in graph.edges(data=True)
                )
            )
            snapshots.append((time_step, positions, nodes, edges))
    return tuple(snapshots)


def test_ground_designs_leave_rollout_hash_failures_metrics_and_labels_identical() -> None:
    config = satellite_config()
    original_hash = config.config_hash()
    baseline = run_tier1_rollout(config)
    disabled, enabled = scenarios(config)
    disabled_result = run_tier1_rollout(disabled.satellite_config)
    enabled_result = run_tier1_rollout(enabled.satellite_config)
    assert disabled.satellite_config is config
    assert enabled.satellite_config is config
    assert config.config_hash() == original_hash
    assert disabled_result == baseline
    assert enabled_result == baseline


def test_ground_designs_leave_sgp4_positions_nodes_edges_and_attributes_identical() -> None:
    config = satellite_config()
    baseline = graph_and_position_snapshot(config)
    disabled, enabled = scenarios(config)
    assert graph_and_position_snapshot(disabled.satellite_config) == baseline
    assert graph_and_position_snapshot(enabled.satellite_config) == baseline


def test_ground_designs_leave_satellite_cache_identity_identical() -> None:
    config = satellite_config()
    _, _, failures = run_tier1_rollout(config)
    failed_nodes_json, failed_edges_json = failures.to_json_strings()
    sample_config = {
        "num_planes": config.num_planes,
        "sats_per_plane": config.sats_per_plane,
        "inclination_deg": config.inclination_deg,
        "altitude_km": config.altitude_km,
        "phasing_factor": config.phasing_factor,
        "duration_minutes": config.duration_minutes,
        "step_seconds": config.step_seconds,
        "num_steps": config.num_steps,
        "max_isl_distance_km": config.max_isl_distance_km,
        "isl_policy": config.isl_policy,
        "adjacent_search_k": config.adjacent_search_k,
        "max_inter_plane_links_per_sat": config.max_inter_plane_links_per_sat,
        "node_failure_prob": config.node_failure_prob,
        "edge_failure_prob": config.edge_failure_prob,
        "failure_model": config.failure_model,
        "seed": config.seed,
        "epoch_iso": config.epoch_iso,
        "failed_nodes_json": failed_nodes_json,
        "failed_edges_json": failed_edges_json,
        "schema_version": 2,
        "dataset_version": "tier1_temporal_connectivity_v2",
        "orbital_engine": config.orbital_engine,
        "physics_model_version": PHYSICS_MODEL_VERSION,
        "link_budget_config": LinkBudgetEngine().to_config(),
    }
    baseline_key = make_sample_cache_key(sample_config)
    disabled, enabled = scenarios(config)
    for scenario in (disabled, enabled):
        assert scenario.satellite_config is config
        assert make_sample_cache_key(sample_config) == baseline_key
    extracted = extract_cache_key_config(sample_config)
    assert not any("ground" in field for field in extracted)


def test_ground_designs_leave_tgnn_reconstruction_identical(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from satnet.models.gnn_dataset import SatNetTemporalDataset

    monte_carlo_config = Tier1MonteCarloConfig(
        num_runs=1,
        num_planes_range=(3, 3),
        sats_per_plane_range=(4, 4),
        inclination_deg_range=(53.0, 53.0),
        altitude_km_range=(550.0, 550.0),
        duration_minutes=1,
        step_seconds=60,
        max_isl_distance_km=10000.0,
        isl_policy="grid_adaptive",
        adjacent_search_k=1,
        max_inter_plane_links_per_sat=1,
        node_failure_prob_range=(0.1, 0.1),
        edge_failure_prob_range=(0.1, 0.1),
        seed=919,
    )
    runs, steps = generate_tier1_temporal_dataset(monte_carlo_config)
    write_tier1_dataset_csv(
        runs,
        steps,
        tmp_path / "tier1_design_runs.csv",
        tmp_path / "tier1_design_steps.csv",
    )
    dataset = SatNetTemporalDataset(root=str(tmp_path), target_name="partition_any")
    baseline_sequence = dataset.get(0)
    row = runs[0]
    config = Tier1RolloutConfig(
        num_planes=row.num_planes,
        sats_per_plane=row.sats_per_plane,
        inclination_deg=row.inclination_deg,
        altitude_km=row.altitude_km,
        phasing_factor=row.phasing_factor,
        duration_minutes=row.duration_minutes,
        step_seconds=row.step_seconds,
        max_isl_distance_km=row.max_isl_distance_km,
        isl_policy=row.isl_policy,
        adjacent_search_k=row.adjacent_search_k,
        max_inter_plane_links_per_sat=row.max_inter_plane_links_per_sat,
        node_failure_prob=row.node_failure_prob,
        edge_failure_prob=row.edge_failure_prob,
        failure_model=row.failure_model,
        seed=row.seed,
        epoch_iso=row.epoch_iso,
        orbital_engine=row.orbital_engine,
    )
    disabled, enabled = scenarios(config)
    assert disabled.satellite_config is config
    assert enabled.satellite_config is config
    for scenario in (disabled, enabled):
        reconstructed_sequence = dataset.get(0)
        assert scenario.satellite_config.config_hash() == row.config_hash
        assert len(reconstructed_sequence) == len(baseline_sequence)
        for baseline, reconstructed in zip(
            baseline_sequence, reconstructed_sequence, strict=True
        ):
            assert baseline.time_step.item() == reconstructed.time_step.item()
            assert baseline.x.equal(reconstructed.x)
            assert baseline.edge_index.equal(reconstructed.edge_index)
            assert baseline.edge_attr.equal(reconstructed.edge_attr)
            assert baseline.y.equal(reconstructed.y)
