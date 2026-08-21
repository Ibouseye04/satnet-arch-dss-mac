from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import networkx as nx
import pytest

from satnet.dss.schemas import DSSArchitectureRequest
from satnet.dss import scenario_builder
from satnet.experiments.final_generation import adaptive_contract
from satnet.experiments.final_generation.adaptive_contract import (
    load_adaptive_contract,
    map_adaptive_run,
)
from satnet.experiments.production_profile import FINAL_ADAPTIVE_PRODUCTION_PROFILE
from satnet.ground.satellite_graph_adapter import (
    reconstruct_operational_satellite_graph_sequence,
)
from satnet.metrics.labels import compute_gcc_size, compute_num_components
from satnet.network import hypatia_adapter
from satnet.network.hypatia_adapter import (
    SatellitePosition,
    WalkerDeltaConfig,
    _compute_grid_plus_isls,
    HypatiaAdapter,
)
from satnet.simulation.tier1_rollout import (
    DEFAULT_FAILURE_MODEL,
    Tier1FailureRealization,
    Tier1RolloutConfig,
    run_tier1_rollout,
)


CONTRACT_ROOT = (
    Path(__file__).parents[2]
    / "artifacts"
    / "final_integrated_dataset_10k_adaptive_v2_contract"
)


class SelectiveControlledBudget:
    def __init__(self) -> None:
        self.evaluated_distances: list[float] = []

    def evaluate_link(self, distance_km: float):
        self.evaluated_distances.append(distance_km)
        if abs(distance_km - 90.0) < 1e-6:
            return "optical", -20.0, -1.0, False
        return "optical", -20.0, 25.0, True


def controlled_positions() -> list[SatellitePosition]:
    return [
        SatellitePosition(0, 7000.0, 0.0, 0.0),
        SatellitePosition(1, 7000.0, 10.0, 0.0),
        SatellitePosition(2, 7000.0, -10.0, 0.0),
        SatellitePosition(3, -500.0, 0.0, 0.0),
        SatellitePosition(4, 7000.0, 100.0, 0.0),
        SatellitePosition(5, 7000.0, -100.0, 0.0),
    ]


def edge_set(graph: nx.Graph) -> set[tuple[int, int]]:
    return {tuple(sorted((int(u), int(v)))) for u, v in graph.edges()}


def link_edge_set(links) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(link.sat_id_1), int(link.sat_id_2))))
        for link in links
    }


def interplane_edge_set(graph: nx.Graph) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(u), int(v))))
        for u, v, attributes in graph.edges(data=True)
        if attributes["link_type"] != "intra_plane"
    }


def graph_hash(graph: nx.Graph) -> str:
    payload = {
        "edges": sorted(edge_set(graph)),
        "nodes": sorted(int(node) for node in graph.nodes()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_sgp4_policy_graphs(
    *,
    num_planes: int,
    sats_per_plane: int,
    altitude_km: float,
    inclination_deg: float,
    policy: str,
) -> dict[int, nx.Graph]:
    with HypatiaAdapter(
        num_planes=num_planes,
        sats_per_plane=sats_per_plane,
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        phasing_factor=1,
    ) as adapter:
        adapter.generate_tles()
        adapter.calculate_isls(
            duration_minutes=10,
            step_seconds=60,
            max_isl_distance_km=10_000.0,
            isl_policy=policy,
            adjacent_search_k=1,
            max_inter_plane_links_per_sat=1,
        )
        return {t: graph.copy() for t, graph in adapter.iter_graphs()}


def prepared_mapping():
    contract = load_adaptive_contract(CONTRACT_ROOT)
    designs = {record["design_id"]: record for record in contract["designs"]}
    run = contract["runs"][0]
    return map_adaptive_run(
        designs[run["design_id"]],
        run,
        contract_spec_hash=contract["contract_spec_hash"],
    )


def adaptive_config_for_p03(**overrides: object) -> Tier1RolloutConfig:
    values: dict[str, object] = {
        "num_planes": 5,
        "sats_per_plane": 6,
        "inclination_deg": 55.0,
        "altitude_km": 600.0,
        "phasing_factor": 1,
        "duration_minutes": 10,
        "step_seconds": 60,
        "max_isl_distance_km": 10_000.0,
        "isl_policy": "grid_adaptive",
        "adjacent_search_k": 1,
        "max_inter_plane_links_per_sat": 1,
        "node_failure_prob": 0.0,
        "edge_failure_prob": 0.0,
        "failure_model": DEFAULT_FAILURE_MODEL,
        "seed": 1,
        "epoch_iso": "2000-01-01T12:00:00+00:00",
        "orbital_engine": "sgp4",
    }
    values.update(overrides)
    return Tier1RolloutConfig(**values)


def test_controlled_sentinel_proves_actual_adaptive_edge_selection() -> None:
    config = WalkerDeltaConfig(num_planes=2, sats_per_plane=3)
    positions = controlled_positions()
    fixed_budget = SelectiveControlledBudget()
    adaptive_budget = SelectiveControlledBudget()
    fixed_links, fixed_stats = _compute_grid_plus_isls(
        config,
        positions,
        fixed_budget,
        max_isl_distance_km=10_000.0,
        isl_policy="grid_fixed",
        adjacent_search_k=1,
        max_inter_plane_links_per_sat=1,
    )
    adaptive_links, adaptive_stats = _compute_grid_plus_isls(
        config,
        positions,
        adaptive_budget,
        max_isl_distance_km=10_000.0,
        isl_policy="grid_adaptive",
        adjacent_search_k=1,
        max_inter_plane_links_per_sat=1,
        collect_adaptive_examples=10,
    )

    fixed_edges = link_edge_set(fixed_links)
    adaptive_edges = link_edge_set(adaptive_links)
    fixed_interplane = {
        tuple(sorted((link.sat_id_1, link.sat_id_2)))
        for link in fixed_links
        if link.link_type != "intra_plane"
    }
    adaptive_interplane = {
        tuple(sorted((link.sat_id_1, link.sat_id_2)))
        for link in adaptive_links
        if link.link_type != "intra_plane"
    }
    adaptive_only = adaptive_edges - fixed_edges
    fixed_only = fixed_edges - adaptive_edges

    print(f"FIXED_INTERPLANE_EDGES = {sorted(fixed_interplane)}")
    print(f"ADAPTIVE_INTERPLANE_EDGES = {sorted(adaptive_interplane)}")
    print(f"ADAPTIVE_ONLY_EDGES = {sorted(adaptive_only)}")
    print(f"FIXED_ONLY_EDGES = {sorted(fixed_only)}")

    assert fixed_interplane == set()
    assert adaptive_interplane == {(0, 4), (1, 5)}
    assert adaptive_only == {(0, 4), (1, 5)}
    assert fixed_only == set()
    assert {
        tuple(sorted((link.sat_id_1, link.sat_id_2)))
        for link in fixed_links
        if link.link_type == "intra_plane"
    } == {
        tuple(sorted((link.sat_id_1, link.sat_id_2)))
        for link in adaptive_links
        if link.link_type == "intra_plane"
    }
    assert 90.0 in fixed_budget.evaluated_distances
    assert 90.0 in adaptive_budget.evaluated_distances
    assert fixed_stats.links_rejected_los > 0
    assert adaptive_stats.links_rejected_los > 0
    assert fixed_stats.links_rejected_budget > 0
    assert adaptive_stats.links_rejected_budget > 0

    example = adaptive_stats.adaptive_selection_examples[0]
    outcomes = example["adaptive_candidates"]
    assert [outcome["candidate_offset"] for outcome in outcomes] == [0, -1, 1]
    assert outcomes[0]["rejection_reason"] == "earth_obscuration"
    assert outcomes[0]["viable"] is False
    assert outcomes[1]["rejection_reason"] is None
    assert outcomes[2]["selected"] is True
    assert example["selected"][0]["candidate_offset"] == 1

    degree: Counter[int] = Counter()
    for link in adaptive_links:
        if link.link_type != "intra_plane":
            degree[link.sat_id_1] += 1
            degree[link.sat_id_2] += 1
    assert degree
    assert max(degree.values()) <= 1


def test_real_sgp4_p03_has_an_eleven_timestep_graph_differential() -> None:
    fixed = run_sgp4_policy_graphs(
        num_planes=5,
        sats_per_plane=6,
        altitude_km=600.0,
        inclination_deg=55.0,
        policy="grid_fixed",
    )
    adaptive = run_sgp4_policy_graphs(
        num_planes=5,
        sats_per_plane=6,
        altitude_km=600.0,
        inclination_deg=55.0,
        policy="grid_adaptive",
    )
    assert tuple(fixed) == tuple(range(11))
    assert tuple(adaptive) == tuple(range(11))
    differences = []
    print(
        "timestep,fixed_interplane_count,adaptive_interplane_count,"
        "adaptive_only_count,fixed_components,adaptive_components,"
        "fixed_gcc_original,adaptive_gcc_original"
    )
    for timestep in range(11):
        fixed_edges = edge_set(fixed[timestep])
        adaptive_edges = edge_set(adaptive[timestep])
        adaptive_only = adaptive_edges - fixed_edges
        differences.append(adaptive_only)
        print(
            timestep,
            len(interplane_edge_set(fixed[timestep])),
            len(interplane_edge_set(adaptive[timestep])),
            len(adaptive_only),
            compute_num_components(fixed[timestep]),
            compute_num_components(adaptive[timestep]),
            f"{compute_gcc_size(fixed[timestep]) / 30:.6f}",
            f"{compute_gcc_size(adaptive[timestep]) / 30:.6f}",
            sep=",",
        )
    assert all(differences)
    assert len(fixed[0].edges()) == 2
    assert len(adaptive[0].edges()) == 8
    print(f"P03_T0_FIXED_HASH = {graph_hash(fixed[0])}")
    print(f"P03_T0_ADAPTIVE_HASH = {graph_hash(adaptive[0])}")
    print(f"P03_T0_ADAPTIVE_ONLY = {sorted(differences[0])}")


def test_real_sgp4_adaptive_trace_reports_nonzero_candidate_offset() -> None:
    with HypatiaAdapter(
        num_planes=5,
        sats_per_plane=6,
        altitude_km=600.0,
        inclination_deg=55.0,
        phasing_factor=1,
    ) as adapter:
        adapter.generate_tles()
        _, stats = adapter.calculate_isls(
            duration_minutes=10,
            step_seconds=60,
            max_isl_distance_km=10_000.0,
            isl_policy="grid_adaptive",
            adjacent_search_k=1,
            max_inter_plane_links_per_sat=1,
            collect_adaptive_examples=1,
        )
    example = stats.adaptive_selection_examples[0]
    same_index = example["adaptive_candidates"][0]
    selected = example["selected"][0]
    selected_outcome = next(
        outcome
        for outcome in example["adaptive_candidates"]
        if outcome["sat_id"] == selected["sat_id"]
    )
    print(
        "PRODUCTION_ALTERNATE_TRACE = "
        + json.dumps(
            {
                "timestep": example["time_step"],
                "source_satellite": example["sat_id"],
                "same_index_candidate": same_index["sat_id"],
                "same_index_reason": same_index["rejection_reason"],
                "selected_alternate": selected["sat_id"],
                "alternate_offset": selected["candidate_offset"],
                "los": selected_outcome["los"],
                "distance_km": selected["distance_km"],
                "margin_db": selected["margin_db"],
                "final_edge": sorted((example["sat_id"], selected["sat_id"])),
            },
            sort_keys=True,
        )
    )
    assert example["time_step"] == 0
    assert same_index["candidate_offset"] == 0
    assert same_index["los"] is False
    assert selected["candidate_offset"] == -1
    assert sorted((example["sat_id"], selected["sat_id"])) == [2, 7]


def test_prepared_adaptive_mapping_reaches_production_rollout_and_graphs() -> None:
    mapping = prepared_mapping()
    config = mapping.satellite_config
    runtime = {
        "isl_policy": config.isl_policy,
        "adjacent_search_k": config.adjacent_search_k,
        "max_inter_plane_links_per_sat": config.max_inter_plane_links_per_sat,
        "failure_model": config.failure_model,
    }
    print(f"PRODUCTION_RUNTIME_FIELDS = {runtime}")
    steps, summary, failures = run_tier1_rollout(config)
    assert runtime == {
        "isl_policy": "grid_adaptive",
        "adjacent_search_k": 1,
        "max_inter_plane_links_per_sat": 1,
        "failure_model": DEFAULT_FAILURE_MODEL,
    }
    assert len(steps) == 11
    assert summary.num_failed_nodes == 0
    assert summary.num_failed_edges == 0
    assert failures == Tier1FailureRealization(set(), set())

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
        _, stats = adapter.calculate_isls(
            duration_minutes=config.duration_minutes,
            step_seconds=config.step_seconds,
            max_isl_distance_km=config.max_isl_distance_km,
            isl_policy=config.isl_policy,
            adjacent_search_k=config.adjacent_search_k,
            max_inter_plane_links_per_sat=config.max_inter_plane_links_per_sat,
            collect_adaptive_examples=1,
        )
    example = stats.adaptive_selection_examples[0]
    same_index = example["adaptive_candidates"][0]
    selected = example["selected"][0]
    selected_outcome = next(
        outcome
        for outcome in example["adaptive_candidates"]
        if outcome["sat_id"] == selected["sat_id"]
    )
    print(
        "PREPARED_RUN_ALTERNATE_TRACE = "
        + json.dumps(
            {
                "timestep": example["time_step"],
                "source_satellite": example["sat_id"],
                "same_index_candidate": same_index["sat_id"],
                "same_index_reason": same_index["rejection_reason"],
                "selected_alternate": selected["sat_id"],
                "alternate_offset": selected["candidate_offset"],
                "los": selected_outcome["los"],
                "distance_km": selected["distance_km"],
                "margin_db": selected["margin_db"],
                "final_edge": sorted((example["sat_id"], selected["sat_id"])),
            },
            sort_keys=True,
        )
    )
    assert selected["candidate_offset"] != 0


def test_tier1_g3_tgnn_edge_parity_for_prepared_adaptive_run() -> None:
    pytest.importorskip("torch")
    import sys

    sys.modules.setdefault("torch_scatter", None)
    sys.modules.setdefault("torch_sparse", None)
    pytest.importorskip("torch_geometric")
    from satnet.models.gnn_dataset import SatNetTemporalDataset

    mapping = prepared_mapping()
    config = mapping.satellite_config
    nofailures = Tier1FailureRealization(set(), set())
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
        tier1 = adapter.get_graph_at_step(0)
    g3 = reconstruct_operational_satellite_graph_sequence(
        satellite_config=config,
        failure_realization=nofailures,
    )[0].to_networkx()
    tgnn = SatNetTemporalDataset._networkx_to_pyg_data(
        SatNetTemporalDataset.__new__(SatNetTemporalDataset),
        G=g3,
        time_step=0,
        num_planes=config.num_planes,
        sats_per_plane=config.sats_per_plane,
    )
    tgnn_edges = {
        tuple(sorted((int(u), int(v))))
        for u, v in zip(
            tgnn.edge_index[0].tolist(),
            tgnn.edge_index[1].tolist(),
            strict=True,
        )
    }
    tier1_edges = edge_set(tier1)
    g3_edges = edge_set(g3)
    print(
        "PARITY = "
        + json.dumps(
            {
                "tier1_count": len(tier1_edges),
                "g3_count": len(g3_edges),
                "tgnn_directed_count": len(tgnn_edges),
                "tier1_hash": hashlib.sha256(repr(sorted(tier1_edges)).encode()).hexdigest(),
                "g3_hash": hashlib.sha256(repr(sorted(g3_edges)).encode()).hexdigest(),
                "tgnn_hash": hashlib.sha256(repr(sorted(tgnn_edges)).encode()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    adaptive_only = sorted({(1, 40)} & tier1_edges)
    assert tier1_edges == g3_edges == tgnn_edges
    assert adaptive_only == [(1, 40)]
    print("PARITY_ADAPTIVE_ONLY_EDGE = Tier1:PRESENT G3:PRESENT TGNN:PRESENT")


def test_temporal_union_adaptive_failure_overlay_is_persistent_and_replayable() -> None:
    config = adaptive_config_for_p03(
        node_failure_prob=0.15,
        edge_failure_prob=0.08,
        seed=1,
    )
    first = run_tier1_rollout(config)
    second = run_tier1_rollout(config)
    steps, summary, failures = first
    replay_steps, replay_summary, replay_failures = second
    assert summary == replay_summary
    assert failures == replay_failures
    assert steps == replay_steps

    adaptive_graphs = run_sgp4_policy_graphs(
        num_planes=config.num_planes,
        sats_per_plane=config.sats_per_plane,
        altitude_km=config.altitude_km,
        inclination_deg=config.inclination_deg,
        policy="grid_adaptive",
    )
    fixed_graphs = run_sgp4_policy_graphs(
        num_planes=config.num_planes,
        sats_per_plane=config.sats_per_plane,
        altitude_km=config.altitude_km,
        inclination_deg=config.inclination_deg,
        policy="grid_fixed",
    )
    adaptive_union = {
        edge for graph in adaptive_graphs.values() for edge in edge_set(graph)
    }
    fixed_union = {
        edge for graph in fixed_graphs.values() for edge in edge_set(graph)
    }
    assert failures.failed_edges <= adaptive_union
    assert failures.failed_edges & (adaptive_union - fixed_union)
    failed_edge = sorted(failures.failed_edges & (adaptive_union - fixed_union))[0]
    baseline_graphs = reconstruct_operational_satellite_graph_sequence(
        satellite_config=config,
        failure_realization=Tier1FailureRealization(set(), set()),
    )
    effective_graphs = reconstruct_operational_satellite_graph_sequence(
        satellite_config=config,
        failure_realization=failures,
    )
    active_timesteps = [
        timestep
        for timestep, snapshot in enumerate(baseline_graphs)
        if snapshot.to_networkx().has_edge(*failed_edge)
    ]
    assert active_timesteps
    assert all(
        not snapshot.to_networkx().has_edge(*failed_edge)
        for snapshot in effective_graphs
        if snapshot.to_networkx().has_edge(*failed_edge)
    )
    assert all(
        not snapshot.to_networkx().has_node(node)
        for snapshot in effective_graphs
        for node in failures.failed_nodes
    )
    print(f"ADAPTIVE_ACCEPTED_EDGE_UNION = {sorted(adaptive_union)}")
    print(f"FIXED_RECONSTRUCTION_UNION = {sorted(fixed_union)}")
    print(f"TEMPORAL_FAILED_EDGES = {sorted(failures.failed_edges)}")
    print(
        f"FAILED_EDGE_PROOF = edge={failed_edge}, active_timesteps={active_timesteps}, "
        "removed_from_effective_graph=True"
    )
    print(f"PERSISTENT_FAILED_NODES = {sorted(failures.failed_nodes)}")
    print("DETERMINISTIC_REPLAY = identical=True")


def test_string_only_adaptive_claim_cannot_hide_forced_fixed_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = adaptive_config_for_p03()
    expected = run_sgp4_policy_graphs(
        num_planes=config.num_planes,
        sats_per_plane=config.sats_per_plane,
        altitude_km=config.altitude_km,
        inclination_deg=config.inclination_deg,
        policy="grid_adaptive",
    )
    original_builder = hypatia_adapter._compute_grid_plus_isls

    def fixed_substitution(*args, **kwargs):
        positional = list(args)
        positional[4] = "grid_fixed"
        return original_builder(*positional, **kwargs)

    monkeypatch.setattr(hypatia_adapter, "_compute_grid_plus_isls", fixed_substitution)
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
        forced_fixed = {t: edge_set(graph) for t, graph in adapter.iter_graphs()}
    assert forced_fixed[0] != edge_set(expected[0])
    fixed_reference = run_sgp4_policy_graphs(
        num_planes=config.num_planes,
        sats_per_plane=config.sats_per_plane,
        altitude_km=config.altitude_km,
        inclination_deg=config.inclination_deg,
        policy="grid_fixed",
    )
    assert forced_fixed[0] == edge_set(fixed_reference[0])
    print("STRING_ONLY_ANTIFRAUD = forced_fixed_edge_set_mismatch_detected=True")


def test_fixed_policy_drift_fails_contract_mapping_replay_mapping_and_dss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = prepared_mapping()
    original_read_json = adaptive_contract.read_json

    def fixed_contract_specification(path):
        value = original_read_json(path)
        if Path(path).name == "contract_specification.json":
            value = deepcopy(value)
            value["fixed_profile"]["isl_policy"] = "grid_fixed"
        return value

    monkeypatch.setattr(adaptive_contract, "read_json", fixed_contract_specification)
    with pytest.raises(ValueError, match="grid_fixed|Production profile mismatch"):
        adaptive_contract.load_adaptive_contract(CONTRACT_ROOT)
    monkeypatch.setattr(adaptive_contract, "read_json", original_read_json)

    fixed_design = dict(mapping.design)
    fixed_design["isl_policy"] = "grid_fixed"
    with pytest.raises(ValueError, match="Production profile mismatch|grid_fixed"):
        map_adaptive_run(
            fixed_design,
            mapping.run,
            contract_spec_hash=mapping.run["contract_spec_hash"],
        )
    with pytest.raises(ValueError, match="Production profile mismatch|grid_fixed"):
        FINAL_ADAPTIVE_PRODUCTION_PROFILE.assert_matches(
            {**FINAL_ADAPTIVE_PRODUCTION_PROFILE.as_dict(), "isl_policy": "grid_fixed"}
        )

    monkeypatch.setattr(scenario_builder, "DSS_ISL_POLICY", "grid_fixed")
    request = DSSArchitectureRequest(
        num_planes=5,
        sats_per_plane=6,
        altitude_km=600.0,
        inclination_deg=55.0,
        satellite_node_failure_probability=0.0,
        satellite_edge_failure_probability=0.0,
        civilian_count=1,
        government_count=1,
        military_count=1,
        ground_station_failure_probability=0.0,
    )
    with pytest.raises(ValueError, match="grid_fixed"):
        scenario_builder.build_rollout_config(request, satellite_failure_seed=1)
    print(
        "DRIFT_GUARDRAILS = "
        "adaptive_contract_loader:CONFIGURATION_GUARDRAIL; "
        "final_mapping:CONFIGURATION_GUARDRAIL; "
        "replay_mapping:CONFIGURATION_GUARDRAIL; "
        "dss_scenario:CONFIGURATION_GUARDRAIL"
    )
