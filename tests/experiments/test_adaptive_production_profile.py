from __future__ import annotations

import pytest

from satnet.dss.scenario_builder import (
    DSS_ADJACENT_SEARCH_K,
    DSS_ISL_POLICY,
    DSS_MAX_INTER_PLANE_LINKS_PER_SAT,
    DSS_SEQUENCE_LENGTH,
    build_rollout_config,
)
from satnet.dss.schemas import DSSArchitectureRequest
from satnet.experiments.final_generation.adaptive_contract import (
    load_adaptive_contract,
    map_adaptive_contract_runs,
)
from satnet.experiments.final_dataset.specification import (
    build_adaptive_contract_specification,
    build_contract_specification,
    validate_adaptive_contract_specification,
)
from satnet.experiments.integrated_ground_manifest import build_adaptive_pilot_designs
from satnet.experiments.production_profile import FINAL_ADAPTIVE_PRODUCTION_PROFILE
from satnet.ground.canonical import canonical_hash
from satnet.simulation.tier1_rollout import (
    FAILURE_MODEL_PERSISTENT_TEMPORAL_UNION_EDGES_V1,
)


def test_adaptive_profile_is_explicit_and_temporal() -> None:
    profile = FINAL_ADAPTIVE_PRODUCTION_PROFILE
    assert profile.isl_policy == "grid_adaptive"
    assert profile.adjacent_search_k == 1
    assert profile.max_inter_plane_links_per_sat == 1
    assert profile.inclusive_timestep_count == 11
    assert profile.failure_model == FAILURE_MODEL_PERSISTENT_TEMPORAL_UNION_EDGES_V1


def test_adaptive_contract_has_new_hash_and_guarded_topology() -> None:
    adaptive = build_adaptive_contract_specification()
    historical = build_contract_specification()
    validate_adaptive_contract_specification(adaptive)
    assert adaptive["production_profile"] == "final_integrated_dataset_10k_adaptive_v2"
    assert adaptive["contract_spec_hash"] != historical["contract_spec_hash"]
    assert adaptive["fixed_profile"]["isl_policy"] == "grid_adaptive"
    assert adaptive["fixed_profile"]["adjacent_search_k"] == 1
    assert adaptive["fixed_profile"]["max_inter_plane_links_per_sat"] == 1


def test_contract_hash_changes_for_each_topology_field() -> None:
    baseline = build_adaptive_contract_specification()
    for field, value in (
        ("isl_policy", "grid_fixed"),
        ("adjacent_search_k", 2),
        ("max_inter_plane_links_per_sat", 2),
    ):
        changed = dict(baseline["fixed_profile"])
        changed[field] = value
        payload = {key: item for key, item in baseline.items() if key != "contract_spec_hash"}
        payload["fixed_profile"] = changed
        assert canonical_hash(payload) != baseline["contract_spec_hash"]


def test_adaptive_pilot_anchors_do_not_inherit_generic_fixed_defaults() -> None:
    designs = build_adaptive_pilot_designs()
    assert len(designs) == 5
    assert {
        (design.isl_policy, design.adjacent_search_k, design.max_inter_plane_links_per_sat)
        for design in designs
    } == {("grid_adaptive", 1, 1)}
    assert all(
        design.satellite_config(satellite_seed=0).failure_model
        == FAILURE_MODEL_PERSISTENT_TEMPORAL_UNION_EDGES_V1
        for design in designs
    )


def test_dss_profile_is_adaptive_and_uses_eleven_snapshots() -> None:
    request = DSSArchitectureRequest(
        num_planes=4,
        sats_per_plane=5,
        altitude_km=550.0,
        inclination_deg=53.0,
        satellite_node_failure_probability=0.0,
        satellite_edge_failure_probability=0.0,
        civilian_count=1,
        government_count=1,
        military_count=1,
        ground_station_failure_probability=0.0,
    )
    config = build_rollout_config(request, satellite_failure_seed=7)
    assert (DSS_ISL_POLICY, DSS_ADJACENT_SEARCH_K, DSS_MAX_INTER_PLANE_LINKS_PER_SAT) == (
        "grid_adaptive",
        1,
        1,
    )
    assert config.isl_policy == "grid_adaptive"
    assert config.num_steps == DSS_SEQUENCE_LENGTH == 11
    assert config.failure_model == FAILURE_MODEL_PERSISTENT_TEMPORAL_UNION_EDGES_V1


def test_prepared_adaptive_contract_maps_all_runs_without_simulation() -> None:
    root = "artifacts/final_integrated_dataset_10k_adaptive_v2_contract"
    contract = load_adaptive_contract(root)
    mappings = map_adaptive_contract_runs(contract)
    assert len(mappings) == 10_000
    assert {mapping.satellite_config.isl_policy for mapping in mappings} == {"grid_adaptive"}
    assert {mapping.satellite_config.adjacent_search_k for mapping in mappings} == {1}
    assert {mapping.satellite_config.max_inter_plane_links_per_sat for mapping in mappings} == {1}
    assert {mapping.satellite_config.num_steps for mapping in mappings} == {11}


def test_profile_rejects_fixed_policy_in_adaptive_mapping() -> None:
    values = FINAL_ADAPTIVE_PRODUCTION_PROFILE.as_dict()
    values["isl_policy"] = "grid_fixed"
    with pytest.raises(ValueError, match="grid_fixed"):
        FINAL_ADAPTIVE_PRODUCTION_PROFILE.assert_matches(values)
