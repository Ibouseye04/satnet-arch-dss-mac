from __future__ import annotations

from copy import deepcopy

import pytest

from satnet.utils.graph_cache import (
    CACHE_SCHEMA_VERSION,
    make_cache_metadata,
    make_sample_cache_key,
    validate_cache_entry,
)


BASE_CONFIG = {
    "num_planes": 4,
    "sats_per_plane": 6,
    "inclination_deg": 53.0,
    "altitude_km": 550.0,
    "phasing_factor": 1,
    "duration_minutes": 5,
    "step_seconds": 60,
    "num_steps": 6,
    "max_isl_distance_km": 10_000.0,
    "isl_policy": "grid_adaptive",
    "adjacent_search_k": 1,
    "max_inter_plane_links_per_sat": 1,
    "node_failure_prob": 0.01,
    "edge_failure_prob": 0.02,
    "failure_model": "persistent_temporal_union_edges_v1",
    "seed": 42,
    "epoch_iso": "2000-01-01T12:00:00+00:00",
    "failed_nodes_json": "[1]",
    "failed_edges_json": "[[2,3]]",
    "schema_version": 2,
    "dataset_version": "tier1_temporal_connectivity_v2",
    "orbital_engine": "sgp4",
    "physics_model_version": "physics-v1",
    "link_budget_config": {"optical_wavelength_nm": 1550.0},
}


@pytest.mark.parametrize(
    ("field", "mutated_value"),
    [
        ("max_isl_distance_km", 5_000.0),
        ("isl_policy", "grid_fixed"),
        ("epoch_iso", "2000-01-01T12:01:00+00:00"),
        ("phasing_factor", 2),
        ("failed_nodes_json", "[2]"),
        ("seed", 43),
        ("max_inter_plane_links_per_sat", 2),
    ],
)
def test_v05_included_graph_inputs_change_cache_key(field: str, mutated_value) -> None:
    mutated = deepcopy(BASE_CONFIG)
    mutated[field] = mutated_value
    assert make_sample_cache_key(BASE_CONFIG) != make_sample_cache_key(mutated)


@pytest.mark.parametrize(
    ("field", "mutated_value"),
    [
        ("orbital_engine", "keplerian"),
        ("physics_model_version", "physics-v2"),
        ("link_budget_config", {"optical_wavelength_nm": 1310.0}),
    ],
)
def test_v05_graph_affecting_inputs_change_cache_key(
    field: str,
    mutated_value,
) -> None:
    mutated = deepcopy(BASE_CONFIG)
    mutated[field] = mutated_value
    assert make_sample_cache_key(BASE_CONFIG) != make_sample_cache_key(mutated)


def test_v05_generator_provenance_mismatch_is_rejected() -> None:
    key = make_sample_cache_key(BASE_CONFIG)
    metadata = make_cache_metadata(
        sample_cache_key=key,
        generator_provenance="generator-v1",
        generator_config={},
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_cache_entry(
            [],
            metadata,
            expected_sample_cache_key=key,
            expected_generator_provenance="generator-v2",
        )


def test_v05_incomplete_and_version_mismatched_metadata_are_rejected() -> None:
    key = make_sample_cache_key(BASE_CONFIG)
    with pytest.raises(ValueError, match="metadata is missing"):
        validate_cache_entry(
            [],
            None,
            expected_sample_cache_key=key,
            expected_generator_provenance="generator-v1",
        )

    stale = make_cache_metadata(
        sample_cache_key=key,
        generator_provenance="generator-v1",
        generator_config={},
    )
    stale["cache_schema_version"] = CACHE_SCHEMA_VERSION - 1
    with pytest.raises(ValueError, match="Unsupported cache schema version"):
        validate_cache_entry(
            [],
            stale,
            expected_sample_cache_key=key,
            expected_generator_provenance="generator-v1",
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "dataset_version",
        "link_budget_config",
        "orbital_engine",
        "physics_model_version",
        "schema_version",
    ],
)
def test_v05_missing_scientific_identity_field_is_rejected(missing_field: str) -> None:
    incomplete = deepcopy(BASE_CONFIG)
    incomplete.pop(missing_field)
    with pytest.raises(ValueError, match=missing_field):
        make_sample_cache_key(incomplete)


def test_v05_structurally_malformed_payload_is_rejected() -> None:
    key = make_sample_cache_key(BASE_CONFIG)
    metadata = make_cache_metadata(
        sample_cache_key=key,
        generator_provenance="generator-v1",
        generator_config={},
    )
    with pytest.raises(ValueError, match="missing required attributes"):
        validate_cache_entry(
            [object()],
            metadata,
            expected_sample_cache_key=key,
            expected_generator_provenance="generator-v1",
        )
