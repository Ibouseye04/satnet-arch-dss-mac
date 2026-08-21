from __future__ import annotations

import math

import pytest

from satnet.dss.analysis_service import aggregate_space_predictions
from satnet.dss.realization import derive_all_realization_seeds
from satnet.dss.schemas import DSSArchitectureRequest, DSS_DEFAULT_THRESHOLD


def valid_request(**overrides):
    values = {
        "num_planes": 4,
        "sats_per_plane": 5,
        "altitude_km": 550.0,
        "inclination_deg": 53.0,
        "satellite_node_failure_probability": 0.05,
        "satellite_edge_failure_probability": 0.10,
        "civilian_count": 1,
        "government_count": 1,
        "military_count": 1,
        "ground_station_failure_probability": 0.20,
    }
    values.update(overrides)
    return DSSArchitectureRequest(**values)


def test_valid_schema_and_default_threshold() -> None:
    request = valid_request()
    assert request.required_minimum_connectivity == DSS_DEFAULT_THRESHOLD
    assert request.to_dict()["altitude_km"] == 550.0
    assert request.physical_architecture_hash


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_planes", 3),
        ("num_planes", 7),
        ("sats_per_plane", 4),
        ("sats_per_plane", 9),
        ("altitude_km", 299.999),
        ("altitude_km", 1200.001),
        ("inclination_deg", 29.9),
        ("inclination_deg", 98.1),
        ("satellite_node_failure_probability", -0.01),
        ("satellite_edge_failure_probability", 0.251),
        ("ground_station_failure_probability", 0.401),
        ("required_minimum_connectivity", -0.01),
        ("required_minimum_connectivity", 1.01),
    ],
)
def test_invalid_ranges_are_rejected(field, value) -> None:
    with pytest.raises((TypeError, ValueError)):
        valid_request(**{field: value})


def test_threshold_is_configurable_but_excluded_from_seed_derivation() -> None:
    first = valid_request(required_minimum_connectivity=0.80)
    second = valid_request(required_minimum_connectivity=0.70)
    assert first.architecture_hash != second.architecture_hash
    assert first.physical_architecture_hash == second.physical_architecture_hash
    assert derive_all_realization_seeds(first) == derive_all_realization_seeds(second)


def test_expected_minimum_gcc_is_arithmetic_mean_without_clipping() -> None:
    result = aggregate_space_predictions([0.84, 0.81, 0.76, 0.83, 1.04], 0.80)
    assert math.isclose(result.expected_minimum_gcc, (0.84 + 0.81 + 0.76 + 0.83 + 1.04) / 5)
    assert result.lowest_modeled_gcc == 0.76
    assert result.highest_modeled_gcc == 1.04
    assert result.expected_margin == result.expected_minimum_gcc - 0.80
    assert result.realizations_meeting_requirement == 4
    assert result.realization_risk_flag is True


def test_aggregation_requires_exactly_five_realizations() -> None:
    with pytest.raises(ValueError, match="exactly five"):
        aggregate_space_predictions([0.8] * 4, 0.8)
