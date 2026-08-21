"""Authoritative scientific profiles for production experiment lineages.

Generic Tier 1 configuration objects retain their historical defaults for
backward compatibility. Production contracts and scenario builders must pass
one of these explicit profiles instead of inheriting a generic default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from satnet.ground.canonical import canonical_float_string
from satnet.simulation.tier1_rollout import DEFAULT_EPOCH_ISO, DEFAULT_FAILURE_MODEL


@dataclass(frozen=True)
class ProductionTopologyProfile:
    """Immutable topology and temporal identity for a production lineage."""

    profile_id: str
    isl_policy: str
    adjacent_search_k: int
    max_inter_plane_links_per_sat: int
    duration_minutes: int
    step_seconds: int
    phasing_factor: int
    max_isl_distance_km: float
    epoch_iso: str
    orbital_engine: str
    failure_model: str

    @property
    def inclusive_timestep_count(self) -> int:
        return self.duration_minutes * 60 // self.step_seconds + 1

    def as_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "isl_policy": self.isl_policy,
            "adjacent_search_k": self.adjacent_search_k,
            "max_inter_plane_links_per_sat": self.max_inter_plane_links_per_sat,
            "duration_minutes": self.duration_minutes,
            "step_seconds": self.step_seconds,
            "inclusive_timestep_count": self.inclusive_timestep_count,
            "phasing_factor": self.phasing_factor,
            "max_isl_distance_km": self.max_isl_distance_km,
            "epoch_iso": self.epoch_iso,
            "orbital_engine": self.orbital_engine,
            "failure_model": self.failure_model,
        }

    def validate(self) -> None:
        if self.isl_policy not in {"grid_fixed", "grid_adaptive"}:
            raise ValueError(f"Unsupported ISL policy: {self.isl_policy}")
        if self.adjacent_search_k < 0:
            raise ValueError("adjacent_search_k must be non-negative")
        if self.max_inter_plane_links_per_sat not in {1, 2}:
            raise ValueError("max_inter_plane_links_per_sat must be 1 or 2")
        if self.duration_minutes * 60 % self.step_seconds:
            raise ValueError("duration must be an integer number of steps")
        if self.failure_model != DEFAULT_FAILURE_MODEL:
            raise ValueError("Production profile must use the accepted temporal-union failure model")

    def assert_matches(self, values: Mapping[str, object]) -> None:
        """Reject missing or drifted scientific identity fields."""

        expected = self.as_dict()
        for name, expected_value in expected.items():
            if name not in values:
                continue
            actual_value = values[name]
            if isinstance(expected_value, float) and isinstance(actual_value, str):
                matches = actual_value == canonical_float_string(expected_value)
            else:
                matches = actual_value == expected_value
            if not matches:
                raise ValueError(
                    f"Production profile mismatch for {name}: "
                    f"expected {expected_value!r}, got {actual_value!r}"
                )


FINAL_ADAPTIVE_PRODUCTION_PROFILE = ProductionTopologyProfile(
    profile_id="final_integrated_dataset_10k_adaptive_v2",
    isl_policy="grid_adaptive",
    adjacent_search_k=1,
    max_inter_plane_links_per_sat=1,
    duration_minutes=10,
    step_seconds=60,
    phasing_factor=1,
    max_isl_distance_km=10_000.0,
    epoch_iso=DEFAULT_EPOCH_ISO,
    orbital_engine="sgp4",
    failure_model=DEFAULT_FAILURE_MODEL,
)

# This is retained solely to reconstruct and validate the immutable historical
# fixed-policy pilot and contract. It is not a production default.
HISTORICAL_FIXED_PROFILE = ProductionTopologyProfile(
    profile_id="historical_fixed_policy_evidence",
    isl_policy="grid_fixed",
    adjacent_search_k=1,
    max_inter_plane_links_per_sat=1,
    duration_minutes=10,
    step_seconds=60,
    phasing_factor=1,
    max_isl_distance_km=10_000.0,
    epoch_iso=DEFAULT_EPOCH_ISO,
    orbital_engine="sgp4",
    failure_model=DEFAULT_FAILURE_MODEL,
)

FINAL_ADAPTIVE_PRODUCTION_PROFILE.validate()
HISTORICAL_FIXED_PROFILE.validate()


def assert_adaptive_production_values(values: Mapping[str, object]) -> None:
    """Guard a production mapping against silent fixed-policy reversion."""

    FINAL_ADAPTIVE_PRODUCTION_PROFILE.assert_matches(values)
    required = (
        "isl_policy",
        "adjacent_search_k",
        "max_inter_plane_links_per_sat",
        "failure_model",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Adaptive production profile is missing fields: {missing}")
    if values["isl_policy"] != "grid_adaptive":
        raise ValueError("Adaptive production profile cannot use grid_fixed")
