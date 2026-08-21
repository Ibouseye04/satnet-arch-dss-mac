"""Validated request schema for the SATNET Phase 1 DSS."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from satnet.ground.canonical import canonical_float_string, canonical_hash

DSS_DEFAULT_THRESHOLD = 0.80
DSS_REALIZATION_COUNT = 5
DSS_SEQUENCE_LENGTH = 11
DSS_MASTER_SEED_DOMAIN = "satnet_dss_phase1_v1"

SPACE_BOUNDS: dict[str, tuple[float, float]] = {
    "altitude_km": (300.0, 1200.0),
    "inclination_deg": (30.0, 98.0),
    "satellite_node_failure_probability": (0.0, 0.20),
    "satellite_edge_failure_probability": (0.0, 0.25),
    "ground_station_failure_probability": (0.0, 0.40),
    "required_minimum_connectivity": (0.0, 1.0),
}


class DSSValidationError(ValueError):
    """Raised when a DSS request is outside its validated contract."""


def _bounded_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an integer or float")
    normalized = float(value)
    if not math.isfinite(normalized) or not minimum <= normalized <= maximum:
        raise DSSValidationError(f"{name} must be finite and within [{minimum}, {maximum}]")
    return 0.0 if normalized == 0.0 else normalized


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise DSSValidationError(f"{name} must be an integer within [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True)
class DSSArchitectureRequest:
    """User-entered architecture and post-inference decision requirement."""

    num_planes: int
    sats_per_plane: int
    altitude_km: float
    inclination_deg: float
    satellite_node_failure_probability: float
    satellite_edge_failure_probability: float
    civilian_count: int
    government_count: int
    military_count: int
    ground_station_failure_probability: float
    required_minimum_connectivity: float = DSS_DEFAULT_THRESHOLD

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_planes", _bounded_int(self.num_planes, "num_planes", 4, 6))
        object.__setattr__(
            self, "sats_per_plane", _bounded_int(self.sats_per_plane, "sats_per_plane", 5, 8)
        )
        for name in (
            "altitude_km",
            "inclination_deg",
            "satellite_node_failure_probability",
            "satellite_edge_failure_probability",
            "ground_station_failure_probability",
            "required_minimum_connectivity",
        ):
            minimum, maximum = SPACE_BOUNDS[name]
            object.__setattr__(self, name, _bounded_float(getattr(self, name), name, minimum, maximum))
        for name in ("civilian_count", "government_count", "military_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise DSSValidationError(f"{name} must be a nonnegative integer")
        if self.civilian_count + self.government_count + self.military_count <= 0:
            raise DSSValidationError("At least one ground station must be requested")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DSSArchitectureRequest":
        if not isinstance(payload, dict):
            raise TypeError("DSS request must be a mapping")
        return cls(**payload)

    def canonical_payload(self, *, include_threshold: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "altitude_km": canonical_float_string(self.altitude_km),
            "civilian_count": self.civilian_count,
            "government_count": self.government_count,
            "ground_station_failure_probability": canonical_float_string(
                self.ground_station_failure_probability
            ),
            "inclination_deg": canonical_float_string(self.inclination_deg),
            "military_count": self.military_count,
            "num_planes": self.num_planes,
            "satellite_edge_failure_probability": canonical_float_string(
                self.satellite_edge_failure_probability
            ),
            "satellite_node_failure_probability": canonical_float_string(
                self.satellite_node_failure_probability
            ),
            "sats_per_plane": self.sats_per_plane,
        }
        if include_threshold:
            payload["required_minimum_connectivity"] = canonical_float_string(
                self.required_minimum_connectivity
            )
        return payload

    @property
    def architecture_hash(self) -> str:
        return canonical_hash(
            {
                "architecture": self.canonical_payload(include_threshold=True),
                "identity_domain": DSS_MASTER_SEED_DOMAIN,
                "identity_version": "1",
            }
        )

    @property
    def physical_architecture_hash(self) -> str:
        """Hash used for physical scenarios; intentionally excludes the threshold."""
        return canonical_hash(
            {
                "architecture": self.canonical_payload(include_threshold=False),
                "identity_domain": f"{DSS_MASTER_SEED_DOMAIN}_physical",
                "identity_version": "1",
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "num_planes": self.num_planes,
            "sats_per_plane": self.sats_per_plane,
            "altitude_km": self.altitude_km,
            "inclination_deg": self.inclination_deg,
            "satellite_node_failure_probability": self.satellite_node_failure_probability,
            "satellite_edge_failure_probability": self.satellite_edge_failure_probability,
            "civilian_count": self.civilian_count,
            "government_count": self.government_count,
            "military_count": self.military_count,
            "ground_station_failure_probability": self.ground_station_failure_probability,
            "required_minimum_connectivity": self.required_minimum_connectivity,
        }
