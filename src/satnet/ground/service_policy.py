from __future__ import annotations

from dataclasses import dataclass, field
import math

from satnet.ground.canonical import canonical_float_string, canonical_hash

GROUND_SERVICE_MODEL_VERSION = "1"
GROUND_SERVICE_POLICY_VERSION = "1"
GROUND_SERVICE_POLICY_IDENTITY_DOMAIN = "satnet_ground_service_policy"
GROUND_SERVICE_POLICY_IDENTITY_VERSION = "1"


def _normalize_threshold(value: int | float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be an integer or float")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field_name} must be within [0.0, 1.0]")
    return 0.0 if normalized == 0.0 else normalized


@dataclass(frozen=True)
class GroundServicePolicy:
    space_gcc_threshold: float
    ground_service_threshold: float
    ground_service_policy_hash: str = field(init=False)

    def __post_init__(self) -> None:
        space_threshold = _normalize_threshold(
            self.space_gcc_threshold, "space_gcc_threshold"
        )
        ground_threshold = _normalize_threshold(
            self.ground_service_threshold, "ground_service_threshold"
        )
        object.__setattr__(self, "space_gcc_threshold", space_threshold)
        object.__setattr__(self, "ground_service_threshold", ground_threshold)
        object.__setattr__(
            self,
            "ground_service_policy_hash",
            canonical_hash(
                {
                    "ground_service_model_version": GROUND_SERVICE_MODEL_VERSION,
                    "ground_service_policy_version": GROUND_SERVICE_POLICY_VERSION,
                    "ground_service_threshold": canonical_float_string(ground_threshold),
                    "identity_domain": GROUND_SERVICE_POLICY_IDENTITY_DOMAIN,
                    "identity_version": GROUND_SERVICE_POLICY_IDENTITY_VERSION,
                    "space_gcc_threshold": canonical_float_string(space_threshold),
                }
            ),
        )
