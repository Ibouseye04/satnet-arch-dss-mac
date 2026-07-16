from __future__ import annotations

from dataclasses import dataclass, field
import math

from satnet.ground.canonical import canonical_float_string, canonical_hash

GROUND_FAILURE_MODEL_VERSION = "1"
GROUND_FAILURE_POLICY_VERSION = "1"
GROUND_FAILURE_SAMPLING_VERSION = "1"
GROUND_FAILURE_POLICY_IDENTITY_DOMAIN = "satnet_ground_failure_policy"
GROUND_FAILURE_POLICY_IDENTITY_VERSION = "1"
MAX_GROUND_FAILURE_SEED = 2**63 - 1


def validate_ground_failure_seed(value: object) -> int:
    if type(value) is not int:
        raise TypeError("ground_failure_seed must be an integer")
    if not 0 <= value <= MAX_GROUND_FAILURE_SEED:
        raise ValueError(
            f"ground_failure_seed must be within [0, {MAX_GROUND_FAILURE_SEED}]"
        )
    return value


def _normalize_probability(value: int | float) -> float:
    if type(value) not in (int, float):
        raise TypeError("ground_station_failure_probability must be an integer or float")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError("ground_station_failure_probability must be finite") from exc
    if not math.isfinite(normalized):
        raise ValueError("ground_station_failure_probability must be finite")
    if not 0.0 <= normalized <= 1.0:
        raise ValueError("ground_station_failure_probability must be within [0.0, 1.0]")
    return 0.0 if normalized == 0.0 else normalized


@dataclass(frozen=True)
class GroundFailurePolicy:
    ground_station_failure_probability: float
    ground_failure_policy_hash: str = field(init=False)

    def __post_init__(self) -> None:
        probability = _normalize_probability(self.ground_station_failure_probability)
        object.__setattr__(self, "ground_station_failure_probability", probability)
        object.__setattr__(
            self,
            "ground_failure_policy_hash",
            canonical_hash(
                {
                    "ground_failure_model_version": GROUND_FAILURE_MODEL_VERSION,
                    "ground_failure_policy_version": GROUND_FAILURE_POLICY_VERSION,
                    "ground_failure_sampling_version": GROUND_FAILURE_SAMPLING_VERSION,
                    "ground_station_failure_probability": canonical_float_string(probability),
                    "identity_domain": GROUND_FAILURE_POLICY_IDENTITY_DOMAIN,
                    "identity_version": GROUND_FAILURE_POLICY_IDENTITY_VERSION,
                }
            ),
        )
