"""Deterministic DSS realization seed derivation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from satnet.ground.canonical import canonical_json
from satnet.dss.schemas import DSSArchitectureRequest, DSS_MASTER_SEED_DOMAIN, DSS_REALIZATION_COUNT

SATELLITE_FAILURE_DOMAIN = "satellite_failure_realization"
GROUND_FAILURE_DOMAIN = "ground_failure_realization"
GROUND_SELECTION_DOMAIN = "ground_station_selection"
MAX_SIGNED_SEED = 2**63 - 1


@dataclass(frozen=True)
class DSSRealizationSeeds:
    realization_index: int
    satellite_failure_seed: int
    ground_failure_seed: int
    ground_station_selection_seed: int


def _seed_from_payload(payload: dict[str, object]) -> int:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % MAX_SIGNED_SEED


def derive_seed(
    architecture: DSSArchitectureRequest,
    *,
    domain: str,
    realization_index: int | None = None,
) -> int:
    if not isinstance(architecture, DSSArchitectureRequest):
        raise TypeError("architecture must be a DSSArchitectureRequest")
    if realization_index is not None and not 0 <= realization_index < DSS_REALIZATION_COUNT:
        raise ValueError(f"realization_index must be within [0, {DSS_REALIZATION_COUNT - 1}]")
    payload: dict[str, object] = {
        "architecture": architecture.canonical_payload(include_threshold=False),
        "identity_domain": DSS_MASTER_SEED_DOMAIN,
        "identity_version": "1",
        "seed_domain": domain,
    }
    if realization_index is not None:
        payload["realization_index"] = realization_index
    return _seed_from_payload(payload)


def derive_realization_seeds(
    architecture: DSSArchitectureRequest, realization_index: int
) -> DSSRealizationSeeds:
    if not 0 <= realization_index < DSS_REALIZATION_COUNT:
        raise ValueError(f"realization_index must be within [0, {DSS_REALIZATION_COUNT - 1}]")
    return DSSRealizationSeeds(
        realization_index=realization_index,
        satellite_failure_seed=derive_seed(
            architecture, domain=SATELLITE_FAILURE_DOMAIN, realization_index=realization_index
        ),
        ground_failure_seed=derive_seed(
            architecture, domain=GROUND_FAILURE_DOMAIN, realization_index=realization_index
        ),
        ground_station_selection_seed=derive_seed(architecture, domain=GROUND_SELECTION_DOMAIN),
    )


def derive_all_realization_seeds(
    architecture: DSSArchitectureRequest,
) -> tuple[DSSRealizationSeeds, ...]:
    return tuple(derive_realization_seeds(architecture, index) for index in range(DSS_REALIZATION_COUNT))
