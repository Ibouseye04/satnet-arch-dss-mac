from __future__ import annotations

from satnet.dss.realization import (
    GROUND_FAILURE_DOMAIN,
    GROUND_SELECTION_DOMAIN,
    SATELLITE_FAILURE_DOMAIN,
    derive_all_realization_seeds,
    derive_seed,
)
from satnet.dss.schemas import DSS_REALIZATION_COUNT
from tests.dss.test_schemas_and_policy import valid_request


def test_exactly_five_deterministic_realization_seed_sets() -> None:
    seeds = derive_all_realization_seeds(valid_request())
    assert len(seeds) == DSS_REALIZATION_COUNT == 5
    assert [seed.realization_index for seed in seeds] == list(range(5))
    assert len({seed.satellite_failure_seed for seed in seeds}) == 5
    assert len({seed.ground_failure_seed for seed in seeds}) == 5
    assert len({seed.ground_station_selection_seed for seed in seeds}) == 1


def test_seed_domains_are_separate() -> None:
    request = valid_request()
    assert derive_seed(request, domain=SATELLITE_FAILURE_DOMAIN, realization_index=0) != derive_seed(
        request, domain=GROUND_FAILURE_DOMAIN, realization_index=0
    )
    assert derive_seed(request, domain=GROUND_SELECTION_DOMAIN) != derive_seed(
        request, domain=SATELLITE_FAILURE_DOMAIN, realization_index=0
    )


def test_repeated_seed_derivation_is_byte_stable() -> None:
    request = valid_request()
    assert derive_all_realization_seeds(request) == derive_all_realization_seeds(request)
