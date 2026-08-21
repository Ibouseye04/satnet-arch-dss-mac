from __future__ import annotations

from dataclasses import replace
import math

import pytest

from satnet.ground.canonical import canonical_json
from satnet.ground.catalog import GroundStation, GroundStationCatalog, GroundStationClass
from satnet.ground.failure_policy import (
    MAX_GROUND_FAILURE_SEED,
    GroundFailurePolicy,
)
from satnet.ground.failure_realization import (
    GroundFailureRealization,
    GroundFailureRealizationRecord,
    create_ground_failure_realization_record,
    ground_failure_trial_bytes,
    ground_failure_trial_digest,
    ground_failure_trial_payload,
    ground_station_fails,
    sample_ground_failure_realization,
    validate_ground_failure_realization_context,
    validate_ground_failure_realization_record_context,
)
from satnet.ground.persistence import make_enabled_ground_design_record
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from tests.ground.test_g4_service_metrics import make_context


def test_policy_validation_identity_and_negative_zero() -> None:
    negative = GroundFailurePolicy(-0.0)
    positive = GroundFailurePolicy(0)
    assert negative == positive
    assert negative.ground_failure_policy_hash == positive.ground_failure_policy_hash
    assert math.copysign(1.0, negative.ground_station_failure_probability) == 1.0
    for value in (True, "0.5", float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises((TypeError, ValueError)):
            GroundFailurePolicy(value)


def test_locked_station_trial_golden_vectors() -> None:
    payload = ground_failure_trial_payload(
        station_id="CIV_G5_001", ground_failure_seed=42
    )
    expected_json = (
        '{"ground_failure_model_version":"1",'
        '"ground_failure_sampling_version":"1",'
        '"ground_failure_seed":42,'
        '"identity_domain":"satnet_ground_failure_trial",'
        '"identity_version":"1",'
        '"station_id":"CIV_G5_001"}'
    )
    expected_digest = "13f9592fbd15640525e827b6534fef0f38844469c702f221a89af958e2b73d14"
    assert canonical_json(payload) == expected_json
    assert ground_failure_trial_bytes(
        station_id="CIV_G5_001", ground_failure_seed=42
    ) == expected_json.encode("utf-8")
    assert ground_failure_trial_digest(
        station_id="CIV_G5_001", ground_failure_seed=42
    ).hex() == expected_digest
    assert not ground_station_fails(
        station_id="CIV_G5_001",
        ground_failure_seed=42,
        policy=GroundFailurePolicy(0.0),
    )
    assert ground_station_fails(
        station_id="CIV_G5_001",
        ground_failure_seed=42,
        policy=GroundFailurePolicy(1.0),
    )
    interior_nonfailure = GroundFailurePolicy(0.05)
    interior_failure = GroundFailurePolicy(0.1)
    assert interior_nonfailure.ground_station_failure_probability.as_integer_ratio() == (
        3602879701896397,
        72057594037927936,
    )
    assert interior_failure.ground_station_failure_probability.as_integer_ratio() == (
        3602879701896397,
        36028797018963968,
    )
    assert not ground_station_fails(
        station_id="CIV_G5_001",
        ground_failure_seed=42,
        policy=interior_nonfailure,
    )
    assert ground_station_fails(
        station_id="CIV_G5_001",
        ground_failure_seed=42,
        policy=interior_failure,
    )


def test_probability_endpoints_sample_exact_populations() -> None:
    catalog, design, selection, _, _, _ = make_context()
    none_failed = sample_ground_failure_realization(
        ground_design=design,
        catalog=catalog,
        policy=GroundFailurePolicy(0.0),
        ground_failure_seed=8,
    )
    all_failed = sample_ground_failure_realization(
        ground_design=design,
        catalog=catalog,
        policy=GroundFailurePolicy(1.0),
        ground_failure_seed=8,
    )
    expected = tuple(sorted(selection.selected_station_ids))
    assert none_failed.selected_station_ids == expected
    assert none_failed.failed_station_ids == ()
    assert none_failed.operational_station_ids == expected
    assert all_failed.failed_station_ids == expected
    assert all_failed.operational_station_ids == ()


def test_g1_class_order_binds_by_population_to_g5_global_order() -> None:
    stations = (
        GroundStation(
            station_id="ZZZ_CIV_001",
            name="Civilian",
            station_class=GroundStationClass.CIVILIAN,
            latitude_deg=0.0,
            longitude_deg=0.0,
            altitude_m=0.0,
            region="region_test",
            country_code="ZZ",
        ),
        GroundStation(
            station_id="AAA_GOV_001",
            name="Government",
            station_class=GroundStationClass.GOVERNMENT,
            latitude_deg=1.0,
            longitude_deg=1.0,
            altitude_m=0.0,
            region="region_test",
            country_code="ZZ",
        ),
    )
    catalog = GroundStationCatalog(stations)
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(1, 1, 0, station_selection_seed=2),
    )
    design = make_enabled_ground_design_record(
        run_id=3,
        satellite_config_hash="a" * 64,
        selection=selection,
    )
    realization = sample_ground_failure_realization(
        ground_design=design,
        catalog=catalog,
        policy=GroundFailurePolicy(0.0),
        ground_failure_seed=4,
    )
    assert selection.selected_station_ids == ("ZZZ_CIV_001", "AAA_GOV_001")
    assert realization.selected_station_ids == ("AAA_GOV_001", "ZZZ_CIV_001")
    assert set(realization.selected_station_ids) == set(selection.selected_station_ids)


def test_seed_and_probability_change_identity_without_outcome_requirement() -> None:
    catalog, design, _, _, _, _ = make_context()
    first = sample_ground_failure_realization(
        ground_design=design,
        catalog=catalog,
        policy=GroundFailurePolicy(0.0),
        ground_failure_seed=1,
    )
    second = sample_ground_failure_realization(
        ground_design=design,
        catalog=catalog,
        policy=GroundFailurePolicy(0.0),
        ground_failure_seed=2,
    )
    different_policy = sample_ground_failure_realization(
        ground_design=design,
        catalog=catalog,
        policy=GroundFailurePolicy(0.01),
        ground_failure_seed=1,
    )
    assert first.failed_station_ids == second.failed_station_ids == ()
    assert first.ground_failure_realization_hash != second.ground_failure_realization_hash
    assert first.ground_failure_realization_hash != different_policy.ground_failure_realization_hash


@pytest.mark.parametrize("seed", [True, -1, MAX_GROUND_FAILURE_SEED + 1, 1.0, "1"])
def test_invalid_seeds_fail(seed: object) -> None:
    catalog, design, _, _, _, _ = make_context()
    with pytest.raises((TypeError, ValueError)):
        sample_ground_failure_realization(
            ground_design=design,
            catalog=catalog,
            policy=GroundFailurePolicy(0.5),
            ground_failure_seed=seed,
        )


def test_realization_relational_corruption_fails_even_with_replacement_hash() -> None:
    catalog, design, _, _, _, _ = make_context()
    realization = sample_ground_failure_realization(
        ground_design=design,
        catalog=catalog,
        policy=GroundFailurePolicy(0.0),
        ground_failure_seed=1,
    )
    with pytest.raises(ValueError, match="partition"):
        replace(
            realization,
            operational_station_ids=realization.operational_station_ids[:-1],
            operational_ground_station_count=(
                realization.operational_ground_station_count - 1
            ),
        )


def test_contextual_resampling_rejects_wrong_policy_and_design() -> None:
    catalog, design, _, _, _, _ = make_context()
    policy = GroundFailurePolicy(0.25)
    realization = sample_ground_failure_realization(
        ground_design=design,
        catalog=catalog,
        policy=policy,
        ground_failure_seed=5,
    )
    with pytest.raises(ValueError, match="authoritative context"):
        validate_ground_failure_realization_context(
            realization=realization,
            ground_design=design,
            catalog=catalog,
            policy=GroundFailurePolicy(0.5),
        )


def test_realization_record_factory_binds_run_and_context() -> None:
    catalog, design, _, _, _, _ = make_context()
    policy = GroundFailurePolicy(0.25)
    record = create_ground_failure_realization_record(
        run_id=design.run_id,
        ground_design=design,
        catalog=catalog,
        policy=policy,
        ground_failure_seed=9,
    )
    validate_ground_failure_realization_record_context(
        record=record,
        ground_design=design,
        catalog=catalog,
        policy=policy,
    )
    with pytest.raises(ValueError, match="run ID"):
        create_ground_failure_realization_record(
            run_id=design.run_id + 1,
            ground_design=design,
            catalog=catalog,
            policy=policy,
            ground_failure_seed=9,
        )
    with pytest.raises(ValueError, match="record_hash"):
        GroundFailureRealizationRecord(
            ground_failure_realization_schema_version=(
                record.ground_failure_realization_schema_version
            ),
            run_id=record.run_id,
            realization=record.realization,
            record_hash="0" * 64,
        )


def test_disabled_design_fails_before_sampling() -> None:
    catalog, design, _, _, _, _ = make_context()
    forged = object.__new__(type(design))
    for name, value in design.__dict__.items():
        object.__setattr__(forged, name, value)
    object.__setattr__(forged, "ground_segment_enabled", False)
    with pytest.raises(ValueError, match="enabled"):
        sample_ground_failure_realization(
            ground_design=forged,
            catalog=catalog,
            policy=GroundFailurePolicy(0.5),
            ground_failure_seed=1,
        )


def test_direct_self_consistent_realization_still_requires_context() -> None:
    catalog, design, _, _, _, _ = make_context()
    policy = GroundFailurePolicy(0.0)
    expected = sample_ground_failure_realization(
        ground_design=design,
        catalog=catalog,
        policy=policy,
        ground_failure_seed=1,
    )
    alternate = GroundFailureRealization(
        ground_failure_model_version=expected.ground_failure_model_version,
        ground_failure_sampling_version=expected.ground_failure_sampling_version,
        ground_design_hash=expected.ground_design_hash,
        ground_failure_policy_hash=expected.ground_failure_policy_hash,
        ground_failure_seed=expected.ground_failure_seed,
        selected_station_ids=expected.selected_station_ids,
        failed_station_ids=expected.failed_station_ids,
        operational_station_ids=expected.operational_station_ids,
        total_ground_station_count=expected.total_ground_station_count,
        failed_ground_station_count=expected.failed_ground_station_count,
        operational_ground_station_count=expected.operational_ground_station_count,
        ground_failure_realization_hash=expected.ground_failure_realization_hash,
    )
    validate_ground_failure_realization_context(
        realization=alternate,
        ground_design=design,
        catalog=catalog,
        policy=policy,
    )
