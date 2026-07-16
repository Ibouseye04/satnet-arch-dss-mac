from __future__ import annotations

from dataclasses import asdict
import math

import pytest

from satnet.ground.failure_policy import GroundFailurePolicy
from satnet.ground.failure_realization import create_ground_failure_realization_record
from satnet.ground.failure_service_metrics import (
    FailureAdjustedGroundServiceStepMetrics,
    FailureAdjustedGroundServiceStepRecord,
    _compute_failure_adjusted_step_hash,
    _step_record_hash,
    compute_failure_adjusted_ground_service_step,
    create_failure_adjusted_step_record,
    validate_failure_adjusted_ground_service_step_context,
)
from satnet.ground.service_aggregation import (
    GROUND_SERVICE_STEP_SCHEMA_VERSION,
    GroundServiceStepRecord,
    _step_record_hash as g4_step_record_hash,
)
from satnet.ground.service_policy import GroundServicePolicy
from tests.ground.test_g4_service_metrics import make_context


def g4_record(metrics, run_id: int = 11) -> GroundServiceStepRecord:
    return GroundServiceStepRecord(
        ground_service_step_schema_version=GROUND_SERVICE_STEP_SCHEMA_VERSION,
        run_id=run_id,
        metrics=metrics,
        record_hash=g4_step_record_hash(run_id=run_id, metrics=metrics),
    )


def context_with_service(*, policy: GroundServicePolicy | None = None):
    catalog, design, selection, _, default_policy, _ = make_context(policy=policy)
    attachments = tuple((0, station_id) for station_id in selection.selected_station_ids[:3])
    return make_context(attachments=attachments, policy=policy or default_policy)


def adjusted(*, probability: float, seed: int = 3, policy: GroundServicePolicy | None = None):
    catalog, design, selection, _, service_policy, baseline = context_with_service(policy=policy)
    failure_policy = GroundFailurePolicy(probability)
    realization = create_ground_failure_realization_record(
        run_id=design.run_id,
        ground_design=design,
        catalog=catalog,
        policy=failure_policy,
        ground_failure_seed=seed,
    )
    record = create_failure_adjusted_step_record(
        run_id=design.run_id,
        verified_g4_step_record=g4_record(baseline),
        ground_design=design,
        catalog=catalog,
        policy=failure_policy,
        realization_record=realization,
        ground_service_policy=service_policy,
    )
    return catalog, design, selection, service_policy, baseline, failure_policy, realization, record


def test_no_failures_exactly_preserve_baseline_service() -> None:
    *_, baseline, _, realization, record = adjusted(probability=0.0)
    metrics = record.metrics
    assert realization.realization.failed_station_ids == ()
    assert metrics.failure_adjusted_serviced_ground_station_ids == baseline.serviced_ground_station_ids
    assert metrics.failure_adjusted_ground_service_fraction == baseline.ground_service_fraction
    assert metrics.failure_adjusted_overall_service_fraction == baseline.overall_service_fraction
    assert metrics.ground_service_loss_due_to_failures == 0.0
    assert metrics.overall_service_loss_due_to_ground_failures == 0.0
    assert math.copysign(1.0, metrics.ground_service_loss_due_to_failures) == 1.0


def test_all_failures_zero_ground_service_and_preserve_satellite_evidence() -> None:
    *_, baseline, _, realization, record = adjusted(probability=1.0)
    metrics = record.metrics
    assert realization.realization.failed_ground_station_count == metrics.total_ground_station_count
    assert metrics.operational_ground_station_count == 0
    assert metrics.failure_adjusted_serviced_ground_station_count == 0
    assert metrics.failure_adjusted_ground_service_fraction == 0.0
    assert metrics.failure_adjusted_overall_service_fraction == 0.0
    for name in (
        "configured_satellite_count",
        "operational_satellite_count",
        "satellite_component_count",
        "satellite_gcc_size",
        "satellite_gcc_ids",
        "space_gcc_fraction_original",
        "space_gcc_fraction_surviving",
        "space_threshold_met",
    ):
        assert getattr(metrics, name) == getattr(baseline, name)


def test_partial_failure_is_persistent_set_subtraction_with_original_denominator() -> None:
    values = None
    for seed in range(100):
        candidate = adjusted(probability=0.5, seed=seed)
        failed = candidate[-2].realization.failed_station_ids
        if failed and len(failed) < candidate[-2].realization.total_ground_station_count:
            values = candidate
            break
    assert values is not None
    _, _, selection, _, baseline, _, realization, record = values
    metrics = record.metrics
    expected_serviced = tuple(
        sorted(set(baseline.serviced_ground_station_ids) - set(realization.realization.failed_station_ids))
    )
    assert metrics.failure_adjusted_serviced_ground_station_ids == expected_serviced
    assert metrics.failure_adjusted_ground_service_fraction == len(expected_serviced) / len(
        selection.selected_station_ids
    )
    assert set(metrics.failed_ground_station_ids).isdisjoint(
        metrics.failure_adjusted_serviced_ground_station_ids
    )


def test_failed_already_unserviced_station_does_not_change_service_fraction() -> None:
    catalog, design, selection, _, service_policy, _ = make_context()
    serviced_id = selection.selected_station_ids[0]
    catalog, design, selection, _, service_policy, baseline = make_context(
        attachments=((0, serviced_id),)
    )
    chosen = None
    for seed in range(1000):
        policy = GroundFailurePolicy(0.5)
        realization = create_ground_failure_realization_record(
            run_id=design.run_id,
            ground_design=design,
            catalog=catalog,
            policy=policy,
            ground_failure_seed=seed,
        )
        failed = set(realization.realization.failed_station_ids)
        if failed and serviced_id not in failed:
            chosen = (policy, realization)
            break
    assert chosen is not None
    policy, realization = chosen
    record = create_failure_adjusted_step_record(
        run_id=design.run_id,
        verified_g4_step_record=g4_record(baseline),
        ground_design=design,
        catalog=catalog,
        policy=policy,
        realization_record=realization,
        ground_service_policy=service_policy,
    )
    assert record.metrics.failed_ground_station_count > 0
    assert record.metrics.failure_adjusted_ground_service_fraction == baseline.ground_service_fraction


def test_class_counts_use_selected_class_denominators_and_absent_class_none() -> None:
    catalog, design, selection, _, service_policy, baseline = make_context(
        class_counts=(2, 1, 0)
    )
    failure_policy = GroundFailurePolicy(1.0)
    realization = create_ground_failure_realization_record(
        run_id=design.run_id,
        ground_design=design,
        catalog=catalog,
        policy=failure_policy,
        ground_failure_seed=2,
    )
    metrics = compute_failure_adjusted_ground_service_step(
        baseline_metrics=baseline,
        ground_design=design,
        catalog=catalog,
        failure_realization=realization.realization,
        ground_service_policy=service_policy,
    )
    assert metrics.failed_civilian_count == 2
    assert metrics.failed_government_count == 1
    assert metrics.failure_adjusted_civilian_service_fraction == 0.0
    assert metrics.failure_adjusted_government_service_fraction == 0.0
    assert metrics.failure_adjusted_military_service_fraction is None


def test_failure_can_transition_ground_and_overall_thresholds_only_downward() -> None:
    policy = GroundServicePolicy(0.5, 0.5)
    *_, baseline, _, _, record = adjusted(probability=1.0, policy=policy)
    assert baseline.space_threshold_met
    assert baseline.ground_threshold_met
    assert baseline.overall_threshold_met
    assert record.metrics.space_threshold_met
    assert not record.metrics.ground_threshold_met
    assert not record.metrics.overall_threshold_met


def test_zero_threshold_retains_policy_compliance_with_zero_service() -> None:
    policy = GroundServicePolicy(0.0, 0.0)
    *_, record = adjusted(probability=1.0, policy=policy)
    assert record.metrics.failure_adjusted_ground_service_fraction == 0.0
    assert record.metrics.space_threshold_met
    assert record.metrics.ground_threshold_met
    assert record.metrics.overall_threshold_met


def test_contextual_validation_rejects_wrong_policy_and_baseline() -> None:
    catalog, design, _, _, baseline, _, realization, record = adjusted(probability=0.5)
    with pytest.raises(ValueError):
        validate_failure_adjusted_ground_service_step_context(
            metrics=record.metrics,
            baseline_metrics=baseline,
            ground_design=design,
            catalog=catalog,
            failure_realization=realization.realization,
            ground_service_policy=GroundServicePolicy(0.1, 0.1),
        )


def test_negative_zero_scientific_evidence_is_rejected() -> None:
    *_, record = adjusted(probability=0.0)
    values = asdict(record.metrics)
    values["ground_service_loss_due_to_failures"] = -0.0
    values["failure_adjusted_step_hash"] = _compute_failure_adjusted_step_hash(values)
    with pytest.raises(ValueError, match="positive zero"):
        FailureAdjustedGroundServiceStepMetrics(**values)


def test_step_record_identity_binds_run_and_rejects_corruption() -> None:
    *_, record = adjusted(probability=0.5)
    alternate = FailureAdjustedGroundServiceStepRecord(
        ground_failure_service_step_schema_version=(
            record.ground_failure_service_step_schema_version
        ),
        run_id=99,
        metrics=record.metrics,
        record_hash=_step_record_hash(run_id=99, metrics=record.metrics),
    )
    assert alternate.metrics.failure_adjusted_step_hash == record.metrics.failure_adjusted_step_hash
    assert alternate.record_hash != record.record_hash
    with pytest.raises(ValueError, match="record_hash"):
        FailureAdjustedGroundServiceStepRecord(
            ground_failure_service_step_schema_version=(
                record.ground_failure_service_step_schema_version
            ),
            run_id=record.run_id,
            metrics=record.metrics,
            record_hash="0" * 64,
        )


def test_replacement_hash_does_not_validate_wrong_adjusted_service() -> None:
    *_, record = adjusted(probability=0.0)
    values = asdict(record.metrics)
    values["failure_adjusted_serviced_ground_station_ids"] = ()
    values["failure_adjusted_serviced_ground_station_count"] = 0
    values["failure_adjusted_unserviced_ground_station_ids"] = tuple(
        sorted(
            set(record.metrics.failed_ground_station_ids)
            | set(record.metrics.operational_ground_station_ids)
        )
    )
    values["failure_adjusted_unserviced_ground_station_count"] = len(
        values["failure_adjusted_unserviced_ground_station_ids"]
    )
    values["failure_adjusted_ground_service_fraction"] = 0.0
    values["failure_adjusted_overall_service_fraction"] = 0.0
    values["ground_service_loss_due_to_failures"] = values[
        "baseline_ground_service_fraction"
    ]
    values["overall_service_loss_due_to_ground_failures"] = values[
        "baseline_overall_service_fraction"
    ]
    values["ground_threshold_met"] = False
    values["overall_threshold_met"] = False
    for name in ("civilian", "government", "military"):
        values[f"failure_adjusted_serviced_{name}_count"] = 0
        values[f"failure_adjusted_{name}_service_fraction"] = (
            None if values[f"total_{name}_count"] == 0 else 0.0
        )
    values["failure_adjusted_step_hash"] = _compute_failure_adjusted_step_hash(values)
    corrupted = FailureAdjustedGroundServiceStepMetrics(**values)
    catalog, design, _, _, baseline, _, realization, _ = adjusted(probability=0.0)
    with pytest.raises(ValueError, match="contextual mismatch"):
        validate_failure_adjusted_ground_service_step_context(
            metrics=corrupted,
            baseline_metrics=baseline,
            ground_design=design,
            catalog=catalog,
            failure_realization=realization.realization,
            ground_service_policy=GroundServicePolicy(0.5, 0.5),
        )
