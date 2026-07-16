from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
import math

import pytest

from satnet.ground.service_aggregation import (
    GROUND_SERVICE_STEP_SCHEMA_VERSION,
    GroundServiceRunSummary,
    GroundServiceStepRecord,
    _compute_run_summary_hash,
    _step_record_hash,
    summarize_ground_service_run,
    validate_ground_service_run_summary,
)
from satnet.ground.service_metrics import (
    GroundServiceStepMetrics,
    _compute_step_metrics_hash,
)
from satnet.ground.service_policy import GroundServicePolicy
from tests.ground.test_g4_service_metrics import TIMESTAMP, make_context


def rekey_metrics(
    metrics: GroundServiceStepMetrics,
    *,
    timestep_index: int,
    seconds: int,
) -> GroundServiceStepMetrics:
    values = asdict(metrics)
    values["timestep_index"] = timestep_index
    values["timestamp_utc"] = TIMESTAMP + timedelta(seconds=seconds)
    values["step_metrics_hash"] = _compute_step_metrics_hash(values)
    return GroundServiceStepMetrics(**values)


def record(metrics: GroundServiceStepMetrics, run_id: int = 11) -> GroundServiceStepRecord:
    return GroundServiceStepRecord(
        ground_service_step_schema_version=GROUND_SERVICE_STEP_SCHEMA_VERSION,
        run_id=run_id,
        metrics=metrics,
        record_hash=_step_record_hash(run_id=run_id, metrics=metrics),
    )


def three_records(*, absent_classes: bool = False) -> tuple[GroundServiceStepRecord, ...]:
    class_counts = (2, 0, 0) if absent_classes else (2, 1, 1)
    policy = GroundServicePolicy(0.5, 0.5)
    _, _, selection, _, _, _ = make_context(class_counts=class_counts, policy=policy)
    selected = selection.selected_station_ids
    first = make_context(
        class_counts=class_counts,
        satellite_ids=(0, 1, 2, 3),
        isl_edges=((0, 1), (1, 2), (2, 3)),
        attachments=tuple((0, station_id) for station_id in selected),
        policy=policy,
    )[-1]
    second = make_context(
        class_counts=class_counts,
        satellite_ids=(0, 1, 2, 3),
        isl_edges=((0, 1), (2, 3)),
        attachments=((0, selected[0]),),
        policy=policy,
    )[-1]
    third = make_context(
        class_counts=class_counts,
        satellite_ids=(0,),
        isl_edges=(),
        attachments=(),
        policy=policy,
    )[-1]
    return (
        record(rekey_metrics(first, timestep_index=4, seconds=0)),
        record(rekey_metrics(second, timestep_index=5, seconds=60)),
        record(rekey_metrics(third, timestep_index=6, seconds=120)),
    )


def rebuilt_summary(summary: GroundServiceRunSummary, **changes: object) -> GroundServiceRunSummary:
    values = asdict(summary)
    values.update(changes)
    values["run_summary_hash"] = _compute_run_summary_hash(values)
    return GroundServiceRunSummary(**values)


def test_run_aggregation_exact_minimum_fsum_mean_and_breaches() -> None:
    records = three_records()
    summary = summarize_ground_service_run(step_records=records)
    original = [record.metrics.space_gcc_fraction_original for record in records]
    ground = [record.metrics.ground_service_fraction for record in records]
    overall = [record.metrics.overall_service_fraction for record in records]
    assert summary.first_timestep_index == 4
    assert summary.last_timestep_index == 6
    assert summary.timestep_count == 3
    assert summary.space_gcc_fraction_original_min == min(original) == 0.25
    assert summary.space_gcc_fraction_original_mean == math.fsum(original) / 3
    assert summary.ground_service_fraction_min == min(ground) == 0.0
    assert summary.ground_service_fraction_mean == math.fsum(ground) / 3
    assert summary.overall_service_fraction_min == min(overall) == 0.0
    assert summary.overall_service_fraction_mean == math.fsum(overall) / 3
    assert summary.space_threshold_breach_any
    assert summary.space_threshold_breach_timestep_count == 1
    assert summary.first_space_threshold_breach_timestep == 6
    assert summary.ground_threshold_breach_any
    assert summary.ground_threshold_breach_timestep_count == 2
    assert summary.first_ground_threshold_breach_timestep == 5
    assert summary.overall_threshold_breach_any
    assert summary.overall_threshold_breach_timestep_count == 2
    assert summary.first_overall_threshold_breach_timestep == 5
    validate_ground_service_run_summary(summary=summary, step_records=records)


def test_absent_class_run_aggregates_remain_none() -> None:
    summary = summarize_ground_service_run(step_records=three_records(absent_classes=True))
    assert summary.total_government_count == 0
    assert summary.government_service_fraction_min is None
    assert summary.government_service_fraction_mean is None
    assert summary.total_military_count == 0
    assert summary.military_service_fraction_min is None
    assert summary.military_service_fraction_mean is None


def test_step_sequence_hash_is_scientific_and_order_bound() -> None:
    records = three_records()
    summary = summarize_ground_service_run(step_records=records)
    different_run = tuple(record(step.metrics, run_id=99) for step in records)
    equivalent = summarize_ground_service_run(step_records=different_run)
    assert equivalent.run_id == 99
    assert equivalent.step_sequence_hash == summary.step_sequence_hash
    assert equivalent.run_summary_hash == summary.run_summary_hash
    with pytest.raises(ValueError, match="increasing order"):
        summarize_ground_service_run(step_records=(records[1], records[0], records[2]))
    reassigned = (
        records[0],
        record(rekey_metrics(records[2].metrics, timestep_index=5, seconds=60)),
        record(rekey_metrics(records[1].metrics, timestep_index=6, seconds=120)),
    )
    changed = summarize_ground_service_run(step_records=reassigned)
    assert changed.step_sequence_hash != summary.step_sequence_hash


@pytest.mark.parametrize(
    ("records_factory", "message"),
    [
        (lambda records: (), "At least one"),
        (lambda records: (records[0], record(records[1].metrics, run_id=12)), "mixed run"),
        (lambda records: (records[0], records[0]), "duplicate"),
        (lambda records: (records[0], records[2]), "contiguous"),
        (lambda records: (records[1], records[0]), "increasing order"),
    ],
)
def test_run_aggregation_rejects_invalid_step_collections(records_factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_ground_service_run(step_records=records_factory(three_records()))


def test_timestamp_reversal_is_rejected() -> None:
    records = three_records()
    reversed_time = record(rekey_metrics(records[1].metrics, timestep_index=5, seconds=-60))
    with pytest.raises(ValueError, match="timestamps"):
        summarize_ground_service_run(step_records=(records[0], reversed_time, records[2]))


@pytest.mark.parametrize(
    "field_name",
    [
        "satellite_config_hash",
        "ground_design_hash",
        "visibility_policy_hash",
        "ground_service_policy_hash",
        "ground_service_model_version",
        "configured_satellite_count",
        "total_ground_station_count",
        "total_civilian_count",
    ],
)
def test_constant_scientific_fields_are_required(field_name: str) -> None:
    records = list(three_records())
    values = asdict(records[1].metrics)
    current = values[field_name]
    if field_name.endswith("_hash"):
        values[field_name] = "f" * 64
    elif field_name == "ground_service_model_version":
        values[field_name] = "2"
    else:
        values[field_name] = current + 1
    if field_name in {
        "ground_service_model_version",
        "configured_satellite_count",
        "total_ground_station_count",
        "total_civilian_count",
    }:
        with pytest.raises((TypeError, ValueError)):
            values["step_metrics_hash"] = _compute_step_metrics_hash(values)
            GroundServiceStepMetrics(**values)
        return
    values["step_metrics_hash"] = _compute_step_metrics_hash(values)
    metrics = GroundServiceStepMetrics(**values)
    records[1] = record(metrics)
    with pytest.raises(ValueError, match="constant"):
        summarize_ground_service_run(step_records=records)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("space_gcc_fraction_original_min", 0.0),
        ("space_gcc_fraction_original_mean", 0.0),
        ("ground_service_fraction_min", 0.1),
        ("overall_service_fraction_mean", 0.1),
        ("first_space_threshold_breach_timestep", 5),
        ("space_threshold_breach_timestep_count", 2),
        ("step_sequence_hash", "e" * 64),
    ],
)
def test_summary_corruption_fails_with_replacement_hash_contextually(
    field_name: str, value: object
) -> None:
    records = three_records()
    summary = summarize_ground_service_run(step_records=records)
    corrupted = rebuilt_summary(summary, **{field_name: value})
    with pytest.raises(ValueError, match="summary mismatch"):
        validate_ground_service_run_summary(summary=corrupted, step_records=records)


def test_breach_relational_corruption_fails_intrinsically() -> None:
    summary = summarize_ground_service_run(step_records=three_records())
    with pytest.raises(ValueError, match="breach flag"):
        rebuilt_summary(
            summary,
            space_threshold_breach_any=False,
        )


def test_step_record_rejects_boolean_run_id_and_hash_corruption() -> None:
    metrics = three_records()[0].metrics
    with pytest.raises(TypeError, match="run_id"):
        record(metrics, run_id=True)
    with pytest.raises(ValueError, match="record_hash"):
        GroundServiceStepRecord(
            ground_service_step_schema_version=GROUND_SERVICE_STEP_SCHEMA_VERSION,
            run_id=11,
            metrics=metrics,
            record_hash="0" * 64,
        )
