from __future__ import annotations

from dataclasses import asdict, replace
import math

import pytest

from satnet.ground.failure_policy import GroundFailurePolicy
from satnet.ground.failure_realization import create_ground_failure_realization_record
from satnet.ground.failure_service_aggregation import (
    FailureAdjustedGroundServiceRunRecord,
    FailureAdjustedGroundServiceRunSummary,
    _compute_run_summary_hash,
    _run_record_hash,
    create_failure_adjusted_run_record,
    summarize_failure_adjusted_ground_service_run,
    validate_failure_adjusted_ground_service_run_summary,
)
from satnet.ground.failure_service_metrics import (
    FailureAdjustedGroundServiceStepMetrics,
    FailureAdjustedGroundServiceStepRecord,
    _compute_failure_adjusted_step_hash,
    _step_record_hash as g5_step_record_hash,
    create_failure_adjusted_step_record,
)
from satnet.ground.service_aggregation import summarize_ground_service_run
from satnet.ground.service_persistence import make_ground_service_run_record
from satnet.ground.service_policy import GroundServicePolicy
from tests.ground.test_g4_service_aggregation import record as g4_record
from tests.ground.test_g4_service_aggregation import rekey_metrics
from tests.ground.test_g4_service_metrics import make_context


def run_context(*, probability: float = 0.5, seed: int = 7):
    service_policy = GroundServicePolicy(0.5, 0.5)
    catalog, design, selection, _, _, _ = make_context(policy=service_policy)
    attachments = (
        tuple((0, station_id) for station_id in selection.selected_station_ids),
        ((0, selection.selected_station_ids[0]),),
        (),
    )
    g4_steps = []
    for offset, step_attachments in enumerate(attachments):
        baseline = make_context(
            attachments=step_attachments,
            policy=service_policy,
        )[-1]
        baseline = rekey_metrics(
            baseline,
            timestep_index=4 + offset,
            seconds=60 * offset,
        )
        g4_steps.append(g4_record(baseline, run_id=design.run_id))
    g4_steps_tuple = tuple(g4_steps)
    g4_summary = summarize_ground_service_run(step_records=g4_steps_tuple)
    g4_run = make_ground_service_run_record(
        summary=g4_summary,
        step_records=g4_steps_tuple,
    )
    failure_policy = GroundFailurePolicy(probability)
    realization = create_ground_failure_realization_record(
        run_id=design.run_id,
        ground_design=design,
        catalog=catalog,
        policy=failure_policy,
        ground_failure_seed=seed,
    )
    g5_steps = tuple(
        create_failure_adjusted_step_record(
            run_id=design.run_id,
            verified_g4_step_record=step,
            ground_design=design,
            catalog=catalog,
            policy=failure_policy,
            realization_record=realization,
            ground_service_policy=service_policy,
        )
        for step in g4_steps_tuple
    )
    return catalog, design, service_policy, failure_policy, realization, g4_steps_tuple, g4_run, g5_steps


def rebuilt_summary(
    summary: FailureAdjustedGroundServiceRunSummary, **changes: object
) -> FailureAdjustedGroundServiceRunSummary:
    values = asdict(summary)
    values.update(changes)
    values["failure_adjusted_run_summary_hash"] = _compute_run_summary_hash(values)
    return FailureAdjustedGroundServiceRunSummary(**values)


def test_run_aggregation_binds_exact_g4_summary_and_sequence() -> None:
    *_, realization, g4_steps, g4_run, g5_steps = run_context()
    summary = summarize_failure_adjusted_ground_service_run(
        step_records=g5_steps,
        failure_realization_record=realization,
        verified_g4_step_records=g4_steps,
        verified_g4_run_record=g4_run,
    )
    assert summary.baseline_ground_service_run_summary_hash == g4_run.summary.run_summary_hash
    assert summary.baseline_ground_service_step_sequence_hash == g4_run.summary.step_sequence_hash
    assert summary.first_timestep_index == 4
    assert summary.last_timestep_index == 6
    assert summary.timestep_count == 3
    values = [step.metrics.failure_adjusted_ground_service_fraction for step in g5_steps]
    losses = [step.metrics.ground_service_loss_due_to_failures for step in g5_steps]
    assert summary.failure_adjusted_ground_service_fraction_min == min(values)
    assert summary.failure_adjusted_ground_service_fraction_mean == math.fsum(values) / 3
    assert summary.ground_service_loss_due_to_failures_max == max(losses)
    assert summary.ground_service_loss_due_to_failures_mean == math.fsum(losses) / 3


def test_baseline_aggregates_equal_authoritative_g4_summary() -> None:
    *_, realization, g4_steps, g4_run, g5_steps = run_context()
    summary = summarize_failure_adjusted_ground_service_run(
        step_records=g5_steps,
        failure_realization_record=realization,
        verified_g4_step_records=g4_steps,
        verified_g4_run_record=g4_run,
    )
    assert summary.space_gcc_fraction_original_min == g4_run.summary.space_gcc_fraction_original_min
    assert summary.space_gcc_fraction_original_mean == g4_run.summary.space_gcc_fraction_original_mean
    assert summary.space_gcc_fraction_surviving_min == g4_run.summary.space_gcc_fraction_surviving_min
    assert summary.space_gcc_fraction_surviving_mean == g4_run.summary.space_gcc_fraction_surviving_mean
    assert summary.baseline_ground_service_fraction_min == g4_run.summary.ground_service_fraction_min
    assert summary.baseline_ground_service_fraction_mean == g4_run.summary.ground_service_fraction_mean
    assert summary.baseline_overall_service_fraction_min == g4_run.summary.overall_service_fraction_min
    assert summary.baseline_overall_service_fraction_mean == g4_run.summary.overall_service_fraction_mean
    assert summary.space_threshold_breach_any == g4_run.summary.space_threshold_breach_any
    assert (
        summary.first_space_threshold_breach_timestep
        == g4_run.summary.first_space_threshold_breach_timestep
    )
    assert (
        summary.space_threshold_breach_timestep_count
        == g4_run.summary.space_threshold_breach_timestep_count
    )


def test_run_record_factory_performs_contextual_validation() -> None:
    *_, realization, g4_steps, g4_run, g5_steps = run_context()
    record = create_failure_adjusted_run_record(
        verified_g4_step_records=g4_steps,
        verified_g4_run_record=g4_run,
        g5_step_records=g5_steps,
        realization_record=realization,
    )
    validate_failure_adjusted_ground_service_run_summary(
        summary=record.summary,
        step_records=g5_steps,
        failure_realization_record=realization,
        verified_g4_step_records=g4_steps,
        verified_g4_run_record=g4_run,
    )
    assert record.run_id == realization.run_id


def test_rejects_g4_run_record_that_does_not_summarize_steps() -> None:
    *_, realization, g4_steps, _, g5_steps = run_context()
    wrong_summary = summarize_ground_service_run(step_records=g4_steps[:-1])
    wrong_run = make_ground_service_run_record(
        summary=wrong_summary,
        step_records=g4_steps[:-1],
    )
    with pytest.raises(ValueError, match="summary mismatch"):
        summarize_failure_adjusted_ground_service_run(
            step_records=g5_steps,
            failure_realization_record=realization,
            verified_g4_step_records=g4_steps,
            verified_g4_run_record=wrong_run,
        )


def test_rejects_unequal_g4_g5_keys_and_wrong_baseline_step_hash() -> None:
    *_, realization, g4_steps, g4_run, g5_steps = run_context()
    with pytest.raises(ValueError, match="keys do not match"):
        summarize_failure_adjusted_ground_service_run(
            step_records=g5_steps[:-1],
            failure_realization_record=realization,
            verified_g4_step_records=g4_steps,
            verified_g4_run_record=g4_run,
        )
    values = asdict(g5_steps[0].metrics)
    values["baseline_ground_service_step_hash"] = "f" * 64
    values["failure_adjusted_step_hash"] = _compute_failure_adjusted_step_hash(values)
    corrupted_metrics = FailureAdjustedGroundServiceStepMetrics(**values)
    corrupted = FailureAdjustedGroundServiceStepRecord(
        ground_failure_service_step_schema_version=(
            g5_steps[0].ground_failure_service_step_schema_version
        ),
        run_id=g5_steps[0].run_id,
        metrics=corrupted_metrics,
        record_hash=g5_step_record_hash(
            run_id=g5_steps[0].run_id,
            metrics=corrupted_metrics,
        ),
    )
    with pytest.raises(ValueError):
        summarize_failure_adjusted_ground_service_run(
            step_records=(corrupted, *g5_steps[1:]),
            failure_realization_record=realization,
            verified_g4_step_records=g4_steps,
            verified_g4_run_record=g4_run,
        )


def test_rejects_duplicate_missing_reordered_and_mixed_run_steps() -> None:
    *_, realization, g4_steps, g4_run, g5_steps = run_context()
    mixed = FailureAdjustedGroundServiceStepRecord(
        ground_failure_service_step_schema_version=(
            g5_steps[1].ground_failure_service_step_schema_version
        ),
        run_id=99,
        metrics=g5_steps[1].metrics,
        record_hash=g5_step_record_hash(run_id=99, metrics=g5_steps[1].metrics),
    )
    cases = (
        (g5_steps[0], g5_steps[0]),
        (g5_steps[0], g5_steps[2]),
        tuple(reversed(g5_steps)),
        (g5_steps[0], mixed),
    )
    for steps in cases:
        with pytest.raises((TypeError, ValueError)):
            summarize_failure_adjusted_ground_service_run(
                step_records=steps,
                failure_realization_record=realization,
                verified_g4_step_records=g4_steps,
                verified_g4_run_record=g4_run,
            )


def test_summary_corruption_with_replacement_hash_fails_contextually() -> None:
    *_, realization, g4_steps, g4_run, g5_steps = run_context()
    summary = summarize_failure_adjusted_ground_service_run(
        step_records=g5_steps,
        failure_realization_record=realization,
        verified_g4_step_records=g4_steps,
        verified_g4_run_record=g4_run,
    )
    corrupted = rebuilt_summary(
        summary,
        baseline_ground_service_run_summary_hash="e" * 64,
    )
    with pytest.raises(ValueError, match="summary mismatch"):
        validate_failure_adjusted_ground_service_run_summary(
            summary=corrupted,
            step_records=g5_steps,
            failure_realization_record=realization,
            verified_g4_step_records=g4_steps,
            verified_g4_run_record=g4_run,
        )


def test_negative_zero_run_aggregate_is_rejected() -> None:
    *_, realization, g4_steps, g4_run, g5_steps = run_context(probability=0.0)
    summary = summarize_failure_adjusted_ground_service_run(
        step_records=g5_steps,
        failure_realization_record=realization,
        verified_g4_step_records=g4_steps,
        verified_g4_run_record=g4_run,
    )
    values = asdict(summary)
    values["ground_service_loss_due_to_failures_mean"] = -0.0
    values["failure_adjusted_run_summary_hash"] = _compute_run_summary_hash(values)
    with pytest.raises(ValueError, match="positive zero"):
        FailureAdjustedGroundServiceRunSummary(**values)


def test_run_record_hash_binds_run_id_and_rejects_corruption() -> None:
    *_, realization, g4_steps, g4_run, g5_steps = run_context()
    record = create_failure_adjusted_run_record(
        verified_g4_step_records=g4_steps,
        verified_g4_run_record=g4_run,
        g5_step_records=g5_steps,
        realization_record=realization,
    )
    alternate_summary = replace(record.summary, run_id=99)
    alternate = FailureAdjustedGroundServiceRunRecord(
        ground_failure_service_run_schema_version=(
            record.ground_failure_service_run_schema_version
        ),
        run_id=99,
        summary=alternate_summary,
        record_hash=_run_record_hash(run_id=99, summary=alternate_summary),
    )
    assert alternate.summary.failure_adjusted_run_summary_hash == record.summary.failure_adjusted_run_summary_hash
    assert alternate.record_hash != record.record_hash
    with pytest.raises(ValueError, match="record_hash"):
        FailureAdjustedGroundServiceRunRecord(
            ground_failure_service_run_schema_version=(
                record.ground_failure_service_run_schema_version
            ),
            run_id=record.run_id,
            summary=record.summary,
            record_hash="0" * 64,
        )
