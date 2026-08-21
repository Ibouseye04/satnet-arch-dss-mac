from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

from satnet.ground.catalog import GroundStation, GroundStationCatalog, GroundStationClass
from satnet.ground.failure_policy import GroundFailurePolicy
from satnet.ground.failure_realization import (
    create_ground_failure_realization_record,
    ground_station_fails,
)
from satnet.ground.failure_service_aggregation import create_failure_adjusted_run_record
from satnet.ground.failure_service_metrics import create_failure_adjusted_step_record
from satnet.ground.integrated_graph import (
    CanonicalIntegratedEdge,
    CanonicalIntegratedNode,
    IntegratedEdgeKind,
    IntegratedNodeRef,
    create_integrated_graph_snapshot,
)
from satnet.ground.persistence import make_enabled_ground_design_record
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from satnet.ground.service_aggregation import make_ground_service_step_record, summarize_ground_service_run
from satnet.ground.service_metrics import compute_ground_service_step
from satnet.ground.service_persistence import make_ground_service_run_record
from satnet.ground.service_policy import GroundServicePolicy

SATELLITE_HASH = "a" * 64
VISIBILITY_HASH = "b" * 64
VISIBILITY_SNAPSHOT_HASH = "c" * 64
START = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def _station(
    station_id: str, station_class: GroundStationClass, latitude: float
) -> GroundStation:
    return GroundStation(
        station_id=station_id,
        name=station_id,
        station_class=station_class,
        latitude_deg=latitude,
        longitude_deg=latitude,
        altitude_m=0.0,
        region="region_validation",
        country_code="ZZ",
    )


def _context():
    catalog = GroundStationCatalog(
        (
            _station("CIV_G5_DIAG_001", GroundStationClass.CIVILIAN, 1.0),
            _station("CIV_G5_DIAG_002", GroundStationClass.CIVILIAN, 2.0),
            _station("GOV_G5_DIAG_001", GroundStationClass.GOVERNMENT, 3.0),
            _station("MIL_G5_DIAG_001", GroundStationClass.MILITARY, 4.0),
        )
    )
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(2, 1, 1, station_selection_seed=7),
    )
    design = make_enabled_ground_design_record(
        run_id=17,
        satellite_config_hash=SATELLITE_HASH,
        selection=selection,
    )
    service_policy = GroundServicePolicy(0.5, 0.5)
    attachments = (
        tuple((0, station_id) for station_id in selection.selected_station_ids),
        ((0, selection.selected_station_ids[0]), (0, selection.selected_station_ids[2])),
        (),
    )
    snapshots = []
    g4_steps = []
    for timestep, step_attachments in enumerate(attachments):
        nodes = (
            CanonicalIntegratedNode(IntegratedNodeRef.for_satellite(0), ()),
            CanonicalIntegratedNode(IntegratedNodeRef.for_satellite(1), ()),
            *(
                CanonicalIntegratedNode(
                    IntegratedNodeRef.for_ground_station(station_id), ()
                )
                for station_id in selection.selected_station_ids
            ),
        )
        edges = (
            CanonicalIntegratedEdge(
                IntegratedEdgeKind.INTER_SATELLITE,
                IntegratedNodeRef.for_satellite(0),
                IntegratedNodeRef.for_satellite(1),
                (),
            ),
            *(
                CanonicalIntegratedEdge(
                    IntegratedEdgeKind.SATELLITE_GROUND,
                    IntegratedNodeRef.for_satellite(satellite_id),
                    IntegratedNodeRef.for_ground_station(station_id),
                    (),
                )
                for satellite_id, station_id in step_attachments
            ),
        )
        snapshot = create_integrated_graph_snapshot(
            timestep_index=timestep,
            timestamp_utc=START + timedelta(minutes=timestep),
            satellite_config_hash=SATELLITE_HASH,
            ground_design_hash=design.ground_design_hash,
            visibility_policy_hash=VISIBILITY_HASH,
            visibility_snapshot_hash=VISIBILITY_SNAPSHOT_HASH,
            graph_attributes=(),
            nodes=nodes,
            edges=edges,
        )
        metrics = compute_ground_service_step(
            integrated_snapshot=snapshot,
            ground_design=design,
            catalog=catalog,
            configured_satellite_count=2,
            policy=service_policy,
        )
        g4_steps.append(
            make_ground_service_step_record(
                run_id=design.run_id,
                metrics=metrics,
                integrated_snapshot=snapshot,
                ground_design=design,
                catalog=catalog,
                configured_satellite_count=2,
                policy=service_policy,
            )
        )
        snapshots.append(snapshot)
    g4_steps_tuple = tuple(g4_steps)
    g4_summary = summarize_ground_service_run(step_records=g4_steps_tuple)
    g4_run = make_ground_service_run_record(
        summary=g4_summary,
        step_records=g4_steps_tuple,
    )
    return catalog, design, selection, service_policy, g4_steps_tuple, g4_run


def _scenario(probability: float, seed: int) -> dict[str, object]:
    catalog, design, selection, service_policy, g4_steps, g4_run = _context()
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
            verified_g4_step_record=record,
            ground_design=design,
            catalog=catalog,
            policy=failure_policy,
            realization_record=realization,
            ground_service_policy=service_policy,
        )
        for record in g4_steps
    )
    g5_run = create_failure_adjusted_run_record(
        verified_g4_step_records=g4_steps,
        verified_g4_run_record=g4_run,
        g5_step_records=g5_steps,
        realization_record=realization,
    )
    first = g5_steps[0].metrics
    summary = g5_run.summary
    return {
        "adjusted_ground_fraction": first.failure_adjusted_ground_service_fraction,
        "adjusted_ground_threshold_met": first.ground_threshold_met,
        "adjusted_overall_fraction": first.failure_adjusted_overall_service_fraction,
        "adjusted_overall_threshold_met": first.overall_threshold_met,
        "adjusted_serviced_station_ids": first.failure_adjusted_serviced_ground_station_ids,
        "baseline_ground_fraction": first.baseline_ground_service_fraction,
        "baseline_overall_fraction": first.baseline_overall_service_fraction,
        "baseline_serviced_station_ids": g4_steps[0].metrics.serviced_ground_station_ids,
        "class_failure_counts": {
            "civilian": first.failed_civilian_count,
            "government": first.failed_government_count,
            "military": first.failed_military_count,
        },
        "failed_station_ids": realization.realization.failed_station_ids,
        "failure_probability": probability,
        "failure_seed": seed,
        "ground_service_loss": first.ground_service_loss_due_to_failures,
        "operational_station_ids": realization.realization.operational_station_ids,
        "overall_service_loss": first.overall_service_loss_due_to_ground_failures,
        "realization_hash": realization.realization.ground_failure_realization_hash,
        "run_breach_counts": {
            "space": summary.space_threshold_breach_timestep_count,
            "ground": summary.ground_threshold_breach_timestep_count,
            "overall": summary.overall_threshold_breach_timestep_count,
        },
        "run_first_breaches": {
            "space": summary.first_space_threshold_breach_timestep,
            "ground": summary.first_ground_threshold_breach_timestep,
            "overall": summary.first_overall_threshold_breach_timestep,
        },
        "run_ground_loss_max": summary.ground_service_loss_due_to_failures_max,
        "run_ground_loss_mean": summary.ground_service_loss_due_to_failures_mean,
        "run_ground_min": summary.failure_adjusted_ground_service_fraction_min,
        "run_ground_mean": summary.failure_adjusted_ground_service_fraction_mean,
        "run_summary_hash": summary.failure_adjusted_run_summary_hash,
        "selected_station_ids": tuple(sorted(selection.selected_station_ids)),
        "space_threshold_met": first.space_threshold_met,
        "step_sequence_hash": summary.step_sequence_hash,
    }


def build_diagnostics() -> dict[str, object]:
    partial_seed = next(
        seed
        for seed in range(10_000)
        if 0
        < sum(
            ground_station_fails(
                station_id=station_id,
                ground_failure_seed=seed,
                policy=GroundFailurePolicy(0.5),
            )
            for station_id in (
                "CIV_G5_DIAG_001",
                "CIV_G5_DIAG_002",
                "GOV_G5_DIAG_001",
                "MIL_G5_DIAG_001",
            )
        )
        < 4
    )
    zero = _scenario(0.0, 11)
    full = _scenario(1.0, 11)
    partial = _scenario(0.5, partial_seed)
    assert not zero["failed_station_ids"]
    assert len(full["failed_station_ids"]) == len(full["selected_station_ids"])
    assert partial["failed_station_ids"]
    assert partial["failed_station_ids"] == _scenario(0.5, partial_seed)["failed_station_ids"]
    paired = ground_station_fails(
        station_id="CIV_G5_DIAG_001",
        ground_failure_seed=partial_seed,
        policy=GroundFailurePolicy(0.5),
    )
    assert paired == ground_station_fails(
        station_id="CIV_G5_DIAG_001",
        ground_failure_seed=partial_seed,
        policy=GroundFailurePolicy(0.5),
    )
    return {
        "full": full,
        "partial": partial,
        "paired_station_outcome": paired,
        "status": "passed",
        "zero": zero,
    }


def main() -> None:
    print(json.dumps(build_diagnostics(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
