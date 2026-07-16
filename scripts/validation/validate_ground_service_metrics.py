from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

from satnet.ground.catalog import GroundStation, GroundStationCatalog, GroundStationClass
from satnet.ground.integrated_graph import (
    CanonicalIntegratedEdge,
    CanonicalIntegratedNode,
    IntegratedEdgeKind,
    IntegratedNodeRef,
    create_integrated_graph_snapshot,
)
from satnet.ground.persistence import make_enabled_ground_design_record
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from satnet.ground.service_aggregation import (
    make_ground_service_step_record,
    summarize_ground_service_run,
)
from satnet.ground.service_metrics import compute_ground_service_step
from satnet.ground.service_policy import GroundServicePolicy

TIMESTAMP = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
SATELLITE_HASH = "4" * 64
VISIBILITY_POLICY_HASH = "5" * 64
VISIBILITY_SNAPSHOT_HASH = "6" * 64


def _station(station_id: str, station_class: GroundStationClass, offset: int) -> GroundStation:
    return GroundStation(
        station_id=station_id,
        name=f"G4 Diagnostic {station_id}",
        station_class=station_class,
        latitude_deg=float(offset),
        longitude_deg=float(offset),
        altitude_m=0.0,
        region="diagnostic_region",
        country_code="ZZ",
    )


def _catalog() -> GroundStationCatalog:
    return GroundStationCatalog(
        (
            _station("CIV_G4_DIAG_001", GroundStationClass.CIVILIAN, 1),
            _station("CIV_G4_DIAG_002", GroundStationClass.CIVILIAN, 2),
            _station("GOV_G4_DIAG_001", GroundStationClass.GOVERNMENT, 3),
            _station("MIL_G4_DIAG_001", GroundStationClass.MILITARY, 4),
        )
    )


def _case(
    name: str,
    *,
    class_counts: tuple[int, int, int] = (2, 1, 1),
    satellite_ids: tuple[int, ...],
    isl_edges: tuple[tuple[int, int], ...],
    attachment_specs: tuple[tuple[int, str], ...],
    configured_satellite_count: int,
    policy: GroundServicePolicy,
    timestep_index: int = 0,
):
    catalog = _catalog()
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(*class_counts, station_selection_seed=19),
    )
    design = make_enabled_ground_design_record(
        run_id=0,
        satellite_config_hash=SATELLITE_HASH,
        selection=selection,
    )
    selected_by_label = {
        "civilian_0": selection.civilian_station_ids[0]
        if selection.civilian_station_ids
        else None,
        "civilian_1": selection.civilian_station_ids[1]
        if len(selection.civilian_station_ids) > 1
        else None,
        "government_0": selection.government_station_ids[0]
        if selection.government_station_ids
        else None,
        "military_0": selection.military_station_ids[0]
        if selection.military_station_ids
        else None,
    }
    attachments = tuple(
        (satellite_id, selected_by_label[label])
        for satellite_id, label in attachment_specs
        if selected_by_label[label] is not None
    )
    nodes = tuple(
        CanonicalIntegratedNode(IntegratedNodeRef.for_satellite(satellite_id), ())
        for satellite_id in satellite_ids
    ) + tuple(
        CanonicalIntegratedNode(IntegratedNodeRef.for_ground_station(station_id), ())
        for station_id in selection.selected_station_ids
    )
    edges = tuple(
        CanonicalIntegratedEdge(
            IntegratedEdgeKind.INTER_SATELLITE,
            IntegratedNodeRef.for_satellite(left),
            IntegratedNodeRef.for_satellite(right),
            (),
        )
        for left, right in isl_edges
    ) + tuple(
        CanonicalIntegratedEdge(
            IntegratedEdgeKind.SATELLITE_GROUND,
            IntegratedNodeRef.for_satellite(satellite_id),
            IntegratedNodeRef.for_ground_station(station_id),
            (),
        )
        for satellite_id, station_id in attachments
    )
    snapshot = create_integrated_graph_snapshot(
        timestep_index=timestep_index,
        timestamp_utc=TIMESTAMP + timedelta(minutes=timestep_index),
        satellite_config_hash=SATELLITE_HASH,
        ground_design_hash=design.ground_design_hash,
        visibility_policy_hash=VISIBILITY_POLICY_HASH,
        visibility_snapshot_hash=VISIBILITY_SNAPSHOT_HASH,
        graph_attributes=(),
        nodes=nodes,
        edges=edges,
    )
    metrics = compute_ground_service_step(
        integrated_snapshot=snapshot,
        ground_design=design,
        catalog=catalog,
        configured_satellite_count=configured_satellite_count,
        policy=policy,
    )
    return name, catalog, design, snapshot, policy, metrics


def build_diagnostic_cases():
    standard = GroundServicePolicy(0.5, 0.5)
    all_labels = (
        (0, "civilian_0"),
        (0, "civilian_1"),
        (1, "government_0"),
        (1, "military_0"),
    )
    return (
        _case(
            "fully_serviced",
            satellite_ids=(0, 1),
            isl_edges=((0, 1),),
            attachment_specs=all_labels,
            configured_satellite_count=2,
            policy=standard,
        ),
        _case(
            "station_with_no_access",
            satellite_ids=(0, 1),
            isl_edges=((0, 1),),
            attachment_specs=all_labels[:-1],
            configured_satellite_count=2,
            policy=standard,
        ),
        _case(
            "station_connected_outside_gcc",
            satellite_ids=(0, 1, 2),
            isl_edges=((0, 1),),
            attachment_specs=((2, "civilian_0"),),
            configured_satellite_count=3,
            policy=standard,
        ),
        _case(
            "split_satellite_network",
            satellite_ids=(0, 1, 2, 3),
            isl_edges=((0, 1), (2, 3)),
            attachment_specs=((0, "civilian_0"),),
            configured_satellite_count=4,
            policy=standard,
        ),
        _case(
            "equal_size_gcc_tie_different_attachments",
            satellite_ids=(0, 1, 2, 3),
            isl_edges=((0, 1), (2, 3)),
            attachment_specs=((0, "civilian_0"), (2, "government_0")),
            configured_satellite_count=4,
            policy=standard,
        ),
        _case(
            "catastrophic_satellite_attrition",
            satellite_ids=(7,),
            isl_edges=(),
            attachment_specs=((7, "civilian_0"),),
            configured_satellite_count=100,
            policy=GroundServicePolicy(0.01, 0.25),
        ),
        _case(
            "mixed_class_service",
            satellite_ids=(0, 1),
            isl_edges=((0, 1),),
            attachment_specs=((0, "civilian_0"), (1, "government_0")),
            configured_satellite_count=2,
            policy=standard,
        ),
        _case(
            "zero_operational_satellites",
            satellite_ids=(),
            isl_edges=(),
            attachment_specs=(),
            configured_satellite_count=4,
            policy=standard,
        ),
        _case(
            "threshold_boundary",
            satellite_ids=(0, 1, 2, 3),
            isl_edges=((0, 1), (2, 3)),
            attachment_specs=((0, "civilian_0"), (1, "government_0")),
            configured_satellite_count=4,
            policy=standard,
        ),
        _case(
            "zero_threshold_semantics",
            satellite_ids=(),
            isl_edges=(),
            attachment_specs=(),
            configured_satellite_count=4,
            policy=GroundServicePolicy(0.0, 0.0),
        ),
    )


def _metric_result(case) -> dict[str, object]:
    name, _, _, _, _, metrics = case
    return {
        "case": name,
        "configured_satellite_count": metrics.configured_satellite_count,
        "operational_satellite_count": metrics.operational_satellite_count,
        "satellite_component_count": metrics.satellite_component_count,
        "satellite_gcc_ids": list(metrics.satellite_gcc_ids),
        "space_gcc_fraction_original": metrics.space_gcc_fraction_original,
        "space_gcc_fraction_surviving": metrics.space_gcc_fraction_surviving,
        "serviced_ground_station_ids": list(metrics.serviced_ground_station_ids),
        "unserviced_ground_station_ids": list(metrics.unserviced_ground_station_ids),
        "civilian_service_fraction": metrics.civilian_service_fraction,
        "government_service_fraction": metrics.government_service_fraction,
        "military_service_fraction": metrics.military_service_fraction,
        "ground_service_fraction": metrics.ground_service_fraction,
        "overall_service_fraction": metrics.overall_service_fraction,
        "space_threshold_met": metrics.space_threshold_met,
        "ground_threshold_met": metrics.ground_threshold_met,
        "overall_threshold_met": metrics.overall_threshold_met,
        "step_metrics_hash": metrics.step_metrics_hash,
    }


def _temporal_diagnostic() -> dict[str, object]:
    policy = GroundServicePolicy(0.5, 0.5)
    cases = (
        _case(
            "temporal_connected",
            satellite_ids=(0, 1, 2, 3),
            isl_edges=((0, 1), (1, 2), (2, 3)),
            attachment_specs=((0, "civilian_0"), (0, "civilian_1"), (1, "government_0"), (1, "military_0")),
            configured_satellite_count=4,
            policy=policy,
            timestep_index=0,
        ),
        _case(
            "temporal_disconnect",
            satellite_ids=(0, 1, 2, 3),
            isl_edges=((0, 1), (2, 3)),
            attachment_specs=((2, "civilian_0"),),
            configured_satellite_count=4,
            policy=policy,
            timestep_index=1,
        ),
        _case(
            "temporal_recovery",
            satellite_ids=(0, 1, 2, 3),
            isl_edges=((0, 1), (1, 2), (2, 3)),
            attachment_specs=((0, "civilian_0"), (0, "civilian_1"), (1, "government_0"), (1, "military_0")),
            configured_satellite_count=4,
            policy=policy,
            timestep_index=2,
        ),
    )
    records = tuple(
        make_ground_service_step_record(
            run_id=case[2].run_id,
            metrics=case[5],
            integrated_snapshot=case[3],
            ground_design=case[2],
            catalog=case[1],
            configured_satellite_count=4,
            policy=case[4],
        )
        for case in cases
    )
    summary = summarize_ground_service_run(step_records=records)
    return {
        "case": "multi_timestep_disconnect_and_recovery",
        "space_gcc_fraction_original_min": summary.space_gcc_fraction_original_min,
        "space_gcc_fraction_original_mean": summary.space_gcc_fraction_original_mean,
        "space_gcc_fraction_surviving_min": summary.space_gcc_fraction_surviving_min,
        "space_gcc_fraction_surviving_mean": summary.space_gcc_fraction_surviving_mean,
        "ground_service_fraction_min": summary.ground_service_fraction_min,
        "ground_service_fraction_mean": summary.ground_service_fraction_mean,
        "overall_service_fraction_min": summary.overall_service_fraction_min,
        "overall_service_fraction_mean": summary.overall_service_fraction_mean,
        "space_threshold_breach_any": summary.space_threshold_breach_any,
        "ground_threshold_breach_any": summary.ground_threshold_breach_any,
        "overall_threshold_breach_any": summary.overall_threshold_breach_any,
        "space_threshold_breach_timestep_count": summary.space_threshold_breach_timestep_count,
        "ground_threshold_breach_timestep_count": summary.ground_threshold_breach_timestep_count,
        "overall_threshold_breach_timestep_count": summary.overall_threshold_breach_timestep_count,
        "first_space_threshold_breach_timestep": summary.first_space_threshold_breach_timestep,
        "first_ground_threshold_breach_timestep": summary.first_ground_threshold_breach_timestep,
        "first_overall_threshold_breach_timestep": summary.first_overall_threshold_breach_timestep,
        "step_sequence_hash": summary.step_sequence_hash,
        "run_summary_hash": summary.run_summary_hash,
    }


def build_diagnostics() -> dict[str, object]:
    return {
        "steps": [_metric_result(case) for case in build_diagnostic_cases()],
        "run": _temporal_diagnostic(),
    }


def main() -> None:
    print(json.dumps(build_diagnostics(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
