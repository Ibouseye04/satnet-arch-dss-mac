from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone

import networkx as nx
import pytest

from satnet.ground.catalog import GroundStation, GroundStationCatalog, GroundStationClass
from satnet.ground.coordinates import (
    CoordinateFrame,
    FramedSatellitePosition,
    OperationalSatellitePositionSnapshot,
    WGS84_SEMI_MAJOR_AXIS_KM,
)
from satnet.ground.integrated_builder import build_integrated_ground_graph
from satnet.ground.integrated_graph import operational_snapshot_from_networkx
from satnet.ground.integrated_persistence import make_integrated_graph_record
from satnet.ground.persistence import make_enabled_ground_design_record
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from satnet.ground.service_aggregation import (
    GroundServiceStepRecord,
    _step_record_hash,
)
from satnet.ground.service_metrics import (
    GroundServiceStepMetrics,
    _compute_step_metrics_hash,
    compute_ground_service_step,
    validate_ground_service_step_context,
)
from satnet.ground.service_persistence import (
    GroundServiceRunRecord,
    replay_ground_service_records,
)
from satnet.ground.service_policy import GroundServicePolicy
from satnet.ground.visibility import (
    GroundVisibilityPolicy,
    GroundVisibilitySnapshot,
    evaluate_ground_design_visibility,
)
from satnet.ground.visibility_persistence import make_ground_visibility_record
from tests.ground.test_g4_service_metrics import make_context
from tests.ground.test_g4_service_persistence import replay_kwargs, verified_context

TIMESTAMP = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
SATELLITE_HASH = "9" * 64


def valid_chain(*, station_class: GroundStationClass, altitude_m: float):
    prefix = {
        GroundStationClass.CIVILIAN: "CIV",
        GroundStationClass.GOVERNMENT: "GOV",
        GroundStationClass.MILITARY: "MIL",
    }[station_class]
    station = GroundStation(
        station_id=f"{prefix}_G4_ALT_001",
        name="G4 Alternate Evidence Station",
        station_class=station_class,
        latitude_deg=0.0,
        longitude_deg=0.0,
        altitude_m=altitude_m,
        region="region_test",
        country_code="ZZ",
    )
    catalog = GroundStationCatalog((station,))
    counts = {
        GroundStationClass.CIVILIAN: (1, 0, 0),
        GroundStationClass.GOVERNMENT: (0, 1, 0),
        GroundStationClass.MILITARY: (0, 0, 1),
    }[station_class]
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(*counts, station_selection_seed=3),
    )
    design = make_enabled_ground_design_record(
        run_id=14,
        satellite_config_hash=SATELLITE_HASH,
        selection=selection,
    )
    graph = nx.Graph()
    graph.add_node(0)
    graph_source = operational_snapshot_from_networkx(
        timestep_index=0,
        timestamp_utc=TIMESTAMP,
        satellite_config_hash=SATELLITE_HASH,
        graph=graph,
    )
    position_source = OperationalSatellitePositionSnapshot(
        timestep_index=0,
        timestamp_utc=TIMESTAMP,
        satellite_config_hash=SATELLITE_HASH,
        positions=(
            FramedSatellitePosition(
                satellite_id=0,
                timestamp_utc=TIMESTAMP,
                x_km=WGS84_SEMI_MAJOR_AXIS_KM + 500.0,
                y_km=0.0,
                z_km=0.0,
                frame=CoordinateFrame.ECEF,
            ),
        ),
    )
    visibility_policy = GroundVisibilityPolicy(0.0)
    visibility = evaluate_ground_design_visibility(
        ground_design=design,
        catalog=catalog,
        satellite_snapshot=position_source,
        policy=visibility_policy,
    )
    integrated = build_integrated_ground_graph(
        ground_design=design,
        catalog=catalog,
        satellite_graph_snapshot=graph_source,
        verified_visibility_snapshot=visibility,
    )
    metrics = compute_ground_service_step(
        integrated_snapshot=integrated,
        ground_design=design,
        catalog=catalog,
        configured_satellite_count=1,
        policy=GroundServicePolicy(1.0, 1.0),
    )
    return catalog, design, visibility, integrated, metrics


def test_valid_alternate_g1_g2_g3_class_evidence_changes_class_reporting() -> None:
    civilian = valid_chain(station_class=GroundStationClass.CIVILIAN, altitude_m=0.0)
    government = valid_chain(station_class=GroundStationClass.GOVERNMENT, altitude_m=0.0)
    assert civilian[-1].civilian_service_fraction == 1.0
    assert civilian[-1].government_service_fraction is None
    assert government[-1].civilian_service_fraction is None
    assert government[-1].government_service_fraction == 1.0
    assert civilian[1].ground_design_hash != government[1].ground_design_hash
    assert civilian[3].graph_hash != government[3].graph_hash
    assert civilian[-1].step_metrics_hash != government[-1].step_metrics_hash


def test_valid_alternate_g1_g2_g3_elevation_evidence_changes_identity_not_service() -> None:
    baseline = valid_chain(station_class=GroundStationClass.CIVILIAN, altitude_m=0.0)
    elevated = valid_chain(station_class=GroundStationClass.CIVILIAN, altitude_m=100.0)
    assert baseline[2].visible_links[0].elevation_deg == 90.0
    assert elevated[2].visible_links[0].elevation_deg == 90.0
    assert baseline[2].visible_links[0].slant_range_km != elevated[2].visible_links[0].slant_range_km
    assert baseline[-1].ground_service_fraction == elevated[-1].ground_service_fraction == 1.0
    assert baseline[2].snapshot_hash != elevated[2].snapshot_hash
    assert baseline[3].graph_hash != elevated[3].graph_hash
    assert baseline[-1].step_metrics_hash != elevated[-1].step_metrics_hash


@pytest.mark.parametrize(
    ("changes", "contextual"),
    [
        ({"satellite_component_count": 3}, True),
        ({"serviced_ground_station_ids": ("CIV_G4_FAKE",)}, True),
        ({"unserviced_ground_station_ids": ("CIV_G4_FAKE",)}, True),
        ({"space_threshold_met": False, "overall_threshold_met": False}, True),
    ],
)
def test_additional_step_corruption_fails_with_replacement_hash(
    changes: dict[str, object], contextual: bool
) -> None:
    catalog, design, selection, snapshot, policy, _ = make_context()
    catalog, design, selection, snapshot, policy, metrics = make_context(
        attachments=((0, selection.selected_station_ids[0]),)
    )
    values = asdict(metrics)
    values.update(changes)
    values["step_metrics_hash"] = _compute_step_metrics_hash(values)
    if not contextual:
        with pytest.raises(ValueError):
            GroundServiceStepMetrics(**values)
        return
    try:
        corrupted = GroundServiceStepMetrics(**values)
    except ValueError:
        return
    with pytest.raises(ValueError, match="contextual mismatch"):
        validate_ground_service_step_context(
            metrics=corrupted,
            integrated_snapshot=snapshot,
            ground_design=design,
            catalog=catalog,
            configured_satellite_count=4,
            policy=policy,
        )


def test_replay_rejects_reordered_g4_steps() -> None:
    values = verified_context()
    kwargs = replay_kwargs(values)
    kwargs["step_records"] = tuple(reversed(values[-2]))
    with pytest.raises(ValueError, match="canonical.*order"):
        replay_ground_service_records(**kwargs)


def test_cross_stage_g2_and_g3_run_mismatches_fail() -> None:
    values = verified_context()
    kwargs = replay_kwargs(values)
    kwargs["visibility_records"] = tuple(
        make_ground_visibility_record(
            run_id=99,
            snapshot=GroundVisibilitySnapshot(
                timestep_index=record.timestep_index,
                timestamp_utc=record.timestamp_utc,
                satellite_config_hash=record.satellite_config_hash,
                ground_design_hash=record.ground_design_hash,
                visibility_policy_hash=record.visibility_policy_hash,
                visibility_model_version=record.visibility_model_version,
                frame_contract_version=record.frame_contract_version,
                wgs84_model_version=record.wgs84_model_version,
                link_observations=record.link_observations,
                visible_links=tuple(
                    observation
                    for observation in record.link_observations
                    if observation.is_visible
                ),
                visible_satellite_ids_by_station=record.visible_satellite_ids_by_station,
                snapshot_hash=record.snapshot_hash,
            ),
        )
        for record in values[5]
    )
    with pytest.raises(ValueError, match="G2 record run ID"):
        replay_ground_service_records(**kwargs)
    kwargs = replay_kwargs(values)
    wrong_design = replace(values[3], run_id=99)
    kwargs["integrated_records"] = tuple(
        make_integrated_graph_record(
            ground_design=wrong_design,
            snapshot=record.to_snapshot(),
        )
        for record in values[7]
    )
    with pytest.raises(ValueError, match="G3 record run ID"):
        replay_ground_service_records(**kwargs)


def test_step_and_run_record_hash_corruption_fail() -> None:
    values = verified_context()
    step = values[-2][0]
    with pytest.raises(ValueError, match="record_hash"):
        GroundServiceStepRecord(
            ground_service_step_schema_version=step.ground_service_step_schema_version,
            run_id=step.run_id,
            metrics=step.metrics,
            record_hash="0" * 64,
        )
    run = values[-1]
    with pytest.raises(ValueError, match="record_hash"):
        GroundServiceRunRecord(
            ground_service_run_schema_version=run.ground_service_run_schema_version,
            run_id=run.run_id,
            summary=run.summary,
            record_hash="0" * 64,
        )


def test_step_scientific_identity_excludes_run_but_record_identity_binds_it() -> None:
    step = verified_context()[-2][0]
    alternate = GroundServiceStepRecord(
        ground_service_step_schema_version=step.ground_service_step_schema_version,
        run_id=99,
        metrics=step.metrics,
        record_hash=_step_record_hash(run_id=99, metrics=step.metrics),
    )
    assert alternate.metrics.step_metrics_hash == step.metrics.step_metrics_hash
    assert alternate.record_hash != step.record_hash
