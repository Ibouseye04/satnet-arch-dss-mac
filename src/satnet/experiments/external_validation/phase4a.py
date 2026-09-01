"""SATNET Phase 4A: frozen real-data external validation construction.

This module intentionally contains no model imports, checkpoint loading, training,
or inference.  It freezes public inputs, adapts real orbital/status observations
through the existing SATNET physics functions, and writes an auditable
pre-inference package.

The external experiment is not a claim about proprietary Starlink routing.  Its
outputs are real orbital observations transformed through the frozen SATNET
feature and network-construction methodology.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import io
import json
import math
import os
import struct
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable, Iterator, Mapping, Sequence

import networkx as nx
import numpy as np

try:
    import pandas as pd
    import pyarrow.dataset as ds
except ImportError as exc:
    raise ImportError(
        "Phase 4A requires pandas and pyarrow to read pinned Parquet sources"
    ) from exc

from sgp4.api import Satrec, WGS72

from satnet.metrics.labels import compute_gcc_size, compute_num_components
from satnet.network.hypatia_adapter import (
    ATMOSPHERE_BUFFER_KM,
    EARTH_RADIUS_KM,
    LinkBudgetEngine,
    SatellitePosition,
    WalkerDeltaConfig,
    _check_line_of_sight,
    _compute_distance_km,
    _compute_gmst,
    _compute_grid_plus_isls,
    _datetime_to_jd,
    _teme_to_ecef,
)


SOURCE_REVISIONS = {
    "juliensimon/starlink-fleet-data": "b9068f0abc217b031db784e052b4aec26e60c8aa",
    "juliensimon/space-track-tle-history": "717fd92350652061dddc396f82470eeb7e2a434b",
}
SOURCE_URLS = {
    name: f"https://huggingface.co/datasets/{name}/tree/{revision}"
    for name, revision in SOURCE_REVISIONS.items()
}

EXTERNAL_RF_FEATURE_ORDER = (
    "num_planes",
    "sats_per_plane",
    "altitude_km",
    "inclination_deg",
    "satellite_node_failure_probability",
    "satellite_edge_failure_probability",
)
EXTERNAL_TGNN_NODE_FEATURE_ORDER = (
    "plane_idx_normalized",
    "sat_in_plane_normalized",
    "node_exists_constant",
)
EXTERNAL_TGNN_EDGE_FEATURE_ORDER = (
    "distance_km_scaled_10000",
    "margin_db_scaled_100",
    "link_type_code_scaled_2",
    "link_mode_binary",
)
STATUS_MAPPING = {
    "operational": True,
    "raising": False,
    "deorbiting": False,
    "decayed": False,
    "anomalous": False,
    "unknown": False,
}
PLANE_COUNTS = (4, 5, 6)
SATS_PER_PLANE_COUNTS = (5, 6, 7, 8)
RAAN_CLUSTER_MAX_GAP_DEG = 1.0
MAX_TLE_AGE_HOURS = 24.0
LOOKBACK_DAYS = 30
MAX_ISL_DISTANCE_KM = 10_000.0
GCC_THRESHOLD = 0.8
DURATION_MINUTES = 10
STEP_SECONDS = 60
NUM_TIMESTEPS = 11
ISL_POLICY = "grid_adaptive"
ADJACENT_SEARCH_K = 1
MAX_INTER_PLANE_LINKS_PER_SAT = 1
FAILURE_MODEL = "persistent_temporal_union_edges_v1"
EXTERNAL_TASKS = (
    "rf_space_classification",
    "tgnn_space_classification",
    "rf_space_regression",
    "tgnn_space_regression",
)
EXTERNAL_PRIMARY_COMPARISON = ("rf_space_regression", "tgnn_space_regression")
EXTERNAL_PRIMARY_SEED = 42
EXTERNAL_BOOTSTRAP_REPLICATES = 2000
EXTERNAL_BOOTSTRAP_SEED = 20260820
EXTERNAL_BOOTSTRAP_INTERVAL = "95% percentile"
EXTERNAL_STATISTICAL_CONTRACT = {
    "primary_comparison": {
        "left_task": EXTERNAL_PRIMARY_COMPARISON[0],
        "right_task": EXTERNAL_PRIMARY_COMPARISON[1],
        "metric": "MAE",
        "difference": "RF MAE minus TGNN MAE",
        "positive_interpretation": "TGNN lower error",
        "seed": EXTERNAL_PRIMARY_SEED,
    },
    "paired_unit": "episode_id",
    "replicates": EXTERNAL_BOOTSTRAP_REPLICATES,
    "bootstrap_seed": EXTERNAL_BOOTSTRAP_SEED,
    "interval": EXTERNAL_BOOTSTRAP_INTERVAL,
    "primary_reporting_seed": EXTERNAL_PRIMARY_SEED,
    "ensemble": False,
    "classification": "descriptive_only",
    "classification_inferential_procedure": None,
    "authorized_descriptive_seeds": [42, 123, 456, 789, 2026],
}
TARGET_START = datetime(2024, 2, 1, 12, tzinfo=timezone.utc)
TARGET_END = datetime(2025, 12, 31, 12, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Phase4AConfig:
    """Immutable construction contract for the Adaptive-v2 external package."""

    output_root: Path
    episode_count: int = 300
    start: datetime = TARGET_START
    end: datetime = TARGET_END
    lookback_days: int = LOOKBACK_DAYS
    max_tle_age_hours: float = MAX_TLE_AGE_HOURS
    raan_cluster_max_gap_deg: float = RAAN_CLUSTER_MAX_GAP_DEG
    gcc_threshold: float = GCC_THRESHOLD
    max_isl_distance_km: float = MAX_ISL_DISTANCE_KM
    duration_minutes: int = DURATION_MINUTES
    step_seconds: int = STEP_SECONDS
    source_root: Path = Path(r"C:\Users\johns\external\satnet-real-external-validation-v1")
    synthetic_dataset_root: Path = Path(r"C:\Users\johns\external\satnet-10k-final-ml-datasets-v2-adaptive")

    def __post_init__(self) -> None:
        if self.episode_count != 300:
            raise ValueError("Phase 4A requires exactly 300 episodes")
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("Phase 4A boundaries must be timezone-aware")
        if self.duration_minutes * 60 // self.step_seconds + 1 != NUM_TIMESTEPS:
            raise ValueError("Phase 4A requires 11 inclusive timesteps")
        if self.lookback_days <= 0 or self.max_tle_age_hours <= 0:
            raise ValueError("lookback and TLE-age limits must be positive")


@dataclass(frozen=True)
class OrbitalRecord:
    norad_id: int
    epoch: datetime
    inclination_deg: float
    raan_deg: float
    eccentricity: float
    arg_perigee_deg: float
    mean_anomaly_deg: float
    mean_motion_rev_day: float
    mean_motion_dot_rev_day2: float
    bstar: float
    altitude_km: float
    satrec: Satrec


@dataclass(frozen=True)
class StatusRecord:
    norad_id: int
    epoch: datetime
    status: str
    is_isl_capable: bool
    shell_id: int
    shell_name: str


@dataclass(frozen=True)
class SelectedSatellite:
    norad_id: int
    plane_idx: int
    sat_in_plane: int
    orbit: OrbitalRecord | None
    status: str
    status_available: bool
    is_isl_capable: bool
    tle_age_seconds: float | None


def _utc(value: Any) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _adapter_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).parents[2] / "network" / "hypatia_adapter.py"):
        digest.update(path.as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _finite(value: float, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field}")
    return result


def choose_episode_timestamps(
    start: datetime = TARGET_START,
    end: datetime = TARGET_END,
    count: int = 300,
) -> tuple[datetime, ...]:
    """Choose evenly spaced timestamps without consulting any outcome."""
    if count != 300 or start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("Phase 4A timestamps require 300 ordered aware boundaries")
    span = (end - start).total_seconds()
    return tuple(
        start + timedelta(seconds=round(index * span / (count - 1)))
        for index in range(count)
    )


def _epoch_days_since_1949(epoch: datetime) -> float:
    return (epoch - datetime(1949, 12, 31, tzinfo=timezone.utc)).total_seconds() / 86_400.0


def _make_satrec(row: Mapping[str, Any]) -> Satrec:
    """Construct the WGS72 SGP4 record from the public orbital elements."""
    satellite = Satrec()
    satellite.sgp4init(
        WGS72,
        "i",
        int(row["norad_id"]),
        _epoch_days_since_1949(_utc(row["epoch"])),
        float(row["bstar"]),
        float(row["mean_motion_dot"]) * 2.0 * math.pi / (1440.0**2),
        0.0,
        float(row["eccentricity"]),
        math.radians(float(row["arg_perigee"])),
        math.radians(float(row["inclination"])),
        math.radians(float(row["mean_anomaly"])),
        float(row["mean_motion"]) * 2.0 * math.pi / 1440.0,
        math.radians(float(row["raan"])),
    )
    return satellite


def _make_orbital_record(row: Mapping[str, Any]) -> OrbitalRecord:
    values = {key: row[key] for key in row}
    return OrbitalRecord(
        norad_id=int(values["norad_id"]),
        epoch=_utc(values["epoch"]),
        inclination_deg=_finite(values["inclination"], "inclination"),
        raan_deg=_finite(values["raan"], "raan") % 360.0,
        eccentricity=_finite(values["eccentricity"], "eccentricity"),
        arg_perigee_deg=_finite(values["arg_perigee"], "arg_perigee") % 360.0,
        mean_anomaly_deg=_finite(values["mean_anomaly"], "mean_anomaly") % 360.0,
        mean_motion_rev_day=_finite(values["mean_motion"], "mean_motion"),
        mean_motion_dot_rev_day2=_finite(values["mean_motion_dot"], "mean_motion_dot"),
        bstar=_finite(values["bstar"], "bstar"),
        altitude_km=_finite(values["altitude_km"], "altitude_km"),
        satrec=_make_satrec(values),
    )


def _propagate(record: OrbitalRecord, target: datetime, sat_id: int) -> SatellitePosition | None:
    jd, fr = _datetime_to_jd(target)
    error, r_teme, _ = record.satrec.sgp4(jd, fr)
    if error != 0 or not all(math.isfinite(float(value)) for value in r_teme):
        return None
    gmst = _compute_gmst(target)
    x, y, z = _teme_to_ecef(float(r_teme[0]), float(r_teme[1]), float(r_teme[2]), gmst)
    radius = math.sqrt(x * x + y * y + z * z)
    if radius <= 0 or not math.isfinite(radius):
        return None
    return SatellitePosition(
        sat_id=sat_id,
        x_km=x,
        y_km=y,
        z_km=z,
        lat_deg=math.degrees(math.asin(z / radius)),
        lon_deg=math.degrees(math.atan2(y, x)),
        alt_km=radius - EARTH_RADIUS_KM,
    )


def infer_raan_planes(
    records: Sequence[OrbitalRecord],
    *,
    max_gap_deg: float = RAAN_CLUSTER_MAX_GAP_DEG,
) -> tuple[tuple[OrbitalRecord, ...], ...]:
    """Infer planes from circular RAAN structure, independent of outcomes.

    RAAN observations are sorted around the circle.  Consecutive observations
    separated by no more than ``max_gap_deg`` are in one structural cluster;
    the first and last cluster are merged when their circular gap also meets
    that criterion.  This is a deterministic input-space rule, not a target,
    prediction, or performance-based assignment.
    """
    if not records or max_gap_deg <= 0 or max_gap_deg >= 180:
        raise ValueError("RAAN clustering requires records and a valid angular gap")
    ordered = sorted(records, key=lambda item: (item.raan_deg, item.norad_id))
    gaps = [
        ordered[index + 1].raan_deg - ordered[index].raan_deg
        for index in range(len(ordered) - 1)
    ]
    gaps.append(ordered[0].raan_deg + 360.0 - ordered[-1].raan_deg)
    split_indices = [index for index, gap in enumerate(gaps) if gap > max_gap_deg]
    if not split_indices:
        return (tuple(ordered),)
    start = (split_indices[0] + 1) % len(ordered)
    rotated = ordered[start:] + ordered[:start]
    clusters: list[tuple[OrbitalRecord, ...]] = []
    current: list[OrbitalRecord] = [rotated[0]]
    for previous, item in zip(rotated, rotated[1:]):
        if (item.raan_deg - previous.raan_deg) % 360.0 <= max_gap_deg:
            current.append(item)
        else:
            clusters.append(tuple(current))
            current = [item]
    clusters.append(tuple(current))
    return tuple(sorted(clusters, key=lambda cluster: (min(item.raan_deg for item in cluster), len(cluster))))


def _phase_key(record: OrbitalRecord) -> tuple[float, int]:
    return ((record.mean_anomaly_deg + record.arg_perigee_deg) % 360.0, record.norad_id)


def _nearest_orbit(
    records: Sequence[OrbitalRecord],
    target: datetime,
    max_age_hours: float,
) -> OrbitalRecord | None:
    if not records:
        return None
    epochs = [item.epoch for item in records]
    index = bisect.bisect_right(epochs, target) - 1
    if index < 0:
        return None
    record = records[index]
    if (target - record.epoch).total_seconds() > max_age_hours * 3600.0:
        return None
    return record


def _latest_status(records: Sequence[StatusRecord], target: datetime) -> StatusRecord | None:
    if not records:
        return None
    epochs = [item.epoch for item in records]
    index = bisect.bisect_right(epochs, target) - 1
    return records[index] if index >= 0 else None


def scale_tgnn_edge_features(
    distance_km: float,
    margin_db: float,
    link_type: str,
    link_mode: str,
) -> tuple[float, float, float, float]:
    """Apply the frozen four-dimensional TGNN edge scaling exactly."""
    type_code = {"intra_plane": 0.0, "inter_plane": 1.0, "seam_link": 2.0}[link_type]
    return (
        _finite(distance_km, "distance_km") / 10_000.0,
        _finite(margin_db, "margin_db") / 100.0,
        type_code / 2.0,
        1.0 if link_mode == "optical" else 0.0,
    )


def _topology_identity(graph: nx.Graph) -> str:
    records = []
    for source, target, data in graph.edges(data=True):
        records.append({
            "source": int(min(source, target)),
            "target": int(max(source, target)),
            "distance_km": float(data["distance_km"]),
            "margin_db": float(data["margin_db"]),
            "link_type": str(data["link_type"]),
            "link_mode": str(data["link_mode"]),
        })
    records.sort(key=lambda item: (item["source"], item["target"], item["link_type"]))
    return hashlib.sha256(_json_bytes(records)).hexdigest()


def _node_features(plane: int, sat: int, planes: int, sats: int, exists: bool) -> list[float]:
    return [
        plane / max(planes - 1, 1),
        sat / max(sats - 1, 1),
        1.0 if exists else 0.0,
    ]


def _build_graph(
    selected: Sequence[SelectedSatellite],
    target: datetime,
    config: Phase4AConfig,
    *,
    positions_override: Sequence[SatellitePosition | None] | None = None,
) -> tuple[nx.Graph, list[dict[str, Any]], list[SatellitePosition | None]]:
    planes = max(item.plane_idx for item in selected) + 1
    sats = max(item.sat_in_plane for item in selected) + 1
    walker = WalkerDeltaConfig(
        num_planes=planes,
        sats_per_plane=sats,
        inclination_deg=53.0,
        altitude_km=550.0,
        phasing_factor=1,
        epoch=target,
    )
    positions: list[SatellitePosition | None] = []
    for index, item in enumerate(selected):
        positions.append(
            positions_override[index]
            if positions_override is not None
            else (_propagate(item.orbit, target, index) if item.orbit is not None else None)
        )
    graph = nx.Graph()
    for index, item in enumerate(selected):
        graph.add_node(
            index,
            plane=item.plane_idx,
            sat_in_plane=item.sat_in_plane,
            norad_id=item.norad_id,
            exists=bool(item.status_available and item.orbit is not None and positions[index] is not None),
        )
    links, topology_stats = _compute_grid_plus_isls(
        walker,
        positions,
        LinkBudgetEngine(),
        max_isl_distance_km=config.max_isl_distance_km,
        isl_policy=ISL_POLICY,
        adjacent_search_k=ADJACENT_SEARCH_K,
        max_inter_plane_links_per_sat=MAX_INTER_PLANE_LINKS_PER_SAT,
        collect_adaptive_examples=1,
    )
    if topology_stats.isl_policy != ISL_POLICY:
        raise RuntimeError(
            f"external topology runtime policy mismatch: {topology_stats.isl_policy!r}"
        )
    graph.graph["topology_stats"] = {
        "total_candidate_links": topology_stats.total_candidate_links,
        "links_rejected_distance": topology_stats.links_rejected_distance,
        "links_rejected_los": topology_stats.links_rejected_los,
        "links_rejected_budget": topology_stats.links_rejected_budget,
        "links_accepted": topology_stats.links_accepted,
        "isl_policy": topology_stats.isl_policy,
        "adaptive_selection_examples": topology_stats.adaptive_selection_examples,
    }
    for link in links:
        if graph.nodes[link.sat_id_1]["exists"] and graph.nodes[link.sat_id_2]["exists"]:
            graph.add_edge(
                link.sat_id_1,
                link.sat_id_2,
                distance_km=link.distance_km,
                margin_db=link.margin_db,
                link_type=link.link_type,
                link_mode=link.link_mode,
            )
    edge_rows = []
    for u, v, data in sorted(graph.edges(data=True)):
        edge_rows.append(
            {
                "source": u,
                "target": v,
                **{name: value for name, value in zip(EXTERNAL_TGNN_EDGE_FEATURE_ORDER, scale_tgnn_edge_features(data["distance_km"], data["margin_db"], data["link_type"], data["link_mode"]))},
                "distance_km": data["distance_km"],
                "margin_db": data["margin_db"],
                "link_type": data["link_type"],
                "link_mode": data["link_mode"],
            }
        )
    return graph, edge_rows, positions


def _load_status_records(root: Path, start: datetime, end: datetime) -> tuple[dict[int, list[StatusRecord]], dict[datetime.date, tuple[int, int]], set[int]]:
    path = root / "raw" / "starlink-fleet-data" / "data" / "tle_snapshots.parquet"
    columns = ["norad_id", "epoch_utc", "status", "is_isl_capable", "shell_id", "shell_name"]
    dataset = ds.dataset(path, format="parquet")
    lower = start - timedelta(days=LOOKBACK_DAYS + 1)
    expression = (
        (ds.field("shell_id") == 2)
        & (ds.field("epoch_utc") >= lower.replace(tzinfo=None))
        & (ds.field("epoch_utc") <= end.replace(tzinfo=None))
    )
    frame = dataset.to_table(columns=columns, filter=expression).to_pandas()
    frame["epoch_utc"] = pd.to_datetime(frame["epoch_utc"], utc=True)
    frame = frame.sort_values(["norad_id", "epoch_utc"], kind="mergesort")
    records: dict[int, list[StatusRecord]] = {}
    daily: dict[datetime.date, list[int]] = {}
    for row in frame.itertuples(index=False):
        epoch = _utc(row.epoch_utc)
        record = StatusRecord(int(row.norad_id), epoch, str(row.status).strip().lower(), bool(row.is_isl_capable), int(row.shell_id), str(row.shell_name))
        records.setdefault(record.norad_id, []).append(record)
        if record.is_isl_capable:
            bucket = daily.setdefault(epoch.date(), [0, 0])
            bucket[1] += 1
            bucket[0] += int(not STATUS_MAPPING.get(record.status, False))
    return records, {key: (value[0], value[1]) for key, value in daily.items()}, set(records)


def _load_orbital_records(root: Path, norad_ids: set[int], start: datetime, end: datetime) -> dict[int, list[OrbitalRecord]]:
    columns = ["norad_id", "epoch", "inclination", "raan", "eccentricity", "arg_perigee", "mean_anomaly", "mean_motion", "mean_motion_dot", "bstar", "altitude_km"]
    result: dict[int, list[OrbitalRecord]] = {}
    for year in (2024, 2025):
        path = root / "raw" / "space-track-tle-history" / "data" / f"tle_{year}.parquet"
        dataset = ds.dataset(path, format="parquet")
        expression = ds.field("norad_id").isin(sorted(norad_ids))
        frame = dataset.to_table(columns=columns, filter=expression).to_pandas()
        if frame.empty:
            continue
        frame["epoch"] = pd.to_datetime(frame["epoch"], utc=True)
        frame = frame[(frame["epoch"] >= start - timedelta(days=LOOKBACK_DAYS + 2)) & (frame["epoch"] <= end)]
        for row in frame.to_dict(orient="records"):
            try:
                record = _make_orbital_record(row)
            except (TypeError, ValueError, OverflowError):
                continue
            result.setdefault(record.norad_id, []).append(record)
    for records in result.values():
        records.sort(key=lambda item: item.epoch)
    return result


def _choose_subgraph(
    status_records: Mapping[int, Sequence[StatusRecord]],
    orbital_records: Mapping[int, Sequence[OrbitalRecord]],
    timestamp: datetime,
    num_planes: int,
    sats_per_plane: int,
    config: Phase4AConfig,
) -> tuple[tuple[SelectedSatellite, ...], dict[str, Any]] | None:
    candidates: list[tuple[OrbitalRecord, StatusRecord]] = []
    for norad_id, statuses in status_records.items():
        status = _latest_status(statuses, timestamp)
        orbit = _nearest_orbit(orbital_records.get(norad_id, ()), timestamp, config.max_tle_age_hours)
        if status is None or orbit is None or not status.is_isl_capable:
            continue
        candidates.append((orbit, status))
    if len(candidates) < num_planes * sats_per_plane:
        return None
    clusters = infer_raan_planes([item[0] for item in candidates], max_gap_deg=config.raan_cluster_max_gap_deg)
    eligible = [cluster for cluster in clusters if len(cluster) >= sats_per_plane]
    if len(eligible) < num_planes:
        return None
    by_id = {item[0].norad_id: item for item in candidates}
    selected: list[SelectedSatellite] = []
    chosen_clusters = sorted(eligible, key=lambda cluster: (min(item.raan_deg for item in cluster), len(cluster)))[:num_planes]
    for plane_idx, cluster in enumerate(chosen_clusters):
        for sat_idx, orbit in enumerate(sorted(cluster, key=_phase_key)[:sats_per_plane]):
            _, status = by_id[orbit.norad_id]
            selected.append(SelectedSatellite(orbit.norad_id, plane_idx, sat_idx, orbit, status.status, STATUS_MAPPING.get(status.status, False), True, (timestamp - orbit.epoch).total_seconds()))
    return tuple(selected), {
        "candidate_operational_isl_satellites": len(candidates),
        "inferred_plane_count": len(clusters),
        "eligible_plane_count": len(eligible),
        "selected_raan_centers_deg": [round(mean(item.raan_deg for item in cluster), 8) for cluster in chosen_clusters],
        "raan_cluster_max_gap_deg": config.raan_cluster_max_gap_deg,
    }


def _status_failure_probability(daily: Mapping[datetime.date, tuple[int, int]], timestamp: datetime, lookback_days: int) -> tuple[float, int, int]:
    numerator = denominator = 0
    for offset in range(1, lookback_days + 1):
        values = daily.get((timestamp - timedelta(days=offset)).date())
        if values:
            numerator += values[0]
            denominator += values[1]
    return (numerator / denominator if denominator else 0.0, numerator, denominator)


def _edge_failure_probability(
    selected: Sequence[SelectedSatellite],
    orbital_records: Mapping[int, Sequence[OrbitalRecord]],
    timestamp: datetime,
    config: Phase4AConfig,
) -> tuple[float, int, int]:
    """Estimate historical link unavailability without using episode outcomes.

    Numerator: historical candidate evaluations rejected by the frozen distance,
    LOS, or link-budget rules.  Denominator: candidate evaluations for which both
    endpoint TLEs existed within the 24-hour validity bound.  Samples are one
    deterministic state per UTC day in the strict 30-day pre-episode window.
    """
    walker = WalkerDeltaConfig(
        num_planes=max(item.plane_idx for item in selected) + 1,
        sats_per_plane=max(item.sat_in_plane for item in selected) + 1,
        inclination_deg=53.0,
        altitude_km=550.0,
        phasing_factor=1,
        epoch=timestamp,
    )
    unavailable = total = 0
    for offset in range(1, config.lookback_days + 1):
        state_time = timestamp - timedelta(days=offset)
        positions: list[SatellitePosition | None] = []
        for item in selected:
            orbit = _nearest_orbit(
                orbital_records.get(item.norad_id, ()),
                state_time,
                config.max_tle_age_hours,
            )
            positions.append(
                _propagate(
                    orbit,
                    state_time,
                    item.plane_idx * walker.sats_per_plane + item.sat_in_plane,
                )
                if orbit
                else None
            )
        _, stats = _compute_grid_plus_isls(
            walker,
            positions,
            LinkBudgetEngine(),
            max_isl_distance_km=config.max_isl_distance_km,
            isl_policy=ISL_POLICY,
            adjacent_search_k=ADJACENT_SEARCH_K,
            max_inter_plane_links_per_sat=MAX_INTER_PLANE_LINKS_PER_SAT,
        )
        total += stats.total_candidate_links
        unavailable += (
            stats.links_rejected_distance
            + stats.links_rejected_los
            + stats.links_rejected_budget
        )
    return (unavailable / total if total else 0.0, unavailable, total)


def _graph_snapshot(
    selected: Sequence[SelectedSatellite],
    timestamp: datetime,
    config: Phase4AConfig,
    *,
    positions_override: Sequence[SatellitePosition | None] | None = None,
) -> tuple[dict[str, Any], nx.Graph, list[SatellitePosition | None]]:
    graph, edges, positions = _build_graph(
        selected, timestamp, config, positions_override=positions_override
    )
    nodes = [{"node_id": i, "norad_id": item.norad_id, "plane_idx": item.plane_idx, "sat_in_plane": item.sat_in_plane, "features": _node_features(item.plane_idx, item.sat_in_plane, max(x.plane_idx for x in selected) + 1, max(x.sat_in_plane for x in selected) + 1, bool(graph.nodes[i]["exists"]))} for i, item in enumerate(selected)]
    directed_edges = [dict(edge) for edge in edges]
    directed_edges.extend({**edge, "source": edge["target"], "target": edge["source"]} for edge in edges)
    inter_plane_edges = [
        (u, v)
        for u, v, data in graph.edges(data=True)
        if data["link_type"] != "intra_plane"
    ]
    inter_plane_degrees: dict[int, int] = {}
    for source, target in inter_plane_edges:
        inter_plane_degrees[source] = inter_plane_degrees.get(source, 0) + 1
        inter_plane_degrees[target] = inter_plane_degrees.get(target, 0) + 1
    return {
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "nodes": nodes,
        "edges": directed_edges,
        "physical_edge_count": len(edges),
        "unavailable_node_count": sum(not bool(data["exists"]) for _, data in graph.nodes(data=True)),
        "connected_component_count": compute_num_components(graph),
        "gcc_size": compute_gcc_size(graph),
        "inter_plane_edge_count": len(inter_plane_edges),
        "max_inter_plane_endpoint_degree": max(inter_plane_degrees.values(), default=0),
        "topology_candidate_statistics": graph.graph["topology_stats"],
        "adaptive_topology_identity": _topology_identity(graph),
        "isl_policy": ISL_POLICY,
        "adjacent_search_k": ADJACENT_SEARCH_K,
        "max_inter_plane_links_per_sat": MAX_INTER_PLANE_LINKS_PER_SAT,
        "temporal_failure_edge_policy": FAILURE_MODEL,
    }, graph, positions


def _target_from_snapshots(snapshots: Sequence[tuple[dict[str, Any], nx.Graph]], nominal_nodes: int, threshold: float) -> tuple[float, bool, list[float]]:
    fractions: list[float] = []
    for _, graph in snapshots:
        gcc = compute_gcc_size(graph)
        fractions.append(gcc / nominal_nodes if nominal_nodes else 0.0)
    return min(fractions), any(value < threshold for value in fractions), fractions


def _source_manifest(root: Path) -> dict[str, Any]:
    files = []
    entries = (
        ("juliensimon/starlink-fleet-data", "README.md", root / "raw/starlink-fleet-data/README.md", "CelesTrak usage policy; source documentation"),
        ("juliensimon/starlink-fleet-data", "data/tle_snapshots.parquet", root / "raw/starlink-fleet-data/data/tle_snapshots.parquet", "CelesTrak-derived Starlink orbital/status snapshots"),
        ("juliensimon/space-track-tle-history", "README.md", root / "raw/space-track-tle-history/README.md", "CC-BY-4.0; Space-Track/USSF source documentation"),
        ("juliensimon/space-track-tle-history", "data/tle_2024.parquet", root / "raw/space-track-tle-history/data/tle_2024.parquet", "CC-BY-4.0; public Space-Track/USSF orbital elements"),
        ("juliensimon/space-track-tle-history", "data/tle_2025.parquet", root / "raw/space-track-tle-history/data/tle_2025.parquet", "CC-BY-4.0; public Space-Track/USSF orbital elements"),
    )
    for repository, relative, path, metadata in entries:
        if not path.is_file():
            raise FileNotFoundError(path)
        revision = SOURCE_REVISIONS[repository]
        files.append({"source_repository": repository, "revision": revision, "file": relative, "bytes": path.stat().st_size, "sha256": _sha256_file(path), "download_timestamp": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(), "license_source_metadata": metadata, "source_url": SOURCE_URLS[repository]})
    manifest = {
        "manifest_version": "phase4a.external_source_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": files,
    }
    ledger_path = root / "contracts/external_source_manifest.json"
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        expected = {
            (str(entry["source_repository"]), str(entry["file"])): str(entry["sha256"])
            for entry in ledger.get("sources", [])
        }
        observed = {
            (str(entry["source_repository"]), str(entry["file"])): str(entry["sha256"])
            for entry in files
        }
        if observed != expected:
            raise RuntimeError("raw source hashes differ from the recovered source ledger")
    return manifest


def _persist_source_manifest(root: Path, source_manifest: Mapping[str, Any]) -> str:
    """Persist canonical source-manifest bytes and verify their on-disk identity."""
    payload = _json_bytes(source_manifest)
    expected = _sha256_bytes(payload)
    path = root / "external_source_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    observed = _sha256_file(path)
    if observed != expected:
        raise RuntimeError("persisted source manifest hash does not match canonical bytes")
    return observed


def _contract(config: Phase4AConfig, source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    source_manifest_sha256 = _sha256_bytes(_json_bytes(source_manifest))
    return {
        "schema_version": "phase4a.external_validation_contract.v1",
        "status": "FROZEN_BEFORE_TARGET_CONSTRUCTION",
        "scope": list(EXTERNAL_TASKS),
        "not_authorized": ["integrated_ground_validation", "RF inference", "TGNN inference", "checkpoint loading", "training", "tuning"],
        "source_revisions": SOURCE_REVISIONS,
        "source_manifest_sha256": source_manifest_sha256,
        "statistical_contract": EXTERNAL_STATISTICAL_CONTRACT,
        "evaluation_period": {"start": config.start.isoformat().replace("+00:00", "Z"), "end": config.end.isoformat().replace("+00:00", "Z")},
        "episode_design": {"count": config.episode_count, "timestamps": "inclusive deterministic rounded linear spacing", "duration_minutes": config.duration_minutes, "cadence_seconds": config.step_seconds, "timesteps": NUM_TIMESTEPS, "schedule": "cycle product order num_planes=(4,5,6), sats_per_plane=(5,6,7,8)"},
        "matched_domain": {"shell": "53-degree / approximately 550-km Starlink shell; shell_id=2", "plane_counts": PLANE_COUNTS, "sats_per_plane_counts": SATS_PER_PLANE_COUNTS},
        "plane_inference": {"algorithm": "sort circular RAAN observations, split gaps > 1 degree, merge wrap-around clusters; sort members by (mean_anomaly + arg_perigee) modulo 360 then NORAD ID", "max_gap_deg": config.raan_cluster_max_gap_deg, "selection": "eligible clusters sorted by minimum RAAN then take requested count; no target/model fields consulted"},
        "status_mapping": STATUS_MAPPING,
        "status_categories_observed": ["anomalous", "decayed", "deorbiting", "operational", "raising", "unknown"],
        "isl_capability_rule": "is_isl_capable=true is required for candidate selection; this is a public heuristic, while status operational remains the service-available mapping",
        "missing_data": {"tle_max_age_hours": config.max_tle_age_hours, "future_tle": "prohibited", "missing_or_stale_state": "node unavailable; no future substitution", "rejected_episode": "if fewer than requested structurally valid operational/ISL-capable planes are available"},
        "physics": {"orbital_engine": "SGP4/WGS72 from source2 elements", "teme_to_ecef": "existing SATNET GMST transform", "earth_radius_km": EARTH_RADIUS_KM, "atmosphere_buffer_km": ATMOSPHERE_BUFFER_KM, "max_isl_distance_km": config.max_isl_distance_km, "isl_policy": ISL_POLICY, "adjacent_search_k": ADJACENT_SEARCH_K, "max_inter_plane_links_per_sat": MAX_INTER_PLANE_LINKS_PER_SAT, "link_budget": "existing LinkBudgetEngine defaults; optical preferred, RF fallback", "temporal_failure_edge_policy": FAILURE_MODEL},
        "rf_features": {"order": EXTERNAL_RF_FEATURE_ORDER, "altitude_rule": "mean SGP4-derived ECEF radius minus Earth radius for nominal selected nodes at episode start", "inclination_rule": "arithmetic mean of latest valid source2 inclination values for nominal selected nodes", "node_failure_probability": "30-day strict-pre-start source1 shell-2 ISL-capable snapshot unavailable fraction; numerator non-operational status rows, denominator all qualifying status rows", "edge_failure_probability": "30-day strict-pre-start daily states; numerator frozen candidate link rejected by distance/LOS/budget, denominator candidate evaluations with both endpoint TLEs <=24h old"},
        "tgnn_features": {"node_order": EXTERNAL_TGNN_NODE_FEATURE_ORDER, "edge_order": EXTERNAL_TGNN_EDGE_FEATURE_ORDER, "node_scaling": "plane/(num_planes-1), sat/(sats_per_plane-1), exists in {0,1}", "edge_scaling": "distance/10000, margin/100, type_code/2 with intra=0 inter=1 seam=2, optical=1 otherwise=0"},
        "targets": {"regression": "space_gcc_fraction_original_min = min_t |GCC_t| / nominal_subgraph_size", "classification": "space_threshold_breach_any = any_t(space GCC original fraction < 0.8)", "threshold": config.gcc_threshold, "denominator": "nominal selected satellite count; unavailable nodes remain in denominator"},
        "provenance_classes": ["OBSERVED_REAL", "DERIVED_FROM_REAL", "PUBLIC_HEURISTIC", "FROZEN_SATNET_DERIVATION"],
        "outcome_blindness": "source hashes, timestamp schedule, subgraph schedule, inference algorithm, status/capability rules, lookback, transformations, targets, threshold, and missing-data rules are frozen before aggregate target balance",
    }


def _provenance() -> list[dict[str, str]]:
    fields = [
        ("norad_id", "starlink-fleet-data/tle_snapshots.parquet", "OBSERVED_REAL", "identity join key", "catalog identifier", "exact NORAD value"),
        ("tle_epoch", "space-track-tle-history/tle_2024.parquet or tle_2025.parquet", "OBSERVED_REAL", "latest record at or before episode start", "UTC", "age <=24h; no future record"),
        ("sgp4_position_ecef", "real source2 orbital elements", "DERIVED_FROM_REAL", "SGP4 WGS72 then existing TEME→ECEF", "km", "qualified SATNET propagation semantics"),
        ("plane_idx", "real RAAN structure", "DERIVED_FROM_REAL", "circular RAAN clustering", "index", "1-degree gap; deterministic sort"),
        ("status_available", "starlink-fleet-data status", "DERIVED_FROM_REAL", "status mapping", "boolean", "operational true; all other observed categories false"),
        ("is_isl_capable", "starlink-fleet-data", "PUBLIC_HEURISTIC", "candidate eligibility filter", "boolean", "true required for selected shell candidates"),
        ("link_viability", "real SGP4 positions", "FROZEN_SATNET_DERIVATION", "grid_adaptive candidate ranking, distance, LOS, link budget, and endpoint capacity", "boolean", "existing SATNET adaptive max-range/LOS/budget/capacity rules"),
        ("distance_km", "real SGP4 positions", "DERIVED_FROM_REAL", "ECEF Euclidean distance", "km", "existing SATNET distance function"),
        ("margin_db", "real SGP4 positions", "FROZEN_SATNET_DERIVATION", "existing LinkBudgetEngine", "dB", "optical preferred; RF fallback"),
        ("num_planes", "frozen episode schedule", "FROZEN_SATNET_DERIVATION", "deterministic 12-cell cycle", "count", "4, 5, 6 only"),
        ("sats_per_plane", "frozen episode schedule", "FROZEN_SATNET_DERIVATION", "deterministic 12-cell cycle", "count", "5, 6, 7, 8 only"),
        ("altitude_km", "real SGP4 positions", "DERIVED_FROM_REAL", "mean nominal ECEF radius minus Earth radius", "km", "episode-start arithmetic mean"),
        ("inclination_deg", "source2 orbital elements", "OBSERVED_REAL", "arithmetic mean over selected nodes", "degree", "latest valid record at episode start"),
        ("satellite_node_failure_probability", "source1 status history", "DERIVED_FROM_REAL", "strict pre-start 30-day unavailable fraction", "fraction", "non-operational rows / all qualifying rows"),
        ("satellite_edge_failure_probability", "source2 orbital history", "FROZEN_SATNET_DERIVATION", "strict pre-start 30-day frozen candidate viability fraction", "fraction", "rejected candidate evaluations / valid candidate evaluations"),
        ("plane_idx_normalized", "derived plane_idx", "FROZEN_SATNET_DERIVATION", "plane/(num_planes-1)", "unitless", "frozen TGNN scaling"),
        ("sat_in_plane_normalized", "derived phase ordering", "FROZEN_SATNET_DERIVATION", "sat/(sats_per_plane-1)", "unitless", "frozen TGNN scaling"),
        ("node_exists_constant", "status and TLE validity", "DERIVED_FROM_REAL", "availability indicator", "boolean", "1 iff operational and valid state exists"),
        ("distance_km_scaled_10000", "distance_km", "FROZEN_SATNET_DERIVATION", "distance/10000", "unitless", "exact frozen edge scaling"),
        ("margin_db_scaled_100", "margin_db", "FROZEN_SATNET_DERIVATION", "margin/100", "unitless", "exact frozen edge scaling"),
        ("link_type_code_scaled_2", "adaptive link type", "FROZEN_SATNET_DERIVATION", "code/2; intra=0 inter=1 seam=2", "unitless", "exact frozen edge scaling"),
        ("link_mode_binary", "frozen link mode", "FROZEN_SATNET_DERIVATION", "optical=1 otherwise=0", "unitless", "exact frozen edge scaling"),
        ("space_gcc_fraction_original_min", "derived temporal SATNET graphs", "FROZEN_SATNET_DERIVATION", "min over 11 snapshots of GCC/nominal nodes", "fraction", "original-denominator GCC"),
        ("space_threshold_breach_any", "derived temporal SATNET graphs", "FROZEN_SATNET_DERIVATION", "any GCC fraction < 0.8", "boolean", "frozen synthetic threshold; no retuning"),
    ]
    return [{"field": field, "source": source, "provenance_class": provenance, "transformation": transformation, "units": units, "frozen_rule": rule} for field, source, provenance, transformation, units, rule in fields]


def _descriptive_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None, "q25": None, "median": None, "q75": None}
    ordered = sorted(clean)
    return {"count": len(clean), "min": min(clean), "max": max(clean), "mean": mean(clean), "std": pstdev(clean), "q25": ordered[int(0.25 * (len(ordered) - 1))], "median": median(ordered), "q75": ordered[int(0.75 * (len(ordered) - 1))]}


def _read_frozen_tgnn_sequence(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    data = path.read_bytes()
    magic = b"SATNET-TGNN-V1\0"
    if not data.startswith(magic):
        raise ValueError(f"invalid frozen TGNN sequence: {path}")
    cursor = len(magic)
    header_length = struct.unpack_from("<Q", data, cursor)[0]
    cursor += 8
    header = json.loads(data[cursor : cursor + header_length].decode("utf-8"))
    cursor += header_length
    arrays: dict[str, np.ndarray] = {}
    for name in ("node_features", "node_identity_index", "edge_index", "edge_attr", "snapshot_node_offsets", "snapshot_edge_offsets", "timestep_index"):
        length = struct.unpack_from("<Q", data, cursor)[0]
        cursor += 8
        arrays[name] = np.lib.format.read_array(io.BytesIO(data[cursor : cursor + length]), allow_pickle=False)
        cursor += length
    if cursor != len(data):
        raise ValueError(f"trailing bytes in frozen TGNN sequence: {path}")
    return header, arrays


def _frozen_train_domain(root: Path) -> dict[str, Any]:
    rf_path = root / "rf_space_classification" / "rf_space_classification.csv"
    if not rf_path.is_file():
        return {"available": False, "reason": f"missing {rf_path}"}
    rf = pd.read_csv(rf_path)
    train = rf[rf["split"].astype(str) == "train"]
    result: dict[str, Any] = {
        "available": True,
        "rf_train_rows": int(len(train)),
        "rf": {field: _descriptive_stats(train[field].astype(float).tolist()) for field in EXTERNAL_RF_FEATURE_ORDER},
    }
    target_manifest = root / "tgnn_space_classification" / "tgnn_space_classification_target_manifest.jsonl"
    sequence_root = root / "tgnn_space_classification" / "sequences"
    train_ids: list[int] = []
    if target_manifest.is_file():
        for line in target_manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("split") == "train":
                train_ids.append(int(row["run_id"]))
    node_counts: list[int] = []
    edge_counts: list[int] = []
    distances: list[float] = []
    margins: list[float] = []
    densities: list[float] = []
    for run_id in train_ids:
        path = sequence_root / f"run_{run_id:04d}.tgnn"
        if not path.is_file():
            continue
        header, arrays = _read_frozen_tgnn_sequence(path)
        node_by_step = header.get("node_count_by_timestep", [])
        edge_by_step = header.get("directed_edge_count_by_timestep", [])
        node_counts.extend(int(value) for value in node_by_step)
        edge_counts.extend(int(value) for value in edge_by_step)
        offsets_n = arrays["snapshot_node_offsets"]
        offsets_e = arrays["snapshot_edge_offsets"]
        for step in range(len(node_by_step)):
            count = int(node_by_step[step])
            edge_count = int(edge_by_step[step])
            distances.extend(arrays["edge_attr"][offsets_e[step] : offsets_e[step + 1], 0].astype(float) * 10_000.0)
            margins.extend(arrays["edge_attr"][offsets_e[step] : offsets_e[step + 1], 1].astype(float) * 100.0)
            densities.append(edge_count / (count * (count - 1)) if count > 1 else 0.0)
    result["tgnn"] = {"train_sequence_count": len(train_ids), "nodes_per_snapshot": _descriptive_stats(node_counts), "edges_per_snapshot": _descriptive_stats(edge_counts), "distance_km": _descriptive_stats(distances), "margin_db": _descriptive_stats(margins), "graph_density": _descriptive_stats(densities)}
    return result


def _bundle_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file() and path.name != "external_validation_inventory.json"):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _canonical_edge_records(edges: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for edge in edges:
        source = int(edge["source"])
        target = int(edge["target"])
        if source == target:
            raise RuntimeError("self-loop in external physical edge artifact")
        key = (min(source, target), max(source, target))
        record = {
            "source": key[0],
            "target": key[1],
            "distance_km": float(edge["distance_km"]),
            "margin_db": float(edge["margin_db"]),
            "link_type": str(edge["link_type"]),
            "link_mode": str(edge["link_mode"]),
        }
        previous = records.get(key)
        if previous is not None and previous != record:
            raise RuntimeError("opposite directed edge records disagree")
        records[key] = record
    return [records[key] for key in sorted(records)]


def audit_constructed_adaptive_artifacts(root: Path, config: Phase4AConfig) -> dict[str, Any]:
    """Recompute targets and topology evidence from persisted construction artifacts."""
    rf_rows = list(csv.DictReader((root / "episodes/external_rf_dataset.csv").open(newline="", encoding="utf-8")))
    if len(rf_rows) != config.episode_count:
        raise RuntimeError("persisted external RF dataset does not contain 300 episodes")
    expected_timestamps = choose_episode_timestamps(config.start, config.end, config.episode_count)
    if [int(row["episode_id"]) for row in rf_rows] != list(range(config.episode_count)):
        raise RuntimeError("persisted external episode IDs are not exactly 0..299")
    if [row["episode_timestamp"] for row in rf_rows] != [value.isoformat().replace("+00:00", "Z") for value in expected_timestamps]:
        raise RuntimeError("persisted external episode timestamp schedule mismatch")
    classifications: list[int] = []
    regressions: list[float] = []
    all_edges: list[dict[str, Any]] = []
    nodes_per: list[int] = []
    edges_per: list[int] = []
    densities: list[float] = []
    distances: list[float] = []
    margins: list[float] = []
    unavailable_counts: list[int] = []
    component_counts: list[int] = []
    inter_plane_counts: list[int] = []
    inter_plane_degrees: list[int] = []
    graph_sequence_parity = True
    target_graph_parity = True
    candidate_filtering_observed = False
    candidate_replacement_observed = False
    for row in rf_rows:
        sequence_path = root / row["tgnn_sequence"]
        graph_path = root / row["graph_edges"]
        sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
        graph_artifact = json.loads(graph_path.read_text(encoding="utf-8"))
        if sequence["episode_id"] != int(row["episode_id"]) or graph_artifact["episode_id"] != int(row["episode_id"]):
            raise RuntimeError("episode identity mismatch across external artifacts")
        fractions: list[float] = []
        for sequence_snapshot, graph_snapshot in zip(sequence["snapshots"], graph_artifact["snapshots"]):
            sequence_edges = _canonical_edge_records(sequence_snapshot["edges"])
            graph_edges = _canonical_edge_records(graph_snapshot["edges"])
            if sequence_edges != graph_edges:
                graph_sequence_parity = False
            graph_for_hash = nx.Graph()
            graph_for_hash.add_edges_from(
                (
                    edge["source"],
                    edge["target"],
                    {
                        "distance_km": edge["distance_km"],
                        "margin_db": edge["margin_db"],
                        "link_type": edge["link_type"],
                        "link_mode": edge["link_mode"],
                    },
                )
                for edge in graph_edges
            )
            if sequence_snapshot["isl_policy"] != ISL_POLICY or graph_snapshot["adaptive_topology_identity"] != _topology_identity(graph_for_hash):
                raise RuntimeError("persisted Adaptive-v2 topology identity mismatch")
            candidate_stats = sequence_snapshot["topology_candidate_statistics"]
            candidate_filtering_observed = candidate_filtering_observed or any(
                int(candidate_stats[field]) > 0
                for field in ("links_rejected_distance", "links_rejected_los", "links_rejected_budget")
            )
            candidate_replacement_observed = candidate_replacement_observed or bool(candidate_stats["adaptive_selection_examples"])
            graph = nx.Graph()
            for node in sequence_snapshot["nodes"]:
                graph.add_node(int(node["node_id"]), exists=bool(float(node["features"][2])))
            graph.add_edges_from((edge["source"], edge["target"]) for edge in graph_edges)
            if graph.number_of_edges() != len(graph_edges):
                raise RuntimeError("persisted physical edge identity is not unique")
            if any(not graph.nodes[node]["exists"] for edge in graph_edges for node in (edge["source"], edge["target"])):
                raise RuntimeError("unavailable node retained an incident external edge")
            if max((int(sequence_snapshot["max_inter_plane_endpoint_degree"]), int(graph_snapshot.get("max_inter_plane_endpoint_degree", 0))), default=0) > MAX_INTER_PLANE_LINKS_PER_SAT:
                raise RuntimeError("external adaptive endpoint capacity exceeded")
            gcc_fraction = compute_gcc_size(graph) / len(graph.nodes) if graph.nodes else 0.0
            fractions.append(gcc_fraction)
            nodes_per.append(graph.number_of_nodes())
            edges_per.append(len(graph_edges) * 2)
            densities.append(2.0 * len(graph_edges) / (len(graph.nodes) * (len(graph.nodes) - 1)) if len(graph.nodes) > 1 else 0.0)
            distances.extend(edge["distance_km"] for edge in graph_edges)
            margins.extend(edge["margin_db"] for edge in graph_edges)
            unavailable_counts.append(sum(not bool(data["exists"]) for _, data in graph.nodes(data=True)))
            component_counts.append(compute_num_components(graph))
            inter = [edge for edge in graph_edges if edge["link_type"] != "intra_plane"]
            inter_plane_counts.append(len(inter))
            degree: dict[int, int] = {}
            for edge in inter:
                degree[edge["source"]] = degree.get(edge["source"], 0) + 1
                degree[edge["target"]] = degree.get(edge["target"], 0) + 1
            inter_plane_degrees.append(max(degree.values(), default=0))
            all_edges.extend(graph_edges)
        recomputed_regression = min(fractions)
        recomputed_classification = int(any(value < config.gcc_threshold for value in fractions))
        if float(row["space_gcc_fraction_original_min"]) != recomputed_regression or int(row["space_threshold_breach_any"]) != recomputed_classification:
            target_graph_parity = False
        regressions.append(recomputed_regression)
        classifications.append(recomputed_classification)
    if not graph_sequence_parity or not target_graph_parity:
        raise RuntimeError("persisted external graph, sequence, and target artifacts disagree")
    if not candidate_filtering_observed or not candidate_replacement_observed:
        raise RuntimeError("persisted external artifacts lack observed adaptive candidate/filter evidence")
    return {
        "audit_version": "phase4a.external_dataset_audit.v2",
        "descriptive_only": True,
        "inference_performed": False,
        "episode_count": len(rf_rows),
        "rejected_episode_count": 0,
        "missing_or_stale_satellite_states": sum(int(float(row["tle_age_max_seconds"]) > config.max_tle_age_hours * 3600.0) for row in rf_rows),
        "classification": {"negative": classifications.count(0), "positive": classifications.count(1), "limited": len(set(classifications)) < 2 or min(classifications.count(0), classifications.count(1)) < 20},
        "regression": _descriptive_stats(regressions),
        "external_inputs": {name: _descriptive_stats([float(row[name]) for row in rf_rows]) for name in (*EXTERNAL_RF_FEATURE_ORDER, "tle_age_max_seconds")},
        "tgnn": {"nodes_per_snapshot": _descriptive_stats(nodes_per), "edges_per_snapshot": _descriptive_stats(edges_per), "distance_km": _descriptive_stats(distances), "margin_db": _descriptive_stats(margins), "graph_density": _descriptive_stats(densities)},
        "node_failure_semantics": {"synthetic_reference": "nominal topology first; failed nodes and incident edges removed from effective graph", "external_representation": "nominal node identity retained with node_exists_constant=0; incident edges removed", "unrelated_available_links_preserved": True, "gcc_denominator": "nominal selected satellite count"},
        "unavailable_nodes_per_snapshot": _descriptive_stats(unavailable_counts),
        "snapshots_containing_unavailable_nodes": sum(value > 0 for value in unavailable_counts),
        "snapshots_with_zero_edges": sum(value == 0 for value in edges_per),
        "snapshots_with_multiple_components": sum(value > 1 for value in component_counts),
        "inter_plane_edge_count": _descriptive_stats(inter_plane_counts),
        "max_inter_plane_endpoint_degree": max(inter_plane_degrees, default=0),
        "source_tle_age_seconds": _descriptive_stats([float(row["tle_age_max_seconds"]) for row in rf_rows]),
        "adaptive_propagation_evidence": {"graph_sequence_edge_parity": graph_sequence_parity, "graph_gcc_target_parity": target_graph_parity, "all_topology_hashes_recomputed": True, "all_persisted_edges_adaptive": all(edge["link_type"] in {"intra_plane", "inter_plane", "seam_link"} for edge in all_edges), "isl_policy": ISL_POLICY, "k": ADJACENT_SEARCH_K, "endpoint_capacity": MAX_INTER_PLANE_LINKS_PER_SAT, "los_budget_candidate_filtering": candidate_filtering_observed, "adaptive_candidate_replacement_observed": candidate_replacement_observed},
        "synthetic_train_domain": _frozen_train_domain(config.synthetic_dataset_root),
        "target_construction": {"classification": "any timestep fraction < 0.8", "regression": "minimum timestep fraction", "threshold": config.gcc_threshold, "denominator": "nominal selected satellite count"},
    }


def _implementation_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parents[4],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "external_validation_inventory.json"
    ]


def freeze_pre_inference_inventory(root: Path, config: Phase4AConfig, audit: Mapping[str, Any]) -> dict[str, Any]:
    source_manifest = json.loads((root / "external_source_manifest.json").read_text(encoding="utf-8"))
    artifacts = _artifact_inventory(root)
    inventory = {"inventory_version": "phase4a.external_validation_inventory.v2", "status": "REAL-DATA EXTERNAL VALIDATION DATASET FROZEN", "scientific_lineage": "Adaptive-v2", "implementation_git_sha": _implementation_git_sha(), "model_inference_performed": False, "external_predictions_generated": False, "external_metrics_calculated": False, "episode_count": int(audit["episode_count"]), "episode_ids": list(range(int(audit["episode_count"]))), "rejected_episode_count": int(audit["rejected_episode_count"]), "external_tasks": list(EXTERNAL_TASKS), "statistical_contract": EXTERNAL_STATISTICAL_CONTRACT, "source_revisions": SOURCE_REVISIONS, "source_raw_sha256": {f'{entry["source_repository"]}:{entry["file"]}': entry["sha256"] for entry in source_manifest["sources"]}, "adapter_sha256": _adapter_sha256(), "grid_fixed_source_count": 0, "adaptive_topology": {"isl_policy": ISL_POLICY, "k": ADJACENT_SEARCH_K, "endpoint_capacity": MAX_INTER_PLANE_LINKS_PER_SAT, "temporal_failure_edge_policy": FAILURE_MODEL}, "classification_balance": audit["classification"], "regression_distribution": audit["regression"], "distribution_shift_audit": audit, "provenance_classes": [item["provenance_class"] for item in _provenance()], "self_excluding_inventory_convention": "inventory SHA excludes external_validation_inventory.json; bundle hashes relative paths, NUL separators, and bytes", "artifacts": artifacts, "external_pre_inference_bundle_sha256": _bundle_hash(root)}
    _write_json(root / "external_validation_inventory.json", inventory)
    inventory["external_pre_inference_inventory_sha256"] = _sha256_file(root / "external_validation_inventory.json")
    return inventory


def build_external_validation(config: Phase4AConfig) -> dict[str, Any]:
    """Build the complete pre-inference package and return its handoff."""
    root = config.output_root
    if root.exists():
        raise RuntimeError(f"Adaptive-v2 external output root must be absent: {root.resolve()}")
    root.mkdir(parents=True, exist_ok=False)
    for directory in ("contracts", "adapter", "episodes", "audits", "manifests"):
        (root / directory).mkdir(exist_ok=True)
    source_manifest = _source_manifest(config.source_root)
    source_manifest_sha256 = _sha256_bytes(_json_bytes(source_manifest))
    contract = _contract(config, source_manifest)
    if contract["source_manifest_sha256"] != source_manifest_sha256:
        raise RuntimeError("source manifest hash changed before contract creation")
    persisted_source_manifest_sha256 = _persist_source_manifest(root, source_manifest)
    if persisted_source_manifest_sha256 != source_manifest_sha256:
        raise RuntimeError("persisted source manifest hash changed before contract write")
    _write_json(root / "contracts/external_validation_contract.json", contract)
    _write_json(root / "contracts/external_source_manifest.json", source_manifest)
    _write_json(root / "contracts/external_statistical_contract.json", EXTERNAL_STATISTICAL_CONTRACT)
    _write_json(root / "contracts/external_adapter_specification.json", {"adapter": "satnet.experiments.external_validation.phase4a", "implementation_sha256": _adapter_sha256(), "physics_source": "satnet.network.hypatia_adapter", "model_inference_performed": False, "provenance": _provenance()})

    timestamps = choose_episode_timestamps(config.start, config.end, config.episode_count)
    _write_json(root / "manifests/episode_timestamps.json", [item.isoformat().replace("+00:00", "Z") for item in timestamps])
    statuses, daily_status, norad_ids = _load_status_records(config.source_root, config.start, config.end)
    orbits = _load_orbital_records(config.source_root, norad_ids, config.start, config.end)
    source_hashes = {
        f'{entry["source_repository"]}:{entry["file"]}': entry["sha256"]
        for entry in source_manifest["sources"]
    }
    schedule = [(planes, sats) for planes in PLANE_COUNTS for sats in SATS_PER_PLANE_COUNTS]
    manifest_rows: list[dict[str, Any]] = []
    rf_rows: list[dict[str, Any]] = []
    all_snapshots: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    target_values: list[tuple[float, bool]] = []
    for episode_id, timestamp in enumerate(timestamps):
        planes, sats = schedule[episode_id % len(schedule)]
        selection = _choose_subgraph(statuses, orbits, timestamp, planes, sats, config)
        if selection is None:
            rejected.append({"episode_id": episode_id, "timestamp": timestamp.isoformat(), "reason": "insufficient structurally valid operational ISL-capable shell-2 records"})
            continue
        selected, plane_audit = selection
        node_failure, node_num, node_den = _status_failure_probability(daily_status, timestamp, config.lookback_days)
        edge_failure, edge_num, edge_den = _edge_failure_probability(selected, orbits, timestamp, config)
        snapshots: list[tuple[dict[str, Any], nx.Graph]] = []
        sequence: list[dict[str, Any]] = []
        for timestep in range(NUM_TIMESTEPS):
            snapshot_time = timestamp + timedelta(seconds=timestep * config.step_seconds)
            snapshot, graph, positions = _graph_snapshot(selected, snapshot_time, config)
            snapshot["timestep"] = timestep
            sequence.append(snapshot)
            snapshots.append((snapshot, graph))
            all_snapshots.append({"episode_id": episode_id, **snapshot})
        regression, classification, fractions = _target_from_snapshots(snapshots, len(selected), config.gcc_threshold)
        target_values.append((regression, classification))
        altitudes = []
        for item in selected:
            position = _propagate(item.orbit, timestamp, item.plane_idx * sats + item.sat_in_plane) if item.orbit else None
            if position is not None:
                altitudes.append(position.alt_km)
        inclinations = [item.orbit.inclination_deg for item in selected if item.orbit]
        episode_key = f"episode_{episode_id:04d}"
        sequence_path = root / "episodes" / "tgnn_sequences" / f"{episode_key}.json"
        _write_json(sequence_path, {"episode_id": episode_id, "timestamps": [row["timestamp"] for row in sequence], "node_feature_order": EXTERNAL_TGNN_NODE_FEATURE_ORDER, "edge_feature_order": EXTERNAL_TGNN_EDGE_FEATURE_ORDER, "snapshots": sequence})
        graph_path = root / "episodes" / "graph_edges" / f"{episode_key}.json"
        _write_json(
            graph_path,
            {
                "episode_id": episode_id,
                "snapshots": [
                    {
                        "timestep": snapshot["timestep"],
                        "timestamp": snapshot["timestamp"],
                        "nodes": snapshot["nodes"],
                        "edges": [
                            edge
                            for edge in snapshot["edges"]
                            if int(edge["source"]) < int(edge["target"])
                        ],
                        "adaptive_topology_identity": snapshot["adaptive_topology_identity"],
                    }
                    for snapshot in sequence
                ],
            },
        )
        topology_ids = [str(snapshot["adaptive_topology_identity"]) for snapshot in sequence]
        source_hash_json = json.dumps(source_hashes, sort_keys=True, separators=(",", ":"))
        rf_row = {"episode_id": episode_id, "episode_timestamp": timestamp.isoformat().replace("+00:00", "Z"), "num_planes": planes, "sats_per_plane": sats, "altitude_km": mean(altitudes) if altitudes else 0.0, "inclination_deg": mean(inclinations) if inclinations else 0.0, "satellite_node_failure_probability": node_failure, "satellite_edge_failure_probability": edge_failure, "space_gcc_fraction_original_min": regression, "space_threshold_breach_any": int(classification), "tgnn_sequence": sequence_path.relative_to(root).as_posix(), "graph_edges": graph_path.relative_to(root).as_posix(), "tle_age_max_seconds": max((item.tle_age_seconds or 0.0) for item in selected), "node_failure_numerator": node_num, "node_failure_denominator": node_den, "edge_failure_numerator": edge_num, "edge_failure_denominator": edge_den, "source_norad_ids": ";".join(str(item.norad_id) for item in selected), "source_artifact_hashes": source_hash_json, "adaptive_topology_identities": json.dumps(topology_ids, separators=(",", ":")), "isl_policy": ISL_POLICY, "adjacent_search_k": ADJACENT_SEARCH_K, "max_inter_plane_links_per_sat": MAX_INTER_PLANE_LINKS_PER_SAT, "temporal_failure_edge_policy": FAILURE_MODEL}
        rf_rows.append(rf_row)
        manifest_rows.append({"episode_id": episode_id, "timestamp": timestamp.isoformat().replace("+00:00", "Z"), "num_planes": planes, "sats_per_plane": sats, "nominal_satellites": len(selected), "sequence": sequence_path.relative_to(root).as_posix(), "graph_edges": graph_path.relative_to(root).as_posix(), "max_tle_age_seconds": rf_row["tle_age_max_seconds"], "node_failure_probability": node_failure, "edge_failure_probability": edge_failure, "regression_target": regression, "classification_target": int(classification), "plane_inference": plane_audit, "source_artifact_hashes": source_hashes, "adaptive_topology_identities": topology_ids, "isl_policy": ISL_POLICY, "adjacent_search_k": ADJACENT_SEARCH_K, "max_inter_plane_links_per_sat": MAX_INTER_PLANE_LINKS_PER_SAT, "temporal_failure_edge_policy": FAILURE_MODEL})

    if rejected:
        raise RuntimeError(f"Phase 4A rejected {len(rejected)} episodes; acceptance requires 300: {rejected[:3]}")
    if len(rf_rows) != 300 or len(manifest_rows) != 300:
        raise RuntimeError("Phase 4A did not construct exactly 300 episodes")
    _write_csv(root / "episodes/external_rf_dataset.csv", rf_rows)
    rf_metadata = ["episode_id", "episode_timestamp"]
    classification_rows = [{key: row[key] for key in (*rf_metadata, *EXTERNAL_RF_FEATURE_ORDER, "space_threshold_breach_any")} for row in rf_rows]
    regression_rows = [{key: row[key] for key in (*rf_metadata, *EXTERNAL_RF_FEATURE_ORDER, "space_gcc_fraction_original_min")} for row in rf_rows]
    _write_csv(root / "episodes/rf_space_classification.csv", classification_rows)
    _write_csv(root / "episodes/rf_space_regression.csv", regression_rows)
    tgnn_targets = [{"episode_id": row["episode_id"], "target_field": "space_threshold_breach_any", "target": row["space_threshold_breach_any"], "regression_target_field": "space_gcc_fraction_original_min", "regression_target": row["space_gcc_fraction_original_min"], "sequence": row["tgnn_sequence"]} for row in rf_rows]
    with (root / "episodes/tgnn_space_target_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in tgnn_targets:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    _write_csv(root / "manifests/external_episode_manifest.csv", manifest_rows)
    _write_json(root / "manifests/external_provenance_catalog.json", _provenance())

    audit = audit_constructed_adaptive_artifacts(root, config)
    _write_json(root / "audits/external_dataset_audit.json", audit)
    _write_json(root / "audits/adaptive_behavioral_evidence.json", audit["adaptive_propagation_evidence"])
    for filename, source in (("external_validation_contract.json", root / "contracts/external_validation_contract.json"), ("external_statistical_contract.json", root / "contracts/external_statistical_contract.json"), ("external_adapter_specification.json", root / "contracts/external_adapter_specification.json"), ("external_source_manifest.json", root / "external_source_manifest.json"), ("external_episode_manifest.csv", root / "manifests/external_episode_manifest.csv"), ("external_dataset_audit.json", root / "audits/external_dataset_audit.json"), ("adaptive_behavioral_evidence.json", root / "audits/adaptive_behavioral_evidence.json")):
        destination = root / filename
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    return freeze_pre_inference_inventory(root, config, audit)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty CSV")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Phase 4A without model inference")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    build_external_validation(Phase4AConfig(args.output_root))
    print("REAL-DATA EXTERNAL VALIDATION DATASET FROZEN — READY FOR EXTERNAL MODEL INFERENCE AUTHORIZATION")
