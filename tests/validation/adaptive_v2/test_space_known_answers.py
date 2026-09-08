"""Permanent Adaptive-v2 space-segment known-answer validation tests.

These tests preserve the independently verified numerical references used for
the dissertation validation campaign. They intentionally exercise the frozen
SATNET production implementation rather than replacing it with a second
implementation.

SGP4 reference:
Vallado et al., "Revisiting Spacetrack Report #3", AIAA 2006-6753,
verification satellite 00005.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sgp4.api import Satrec, WGS72

from satnet.network.hypatia_adapter import (
    SGP4_AVAILABLE,
    _compute_gmst,
    _compute_satellite_positions_sgp4,
    _datetime_to_jd,
    _teme_to_ecef,
)


# Vallado / CelesTrak SGP4 verification case sat00005.
TLE_NAME = "SAT-00005"
TLE_LINE_1 = (
    "1 00005U 58002B   00179.78495062  .00000023  "
    "00000-0  28098-4 0  4753"
)
TLE_LINE_2 = (
    "2 00005  34.2682 348.7242 1859667 331.7664  "
    "19.3264 10.82419157413667"
)

# TLE epoch: year 2000, day 179.78495062 UTC.
EPOCH = datetime(2000, 6, 27, 18, 50, 19, 733568, tzinfo=timezone.utc)

# Published TEME position vectors in km.
VALLADO_TEME_EPOCH_KM = (
    7022.46529266,
    -1400.08296755,
    0.03995155,
)

VALLADO_TEME_PLUS_360_MIN_KM = (
    -7154.03120202,
    -3783.17682504,
    -3536.19412294,
)

# Independently transformed with the frozen SATNET GMST-only
# TEME -> ECEF/PEF convention.
EXPECTED_ECEF_EPOCH_KM = (
    -6198.557667592166,
    3585.1267682150774,
    0.03995155,
)

EXPECTED_ECEF_PLUS_360_MIN_KM = (
    1245.7976368179052,
    -7996.285236101472,
    -3536.19412294,
)


def _assert_vector_close(
    actual: tuple[float, float, float] | list[float],
    expected: tuple[float, float, float],
    *,
    abs_km: float,
) -> None:
    assert actual[0] == pytest.approx(expected[0], abs=abs_km)
    assert actual[1] == pytest.approx(expected[1], abs=abs_km)
    assert actual[2] == pytest.approx(expected[2], abs=abs_km)


def test_canonical_validation_environment_has_sgp4() -> None:
    """Adaptive-v2 canonical execution requires the SGP4 orbital engine."""
    assert SGP4_AVAILABLE is True


@pytest.mark.parametrize(
    ("timestamp", "expected_jd"),
    [
        (EPOCH, 2451723.28495062),
        (EPOCH + timedelta(minutes=360), 2451723.53495062),
    ],
)
def test_datetime_to_jd_matches_vallado_case(
    timestamp: datetime,
    expected_jd: float,
) -> None:
    """SATNET's datetime conversion reproduces the known Julian dates."""
    jd, fraction = _datetime_to_jd(timestamp)
    assert jd + fraction == pytest.approx(expected_jd, abs=1e-10)


@pytest.mark.parametrize(
    ("offset_minutes", "expected_teme"),
    [
        (0, VALLADO_TEME_EPOCH_KM),
        (360, VALLADO_TEME_PLUS_360_MIN_KM),
    ],
)
def test_raw_sgp4_matches_vallado_sat00005(
    offset_minutes: int,
    expected_teme: tuple[float, float, float],
) -> None:
    """The installed SGP4 implementation reproduces Vallado verification vectors."""
    target = EPOCH + timedelta(minutes=offset_minutes)
    jd, fraction = _datetime_to_jd(target)

    satellite = Satrec.twoline2rv(TLE_LINE_1, TLE_LINE_2, WGS72)
    error_code, position_teme, _velocity_teme = satellite.sgp4(jd, fraction)

    assert error_code == 0
    _assert_vector_close(position_teme, expected_teme, abs_km=1e-7)


@pytest.mark.parametrize(
    ("offset_minutes", "expected_ecef"),
    [
        (0, EXPECTED_ECEF_EPOCH_KM),
        (360, EXPECTED_ECEF_PLUS_360_MIN_KM),
    ],
)
def test_satnet_sgp4_wrapper_matches_transformed_vallado_vectors(
    offset_minutes: int,
    expected_ecef: tuple[float, float, float],
) -> None:
    """SATNET SGP4 + GMST transformation matches the verified ECEF vectors."""
    positions = _compute_satellite_positions_sgp4(
        [(TLE_NAME, TLE_LINE_1, TLE_LINE_2)],
        EPOCH,
        offset_minutes * 60.0,
    )

    assert len(positions) == 1
    position = positions[0]

    _assert_vector_close(
        (position.x_km, position.y_km, position.z_km),
        expected_ecef,
        abs_km=2e-6,
    )


@pytest.mark.parametrize(
    ("timestamp", "expected_gmst_rad"),
    [
        (EPOCH, 3.469172342303337),
        (EPOCH + timedelta(minutes=360), 5.044269367050106),
    ],
)
def test_satnet_gmst_known_answers(
    timestamp: datetime,
    expected_gmst_rad: float,
) -> None:
    """Freeze the GMST values used by the authoritative frame transformation."""
    assert _compute_gmst(timestamp) == pytest.approx(
        expected_gmst_rad,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("teme", "timestamp", "expected_ecef"),
    [
        (VALLADO_TEME_EPOCH_KM, EPOCH, EXPECTED_ECEF_EPOCH_KM),
        (
            VALLADO_TEME_PLUS_360_MIN_KM,
            EPOCH + timedelta(minutes=360),
            EXPECTED_ECEF_PLUS_360_MIN_KM,
        ),
    ],
)
def test_teme_to_ecef_rotation_known_answers(
    teme: tuple[float, float, float],
    timestamp: datetime,
    expected_ecef: tuple[float, float, float],
) -> None:
    """Freeze SATNET's GMST-only TEME-to-ECEF/PEF transformation behavior."""
    actual = _teme_to_ecef(*teme, _compute_gmst(timestamp))
    _assert_vector_close(actual, expected_ecef, abs_km=1e-9)


# ---------------------------------------------------------------------------
# Earth-obscuration LOS known answers
# ---------------------------------------------------------------------------

from math import cos, radians, sin

from satnet.network.hypatia_adapter import (
    LinkBudgetEngine,
    SatellitePosition,
    _check_line_of_sight,
)


EFFECTIVE_EARTH_RADIUS_KM = 6451.0
LOS_TEST_ORBIT_RADIUS_KM = 6921.0
LOS_CRITICAL_ANGLE_DEG = 42.473791694096846


def _equal_radius_pair(angle_deg: float) -> tuple[SatellitePosition, SatellitePosition]:
    angle_rad = radians(angle_deg)

    first = SatellitePosition(
        sat_id=0,
        x_km=LOS_TEST_ORBIT_RADIUS_KM,
        y_km=0.0,
        z_km=0.0,
    )
    second = SatellitePosition(
        sat_id=1,
        x_km=LOS_TEST_ORBIT_RADIUS_KM * cos(angle_rad),
        y_km=LOS_TEST_ORBIT_RADIUS_KM * sin(angle_rad),
        z_km=0.0,
    )
    return first, second


@pytest.mark.parametrize(
    ("separation_deg", "expected_visible"),
    [
        (30.0, True),
        (LOS_CRITICAL_ANGLE_DEG - 0.001, True),
        (LOS_CRITICAL_ANGLE_DEG, True),
        (LOS_CRITICAL_ANGLE_DEG + 0.001, False),
        (60.0, False),
    ],
)
def test_buffered_earth_los_known_answers(
    separation_deg: float,
    expected_visible: bool,
) -> None:
    """Freeze the 6371 km Earth + 80 km atmosphere LOS boundary."""
    first, second = _equal_radius_pair(separation_deg)

    assert bool(_check_line_of_sight(first, second)) is expected_visible


def test_los_same_position_is_visible() -> None:
    """A zero-length segment outside the buffered Earth remains unobscured."""
    position = SatellitePosition(
        sat_id=0,
        x_km=LOS_TEST_ORBIT_RADIUS_KM,
        y_km=0.0,
        z_km=0.0,
    )

    assert _check_line_of_sight(position, position) is True


# ---------------------------------------------------------------------------
# RF and optical link-budget known answers
# ---------------------------------------------------------------------------

LINK_BUDGET = LinkBudgetEngine()


@pytest.mark.parametrize(
    ("distance_km", "expected_margin_db", "expected_viable"),
    [
        (100.0, 38.60905615127223, True),
        (1000.0, 18.609056151272227, True),
        (5000.0, 4.6296560645518525, True),
        (10000.0, -1.3909438487277725, False),
    ],
)
def test_rf_link_budget_known_answers(
    distance_km: float,
    expected_margin_db: float,
    expected_viable: bool,
) -> None:
    """Freeze the canonical 28-GHz RF ISL margins."""
    _received_dbm, margin_db, viable = LINK_BUDGET.compute_rf_link_budget(
        distance_km
    )

    assert margin_db == pytest.approx(expected_margin_db, abs=1e-10)
    assert viable is expected_viable


def test_rf_zero_margin_distance_known_answer() -> None:
    """The canonical RF link crosses zero margin near 8520.259 km."""
    zero_margin_distance_km = 8520.25921292311

    _received_dbm, margin_db, viable = LINK_BUDGET.compute_rf_link_budget(
        zero_margin_distance_km
    )

    assert margin_db == pytest.approx(0.0, abs=1e-10)
    assert viable is True


def test_rf_rain_margin_reduces_margin_by_exactly_ten_db() -> None:
    """Rain loss is applied only when explicitly requested for Earth-space use."""
    distance_km = 1000.0

    _rx_clear, margin_clear, _ = LINK_BUDGET.compute_rf_link_budget(
        distance_km,
        include_rain_margin=False,
    )
    _rx_rain, margin_rain, _ = LINK_BUDGET.compute_rf_link_budget(
        distance_km,
        include_rain_margin=True,
    )

    assert margin_clear - margin_rain == pytest.approx(10.0, abs=1e-12)


def test_optical_aperture_gain_known_answer() -> None:
    """Freeze the canonical 10-cm, 1550-nm optical aperture gain."""
    assert LINK_BUDGET._optical_gain_dbi == pytest.approx(
        103.53999038541929,
        abs=1e-10,
    )


@pytest.mark.parametrize(
    ("distance_km", "expected_margin_db"),
    [
        (100.0, 50.902417453802485),
        (1000.0, 30.902417453802457),
        (5000.0, 16.923017367082082),
        (10000.0, 10.902417453802457),
    ],
)
def test_optical_link_budget_known_answers(
    distance_km: float,
    expected_margin_db: float,
) -> None:
    """Freeze the canonical 1550-nm optical ISL margins."""
    _received_dbm, margin_db, viable = LINK_BUDGET.compute_optical_link_budget(
        distance_km
    )

    assert margin_db == pytest.approx(expected_margin_db, abs=1e-10)
    assert viable is True


def test_optical_zero_margin_distance_known_answer() -> None:
    """The canonical optical link crosses zero margin near 35084.951 km."""
    zero_margin_distance_km = 35084.95086791181

    _received_dbm, margin_db, viable = LINK_BUDGET.compute_optical_link_budget(
        zero_margin_distance_km
    )

    assert margin_db == pytest.approx(0.0, abs=1e-10)
    assert viable is True


def test_default_link_evaluation_prefers_optical_within_10k_cap() -> None:
    """At the final model's 10,000-km cap, optical remains viable and is selected."""
    mode, _received_dbm, margin_db, viable = LINK_BUDGET.evaluate_link(10000.0)

    assert mode == "optical"
    assert margin_db == pytest.approx(10.902417453802457, abs=1e-10)
    assert viable is True


# ---------------------------------------------------------------------------
# Walker-Delta constellation / TLE known answers
# ---------------------------------------------------------------------------

from math import degrees, pi

from satnet.network.hypatia_adapter import HypatiaAdapter


WALKER_VALIDATION_EPOCH = datetime(
    2026, 7, 19, 0, 0, 0, tzinfo=timezone.utc
)


def _independent_tle_checksum(line: str) -> int:
    """Compute the NORAD TLE checksum independently of SATNET."""
    total = 0
    for character in line[:68]:
        if character.isdigit():
            total += int(character)
        elif character == "-":
            total += 1
    return total % 10


def test_walker_p4_s5_t20_f1_mean_motion_known_answer(tmp_path) -> None:
    """Freeze the controlled P4/S5/T20/F1 Walker-Delta orbital parameters."""
    with HypatiaAdapter(
        num_planes=4,
        sats_per_plane=5,
        inclination_deg=53.0,
        altitude_km=550.0,
        phasing_factor=1,
        epoch=WALKER_VALIDATION_EPOCH,
        output_dir=tmp_path,
    ) as adapter:
        tle_path = adapter.generate_tles()

        assert adapter.total_satellites == 20
        assert adapter.config.total_satellites == 20
        assert adapter.config.mean_motion_rev_per_day == pytest.approx(
            15.078199602381408,
            abs=1e-12,
        )
        assert tle_path.exists()
        assert len(adapter._tle_lines) == 20


def test_walker_p4_s5_t20_f1_raan_and_phasing_known_answers(tmp_path) -> None:
    """Freeze RAAN spacing, 72-degree slot spacing, and 18-degree plane phasing."""
    with HypatiaAdapter(
        num_planes=4,
        sats_per_plane=5,
        inclination_deg=53.0,
        altitude_km=550.0,
        phasing_factor=1,
        epoch=WALKER_VALIDATION_EPOCH,
        output_dir=tmp_path,
    ) as adapter:
        adapter.generate_tles()

        expected_raan_by_plane = (0.0, 90.0, 180.0, 270.0)

        for sat_id, (_name, line1, line2) in enumerate(adapter._tle_lines):
            plane_index = sat_id // 5
            slot_index = sat_id % 5

            expected_raan_deg = expected_raan_by_plane[plane_index]
            expected_phase_deg = 18.0 * plane_index
            expected_mean_anomaly_deg = (
                72.0 * slot_index + expected_phase_deg
            ) % 360.0

            satellite = Satrec.twoline2rv(line1, line2, WGS72)

            inclination_deg = degrees(satellite.inclo)
            raan_deg = degrees(satellite.nodeo) % 360.0
            argument_of_perigee_deg = degrees(satellite.argpo) % 360.0
            mean_anomaly_deg = degrees(satellite.mo) % 360.0
            parsed_mean_motion_rev_per_day = (
                satellite.no_kozai * 1440.0 / (2.0 * pi)
            )

            assert inclination_deg == pytest.approx(53.0, abs=1e-4)
            assert raan_deg == pytest.approx(expected_raan_deg, abs=1e-4)
            assert satellite.ecco == pytest.approx(0.0001, abs=1e-7)
            assert argument_of_perigee_deg == pytest.approx(0.0, abs=1e-4)
            assert mean_anomaly_deg == pytest.approx(
                expected_mean_anomaly_deg,
                abs=1e-4,
            )
            assert parsed_mean_motion_rev_per_day == pytest.approx(
                15.07819960,
                abs=1e-8,
            )


def test_generated_walker_tles_have_valid_structure_and_checksums(tmp_path) -> None:
    """Freeze TLE cardinality, serialization, catalog numbering, and checksums."""
    with HypatiaAdapter(
        num_planes=4,
        sats_per_plane=5,
        inclination_deg=53.0,
        altitude_km=550.0,
        phasing_factor=1,
        epoch=WALKER_VALIDATION_EPOCH,
        output_dir=tmp_path,
    ) as adapter:
        tle_path = adapter.generate_tles()

        serialized_lines = tle_path.read_text().splitlines()

        assert len(serialized_lines) == 60

        for sat_id, (name, line1, line2) in enumerate(adapter._tle_lines):
            catalog_number = sat_id + 1
            base_index = sat_id * 3

            assert name == f"SAT-{sat_id:05d}"
            assert serialized_lines[base_index] == name
            assert serialized_lines[base_index + 1] == line1
            assert serialized_lines[base_index + 2] == line2

            assert len(line1) == 69
            assert len(line2) == 69

            assert line1.startswith(f"1 {catalog_number:05d}")
            assert line2.startswith(f"2 {catalog_number:05d}")

            assert int(line1[68]) == _independent_tle_checksum(line1)
            assert int(line2[68]) == _independent_tle_checksum(line2)


# ---------------------------------------------------------------------------
# Adaptive-v2 +Grid topology known answers
# ---------------------------------------------------------------------------

from collections import Counter

from satnet.network.hypatia_adapter import WalkerDeltaConfig, _compute_grid_plus_isls


def _adaptive_topology_positions() -> list[SatellitePosition]:
    """Controlled 2x4 geometry with a one-slot physical shift between planes."""
    return [
        # Plane 0
        SatellitePosition(0, 7000.0,   0.0, 0.0),
        SatellitePosition(1, 7000.0, 100.0, 0.0),
        SatellitePosition(2, 7000.0, 200.0, 0.0),
        SatellitePosition(3, 7000.0, 300.0, 0.0),

        # Plane 1: physically shifted by one slot
        SatellitePosition(4, 7000.0, 300.0, 0.0),
        SatellitePosition(5, 7000.0,   0.0, 0.0),
        SatellitePosition(6, 7000.0, 100.0, 0.0),
        SatellitePosition(7, 7000.0, 200.0, 0.0),
    ]


def _inter_plane_edges(links) -> set[tuple[int, int]]:
    return {
        tuple(sorted((link.sat_id_1, link.sat_id_2)))
        for link in links
        if link.link_type != "intra_plane"
    }


def test_grid_fixed_known_answer_uses_same_slot_neighbors() -> None:
    """The fixed +Grid policy preserves same-index adjacent-plane pairing."""
    config = WalkerDeltaConfig(
        num_planes=2,
        sats_per_plane=4,
        inclination_deg=53.0,
        altitude_km=550.0,
        phasing_factor=1,
    )

    links, stats = _compute_grid_plus_isls(
        config,
        _adaptive_topology_positions(),
        LinkBudgetEngine(),
        isl_policy="grid_fixed",
    )

    assert _inter_plane_edges(links) == {
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    }

    assert stats.accepted_inter_plane_links == 4


def test_grid_adaptive_v2_known_answer_selects_shifted_best_links() -> None:
    """Adaptive-v2 K=1/cap=1 selects the physically closest shifted partners."""
    config = WalkerDeltaConfig(
        num_planes=2,
        sats_per_plane=4,
        inclination_deg=53.0,
        altitude_km=550.0,
        phasing_factor=1,
    )

    links, stats = _compute_grid_plus_isls(
        config,
        _adaptive_topology_positions(),
        LinkBudgetEngine(),
        isl_policy="grid_adaptive",
        adjacent_search_k=1,
        max_inter_plane_links_per_sat=1,
    )

    expected_edges = {
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 4),
    }

    inter_plane_links = [
        link for link in links if link.link_type != "intra_plane"
    ]

    assert _inter_plane_edges(links) == expected_edges
    assert stats.accepted_inter_plane_links == 4

    # The shifted partners are colocated in this controlled geometry.
    for link in inter_plane_links:
        assert link.distance_km == pytest.approx(0.0, abs=1e-12)

    # Adaptive-v2 final-science capacity: at most one inter-plane
    # endpoint incident on any satellite.
    degree: Counter[int] = Counter()
    for link in inter_plane_links:
        degree[link.sat_id_1] += 1
        degree[link.sat_id_2] += 1

    assert set(degree) == set(range(8))
    assert max(degree.values()) == 1
