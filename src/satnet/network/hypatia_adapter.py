"""
Hypatia Adapter for SatNet Architecture DSS (Tier 1 Fidelity)

This module provides an adapter between the satnet-arch-dss codebase and the
Hypatia satellite simulation library (satgenpy). It generates Walker Delta
constellations, computes TLEs, calculates ISLs over time, and returns
networkx graphs for each time step.

Tier 1 Physics Engines:
    1. Orbital Engine: SGP4 propagator with WGS72 + TEME-to-ECEF transforms
    2. Link Budget Engine: Optical (1550nm) and RF (Ka-Band 28GHz) analysis
    3. Geometry Engine: Earth obscuration via grazing height calculation

Dependencies:
    - networkx
    - numpy
    - sgp4 (for high-fidelity orbital propagation)
    - satgenpy (optional, falls back to internal implementation)
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SGP4 Import (Orbital Engine)
# ---------------------------------------------------------------------------
try:
    from sgp4.api import Satrec, WGS72
    from sgp4 import exporter
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False
    logger.warning("sgp4 not available. Using simplified Keplerian model.")

# Try to import satgenpy modules
try:
    from satgenpy.tles import generate_tles_from_scratch_with_sgp4
    SATGENPY_AVAILABLE = True
except ImportError:
    SATGENPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Physical Constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0
EARTH_MU = 398600.4418  # km^3/s^2 (gravitational parameter)
J2 = 1.08263e-3  # Earth's J2 perturbation coefficient
SECONDS_PER_DAY = 86400.0
ATMOSPHERE_BUFFER_KM = 80.0  # Buffer for Earth obscuration (grazing height)

# Speed of light
C_M_S = 299792458.0  # m/s
C_KM_S = C_M_S / 1000.0  # km/s

# ---------------------------------------------------------------------------
# Link Budget Constants (Tier 1)
# ---------------------------------------------------------------------------
# Optical ISL (1550nm laser)
OPTICAL_WAVELENGTH_M = 1550e-9  # 1550 nm
OPTICAL_TX_POWER_DBM = 37.0  # 5W = 37 dBm
OPTICAL_APERTURE_M = 0.10  # 10 cm aperture
OPTICAL_SENSITIVITY_DBM = -45.0  # Receiver sensitivity

# RF ISL (Ka-Band 28 GHz)
RF_FREQUENCY_HZ = 28e9  # 28 GHz
RF_TX_POWER_DBM = 30.0  # 1W default
RF_ANTENNA_GAIN_DBI = 40.0  # Typical Ka-band antenna
RF_SENSITIVITY_DBM = -90.0  # Receiver sensitivity
RF_RAIN_MARGIN_DB = 10.0  # Default rain margin


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class WalkerDeltaConfig:
    """Configuration for a Walker Delta constellation."""
    
    num_planes: int = 5
    sats_per_plane: int = 10
    inclination_deg: float = 53.0
    altitude_km: float = 550.0
    phasing_factor: int = 1  # Walker phasing F parameter
    epoch: datetime = field(default_factory=lambda: datetime.utcnow())
    
    @property
    def total_satellites(self) -> int:
        return self.num_planes * self.sats_per_plane
    
    @property
    def semi_major_axis_km(self) -> float:
        return EARTH_RADIUS_KM + self.altitude_km
    
    @property
    def orbital_period_seconds(self) -> float:
        """Compute orbital period using Kepler's third law."""
        a = self.semi_major_axis_km
        return 2 * math.pi * math.sqrt(a**3 / EARTH_MU)
    
    @property
    def mean_motion_rev_per_day(self) -> float:
        """Mean motion in revolutions per day."""
        return SECONDS_PER_DAY / self.orbital_period_seconds


@dataclass
class ISLLink:
    """Represents an inter-satellite link with link budget analysis."""
    sat_id_1: int
    sat_id_2: int
    distance_km: float = 0.0
    link_type: str = "isl"  # "intra_plane", "inter_plane", or "seam_link"
    # Link budget attributes (Tier 1)
    signal_strength_dbm: float = 0.0
    margin_db: float = 0.0
    link_mode: str = "optical"  # "optical" or "rf"


# ---------------------------------------------------------------------------
# Link Budget Engine (Tier 1 - Engine 2)
# ---------------------------------------------------------------------------

class LinkBudgetEngine:
    """
    Link budget calculator for Optical (1550nm) and RF (Ka-Band 28GHz) ISLs.
    
    Implements Free Space Path Loss (FSPL) and antenna gain calculations
    to determine link viability and margin.
    """
    
    def __init__(
        self,
        # Optical parameters
        optical_tx_power_dbm: float = OPTICAL_TX_POWER_DBM,
        optical_aperture_m: float = OPTICAL_APERTURE_M,
        optical_sensitivity_dbm: float = OPTICAL_SENSITIVITY_DBM,
        optical_wavelength_m: float = OPTICAL_WAVELENGTH_M,
        # RF parameters
        rf_tx_power_dbm: float = RF_TX_POWER_DBM,
        rf_antenna_gain_dbi: float = RF_ANTENNA_GAIN_DBI,
        rf_sensitivity_dbm: float = RF_SENSITIVITY_DBM,
        rf_frequency_hz: float = RF_FREQUENCY_HZ,
        rf_rain_margin_db: float = RF_RAIN_MARGIN_DB,
    ):
        # Optical
        self.optical_tx_power_dbm = optical_tx_power_dbm
        self.optical_aperture_m = optical_aperture_m
        self.optical_sensitivity_dbm = optical_sensitivity_dbm
        self.optical_wavelength_m = optical_wavelength_m
        
        # RF
        self.rf_tx_power_dbm = rf_tx_power_dbm
        self.rf_antenna_gain_dbi = rf_antenna_gain_dbi
        self.rf_sensitivity_dbm = rf_sensitivity_dbm
        self.rf_frequency_hz = rf_frequency_hz
        self.rf_rain_margin_db = rf_rain_margin_db
        
        # Precompute optical antenna gain (approximation for circular aperture)
        # G ≈ (π * D / λ)^2 * η, where η ≈ 0.55 (aperture efficiency)
        eta = 0.55
        self._optical_gain_linear = (
            (math.pi * self.optical_aperture_m / self.optical_wavelength_m) ** 2 * eta
        )
        self._optical_gain_dbi = 10 * math.log10(self._optical_gain_linear)
    
    def compute_optical_fspl_db(self, distance_km: float) -> float:
        """
        Compute Free Space Path Loss for optical (1550nm) link.
        
        FSPL = 20*log10(4*π*d/λ) in dB
        
        Args:
            distance_km: Link distance in kilometers
        
        Returns:
            FSPL in dB (positive value representing loss)
        """
        distance_m = distance_km * 1000.0
        if distance_m <= 0:
            return 0.0
        
        fspl = 20 * math.log10(4 * math.pi * distance_m / self.optical_wavelength_m)
        return fspl
    
    def compute_rf_fspl_db(self, distance_km: float) -> float:
        """
        Compute Free Space Path Loss for RF (Ka-Band 28GHz) link.
        
        FSPL = 20*log10(4*π*d*f/c) in dB
        
        Args:
            distance_km: Link distance in kilometers
        
        Returns:
            FSPL in dB (positive value representing loss)
        """
        distance_m = distance_km * 1000.0
        if distance_m <= 0:
            return 0.0
        
        fspl = 20 * math.log10(4 * math.pi * distance_m * self.rf_frequency_hz / C_M_S)
        return fspl
    
    def compute_optical_link_budget(
        self, distance_km: float
    ) -> Tuple[float, float, bool]:
        """
        Compute optical link budget.
        
        Link equation: P_rx = P_tx + G_tx - FSPL + G_rx
        
        Args:
            distance_km: Link distance in kilometers
        
        Returns:
            Tuple of (received_power_dbm, margin_db, link_viable)
        """
        fspl = self.compute_optical_fspl_db(distance_km)
        
        # Both Tx and Rx use same aperture gain (symmetric link)
        received_power_dbm = (
            self.optical_tx_power_dbm
            + self._optical_gain_dbi  # Tx gain
            - fspl
            + self._optical_gain_dbi  # Rx gain
        )
        
        margin_db = received_power_dbm - self.optical_sensitivity_dbm
        link_viable = margin_db >= 0
        
        return received_power_dbm, margin_db, link_viable
    
    def compute_rf_link_budget(
        self, distance_km: float, include_rain_margin: bool = False
    ) -> Tuple[float, float, bool]:
        """
        Compute RF (Ka-Band) link budget.
        
        Args:
            distance_km: Link distance in kilometers
            include_rain_margin: Whether to subtract rain margin.
                Default False for ISLs (space-to-space paths have no rain).
                Set True only for Earth-space links (ground station uplinks).
        
        Returns:
            Tuple of (received_power_dbm, margin_db, link_viable)
        """
        fspl = self.compute_rf_fspl_db(distance_km)
        
        # Link equation: P_rx = P_tx + G_tx - FSPL + G_rx - rain_margin
        received_power_dbm = (
            self.rf_tx_power_dbm
            + self.rf_antenna_gain_dbi  # Tx gain
            - fspl
            + self.rf_antenna_gain_dbi  # Rx gain
        )
        
        if include_rain_margin:
            received_power_dbm -= self.rf_rain_margin_db
        
        margin_db = received_power_dbm - self.rf_sensitivity_dbm
        link_viable = margin_db >= 0
        
        return received_power_dbm, margin_db, link_viable
    
    def evaluate_link(
        self, distance_km: float, prefer_optical: bool = True
    ) -> Tuple[str, float, float, bool]:
        """
        Evaluate link using preferred mode, fallback to alternative if needed.
        
        Args:
            distance_km: Link distance in kilometers
            prefer_optical: If True, try optical first, then RF
        
        Returns:
            Tuple of (mode, signal_strength_dbm, margin_db, viable)
        """
        if prefer_optical:
            rx_power, margin, viable = self.compute_optical_link_budget(distance_km)
            if viable:
                return "optical", rx_power, margin, True
            # Fallback to RF
            rx_power, margin, viable = self.compute_rf_link_budget(distance_km)
            return "rf", rx_power, margin, viable
        else:
            rx_power, margin, viable = self.compute_rf_link_budget(distance_km)
            if viable:
                return "rf", rx_power, margin, True
            # Fallback to optical
            rx_power, margin, viable = self.compute_optical_link_budget(distance_km)
            return "optical", rx_power, margin, viable


# ---------------------------------------------------------------------------
# Orbital Engine (Tier 1 - Engine 1): TEME to ECEF Transform
# ---------------------------------------------------------------------------

def _compute_gmst(dt: datetime) -> float:
    """
    Compute Greenwich Mean Sidereal Time (GMST) in radians.
    
    Uses the IAU 1982 model approximation.
    
    Args:
        dt: UTC datetime
    
    Returns:
        GMST angle in radians
    """
    # Julian date calculation
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + dt.microsecond / 3600e6
    
    if month <= 2:
        year -= 1
        month += 12
    
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    
    jd = (
        int(365.25 * (year + 4716))
        + int(30.6001 * (month + 1))
        + day
        + hour / 24.0
        + B
        - 1524.5
    )
    
    # Julian centuries from J2000.0
    T = (jd - 2451545.0) / 36525.0
    
    # GMST in degrees (IAU 1982 model)
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * T**2
        - T**3 / 38710000.0
    )
    
    # Normalize to [0, 360)
    gmst_deg = gmst_deg % 360.0
    
    return math.radians(gmst_deg)


def _teme_to_ecef(
    x_teme_km: float,
    y_teme_km: float,
    z_teme_km: float,
    gmst_rad: float,
) -> Tuple[float, float, float]:
    """
    Transform coordinates from TEME (True Equator Mean Equinox) to ECEF.
    
    This is a simple Z-axis rotation by the negative GMST angle.
    
    Args:
        x_teme_km, y_teme_km, z_teme_km: Position in TEME frame (km)
        gmst_rad: Greenwich Mean Sidereal Time in radians
    
    Returns:
        Tuple of (x_ecef_km, y_ecef_km, z_ecef_km)
    """
    cos_gmst = math.cos(gmst_rad)
    sin_gmst = math.sin(gmst_rad)
    
    # Rotation matrix R_z(-gmst) applied to TEME vector
    x_ecef = cos_gmst * x_teme_km + sin_gmst * y_teme_km
    y_ecef = -sin_gmst * x_teme_km + cos_gmst * y_teme_km
    z_ecef = z_teme_km  # Z unchanged in Z-axis rotation
    
    return x_ecef, y_ecef, z_ecef


# ---------------------------------------------------------------------------
# TLE Generation Utilities
# ---------------------------------------------------------------------------

def _compute_tle_checksum(line: str) -> int:
    """Compute TLE line checksum (modulo 10 sum of digits, '-' counts as 1)."""
    checksum = 0
    for char in line[:68]:
        if char.isdigit():
            checksum += int(char)
        elif char == '-':
            checksum += 1
    return checksum % 10


def _format_tle_float(value: float, width: int, precision: int) -> str:
    """Format a float for TLE format (no leading zero for decimals)."""
    formatted = f"{value:>{width}.{precision}f}"
    return formatted


def _generate_tle_lines(
    sat_id: int,
    inclination_deg: float,
    raan_deg: float,
    eccentricity: float,
    arg_perigee_deg: float,
    mean_anomaly_deg: float,
    mean_motion: float,
    epoch: datetime,
    name: str = None,
) -> Tuple[str, str, str]:
    """
    Generate TLE (Two-Line Element) for a satellite.
    
    Returns:
        Tuple of (name_line, line1, line2)
    """
    if name is None:
        name = f"SAT-{sat_id:05d}"
    
    # Catalog number (use sat_id, padded)
    catalog_num = sat_id + 1
    
    # Epoch: year (2-digit) + day of year with fractional day
    year_2digit = epoch.year % 100
    # Handle both timezone-aware and naive datetimes
    year_start = datetime(epoch.year, 1, 1, tzinfo=epoch.tzinfo)
    day_of_year = (epoch - year_start).total_seconds() / SECONDS_PER_DAY + 1
    epoch_str = f"{year_2digit:02d}{day_of_year:012.8f}"
    
    # Line 1
    # Format: 1 NNNNNC NNNNNAAA NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNN
    line1 = f"1 {catalog_num:05d}U 00000A   {epoch_str}  .00000000  00000-0  00000-0 0  0000"
    line1 = line1[:68]
    checksum1 = _compute_tle_checksum(line1)
    line1 = line1 + str(checksum1)
    
    # Line 2
    # Format: 2 NNNNN NNN.NNNN NNN.NNNN NNNNNNN NNN.NNNN NNN.NNNN NN.NNNNNNNNNNNNNN
    ecc_str = f"{eccentricity:.7f}"[2:]  # Remove "0."
    
    line2 = (
        f"2 {catalog_num:05d} "
        f"{inclination_deg:8.4f} "
        f"{raan_deg:8.4f} "
        f"{ecc_str} "
        f"{arg_perigee_deg:8.4f} "
        f"{mean_anomaly_deg:8.4f} "
        f"{mean_motion:11.8f}"
    )
    # Pad/truncate to 68 chars before checksum
    line2 = f"{line2:68}"[:68]
    checksum2 = _compute_tle_checksum(line2)
    line2 = line2 + str(checksum2)
    
    name_line = f"{name:24}"
    
    return name_line, line1, line2


# ---------------------------------------------------------------------------
# Satellite Position Computation (Tier 1 - SGP4 Orbital Engine)
# ---------------------------------------------------------------------------

@dataclass
class SatellitePosition:
    """3D position of a satellite in ECEF coordinates."""
    sat_id: int
    x_km: float
    y_km: float
    z_km: float
    lat_deg: float = 0.0
    lon_deg: float = 0.0
    alt_km: float = 0.0


def _compute_satellite_positions_sgp4(
    tle_lines: List[Tuple[str, str, str]],
    epoch: datetime,
    time_offset_seconds: float,
) -> List[SatellitePosition]:
    """
    Compute satellite positions using SGP4 propagator with WGS72 gravity model.
    
    This is the Tier 1 high-fidelity orbital engine.
    
    Args:
        tle_lines: List of (name, line1, line2) tuples for each satellite
        epoch: TLE epoch datetime
        time_offset_seconds: Time offset from epoch in seconds
    
    Returns:
        List of SatellitePosition objects in ECEF frame
    """
    if not SGP4_AVAILABLE:
        raise RuntimeError("SGP4 library not available. Install with: pip install sgp4")
    
    positions = []
    
    # Compute target time
    target_time = epoch + timedelta(seconds=time_offset_seconds)
    
    # Compute GMST for TEME->ECEF transformation
    gmst_rad = _compute_gmst(target_time)
    
    # Convert time to Julian date components for SGP4
    # SGP4 uses (jd, fr) where jd is integer Julian day and fr is fraction
    year = target_time.year
    month = target_time.month
    day = target_time.day
    hour = target_time.hour
    minute = target_time.minute
    second = target_time.second + target_time.microsecond / 1e6
    
    for sat_id, (name, line1, line2) in enumerate(tle_lines):
        # Create Satrec object from TLE using WGS72 gravity model
        satellite = Satrec.twoline2rv(line1, line2, WGS72)
        
        # Propagate to target time
        # SGP4 returns error code, position (km), velocity (km/s) in TEME frame
        jd, fr = _datetime_to_jd(target_time)
        error, r_teme, v_teme = satellite.sgp4(jd, fr)
        
        if error != 0:
            # SGP4 propagation error - use fallback position at origin
            # This shouldn't happen with valid TLEs
            positions.append(SatellitePosition(
                sat_id=sat_id,
                x_km=0.0, y_km=0.0, z_km=0.0,
                lat_deg=0.0, lon_deg=0.0, alt_km=0.0,
            ))
            continue
        
        # Transform from TEME to ECEF
        x_ecef, y_ecef, z_ecef = _teme_to_ecef(
            r_teme[0], r_teme[1], r_teme[2], gmst_rad
        )
        
        # Compute geodetic coordinates
        r = math.sqrt(x_ecef**2 + y_ecef**2 + z_ecef**2)
        lat_rad = math.asin(z_ecef / r) if r > 0 else 0.0
        lon_rad = math.atan2(y_ecef, x_ecef)
        alt_km = r - EARTH_RADIUS_KM
        
        positions.append(SatellitePosition(
            sat_id=sat_id,
            x_km=x_ecef,
            y_km=y_ecef,
            z_km=z_ecef,
            lat_deg=math.degrees(lat_rad),
            lon_deg=math.degrees(lon_rad),
            alt_km=alt_km,
        ))
    
    return positions


def _datetime_to_jd(dt: datetime) -> Tuple[float, float]:
    """
    Convert datetime to Julian Date (jd, fraction) for SGP4.
    
    Returns:
        Tuple of (julian_day_integer, day_fraction)
    """
    year = dt.year
    month = dt.month
    day = dt.day
    
    if month <= 2:
        year -= 1
        month += 12
    
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    
    jd_int = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524
    
    # Fraction of day
    fr = (dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + dt.microsecond / 3600e6) / 24.0
    
    # Adjust for Julian day starting at noon
    jd = jd_int - 0.5
    
    return jd, fr


def _compute_satellite_positions_keplerian(
    config: WalkerDeltaConfig,
    time_offset_seconds: float,
) -> List[SatellitePosition]:
    """
    Compute satellite positions using simplified Keplerian propagation.
    
    This is the fallback when SGP4 is not available.
    
    Args:
        config: Walker Delta constellation configuration
        time_offset_seconds: Time offset from epoch in seconds
    
    Returns:
        List of SatellitePosition objects
    """
    positions = []
    
    a = config.semi_major_axis_km
    inc_rad = math.radians(config.inclination_deg)
    period = config.orbital_period_seconds
    
    # Angular velocity (rad/s)
    n = 2 * math.pi / period
    
    # Compute GMST for proper ECEF transformation
    target_time = config.epoch + timedelta(seconds=time_offset_seconds)
    gmst_rad = _compute_gmst(target_time)
    
    sat_id = 0
    for plane_idx in range(config.num_planes):
        # RAAN for this plane (evenly distributed) - in inertial frame
        raan_rad = 2 * math.pi * plane_idx / config.num_planes
        
        for sat_in_plane in range(config.sats_per_plane):
            # Initial mean anomaly with Walker phasing
            phase_offset = (
                2 * math.pi * config.phasing_factor * plane_idx 
                / config.total_satellites
            )
            initial_mean_anomaly = (
                2 * math.pi * sat_in_plane / config.sats_per_plane
                + phase_offset
            )
            
            # Mean anomaly at current time
            mean_anomaly = initial_mean_anomaly + n * time_offset_seconds
            mean_anomaly = mean_anomaly % (2 * math.pi)
            
            # For circular orbit, true anomaly = mean anomaly
            true_anomaly = mean_anomaly
            
            # Position in orbital plane
            x_orbital = a * math.cos(true_anomaly)
            y_orbital = a * math.sin(true_anomaly)
            
            # Rotate to ECI (inertial frame)
            cos_raan = math.cos(raan_rad)
            sin_raan = math.sin(raan_rad)
            cos_inc = math.cos(inc_rad)
            sin_inc = math.sin(inc_rad)
            
            x_eci = cos_raan * x_orbital - sin_raan * cos_inc * y_orbital
            y_eci = sin_raan * x_orbital + cos_raan * cos_inc * y_orbital
            z_eci = sin_inc * y_orbital
            
            # Transform ECI to ECEF using GMST rotation
            x_ecef, y_ecef, z_ecef = _teme_to_ecef(x_eci, y_eci, z_eci, gmst_rad)
            
            # Compute lat/lon/alt
            r = math.sqrt(x_ecef**2 + y_ecef**2 + z_ecef**2)
            lat_rad = math.asin(z_ecef / r) if r > 0 else 0.0
            lon_rad = math.atan2(y_ecef, x_ecef)
            
            positions.append(SatellitePosition(
                sat_id=sat_id,
                x_km=x_ecef,
                y_km=y_ecef,
                z_km=z_ecef,
                lat_deg=math.degrees(lat_rad),
                lon_deg=math.degrees(lon_rad),
                alt_km=r - EARTH_RADIUS_KM,
            ))
            
            sat_id += 1
    
    return positions


def _compute_distance_km(pos1: SatellitePosition, pos2: SatellitePosition) -> float:
    """Compute Euclidean distance between two satellites."""
    dx = pos1.x_km - pos2.x_km
    dy = pos1.y_km - pos2.y_km
    dz = pos1.z_km - pos2.z_km
    return math.sqrt(dx**2 + dy**2 + dz**2)


# ---------------------------------------------------------------------------
# Objective 2: Earth Obscuration (Line of Sight) Check
# ---------------------------------------------------------------------------

def _check_line_of_sight(pos1: SatellitePosition, pos2: SatellitePosition) -> bool:
    """
    Check if there is a clear line of sight between two satellites.
    
    Returns False if the link would pass through Earth (including atmosphere buffer).
    
    Math:
        Treat the link as a line segment between vectors r1 and r2.
        Calculate the minimum distance (h_min) from Earth's center (0,0,0) to that segment.
        If h_min < (EARTH_RADIUS_KM + ATMOSPHERE_BUFFER_KM), the link is obscured.
    
    Args:
        pos1: Position of first satellite
        pos2: Position of second satellite
    
    Returns:
        True if line of sight is clear, False if obscured by Earth
    """
    # Vector from origin to each satellite
    r1 = np.array([pos1.x_km, pos1.y_km, pos1.z_km])
    r2 = np.array([pos2.x_km, pos2.y_km, pos2.z_km])
    
    # Direction vector of the line segment
    d = r2 - r1
    d_len_sq = np.dot(d, d)
    
    if d_len_sq == 0:
        # Same position (degenerate case)
        return True
    
    # Parameter t for the closest point on the line to origin
    # Closest point P = r1 + t * d, where t is clamped to [0, 1] for segment
    # t = -dot(r1, d) / dot(d, d)
    t = -np.dot(r1, d) / d_len_sq
    
    # Clamp t to [0, 1] to stay within the segment
    t_clamped = max(0.0, min(1.0, t))
    
    # Closest point on segment to Earth's center
    closest_point = r1 + t_clamped * d
    
    # Distance from Earth's center to closest point
    h_min = np.linalg.norm(closest_point)
    
    # Check against Earth radius + atmosphere buffer
    min_clearance = EARTH_RADIUS_KM + ATMOSPHERE_BUFFER_KM
    
    return h_min >= min_clearance


# ---------------------------------------------------------------------------
# ISL Computation (Tier 1 - with Link Budget Engine)
# ---------------------------------------------------------------------------

@dataclass
class ISLComputationStats:
    """Statistics from ISL computation for diagnostics."""
    total_candidate_links: int = 0
    links_rejected_los: int = 0  # Line of sight (Earth obscuration)
    links_rejected_budget: int = 0  # Link budget insufficient
    links_accepted: int = 0
    optical_links: int = 0
    rf_links: int = 0


def _compute_grid_plus_isls(
    config: WalkerDeltaConfig,
    positions: List[SatellitePosition],
    link_budget: LinkBudgetEngine,
    max_isl_distance_km: float = 10000.0,  # Increased - let link budget decide
) -> Tuple[List[ISLLink], ISLComputationStats]:
    """
    Compute +Grid ISL pattern with Tier 1 physics:
    - 2 intra-plane links (to neighbors in same orbital plane)
    - 2 inter-plane links (to satellites in adjacent planes)
    
    This is the standard Hypatia/Starlink ISL pattern.
    
    Tier 1 Physics:
    - Earth obscuration check (Geometry Engine)
    - Link budget analysis (Link Budget Engine)
    - Seam link tagging
    
    Args:
        config: Walker Delta constellation configuration
        positions: List of satellite positions in ECEF
        link_budget: LinkBudgetEngine instance for link analysis
        max_isl_distance_km: Maximum geometric distance (soft limit)
    
    Returns:
        Tuple of (links, stats) where stats contains rejection counts
    """
    links = []
    stats = ISLComputationStats()
    
    num_planes = config.num_planes
    sats_per_plane = config.sats_per_plane
    
    def sat_index(plane: int, sat: int) -> int:
        return plane * sats_per_plane + sat
    
    added_links: Set[Tuple[int, int]] = set()
    
    for plane_idx in range(num_planes):
        for sat_in_plane in range(sats_per_plane):
            current_sat = sat_index(plane_idx, sat_in_plane)
            current_pos = positions[current_sat]
            
            # Intra-plane link: next satellite in same plane
            next_in_plane = sat_index(plane_idx, (sat_in_plane + 1) % sats_per_plane)
            link_key = tuple(sorted([current_sat, next_in_plane]))
            if link_key not in added_links:
                stats.total_candidate_links += 1
                next_pos = positions[next_in_plane]
                dist = _compute_distance_km(current_pos, next_pos)
                
                # Check 1: Earth obscuration (Geometry Engine)
                if not _check_line_of_sight(current_pos, next_pos):
                    stats.links_rejected_los += 1
                else:
                    # Check 2: Link budget (Link Budget Engine)
                    mode, signal_dbm, margin_db, viable = link_budget.evaluate_link(dist)
                    
                    if not viable:
                        stats.links_rejected_budget += 1
                    else:
                        links.append(ISLLink(
                            sat_id_1=current_sat,
                            sat_id_2=next_in_plane,
                            distance_km=dist,
                            link_type="intra_plane",
                            signal_strength_dbm=signal_dbm,
                            margin_db=margin_db,
                            link_mode=mode,
                        ))
                        stats.links_accepted += 1
                        if mode == "optical":
                            stats.optical_links += 1
                        else:
                            stats.rf_links += 1
                added_links.add(link_key)
            
            # Inter-plane link: same position in next plane
            next_plane = (plane_idx + 1) % num_planes
            partner_sat = sat_index(next_plane, sat_in_plane)
            link_key = tuple(sorted([current_sat, partner_sat]))
            if link_key not in added_links:
                stats.total_candidate_links += 1
                partner_pos = positions[partner_sat]
                dist = _compute_distance_km(current_pos, partner_pos)
                
                # Check 1: Earth obscuration (Geometry Engine)
                if not _check_line_of_sight(current_pos, partner_pos):
                    stats.links_rejected_los += 1
                else:
                    # Check 2: Link budget (Link Budget Engine)
                    mode, signal_dbm, margin_db, viable = link_budget.evaluate_link(dist)
                    
                    if not viable:
                        stats.links_rejected_budget += 1
                    else:
                        # Tag seam links (last plane -> first plane)
                        if next_plane == 0 and plane_idx == num_planes - 1:
                            link_type = "seam_link"
                        else:
                            link_type = "inter_plane"
                        
                        links.append(ISLLink(
                            sat_id_1=current_sat,
                            sat_id_2=partner_sat,
                            distance_km=dist,
                            link_type=link_type,
                            signal_strength_dbm=signal_dbm,
                            margin_db=margin_db,
                            link_mode=mode,
                        ))
                        stats.links_accepted += 1
                        if mode == "optical":
                            stats.optical_links += 1
                        else:
                            stats.rf_links += 1
                added_links.add(link_key)
    
    return links, stats


# ---------------------------------------------------------------------------
# Main Adapter Class
# ---------------------------------------------------------------------------

class HypatiaAdapter:
    """
    Adapter between satnet-arch-dss and the Hypatia satellite simulation library.
    
    Tier 1 Fidelity Features:
    - SGP4 orbital propagation with WGS72 gravity model
    - TEME-to-ECEF coordinate transformation
    - Link budget analysis (Optical 1550nm + RF Ka-Band 28GHz)
    - Earth obscuration (grazing height) checks
    
    Generates Walker Delta constellations, computes TLEs, calculates ISLs
    over time, and provides networkx graphs for each time step.
    """
    
    def __init__(
        self,
        num_planes: int = 5,
        sats_per_plane: int = 10,
        inclination_deg: float = 53.0,
        altitude_km: float = 550.0,
        phasing_factor: int = 1,
        output_dir: Optional[Path] = None,
        link_budget: Optional[LinkBudgetEngine] = None,
        epoch: Optional[datetime] = None,
    ):
        """
        Initialize the Hypatia adapter with Walker Delta constellation parameters.
        
        Args:
            num_planes: Number of orbital planes
            sats_per_plane: Number of satellites per plane
            inclination_deg: Orbital inclination in degrees
            altitude_km: Orbital altitude above Earth surface in km
            phasing_factor: Walker phasing factor (F parameter)
            output_dir: Directory for generated files (default: temp directory)
            link_budget: LinkBudgetEngine instance (default: creates one with defaults)
            epoch: TLE epoch datetime (default: current UTC time for backward compat)
        """
        # Use provided epoch or fall back to utcnow for backward compatibility
        effective_epoch = epoch if epoch is not None else datetime.utcnow()
        
        self.config = WalkerDeltaConfig(
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            inclination_deg=inclination_deg,
            altitude_km=altitude_km,
            phasing_factor=phasing_factor,
            epoch=effective_epoch,
        )
        
        if output_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="hypatia_")
            self.output_dir = Path(self._temp_dir)
        else:
            self._temp_dir = None
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Link Budget Engine (Tier 1)
        self.link_budget = link_budget if link_budget else LinkBudgetEngine()
        
        self._tle_file: Optional[Path] = None
        self._tle_lines: List[Tuple[str, str, str]] = []  # Store TLE data for SGP4
        self._isl_data: Dict[int, List[ISLLink]] = {}  # time_step -> links
        self._duration_seconds: int = 0
        self._step_seconds: int = 0
        self._max_isl_distance_km: float = 10000.0  # Increased - link budget decides
    
    @property
    def total_satellites(self) -> int:
        return self.config.total_satellites
    
    def generate_tles(self) -> Path:
        """
        Generate TLE file for the Walker Delta constellation.
        
        Also stores TLE lines internally for SGP4 propagation.
        
        Returns:
            Path to the generated TLE file
        """
        tle_path = self.output_dir / "constellation.tle"
        
        lines = []
        self._tle_lines = []  # Clear and rebuild
        sat_id = 0
        
        for plane_idx in range(self.config.num_planes):
            # RAAN for this plane
            raan_deg = 360.0 * plane_idx / self.config.num_planes
            
            for sat_in_plane in range(self.config.sats_per_plane):
                # Mean anomaly with Walker phasing
                phase_offset = (
                    360.0 * self.config.phasing_factor * plane_idx 
                    / self.config.total_satellites
                )
                mean_anomaly_deg = (
                    360.0 * sat_in_plane / self.config.sats_per_plane
                    + phase_offset
                ) % 360.0
                
                name_line, line1, line2 = _generate_tle_lines(
                    sat_id=sat_id,
                    inclination_deg=self.config.inclination_deg,
                    raan_deg=raan_deg,
                    eccentricity=0.0001,  # Near-circular
                    arg_perigee_deg=0.0,
                    mean_anomaly_deg=mean_anomaly_deg,
                    mean_motion=self.config.mean_motion_rev_per_day,
                    epoch=self.config.epoch,
                    name=f"SAT-{sat_id:05d}",
                )
                
                lines.append(name_line.strip())
                lines.append(line1)
                lines.append(line2)
                
                # Store for SGP4 propagation
                self._tle_lines.append((name_line.strip(), line1, line2))
                
                sat_id += 1
        
        with open(tle_path, 'w') as f:
            f.write('\n'.join(lines))
        
        self._tle_file = tle_path
        logger.info(
            "Generated TLEs for %d satellites: %s (SGP4: %s)",
            self.total_satellites,
            tle_path,
            "enabled" if SGP4_AVAILABLE else "disabled",
        )
        
        return tle_path
    
    def calculate_isls(
        self,
        duration_minutes: int = 60,
        step_seconds: int = 60,
        max_isl_distance_km: float = 10000.0,
    ) -> Tuple[Path, ISLComputationStats]:
        """
        Compute Inter-Satellite Links over time using Tier 1 physics.
        
        Uses +Grid pattern: 2 intra-plane + 2 inter-plane links per satellite.
        
        Tier 1 Physics:
        - SGP4 orbital propagation (WGS72) or Keplerian fallback
        - Link budget analysis (Optical 1550nm / RF Ka-Band 28GHz)
        - Earth obscuration (grazing height) checks
        - Seam link tagging
        
        Args:
            duration_minutes: Simulation duration in minutes
            step_seconds: Time step in seconds
            max_isl_distance_km: Maximum geometric distance (soft limit)
        
        Returns:
            Tuple of (path to ISL data file, aggregated computation stats)
        """
        # Ensure TLEs are generated
        if not self._tle_lines:
            self.generate_tles()
        
        self._duration_seconds = duration_minutes * 60
        self._step_seconds = step_seconds
        self._max_isl_distance_km = max_isl_distance_km
        
        num_steps = self._duration_seconds // step_seconds + 1
        
        isl_path = self.output_dir / "isls.txt"
        
        self._isl_data.clear()
        
        # Aggregate stats across all time steps
        total_stats = ISLComputationStats()
        
        with open(isl_path, 'w') as f:
            f.write(f"# ISL data for Walker Delta constellation (Tier 1 Fidelity)\n")
            f.write(f"# Planes: {self.config.num_planes}, Sats/plane: {self.config.sats_per_plane}\n")
            f.write(f"# Duration: {duration_minutes} min, Step: {step_seconds} s\n")
            f.write(f"# Orbital Engine: {'SGP4 (WGS72)' if SGP4_AVAILABLE else 'Keplerian'}\n")
            f.write(f"# Format: time_step sat1 sat2 distance_km link_type mode margin_db\n")
            
            for step in range(num_steps):
                time_offset = step * step_seconds
                
                # Compute positions using SGP4 or Keplerian fallback
                if SGP4_AVAILABLE and self._tle_lines:
                    positions = _compute_satellite_positions_sgp4(
                        self._tle_lines,
                        self.config.epoch,
                        time_offset,
                    )
                else:
                    positions = _compute_satellite_positions_keplerian(
                        self.config, time_offset
                    )
                
                # Compute ISLs with Link Budget Engine
                links, step_stats = _compute_grid_plus_isls(
                    self.config,
                    positions,
                    self.link_budget,
                    max_isl_distance_km,
                )
                
                # Aggregate stats
                total_stats.total_candidate_links += step_stats.total_candidate_links
                total_stats.links_rejected_los += step_stats.links_rejected_los
                total_stats.links_rejected_budget += step_stats.links_rejected_budget
                total_stats.links_accepted += step_stats.links_accepted
                total_stats.optical_links += step_stats.optical_links
                total_stats.rf_links += step_stats.rf_links
                
                self._isl_data[step] = links
                
                for link in links:
                    f.write(
                        f"{step} {link.sat_id_1} {link.sat_id_2} "
                        f"{link.distance_km:.2f} {link.link_type} "
                        f"{link.link_mode} {link.margin_db:.1f}\n"
                    )
        
        logger.info(
            "Calculated ISLs for %d steps (%d min @ %ds): %s",
            num_steps, duration_minutes, step_seconds, isl_path,
        )
        logger.debug(
            "ISL stats: candidates=%d, accepted=%d, rejected_los=%d, rejected_budget=%d, optical=%d, rf=%d",
            total_stats.total_candidate_links,
            total_stats.links_accepted,
            total_stats.links_rejected_los,
            total_stats.links_rejected_budget,
            total_stats.optical_links,
            total_stats.rf_links,
        )
        
        return isl_path, total_stats
    
    def get_graph_at_step(self, time_step: int) -> nx.Graph:
        """
        Get the network graph at a specific time step.
        
        Args:
            time_step: The time step index (0-based)
        
        Returns:
            networkx.Graph with satellite nodes and ISL edges
        
        Raises:
            ValueError: If ISLs haven't been calculated or time_step is out of range
        """
        if not self._isl_data:
            raise ValueError(
                "ISL data not available. Call calculate_isls() first."
            )
        
        if time_step not in self._isl_data:
            max_step = max(self._isl_data.keys())
            raise ValueError(
                f"Time step {time_step} out of range. "
                f"Available steps: 0 to {max_step}"
            )
        
        G = nx.Graph()
        
        # Add all satellite nodes
        for sat_id in range(self.total_satellites):
            plane_idx = sat_id // self.config.sats_per_plane
            sat_in_plane = sat_id % self.config.sats_per_plane
            G.add_node(
                sat_id,
                type="satellite",
                plane=plane_idx,
                sat_in_plane=sat_in_plane,
                label=f"SAT-{sat_id:05d}",
            )
        
        # Add ISL edges for this time step with Tier 1 attributes
        for link in self._isl_data[time_step]:
            G.add_edge(
                link.sat_id_1,
                link.sat_id_2,
                distance_km=link.distance_km,
                link_type=link.link_type,
                capacity=10.0,  # Default capacity for compatibility
                # Tier 1 Link Budget attributes
                signal_strength_dbm=link.signal_strength_dbm,
                margin_db=link.margin_db,
                link_mode=link.link_mode,
            )
        
        return G
    
    def get_graph_at_time(self, time_seconds: float) -> nx.Graph:
        """
        Get the network graph at a specific time in seconds.
        
        Args:
            time_seconds: Time offset from epoch in seconds
        
        Returns:
            networkx.Graph with satellite nodes and ISL edges
        """
        if self._step_seconds == 0:
            raise ValueError(
                "ISL data not available. Call calculate_isls() first."
            )
        
        time_step = int(time_seconds // self._step_seconds)
        return self.get_graph_at_step(time_step)
    
    def get_all_graphs(self) -> Dict[int, nx.Graph]:
        """
        Get network graphs for all computed time steps.
        
        Returns:
            Dictionary mapping time_step -> networkx.Graph
        """
        return {step: self.get_graph_at_step(step) for step in self._isl_data}

    def iter_graphs(
        self,
        start_step: int = 0,
        end_step: Optional[int] = None,
    ) -> "Iterator[Tuple[int, nx.Graph]]":
        """
        Iterate over network graphs for a range of time steps.
        
        This generator yields (time_step, graph) tuples, reusing the cached
        ISL data computed by calculate_isls(). It avoids re-parsing TLEs
        or recomputing ISLs per step.
        
        Args:
            start_step: First time step to yield (0-indexed, inclusive).
            end_step: Last time step to yield (exclusive). If None, iterates
                      through all available steps.
        
        Yields:
            Tuple of (time_step, nx.Graph) for each step in range.
        
        Raises:
            ValueError: If ISLs haven't been calculated.
        
        Example:
            >>> adapter.calculate_isls(duration_minutes=10, step_seconds=60)
            >>> for t, G in adapter.iter_graphs():
            ...     print(f"Step {t}: {G.number_of_edges()} edges")
        """
        if not self._isl_data:
            raise ValueError(
                "ISL data not available. Call calculate_isls() first."
            )
        
        max_available_step = max(self._isl_data.keys())
        
        if end_step is None:
            end_step = max_available_step + 1
        
        # Clamp to available range
        start_step = max(0, start_step)
        end_step = min(end_step, max_available_step + 1)
        
        for step in range(start_step, end_step):
            if step in self._isl_data:
                yield step, self.get_graph_at_step(step)

    @property
    def num_steps(self) -> int:
        """Return the number of computed time steps, or 0 if ISLs not calculated."""
        return len(self._isl_data)

    @property
    def step_seconds(self) -> int:
        """Return the time step interval in seconds."""
        return self._step_seconds

    @property
    def duration_seconds(self) -> int:
        """Return the total simulation duration in seconds."""
        return self._duration_seconds
    
    def get_positions_at_step(self, time_step: int) -> List[SatellitePosition]:
        """
        Get satellite positions at a specific time step.
        
        Uses SGP4 propagation if available, otherwise Keplerian fallback.
        
        Args:
            time_step: The time step index (0-based)
        
        Returns:
            List of SatellitePosition objects in ECEF frame
        """
        time_offset = time_step * self._step_seconds
        
        if SGP4_AVAILABLE and self._tle_lines:
            return _compute_satellite_positions_sgp4(
                self._tle_lines,
                self.config.epoch,
                time_offset,
            )
        else:
            return _compute_satellite_positions_keplerian(self.config, time_offset)
    
    def summary(self) -> str:
        """Return a summary of the constellation configuration."""
        return (
            f"Walker Delta Constellation:\n"
            f"  Planes: {self.config.num_planes}\n"
            f"  Sats/plane: {self.config.sats_per_plane}\n"
            f"  Total satellites: {self.total_satellites}\n"
            f"  Inclination: {self.config.inclination_deg}°\n"
            f"  Altitude: {self.config.altitude_km} km\n"
            f"  Orbital period: {self.config.orbital_period_seconds:.1f} s "
            f"({self.config.orbital_period_seconds/60:.1f} min)\n"
            f"  Mean motion: {self.config.mean_motion_rev_per_day:.4f} rev/day\n"
            f"  Output dir: {self.output_dir}"
        )


# ---------------------------------------------------------------------------
# Main Test Block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Hypatia Adapter Test - Tier 1 Fidelity")
    print("=" * 70)
    
    print("\n--- Physics Engines Status ---")
    print(f"  1. Orbital Engine (SGP4 + WGS72): {'ENABLED' if SGP4_AVAILABLE else 'DISABLED (Keplerian fallback)'}")
    print(f"  2. Link Budget Engine: ENABLED (Optical 1550nm + RF Ka-Band 28GHz)")
    print(f"  3. Geometry Engine (Earth Obscuration): ENABLED")
    
    # Create a 50-satellite constellation (5 planes x 10 sats)
    adapter = HypatiaAdapter(
        num_planes=5,
        sats_per_plane=10,
        inclination_deg=53.0,
        altitude_km=550.0,
    )
    
    print("\n" + adapter.summary())
    
    # Generate TLEs
    print("\n--- Generating TLEs ---")
    tle_path = adapter.generate_tles()
    
    # Calculate ISLs for 10 minutes at 60-second intervals
    print("\n--- Calculating ISLs (Tier 1 Physics) ---")
    isl_path, stats = adapter.calculate_isls(
        duration_minutes=10,
        step_seconds=60,
        max_isl_distance_km=10000.0,
    )
    
    # Print detailed Tier 1 statistics
    print("\n--- Tier 1 Physics Statistics ---")
    print(f"Total candidate links evaluated: {stats.total_candidate_links}")
    print(f"Links accepted: {stats.links_accepted}")
    print(f"Links REJECTED (Earth obscuration): {stats.links_rejected_los}")
    print(f"Links REJECTED (link budget): {stats.links_rejected_budget}")
    if stats.total_candidate_links > 0:
        los_rate = 100.0 * stats.links_rejected_los / stats.total_candidate_links
        budget_rate = 100.0 * stats.links_rejected_budget / stats.total_candidate_links
        print(f"  Earth obscuration rejection rate: {los_rate:.2f}%")
        print(f"  Link budget rejection rate: {budget_rate:.2f}%")
    
    print(f"\nLink Mode Distribution:")
    print(f"  Optical (1550nm): {stats.optical_links}")
    print(f"  RF (Ka-Band 28GHz): {stats.rf_links}")
    
    # Get graph at t=0
    print("\n--- Network Graph at t=0 ---")
    G = adapter.get_graph_at_step(0)
    
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    
    # Count link types
    intra_plane = sum(1 for _, _, d in G.edges(data=True) if d.get('link_type') == 'intra_plane')
    inter_plane = sum(1 for _, _, d in G.edges(data=True) if d.get('link_type') == 'inter_plane')
    seam_links = sum(1 for _, _, d in G.edges(data=True) if d.get('link_type') == 'seam_link')
    print(f"  Intra-plane links: {intra_plane}")
    print(f"  Inter-plane links: {inter_plane}")
    print(f"  Seam links: {seam_links}")
    
    # Count link modes in graph
    optical_count = sum(1 for _, _, d in G.edges(data=True) if d.get('link_mode') == 'optical')
    rf_count = sum(1 for _, _, d in G.edges(data=True) if d.get('link_mode') == 'rf')
    print(f"  Optical links: {optical_count}")
    print(f"  RF links: {rf_count}")
    
    # Show link budget statistics from graph edges
    margins = [d.get('margin_db', 0) for _, _, d in G.edges(data=True)]
    if margins:
        print(f"\nLink Budget Margins (at t=0):")
        print(f"  Min margin: {min(margins):.1f} dB")
        print(f"  Max margin: {max(margins):.1f} dB")
        print(f"  Avg margin: {sum(margins)/len(margins):.1f} dB")
    
    # Check connectivity
    if nx.is_connected(G):
        print("\nNetwork is CONNECTED")
    else:
        num_components = nx.number_connected_components(G)
        print(f"\nNetwork has {num_components} connected components")
    
    # Show average degree
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    print(f"Average degree: {avg_degree:.2f}")
    
    print("\n" + "=" * 70)
    print("Tier 1 Fidelity Test Complete!")
    print("=" * 70)
