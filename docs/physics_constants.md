# Physics Constants and Link Budget Parameters

This document provides citations and justifications for all physics constants used in the Tier 1 simulation pipeline.

## Orbital Mechanics

| Constant | Value | Source |
|----------|-------|--------|
| `EARTH_RADIUS_KM` | 6371.0 km | WGS84 mean radius |
| `MU_EARTH_KM3_S2` | 398600.4418 km³/s² | WGS84 gravitational parameter |
| `C_M_S` | 299792458 m/s | Speed of light (exact, SI definition) |

## Optical ISL Parameters (1550nm Laser)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `OPTICAL_WAVELENGTH_M` | 1550 nm | Standard telecom C-band wavelength; eye-safe, mature technology. Used by Starlink, EDRS, and most commercial optical ISL systems. |
| `OPTICAL_TX_POWER_DBM` | 37.0 dBm (5W) | Conservative estimate for space-qualified optical terminals. Range: 1-10W typical. [1] |
| `OPTICAL_APERTURE_M` | 0.10 m (10 cm) | Typical small optical terminal aperture. Starlink uses ~5-10cm class. [2] |
| `OPTICAL_SENSITIVITY_DBM` | -45.0 dBm | Coherent receiver sensitivity at ~10 Gbps. APD receivers: -40 to -50 dBm typical. [1] |

**References:**
1. Kaushal, H., & Kaddoum, G. (2017). "Optical Communication in Space: Challenges and Mitigation Techniques." IEEE Communications Surveys & Tutorials.
2. SpaceX Starlink FCC filings (2020-2023) - optical ISL specifications.

## RF ISL Parameters (Ka-Band 28 GHz)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `RF_FREQUENCY_HZ` | 28 GHz | Ka-band ISL frequency; used by O3b, SES, and military systems. |
| `RF_TX_POWER_DBM` | 30.0 dBm (1W) | Conservative for LEO ISL; actual systems may use 2-10W. |
| `RF_ANTENNA_GAIN_DBI` | 40.0 dBi | Typical Ka-band phased array or parabolic antenna gain. |
| `RF_SENSITIVITY_DBM` | -90.0 dBm | Conservative receiver sensitivity for Ka-band. |

### Rain Margin Note (Step 5 Fix)

**`RF_RAIN_MARGIN_DB` is NOT applied to ISLs.**

Rain attenuation affects Earth-space paths where the signal traverses the troposphere. For satellite-to-satellite links (ISLs), the signal path is entirely in vacuum/space, so rain margin is inappropriate.

The `include_rain_margin` parameter in `compute_rf_link_budget()` defaults to `False` for ISL calculations. Rain margin should only be applied for:
- Ground station uplinks/downlinks
- Earth-space feeder links

## Link Budget Equations

### Free Space Path Loss (FSPL)

```
FSPL_dB = 20 * log10(4 * π * d / λ)
```

Where:
- `d` = distance in meters
- `λ` = wavelength in meters

### Optical Link Budget

```
P_rx = P_tx + G_tx - FSPL + G_rx
Margin = P_rx - Sensitivity
```

Optical antenna gain approximation (circular aperture):
```
G = (π * D / λ)² * η
```
Where η ≈ 0.55 (aperture efficiency).

### RF Link Budget

```
P_rx = P_tx + G_tx - FSPL + G_rx [- Rain_Margin if Earth-space]
Margin = P_rx - Sensitivity
```

## Coordinate Transform Fidelity

### TEME to ECEF Transform

The current implementation uses a simplified GMST Z-rotation:
```
θ_GMST = GMST(t)
R_ECEF = Rz(-θ_GMST) * R_TEME
```

**Phase 1 (Satellite-only) Note:**
ISL topology checks are based on inter-satellite distance and Earth-center line-of-sight with a spherical Earth model. These are invariant under any rigid rotation of the entire constellation, so a higher-fidelity TEME→ITRS/ITRF transform does **not** change Phase 1 ISL edge existence.

**Phase 2 (Ground Stations) Note:**
A rigorous TEME→ITRS/ITRF transform becomes critical once we introduce any Earth-fixed quantity (ground station visibility, geodetic computations).

## SGP4 Dependency

The Tier 1 pipeline requires `sgp4>=2.22` for accurate orbital propagation. If SGP4 is unavailable, the adapter falls back to simplified Keplerian propagation, which:
- Does not model J2 perturbations
- May drift significantly over multi-orbit simulations

**Recommendation:** Always use SGP4 for Tier 1 datasets. The fallback is for development/testing only.

---

*Last updated: 2026-01-07 (Step 5 - Physics Fidelity Boundaries)*
