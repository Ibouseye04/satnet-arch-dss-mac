# Ground Segment Stage G2 Coordinate-Frame Contract

**Status:** Approved

**Decision date:** 2026-07-16

**Validated G1 implementation SHA:** `753b28be767b7bcee213e239bbfd086b26b1944b`

**G1 documentation/tagged HEAD:** `abb7843647ffa1ed8e19590ea791c81b26811177`

**G2 branch base SHA:** `abb7843647ffa1ed8e19590ea791c81b26811177`

## Production satellite-position audit

The production SGP4 path returns WGS72 TEME position coordinates in kilometers. `HypatiaAdapter` applies its existing Greenwich Mean Sidereal Time Z-axis rotation and exposes project-ECEF coordinates in kilometers through its production `SatellitePosition` values and `get_positions_at_step()` API.

The adapter's TEME-to-ECEF operation is the satellite frame boundary already used by the validated satellite simulator. Stage G2 consumes that exposed result without modification.

## Canonical G2 frame contract

```text
coordinate frame: project ECEF
frame contract version: hypatia_ecef_gmst_v1
units: kilometers
satellite ID type: integer
timestamp: rollout epoch + timestep_index × step_seconds
timestamp normalization: timezone-aware UTC
Earth rotation: already applied by HypatiaAdapter
ground station behavior: fixed in WGS84 ECEF
additional Earth rotation in G2: prohibited
```

Ground Visibility Model Version 1 accepts only ECEF satellite positions. TEME and ECI inputs fail. Stage G2 does not implement a generalized coordinate transformer, fallback frame conversion, or second sidereal rotation.

## Precision boundary

The project ECEF frame is produced by the existing `HypatiaAdapter` using a GMST Z-axis rotation from SGP4 TEME coordinates. It is not claimed to be a complete high-precision TEME-to-ITRF transformation and does not include full Earth-orientation parameters, polar motion, or UT1 corrections.

Satellite coordinates preserve the existing SGP4/WGS72 and project GMST-to-ECEF pipeline. Ground coordinates use WGS84 geodetic-to-ECEF conversion. Stage G2 does not claim sub-meter geodetic accuracy or complete ITRF compliance.

Changing the satellite frame conversion is a separate satellite-physics migration and is prohibited during Stage G2.

## Ground-coordinate contract

Ground stations use:

```text
latitude: geodetic latitude in degrees
longitude: east-positive geodetic longitude in degrees
altitude: WGS84 ellipsoidal height in metres
output: fixed WGS84 ECEF coordinates in kilometers
```

G1 catalog altitude is interpreted as WGS84 ellipsoidal height for Stage G2. Stage G2 does not claim orthometric or geoid-height precision.

Named WGS84 constants are centralized in the ground geometry module:

```text
semi-major axis: 6378.137 km
flattening: 1 / 298.257223563
first eccentricity squared: f × (2 - f)
model version: wgs84_geodetic_ecef_v1
```

## Timestamp contract

Every operational source snapshot has an explicit timestep index and UTC timestamp:

```text
timestamp = satellite_config.epoch + timestep_index × step_seconds
```

Timestamps must be timezone-aware and have UTC offset exactly zero. Canonical serialization is always:

```text
YYYY-MM-DDTHH:MM:SS.ffffffZ
```

Naive timestamps and nonzero UTC offsets fail. Wall-clock time, local time, and elapsed-time-only persistence are prohibited.

An operational source snapshot remains valid when it contains no positions because all satellites failed. Its timestep, timestamp, satellite configuration identity, and coordinate contract remain explicit.

## Operational-satellite contract

Canonical visibility evaluates only satellites remaining after the existing persistent failed-node realization is applied.

Failed satellite IDs are omitted from every operational position snapshot. Existing failed ISL edges are validated but do not affect geometric satellite-ground visibility. Stage G2 performs no new failure sampling.

Visibility position reconstruction uses the existing production adapter initialization and `get_positions_at_step()` path. It does not duplicate or replace orbital propagation.

## Geometry contract

For each selected ground station and operational satellite in one source snapshot, Stage G2 calculates:

- WGS84 station ECEF position.
- ECEF line-of-sight vector.
- Local East-North-Up components using station geodetic latitude and longitude.
- Euclidean slant range in kilometers.
- Elevation via `atan2(up, horizontal_range)`.
- Exact inclusive visibility decision `elevation_deg >= minimum_elevation_deg`.

No epsilon changes the visibility decision. `MIN_VALID_SLANT_RANGE_KM = 1.0e-9` is a computational-degeneracy threshold, not a physical horizon margin. `GEOMETRY_TEST_ABS_TOL = 1.0e-9` is used only by validation tests.

## Scope boundary

Stage G2 models geometric visibility only. A visible satellite-ground pair is not necessarily a viable communications link because antenna, RF, optical, atmospheric, terrain, interference, capacity, and scheduling effects are outside the current model.

No ground nodes or edges are added to NetworkX or PyG. No ground failures, service metrics, resilience labels, dataset migration, model features, or training are authorized.
