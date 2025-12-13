# SATNET Tier1 Ground-Station Integration — Implementation Plan (Atomic Steps)

Goal: add **ground stations (gateways)** into the Tier1 simulation pipeline so you can compute **gateway reachability labels** and (optionally) **link viability** using a clear, testable geometry + link-budget stack.

This plan is designed to plug into:
- `SimulationEngine` time-stepped loop
- the Tier1 labeler (`satnet/metrics/labels.py`) that consumes `gateway_visible_per_t`

---

## Tier1 scope (what we implement now)

### Tier1 v1 (ship now)
- Ground station data model
- Per-timestep **geometric visibility** (elevation mask)
- Per-timestep `gateway_visible_sats[t]` sets
- Multi-source BFS reachability to gateway-visible sats (already in labeler plan)
- Dataset export of gateway-related labels

### Tier1 v1.5 (optional)
- Add **range limit** / simple viability filter:
  - `visible AND slant_range <= max_gsl_range_km`

### Tier1 v2+ (defer)
- Full link budget (EIRP, G/T, C/No, FSPL, weather losses)
- Earth terrain masks per-azimuth
- Event-driven access intervals (rise/set) to avoid per-step checks
- Gateway placement optimization / clustering / Hungarian

---

## Design decision: don’t introduce new orbital propagators
You already have a Hypatia-based constellation representation. For Tier1, reuse whatever your adapter is already producing for sat positions per timestep (ECI/ECEF). Only add **station geometry** + **topocentric transform**.

---

## Core interfaces (minimum set)

### Ground station model
Create `src/satnet/ground/models.py`:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class GroundStation:
    gs_id: str
    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0
    min_elevation_deg: float = 10.0

    # v1.5/v2 params
    max_range_km: float | None = None
    # link budget fields (later):
    eirp_dbw: float | None = None
    gt_db_per_k: float | None = None
    cno_req_dbhz: float | None = None
    freq_hz: float | None = None
```

**Why:** freeze + simple scalar fields make it hashable and stable for caching + config hashing.

---

## Atomic implementation steps

### Step 0 — Define gateway config + defaults
**Why:** Tier1 datasets must be reproducible; gateway logic must be parameterized.

Add to your run config (engine params or a config object):
- `gateways: List[GroundStation]`
- `min_elevation_deg` default (per station overrides ok)
- `gsl_max_range_km` (optional)
- `earth_radius_m` constant (for spherical model; fine for Tier1)

---

### Step 1 — Add adapter hooks for satellite positions (per timestep)
**Why:** visibility requires sat position vectors.

In your Hypatia adapter (or wrapper), implement:

```python
def get_sat_positions_ecef_m(self, t: int) -> dict[str, tuple[float, float, float]]:
    # Return ECEF position (x,y,z) in meters for each satellite at step t.
    ...
```

If you already store sat positions in ECI, also provide:
- `get_sat_positions_eci_m(t)` and `eci_to_ecef(t, vec)`.

**Reasoning:** ECEF is easiest for topocentric transforms at a fixed ground station.

---

### Step 2 — Implement ECEF → ENU topocentric visibility test
**Why:** elevation mask is the Tier1 workhorse.

Create `src/satnet/ground/visibility.py` with two functions:

1) `geodetic_to_ecef(lat, lon, alt)` (WGS84 or spherical)
2) `ecef_to_enu(gs_ecef, sat_ecef, lat, lon)` and elevation

**Tier1 v1 simplification: spherical Earth** (fast, acceptable for network-level labels).
If you want WGS84 later, swap implementation.

**Implementation (drop-in):**
```python
import math

R_EARTH_M = 6371000.0

def geodetic_to_ecef_spherical(lat_deg: float, lon_deg: float, alt_m: float):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    r = R_EARTH_M + alt_m
    x = r * math.cos(lat) * math.cos(lon)
    y = r * math.cos(lat) * math.sin(lon)
    z = r * math.sin(lat)
    return (x, y, z)

def ecef_to_enu(gs_ecef, sat_ecef, lat_deg: float, lon_deg: float):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    dx = sat_ecef[0] - gs_ecef[0]
    dy = sat_ecef[1] - gs_ecef[1]
    dz = sat_ecef[2] - gs_ecef[2]

    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    # ECEF -> ENU rotation
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat*cos_lon * dx - sin_lat*sin_lon * dy + cos_lat * dz
    u =  cos_lat*cos_lon * dx + cos_lat*sin_lon * dy + sin_lat * dz
    return (e, n, u)

def elevation_deg_from_enu(enu):
    e, n, u = enu
    horiz = math.sqrt(e*e + n*n)
    return math.degrees(math.atan2(u, horiz))

def slant_range_km(gs_ecef, sat_ecef):
    dx = sat_ecef[0] - gs_ecef[0]
    dy = sat_ecef[1] - gs_ecef[1]
    dz = sat_ecef[2] - gs_ecef[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz) / 1000.0
```

**Visibility check:**
- `visible = (elevation_deg >= gs.min_elevation_deg)`

**v1.5 viability check:**
- `visible and (gs.max_range_km is None or slant_range_km <= gs.max_range_km)`

---

### Step 3 — Implement gateway-visible satellites per timestep
**Why:** labeler consumes `Set[sat_id]` per timestep (sources for multi-source BFS).

Create `src/satnet/ground/gateway_visibility.py`:

```python
from typing import Iterable, List, Set
from satnet.ground.models import GroundStation
from satnet.ground.visibility import (
    geodetic_to_ecef_spherical, ecef_to_enu, elevation_deg_from_enu, slant_range_km
)

def gateway_visible_sats_at_step(adapter, t: int, gateways: List[GroundStation]) -> Set[str]:
    sat_pos = adapter.get_sat_positions_ecef_m(t)
    visible = set()

    # precompute station ECEF once per run (store in station struct or cache)
    for gs in gateways:
        gs_ecef = geodetic_to_ecef_spherical(gs.lat_deg, gs.lon_deg, gs.alt_m)
        for sat_id, sat_ecef in sat_pos.items():
            enu = ecef_to_enu(gs_ecef, sat_ecef, gs.lat_deg, gs.lon_deg)
            el = elevation_deg_from_enu(enu)
            if el >= gs.min_elevation_deg:
                if gs.max_range_km is not None:
                    if slant_range_km(gs_ecef, sat_ecef) > gs.max_range_km:
                        continue
                visible.add(sat_id)
    return visible

def iter_gateway_visible_sats(adapter, gateways: List[GroundStation], num_steps: int) -> Iterable[Set[str]]:
    for t in range(num_steps):
        yield gateway_visible_sats_at_step(adapter, t, gateways)
```

**Why this design:** trivial to test; no dependency on networkx; only needs sat positions.

---

### Step 4 — Performance: avoid O(G * V * T) where possible
The naive loop above is `O(num_gateways * num_sats)` per timestep. For Tier1 sizes, it may be fine. But you’ll want headroom.

Add these optimizations in order:

1) **Precompute station ECEF** for each gateway once.
2) **Vectorize** sat position arrays using NumPy (big speedup).
3) **Coarse culling**: quick horizon check before full ENU rotation (optional).
4) **Spatial partitioning**: bucket satellites by subpoint lat/lon (Tier1 v2).
5) **Event-driven access windows** (Tier1 v2): compute rise/set times and fill sets per step.

For Tier1 v1, do (1) and optionally (2).

---

### Step 5 — Integrate into SimulationEngine
**Why:** Tier1 run output should include gateway metrics and labels.

In `SimulationEngine.run()` (or similar):
1) compute `num_steps`
2) compute effective graphs per step (after failures)
3) compute `gateway_visible_per_t = iter_gateway_visible_sats(adapter, gateways, num_steps)`
4) pass into labeler

Example wiring:
```python
from satnet.ground.gateway_visibility import iter_gateway_visible_sats
from satnet.metrics.labels import label_run, LabelConfig

gw_vis = list(iter_gateway_visible_sats(self.adapter, self.gateways, self.num_steps))
graphs = list(self.iter_effective_graphs())

labels = label_run(graphs, sat_nodes, gw_vis, LabelConfig(...))
```

---

### Step 6 — Add ground stations into the network graph (optional)
**Why:** Not required for current Tier1 reachability labels (which treat gateway-visible sats as “sources”). But if you want end-to-end paths GS→GS, you need actual GS nodes.

Tier1 v1: **skip** (simplifies graph sizes and avoids recomputing route tables).
Tier1 v2: add GS nodes to `G_t` and connect edges:
- edge `(gs_id, sat_id)` exists if sat is visible/viable
- then you can measure GS-to-GS connectivity and path length.

---

### Step 7 — Dataset export schema (gateway fields)
Add columns:
- `num_gateways`
- `elevation_mask_deg_default` (and/or per station hash)
- `gsl_max_range_km` (if used)
- Labels:
  - `avg_reachable_sat_frac`
  - `min_reachable_sat_frac`
  - `reachability_fraction` (per-sat SLA pass rate)

Also export a stable `gateways_hash`:
- sha256 of sorted JSON of gateway definitions.

---

### Step 8 — Tests (must-have)
Create `tests/test_gateway_visibility.py`:

1) **Elevation math sanity**
- put GS at equator lon=0
- put sat directly above it: sat_ecef = gs_ecef scaled outward
- expect elevation ~90°

2) **Below-horizon**
- put sat on opposite side of Earth (negated vector)
- expect elevation < 0

3) **Range filter**
- set `max_range_km` low; ensure visible but rejected by range

4) **Integration**
- one timestep, one graph, one visible source; labeler reachable_frac should be 1.0 for connected graph

---

## Full link budget (defer) — what to implement when ready

When you move from “visible” to “viable”:
1) compute slant range `d`
2) compute FSPL(d,f)
3) `C/No = EIRP + G/T - FSPL - k` (plus additional losses)
4) viable if `C/No >= CNo_req`

Implement in `src/satnet/ground/link_budget.py` and feature-flag it.

---

## Definition of done checklist

- [ ] GroundStation model exists and is config-hashable
- [ ] Adapter provides sat positions per timestep (ECEF)
- [ ] `iter_gateway_visible_sats(...)` works and is tested
- [ ] Tier1 engine produces gateway reachability labels (non-zero on plausible configs)
- [ ] Export includes gateway config + hashes for reproducibility
