# Walker Delta ISL Topology Implementation Plan (Hypatia-Compatible)
**Goal:** turn the “Walker Delta 4‑ISL (+Grid) topology” report into code changes your agent can implement *atomically* and *defensibly* in a Hypatia-style simulator (NetworkX snapshots / forwarding state recompute).

This plan assumes you want:
- A deterministic topology builder for **Walker Delta I:T/P/F** constellations.
- A topology mode that supports: **(+Grid baseline)** → optional **seam-handling** → optional **viability-constrained / anti-flap** logic.
- A validation harness to enforce adjacency invariants continuously.

---

## 0) Scope + Non-goals
### Scope
1. Encode Walker Delta **indexing + phasing** consistently.
2. Implement **4‑ISL per sat** topology rules:
   - 2 intra-plane (front/rear, wrap)
   - 2 inter-plane (left/right) with phase mapping
   - seam exceptions (3‑ISL)
3. Add **physical feasibility filters** (range + LoS).
4. Add **stability/anti-flap** constraints (optional but strongly recommended).
5. Add a **validation harness** with invariants and time-series metrics.

### Non-goals
- Full PAT state machine simulation (SDA OISL). You can approximate acquisition delay via “link warmup” timers if needed, but it’s not required for connectivity labels.
- Full congestion / traffic engineering. This is about topology correctness + connectivity.

---

## 1) Data Model: canonical identifiers + constellation parameters

### Step 1.1 — Canonical satellite indexing
**Implement:**
- `plane_index p ∈ [0, P-1]`
- `slot_index i ∈ [0, S-1]` where `S = T / P`
- `sat_id = p*S + i` (or store `(p,i)` directly)

**Why:**
Everything else (neighbors, seam, phasing) becomes trivial and testable if indexing is canonical.

**Code sketch:**
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class WalkerParams:
    I_deg: float
    T: int
    P: int
    F: int  # 0..P-1
    @property
    def S(self) -> int:
        assert self.T % self.P == 0, "T must be divisible by P"
        return self.T // self.P

def sat_id(p: int, i: int, S: int) -> int:
    return p * S + i

def sat_pi(sid: int, S: int) -> tuple[int, int]:
    return sid // S, sid % S
```

### Step 1.2 — Centralize wrap helpers
**Implement:**
- `wrap_plane(p) = p % P`
- `wrap_slot(i) = i % S`

**Why:**
Avoid off-by-one bugs and keep adjacency rules readable.

---

## 2) Baseline Topology: 2 intra-plane links (+2 inter-plane links)

### Step 2.1 — Intra-plane neighbors
**Implement:**
For each satellite `S(p,i)`:
- `front = S(p, i+1)`
- `rear  = S(p, i-1)`  
with wrap in `i`.

**Why:**
This is the stable backbone; relative distance in-plane is ~constant.

**Code sketch:**
```python
def intra_neighbors(p: int, i: int, P: int, S: int) -> list[tuple[int,int]]:
    return [(p, (i-1) % S), (p, (i+1) % S)]
```

### Step 2.2 — Inter-plane “left/right” neighbor mapping (phase-aware)
You need a deterministic mapping from `(p,i)` to a slot index in adjacent planes. Two common choices:

**Option A (simple phase shift):**
- Adjacent plane offset = `Δ = (F * S) // P` (integer, works well when `S` multiple of `P`, otherwise you need fractional handling)
- `left  = S(p-1, i + Δ)`
- `right = S(p+1, i - Δ)` (or both `+Δ` depending on your convention)

**Option B (geometric nearest-neighbor):**
- Evaluate candidate slots around a predicted phase index (e.g., `i + Δ` and `i + Δ ± 1`) and pick the one with smallest distance that passes feasibility filters.

**Why:**
The report references “phase factor influences which sat is closest.” In practice, **Option B** is safer because it uses actual geometry (positions) and avoids brittle integer math when phasing doesn’t divide nicely.

**Recommended:**
- Implement **Option A** as a fast default.
- Implement **Option B** behind a flag: `neighbor_selection="phase_nearest"`.

**Code sketch:**
```python
def inter_candidate_slots(i: int, delta: int, S: int, radius: int = 1) -> list[int]:
    # candidates near i+delta: e.g., [i+delta, i+delta-1, i+delta+1]
    base = (i + delta) % S
    out = [base]
    for r in range(1, radius+1):
        out.append((base - r) % S)
        out.append((base + r) % S)
    return out

def inter_neighbors_phase(p: int, i: int, P: int, S: int, F: int) -> list[tuple[int,int]]:
    # simple integer delta; replace with a better mapping if needed
    delta = (F * S) // P
    left  = ((p - 1) % P, (i + delta) % S)
    right = ((p + 1) % P, (i - delta) % S)
    return [left, right]
```

### Step 2.3 — Enforce symmetric undirected edges
**Implement:**
- Build edges as undirected pairs `(min(u,v), max(u,v))`.
- If neighbor mapping isn’t symmetric by construction, enforce symmetry by adding edges both ways and deduping.

**Why:**
Symmetry is an invariant; link graphs should be undirected for connectivity metrics.

---

## 3) Seam handling (3‑ISL exception)

### Step 3.1 — Define seam semantics in your model
Walker Delta “seam” depends on how you label planes and whether adjacent planes are considered co-rotating vs counter-rotating in the network design. In many simulators, a seam is modeled as the boundary plane pair where cross-plane links are *not* established (or are established only conditionally).

**Implement a policy object:**
- `SeamPolicy.NONE`: no seam, all planes wrap adjacency normally.
- `SeamPolicy.SINGLE_SEAM`: one boundary (e.g., between planes `P-1` and `0`) treated as “counter-rotating boundary.”
- `SeamPolicy.CUSTOM`: explicit set of forbidden cross-plane plane-pairs.

**Why:**
You want seam logic explicit; otherwise it becomes hidden bugs.

**Code sketch:**
```python
from enum import Enum
class SeamPolicy(Enum):
    NONE = "none"
    SINGLE_SEAM = "single_seam"
    CUSTOM = "custom"

def is_seam_pair(p1: int, p2: int, P: int, policy: SeamPolicy, custom_pairs=None) -> bool:
    if policy == SeamPolicy.NONE:
        return False
    if policy == SeamPolicy.SINGLE_SEAM:
        # seam between P-1 and 0
        return {p1, p2} == {0, P-1}
    if policy == SeamPolicy.CUSTOM:
        assert custom_pairs is not None
        return (p1, p2) in custom_pairs or (p2, p1) in custom_pairs
    return False
```

### Step 3.2 — Apply seam rule to inter-plane links
**Implement:**
- When choosing left/right neighbor plane, if `(p, p±1)` is a seam pair, **skip that inter-plane edge**.
- That yields seam-edge satellites with **3 links** (2 intra + 1 inter).

**Why:**
Matches report behavior, and provides a clean knob for experiments.

---

## 4) Physical feasibility filters (range + LoS)

### Step 4.1 — Range filter
**Implement:**
- `distance(u,v) <= d_max_km` (default `6500 km` if you’re aligning to SDA upper bound, but keep configurable).

**Why:**
Prevents invalid edges; also critical for real-world plausibility.

### Step 4.2 — Earth obstruction / line-of-sight filter
**Implement:**
Given position vectors `xu`, `xv` in ECEF/ECI (consistent frame), LoS holds if:
- `||xu × xv|| / ||xu - xv|| > R_e` (Earth radius in same units)

**Why:**
This is the standard “chord misses Earth” test.

**Code sketch:**
```python
import numpy as np

def los_ok(xu: np.ndarray, xv: np.ndarray, R_e: float) -> bool:
    num = np.linalg.norm(np.cross(xu, xv))
    den = np.linalg.norm(xu - xv)
    return (num / den) > R_e
```

### Step 4.3 — Feasibility wrapper
**Implement:**
- `is_feasible(u,v,t) = within_range AND los_ok`
- If you don’t have positions per time-step in this module, pass a `PositionProvider` interface.

**Why:**
Separation of concerns: topology selection shouldn’t own orbital propagation.

---

## 5) Stability + anti-flap (viability-constrained model)

### Step 5.1 — Add “min link lifetime” constraint (optional but recommended)
**Implement:**
When selecting inter-plane links, require the candidate edge to remain feasible for:
- `min_lifetime_steps` (e.g., 60 seconds / N steps), or
- `min_lifetime_fraction_of_orbit` if you model orbits.

**Why:**
Stops link flapping and makes routing tables stable across snapshots.

**Practical approach:**
- For each candidate `(u,v)` at time `t`, check feasibility at `t, t+Δ, t+2Δ, ... t+L` (small L).
- Cache feasibility results to avoid O(T*V^2) blowups.

### Step 5.2 — Hysteresis-based handover
**Implement:**
If a sat already has an inter-plane link to neighbor `v_old`, only switch to `v_new` if:
- `metric(v_new) <= metric(v_old) * (1 - hysteresis_margin)` for `H` consecutive steps.

**Why:**
Prevents oscillation when two candidates have similar distances.

---

## 6) Topology Builder: single responsibility module

### Step 6.1 — Define a builder interface
**Implement:**
`WalkerTopologyBuilder.build_snapshot(t) -> nx.Graph`

Inputs:
- Walker params `I,T,P,F`
- seam policy
- physical constraints (`d_max`, `R_e`)
- selection strategy (phase-only vs phase-nearest)
- optional stability settings

**Why:**
You want a clean, testable unit with no side effects.

**Code sketch:**
```python
import networkx as nx

class WalkerTopologyBuilder:
    def __init__(self, params, position_provider, d_max_km, R_e_km,
                 seam_policy=SeamPolicy.NONE, neighbor_strategy="phase_only",
                 stability=None):
        self.params = params
        self.pos = position_provider
        self.d_max_km = d_max_km
        self.R_e_km = R_e_km
        self.seam_policy = seam_policy
        self.neighbor_strategy = neighbor_strategy
        self.stability = stability

    def build_snapshot(self, t) -> nx.Graph:
        P, S, F = self.params.P, self.params.S, self.params.F
        g = nx.Graph()
        # add nodes
        for p in range(P):
            for i in range(S):
                g.add_node(sat_id(p,i,S), plane=p, slot=i)
        # add edges
        for p in range(P):
            for i in range(S):
                u = sat_id(p,i,S)
                # intra-plane
                for (pp, ii) in intra_neighbors(p,i,P,S):
                    v = sat_id(pp,ii,S)
                    self._try_add_edge(g, u, v, t)
                # inter-plane
                for (pp, ii) in inter_neighbors_phase(p,i,P,S,F):
                    if is_seam_pair(p, pp, P, self.seam_policy):
                        continue
                    v = sat_id(pp,ii,S)
                    # if using "phase_nearest", refine v by checking nearby slots
                    v = self._select_best_inter_neighbor(u, p, i, pp, ii, t)
                    if v is not None:
                        self._try_add_edge(g, u, v, t)
        return g
```

---

## 7) Validation harness: invariants + quick diagnostics

You should treat these as **unit tests** and also as **runtime asserts** (configurable).

### Step 7.1 — Degree invariants
**Implement:**
- Non-seam sats: degree == 4
- Seam-edge sats: degree == 3 (if seam policy active)
- No sat: degree > 4

**Why:**
Catches almost every adjacency bug immediately.

**Code sketch:**
```python
def validate_degrees(g, params, seam_policy):
    P, S = params.P, params.S
    for sid in g.nodes:
        p, i = sat_pi(sid, S)
        deg = g.degree[sid]
        # seam-edge = has exactly one seam-adjacent plane side blocked
        if seam_policy != SeamPolicy.NONE and (p in (0, P-1)):
            # NOTE: depends on seam definition; adjust accordingly
            assert deg in (3,4), f"Unexpected deg at seam region: {sid} deg={deg}"
        else:
            assert deg == 4, f"Non-seam sat must be 4-ISL: {sid} deg={deg}"
        assert deg <= 4
```

### Step 7.2 — Distance + LoS invariants
**Implement:**
For every edge `(u,v)` at time `t`:
- `distance(u,v) <= d_max`
- `los_ok(u,v) == True`

**Why:**
Guards against geometry mismatch and coordinate-frame errors.

### Step 7.3 — Symmetry invariant
**Implement:**
NetworkX `Graph` enforces undirectedness, but ensure you never add directed edges in a DiGraph by accident.

### Step 7.4 — Connectivity invariants
**Implement:**
- `nx.is_connected(g)` should generally be true (for your design regime)
- track `nx.diameter(g)` periodically (expensive; sample snapshots)

**Why:**
Ensures the topology actually behaves as a mesh and supports the reliability labels.

### Step 7.5 — Temporal invariants (churn + flapping)
**Implement:**
Between snapshots `g_t` and `g_{t+1}`:
- `edge_churn = |EΔ| / |E|`  (symmetric difference ratio)
- `per-node churn`: number of changed incident edges

**Why:**
Directly quantifies stability and helps tune hysteresis/min-lifetime.

---

## 8) Integration points with your existing Tier1 label pipeline

### Step 8.1 — Unify “snapshot producer” contract
Your Tier1 labels need `G_t` snapshots. Ensure your topology builder outputs:
- `nx.Graph` with consistent node IDs
- edges optionally carry `weight` (distance) for shortest paths

**Why:**
Everything downstream (giant component, partition_any, temporal BFS, etc.) assumes consistent node identity.

### Step 8.2 — Add topology mode toggles
Expose flags so experiments are easy:
- `topology_mode = {"grid_phase", "grid_phase_nearest", "motif"}`
- `seam_policy = {"none", "single_seam", "custom"}`
- `stability = {"none", "min_lifetime", "hysteresis", "both"}`

**Why:**
You’ll inevitably iterate; make it config-driven.

---

## 9) Recommended “atomic” implementation sequence

### Phase A — correctness first (no physics)
1. Implement Walker indexing `(p,i)` and wrap helpers.
2. Implement intra-plane edges and validate degrees=2 per node initially.
3. Implement inter-plane edges via **simple phase mapping** (Option A).
4. Add seam policy (skip seam inter-plane links).
5. Add invariant tests: degree, symmetry, connectivity.

**Why:**
Quickly locks down adjacency correctness before expensive geometry.

### Phase B — add physics filters
6. Add position provider interface + compute distance per edge.
7. Add `d_max` filter.
8. Add LoS filter.
9. Re-run invariants and add edge validity tests.

**Why:**
Prevents invalid edges; avoids debugging physics + logic simultaneously.

### Phase C — stability controls
10. Add “phase_nearest” neighbor selection (Option B candidates).
11. Add min-lifetime feasibility sampling.
12. Add hysteresis handover and churn metrics.

**Why:**
This is where you kill link flapping and make temporal metrics meaningful.

### Phase D — hardening + perf
13. Cache positions per time-step.
14. Cache feasibility checks `(u,v,t)` in an LRU or dict keyed by timestep.
15. Add profiling hooks (time per snapshot, edges built, feasibility calls).

**Why:**
Without caching, naive neighbor selection can explode.

---

## 10) Deliverables checklist (what your agent should produce)

### Code
- `walker_params.py`: `WalkerParams`, indexing utilities.
- `walker_topology.py`: `WalkerTopologyBuilder`, seam policy, selection strategies.
- `physics_filters.py`: distance + LoS utilities, PositionProvider interface.
- `validate_topology.py`: invariant checks + churn metrics.
- `tests/test_walker_topology.py`: unit tests for indexing, degree invariants, seam behavior.

### Config
- `config/topology.yaml` (or JSON): modes and parameters.

### Outputs
- Optional: write snapshots to disk for offline inspection (pickle / edge list).
- Optional: export degree histograms and churn time series.

---

## 11) Small “gotchas” to proactively handle

1. **Phasing math**: `Δν = F*360/T` is continuous; integer slot offsets can be lossy.
   - Prefer candidate-search around predicted slot (phase_nearest).
2. **Seam identification**: depends on how you define “counter-rotating.”
   - Make it a policy, not a hard-coded assumption.
3. **Coordinate frames**: LoS formula assumes consistent frame + origin at Earth center.
   - If you’re in ECEF vs ECI, be consistent per `t`.
4. **Edge duplication**: if you do `u -> v` and later `v -> u`, use set/dedup.
5. **Diameter**: `nx.diameter` is expensive; sample or approximate on large graphs.

---

## 12) Minimal “Phase A” unit tests to write immediately

```python
def test_index_roundtrip():
    params = WalkerParams(I_deg=53, T=1600, P=32, F=1)
    S = params.S
    for p in range(params.P):
        for i in range(S):
            sid = sat_id(p,i,S)
            pp, ii = sat_pi(sid,S)
            assert (p,i) == (pp,ii)

def test_intra_plane_degrees_only():
    # Build only intra-plane edges
    # Expect degree=2 everywhere
    ...

def test_grid_4isl_degree_no_seam():
    # Build full +Grid without seam and without physics filters
    # Expect degree=4 everywhere (undirected)
    ...

def test_seam_reduces_degree():
    # With SINGLE_SEAM policy, seam-edge satellites should lose one inter-plane link
    ...
```

---

## Appendix: quick reference of invariants
- **Degree**: non-seam=4, seam-edge=3, max degree ≤ 4
- **Range**: `dist ≤ d_max`
- **LoS**: `||xu × xv|| / ||xu - xv|| > R_e`
- **Symmetry**: undirected edges
- **Connectivity**: `is_connected == True` (for your intended designs)
- **Temporal**: churn below threshold; per-node edge swaps stable

