# SATNET Tier1 Time‑Varying Connectivity — Implementation Plan (Atomic Steps)

Goal: implement **time‑varying ISL connectivity analysis** in the Tier1 path (Hypatia adapter + simulation engine) in a way that is:
- defensible (temporal topology, not a t=0 snapshot)
- compute‑feasible (no O(V*T) DAG build unless explicitly opted-in)
- compatible with your current codebase patterns (`HypatiaAdapter`, `SimulationEngine`, `export_design_dataset.py`)

This plan focuses on **Tier1** only.

---

## What this plan is (and is not)

**This plan IS:**
- a concrete implementation plan for time‑stepped graphs and time‑aggregated reliability metrics
- designed to plug into the labeler module you already planned (`satnet/metrics/labels.py`)

**This plan is NOT:**
- an implementation of PICCNIC, all‑pairs temporal efficiency, or full time‑expanded DAG routing (those are Tier1 v2/v3)

---

## Key constraint: don’t build the time-expanded DAG (yet)

The report mentions static expansion / DTEG and all‑pairs temporal metrics. Those are valid but will explode memory/runtime quickly:
- time‑expanded DAG nodes: `|V| * T`
- time‑expanded DAG edges: `~(|E| + |V|) * T`

For Tier1 v1, you can ship *defensible temporal labels* using only per‑step BFS/CC with **O(T*(V+E))**.

If/when you need temporal journeys, implement them as **streaming temporal BFS** (source‑limited), not full DAG.

---

## The minimal Tier1 temporal metrics to implement now

These metrics are compute‑feasible and directly useful:

### M1) Partition probability / partition fraction
Per step:
- compute connected components (sat subgraph)
- `is_partitioned[t] = (gcc_frac[t] < gcc_threshold)`
Aggregate:
- `partition_fraction = mean(is_partitioned)`
- `partition_any = any(is_partitioned)`
- `max_partition_streak_steps`
- `total_partition_steps`

### M2) Worst-case *network* disconnect duration proxy (NOT all-pairs)
The report defines worst‑case over all pairs (very expensive). Instead implement:
- `max_partition_streak_steps` as your Tier1 v1 “worst outage” proxy.
Why: if the network is partitioned, *some* pairs are disconnected.

If you still want “pair‑level” worst disconnect later:
- sample K random satellite pairs and track streaks per pair (Tier1 v2).

### M3) Gateway reachability over time
Per step:
- compute gateway-visible satellites (`gw_vis[t]`)
- multi‑source BFS on sat graph from `gw_vis[t]`
Aggregate:
- `avg_reachable_sat_frac`, `min_reachable_sat_frac`, per-sat SLA fraction.

---

## Atomic implementation steps

### Step 0 — Add config for temporal stepping
**Why:** Tie all temporal behavior to explicit config to avoid accidental `t=0` regressions.

Create/extend a config object (or kwargs) that includes:
- `duration_minutes`
- `timestep_seconds` (recommended: 1–10 seconds for analysis; not 100ms)
- `num_steps = floor(duration_minutes*60 / timestep_seconds)`

Where to place:
- `src/satnet/simulation/engine.py` constructor and/or `run()`.

---

### Step 1 — Ensure HypatiaAdapter supports per-step graph extraction
**Why:** The engine should iterate `t` and request `G_t` each step.

In `src/satnet/hypatia/adapter.py` (or wherever `HypatiaAdapter` lives):
- confirm / implement:
  - `calculate_isls(duration_minutes, timestep_seconds)` (precompute state)
  - `get_graph_at_step(t)` (returns `networkx.Graph`)

**Hard requirement:** `get_graph_at_step(t)` must be deterministic and cheap-ish.
If it’s expensive, add caching in Step 4.

---

### Step 2 — Implement a graph iterator in SimulationEngine
**Why:** Centralize temporal stepping and prevent “use step 0” mistakes.

In `src/satnet/simulation/engine.py`, implement:

```python
def iter_graphs(self):
    for t in range(self.num_steps):
        yield self.adapter.get_graph_at_step(t)
```

Then ensure `run()` uses `iter_graphs()`.

---

### Step 3 — Decide ISL model semantics: static vs dynamic
**Why:** The report claims Hypatia’s default +Grid is static. Your adapter may already compute dynamic ISLs.

Pick **one** for Tier1 v1:
- **Option A (recommended):** keep Hypatia’s static +Grid **BUT** treat the network as time-varying only via failures + (optionally) gateways.  
  *Pros:* cheap, deterministic.  
  *Cons:* misses geometry-driven ISL changes.

- **Option B (recommended if feasible in your adapter):** compute ISLs dynamically per step based on sat positions and constraints (range + LoS).  
  *Pros:* more “Tier1.”  
  *Cons:* potentially O(V^2) per step if naive.

If you choose Option B, implement **neighbor selection** so it is NOT O(V^2):
- per satellite, select candidate neighbors:
  - intra-plane neighbors: prev/next index (O(V))
  - inter-plane neighbors: nearest in adjacent planes (O(V)) using indexing, not full search
  - optional seam handling: deterministic mapping
- then filter candidates by range + LoS.

This preserves Starlink-like adjacency without quadratic scanning.

---

### Step 4 — Add temporal graph caching (edge lists, not full graphs)
**Why:** Storing `networkx.Graph` for every step can be RAM-heavy. You can store edge lists and reconstruct graphs on demand.

Implement in adapter (or engine):
- `edges_by_t: List[List[Tuple[u,v]]]`
- `nodes: List[u]` (constant)

**Pattern:**
1) During `calculate_isls(...)`, compute and store `edges_by_t`.
2) `get_graph_at_step(t)` builds a graph from `nodes` + `edges_by_t[t]`.

If graph build overhead becomes a problem, keep a single graph instance and mutate edges, but that’s more error-prone. Start with reconstruction.

---

### Step 5 — Add gateway visibility provider (streaming)
**Why:** Gateway reachability labels require `gw_vis[t]` per step.

Create `src/satnet/ground/gateway_visibility.py`:

**Interface:**
```python
def iter_gateway_visible_sats(adapter, gateways, num_steps):
    for t in range(num_steps):
        yield adapter.get_gateway_visible_sats_at_step(t, gateways)
```

Implementation options:
- **v1 stub:** always yield `set()`.
- **v1 real:** implement spherical-earth visibility with elevation mask:
  - get satellite ECEF (or geodetic) at step t
  - compute elevation angle from gateway to sat
  - visible if elevation >= mask (e.g., 10°)

If your adapter already has positions, use them. If not, add:
- `adapter.get_sat_positions_at_step(t)` returning ECEF or lat/lon/alt.

---

### Step 6 — Integrate temporal metrics into the Tier1 labeler (no DAG)
**Why:** Keep the computation O(T*(V+E)).

Use the labeler module from the previous plan (`satnet/metrics/labels.py`) and pass:
- `graphs = iter_effective_graphs()` (after failures)
- `sat_nodes = [...]`
- `gateway_visible_per_t = iter_gateway_visible_sats(...)`

**Important:** keep both iterables aligned (same number of steps).
If you need multiple passes, materialize to lists once per run.

---

### Step 7 — Add optional “temporal reachability” (source-limited) without building DAG
**Why:** Sometimes you want a metric like “from a random sat, how long until it can reach a gateway over time.” Do it without a DAG.

Implement a **streaming foremost-reachability** for a *small number of sources*:
- maintain `arrival_time` dict for nodes
- iterate time steps; when a node becomes reachable at time t, propagate on `G_t` to mark neighbors reachable at t (or t+1 depending on your semantics)

Keep this feature-gated:
- `enable_temporal_journeys: bool = False`
- only run for K sampled sources (`K <= 10`)

This avoids O(V*T) graph expansion.

---

### Step 8 — Export dataset fields that encode temporal configuration
**Why:** Temporal metrics depend heavily on timestep selection. Without exporting timestep/duration, your dataset is not reproducible.

Add to Tier1 export:
- `duration_minutes`
- `timestep_seconds`
- `num_steps`
- `gcc_threshold`
- gateway config: `gateway_count`, `elevation_mask_deg` (and location hash if using real coords)

---

### Step 9 — Validation tests / invariants (minimum viable)
**Why:** Temporal bugs are easy to introduce. Add tests that confirm the engine uses all timesteps and labels are sane.

Create tests:

1) **Engine stepping**
- mock adapter with `get_graph_at_step(t)` returning a different graph each t
- assert engine iterates all steps

2) **Partition fraction**
- feed graphs where first half are connected, second half disconnected
- partition_fraction should be ~0.5 given threshold

3) **Reachability**
- provide gw_vis on alternating steps and a connected graph
- avg reachable frac should match

---

## Complexity & practicality notes (to guide choices)

### Snapshot interval
- 1s to 10s is usually enough for topology-level analysis.
- 100ms is overkill unless you’re doing packet-level routing.

### Memory
Avoid storing full graph objects per step. Store `edges_by_t` or derive per step.

### If you need high fidelity ISLs
Prefer deterministic adjacency rules + range/LoS checks over brute-force pairwise feasibility.

---

## Optional Tier1 v2+ upgrades (defer)

### U1) Window-union connectivity time
Define a window W steps:
- `U_t = union(G_t..G_{t+W-1})`
- compute whether U_t is connected (or gcc threshold)
Then:
- `connectivity_time_W = min W such that U_t connected for all t` (or percentile)

Start with fixed W (e.g., 60s window) and compute a score; don’t solve for minimal W globally.

### U2) Link churn
Compute:
- `churn[t] = |E_t Δ E_{t-1}| / |E_{t-1}|`
Aggregate mean/max.
Useful for routing stability.

### U3) Pair-level disconnect sampling
Sample K sat pairs; track disconnect streaks based on component membership per step.
This approximates the report’s all-pairs worst-case.

---

## Definition of done checklist

- [ ] Tier1 engine iterates `t=0..num_steps-1` and does not rely on `get_graph_at_step(0)` only
- [ ] Edge storage is not O(T * full graph object overhead) unless explicitly enabled
- [ ] Labeler computes `partition_fraction` and `max_partition_streak_steps` from per-step GCC
- [ ] Optional gateway reachability works with a stub (and later real geometry)
- [ ] Export includes temporal config columns and seed/config hash
- [ ] Tests confirm stepping + metric sanity

