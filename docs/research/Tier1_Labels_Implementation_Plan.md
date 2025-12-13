# SATNET Tier1 Reliability Labels — Implementation Plan (Atomic Steps)

Goal: upgrade the **Tier1** path to compute **defensible, time-varying reliability labels** from Hypatia time-stepped graphs (and optional gateway visibility), then export a **single canonical dataset schema** for training.

This plan intentionally ignores Tier0/toy code paths.

---

## Outcomes (what “done” looks like)

### A) Tier1 engine uses time steps (not just t=0)
- Uses `adapter.calculate_isls(...)` and then iterates through `t = 0..T-1`
- Produces per-step graphs `G_t` (satellite-only to start)

### B) Tier1 labels are computed from time-stepped graphs
Implement and export these run-level labels (fast + defensible):
- `giant_component_fraction_avg`
- `giant_component_fraction_min`
- `partition_fraction`
- `partition_any`
- `total_partition_steps`
- `max_partition_streak_steps`
- `avg_reachable_sat_frac` *(requires gateways)*
- `min_reachable_sat_frac` *(requires gateways)*
- `reachability_fraction` *(fraction of sats meeting per-sat SLA over time; requires gateways)*

### C) Dataset export is deterministic and reproducible
- Each sample includes a seed + config hash
- Grouped splitting is possible by design id

---

## Design decisions (why this approach)

### Why time-stepped graphs?
Tier1 is about **orbital-driven, time-varying topology**. A snapshot at t=0 is not Tier1; it collapses the problem into a static graph.

### Why these labels (and not k-connectivity / time-expanded DAG)?
- k-connectivity (exact) requires min-cut/max-flow over many node pairs → too slow for large constellations.
- time-expanded DAG temporal diameter explodes graph size O(V*T).
- The chosen label set is **O(T*(V+E))** and produces meaningful outage / reachability signals.

### Why `partition = (gcc_frac < threshold)`?
It’s the simplest partition definition that:
- matches your current sat-only graph representation
- avoids the incorrect “any failure means partition” bug
- can be tuned via a mission parameter (`gcc_threshold`)

### Why multi-source BFS for gateway reachability?
Reachability “to any gateway” is naturally computed by BFS from all gateway-visible satellites at each timestep. That’s **one traversal per step**, not `V` traversals.

---

## Atomic implementation steps

### Step 0 — Baseline: create a new labels module
**Why:** Keep labeling logic pure, testable, and independent of simulation engine details.

Create: `src/satnet/metrics/labels.py`

Add:
- `LabelConfig` dataclass
- `LabelResults` dataclass
- `label_run(graphs, sat_nodes, gateway_visible_per_t, cfg)` function
- helper `_longest_true_streak`

> This module should accept an iterable of graphs and gateway-visible sets; it should not know anything about Hypatia.

**Code skeleton (drop-in):**
```python
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set
import networkx as nx

@dataclass
class LabelConfig:
    gcc_threshold: float = 0.99
    gateway_time_requirement: float = 0.95  # X in [0,1]

@dataclass
class LabelResults:
    giant_component_fraction_avg: float
    giant_component_fraction_min: float
    partition_fraction: float
    partition_any: int
    total_partition_steps: int
    max_partition_streak_steps: int
    avg_reachable_sat_frac: float
    min_reachable_sat_frac: float
    reachability_fraction: float  # fraction sats meeting X over time

def _longest_true_streak(bs: List[bool]) -> int:
    best = cur = 0
    for b in bs:
        if b:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

def label_run(
    graphs: Iterable[nx.Graph],
    sat_nodes: List[str],
    gateway_visible_per_t: Iterable[Set[str]],
    cfg: LabelConfig,
) -> LabelResults:
    n = len(sat_nodes)
    gcc_fracs: List[float] = []
    is_part: List[bool] = []
    reachable_fracs: List[float] = []
    reach_counts = {s: 0 for s in sat_nodes}
    steps = 0

    for G, gw_vis in zip(graphs, gateway_visible_per_t):
        steps += 1
        H = G.subgraph(sat_nodes)

        # GCC fraction (sat-only)
        if H.number_of_nodes() == 0 or n == 0:
            gcc_frac = 0.0
        else:
            gcc_size = max((len(c) for c in nx.connected_components(H)), default=0)
            gcc_frac = gcc_size / n
        gcc_fracs.append(gcc_frac)

        part = (gcc_frac < cfg.gcc_threshold)
        is_part.append(part)

        # Multi-source reachability to any gateway-visible sat
        sources = [s for s in gw_vis if s in H]
        if not sources:
            reachable = set()
        else:
            reachable = set()
            for src in sources:
                reachable.update(nx.node_connected_component(H, src))

        rf = (len(reachable) / n) if n else 0.0
        reachable_fracs.append(rf)
        for s in reachable:
            reach_counts[s] += 1

    meets = 0
    if steps and n:
        req = cfg.gateway_time_requirement
        meets = sum(1 for s in sat_nodes if (reach_counts[s] / steps) >= req)

    return LabelResults(
        giant_component_fraction_avg=(sum(gcc_fracs) / steps) if steps else 0.0,
        giant_component_fraction_min=min(gcc_fracs) if gcc_fracs else 0.0,
        partition_fraction=(sum(is_part) / steps) if steps else 0.0,
        partition_any=int(any(is_part)),
        total_partition_steps=sum(is_part),
        max_partition_streak_steps=_longest_true_streak(is_part),
        avg_reachable_sat_frac=(sum(reachable_fracs) / steps) if steps else 0.0,
        min_reachable_sat_frac=min(reachable_fracs) if reachable_fracs else 0.0,
        reachability_fraction=(meets / n) if n else 0.0,
    )
```

---

### Step 1 — Make SimulationEngine iterate over time
**Why:** Tier1 must compute labels over `G_t` for `t=0..T-1`, not snapshot `t=0`.

In: `src/satnet/simulation/engine.py`
- replace `self.network_graph = adapter.get_graph_at_step(0)` usage with a graph generator

**Implementation sketch:**
1. Keep `adapter.calculate_isls(duration_minutes=..., timestep_seconds=...)`.
2. Add a method or inline generator:

```python
def iter_graphs(self):
    num_steps = int(self.duration_minutes * 60 / self.timestep_seconds)
    for t in range(num_steps):
        yield self.adapter.get_graph_at_step(t)
```

3. Ensure `get_graph_at_step(t)` is cheap enough (or add caching later).

---

### Step 2 — Decide failure persistence semantics (pick one)
**Why:** Label computation depends on whether failures persist or reset each timestep.

Recommended Tier1 v1 (simple + realistic):
- **Node failures are persistent** (sat dies, stays dead).
- **Edge availability is per-step** (ISLs can drop due to pointing/range/weather).

**Action:**
- Implement a `FailureState` sampled once per run:
  - `dead_sats: Set[node]`
  - optionally `dead_edges_static: Set[(u,v)]` (if you want static edge failures too)

In each timestep:
- `G_t_eff = G_t.copy()`
- remove `dead_sats` nodes
- apply per-step edge failures if you already model them (optional)

This keeps Monte Carlo meaningful without temporal graphs becoming inconsistent.

---

### Step 3 — Create gateway visibility provider (stub-first)
**Why:** Reachability labels require `gateway_visible_per_t`, but you can start with a stub to ship the labeler.

Create: `src/satnet/ground/gateway_visibility.py`

**v1 stub options:**
- **Option A (fast unblock):** return empty sets for all steps. This makes reachability metrics 0 and keeps the pipeline running.
- **Option B (better v1):** implement spherical-earth geometry + elevation mask (requires sat ECI/ECEF positions per step).

**Interface:**
```python
def gateway_visible_sats_per_timestep(adapter, gateways, num_steps) -> list[set[str]]:
    ...
```

Where each set contains satellite node IDs that have direct LOS to at least one gateway at that timestep.

---

### Step 4 — Wire engine → labeler
**Why:** Centralize metric computation in one place; `SimulationEngine.run()` should output a dict with metrics + labels.

In `SimulationEngine.run()`:
1. Build `graphs = iter_graphs()` that yields **effective** graphs after applying failures.
2. Compute `sat_nodes` list (stable ordering).
3. Compute `gateway_visible_per_t` via the provider (or stub).
4. Call `label_run(...)`.
5. Return:
   - raw per-run metrics (optional)
   - label fields from `LabelResults`

**Example:**
```python
from satnet.metrics.labels import label_run, LabelConfig
from satnet.ground.gateway_visibility import gateway_visible_sats_per_timestep

cfg = LabelConfig(gcc_threshold=0.99, gateway_time_requirement=0.95)
graphs = list(self.iter_effective_graphs())  # or keep as generator; zip needs finite iterables
gw_vis = gateway_visible_sats_per_timestep(self.adapter, self.gateways, len(graphs))
labels = label_run(graphs, sat_nodes, gw_vis, cfg)

return {**design_params, **labels.__dict__, "seed": seed, ...}
```

> Note: `zip(graphs, gw_vis)` requires both iterables to align. If you want streaming, implement `gw_vis` as a generator too.

---

### Step 5 — Canonical dataset schema for Tier1
**Why:** You need a single dataset format so training and evaluation don’t drift.

Create or update exporter script (your repo likely has `scripts/export_design_dataset.py`):
- output: `data/tier1_design_dataset.parquet` (preferred) or `.csv`

**Recommended columns:**

**Design params**
- `num_planes`
- `sats_per_plane`
- `inclination_deg`
- (add as available) `altitude_km`, `phasing`, `isl_mode`, `max_isl_range_km`

**Simulation config**
- `duration_minutes`
- `timestep_seconds`
- `failure_model` (string)
- `p_node_fail` / or distribution params
- `p_edge_fail_base` / or link-margin parameters
- `gcc_threshold`
- `gateway_time_requirement`

**Labels**
- all fields from `LabelResults`

**Reproducibility**
- `seed`
- `code_version` (git SHA if available)
- `config_hash` (sha256 of JSON config)

---

### Step 6 — Add tests (minimum viable)
**Why:** Prevent regressions and ensure label semantics are correct.

Create `tests/test_labels.py`:

1) **GCC fraction sanity**
- graph fully connected ⇒ gcc_frac = 1.0
- graph split into two equal halves ⇒ gcc_frac = 0.5

2) **Partition streak**
- `is_part = [F, T, T, F, T]` ⇒ max streak = 2

3) **Gateway reachability**
- If gateway-visible sources exist and graph connected ⇒ reachable_frac = 1.0
- If no sources ⇒ reachable_frac = 0.0

4) **SLA requirement**
- if sat reachable 95/100 steps ⇒ meets 0.95 threshold

---

### Step 7 — Replace/remove stale Tier0 “partitioned” logic
**Why:** Avoid confusing output and accidental usage.

Search for:
- `partitioned = int((impact.num_components_after > ...) or ...)`

Either:
- delete Tier0 usage, or
- quarantine it in a `tier0/` directory, or
- rename to `toy_partition_proxy` so it can’t be mistaken for Tier1.

---

## Implementation order (fastest to value)

1) Add `metrics/labels.py` (pure)
2) Update Tier1 engine to iterate timesteps
3) Stub gateway visibility (return empty sets)
4) Wire labeler into Tier1 run output + exporter
5) Add tests
6) Improve gateway visibility (geometry) + link viability
7) Upgrade failures to correlated + margin-conditioned

---

## “Definition of done” checklist

- [ ] Tier1 dataset export produces non-zero `giant_component_fraction_*` metrics across runs
- [ ] `partition_any` is not trivially correlated with “any failure happened”
- [ ] Unit tests for labeler pass
- [ ] Output schema includes seed + config hash
- [ ] Engine uses all timesteps (not t=0)

---

## Notes for later (Tier1 v2+)

- Add **windowed union connectivity** (“connectivity time”) as a cheap temporal metric:
  - For a fixed window `W`, compute connectivity of union graph `U_t = ⋃_{i=t..t+W-1} G_i`
- Add **approx k-connectivity**:
  - sample node pairs + approximate min-cut
- Add **routing metrics** (bottlenecks):
  - demand model + shortest-path edge counts, or a fast multi-commodity approximation

