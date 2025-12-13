# Unified Refactor Plan — Tier 1 Temporal Simulation + Satellite-only GCC Labels (v1) with v2+ Roadmap

## Purpose
Create a single, executable plan that:
- Solves the current repo problem: **eliminate the “Two Worlds” pipeline split** by making the **Tier 1 (Hypatia-based) temporal pipeline** the only supported simulation/data path in `src/`.
- Keeps **strict v1 scope** (temporal satellite-only connectivity + GCC labels) so work can land quickly and safely.
- Provides a **broader v2+ roadmap** (ground stations, correlated failures, probabilistic ISL availability, traffic/routing) in the same document.
- Is written in **atomic steps** that a junior engineer can execute as small PRs.

## Scope

### v1 (strict scope)
- Time-stepped simulation loop: `t = 0..T` at fixed `step_seconds`.
- Tier 1 topology provider: `src/satnet/network/hypatia_adapter.py`.
- **Satellite-only connectivity labels** based on GCC fraction.
- Canonical dataset export contract (schema-stable; reproducible with config + seed).

### Non-goals (explicitly deferred until v2+)
- Ground stations / gateways / service availability labels.
- Probabilistic ISL availability models (weather/pointing/margin → drop probability).
- Correlated failure regimes (plane-level CCF, storms, etc.).
- Traffic / routing / throughput labels.

## Guiding principles (must follow)
- **Tier 1 only in `src/`**: production code must not depend on toy topology generation.
- **Time drives the simulation**: no “compute Tier1 then only use `t=0`”.
- **Determinism**: every dataset row must be reproducible from config + seed + code version.
- **Separation of concerns**:
  - HypatiaAdapter: time-varying topology
  - Failure sampling/process: persistent and/or per-step failures
  - Labeling: pure functions from graphs/time-series → labels
  - Export: schema-stable writer with validation
- **Atomic steps**: each step is a small PR that compiles and has a verification check.

## Baseline (current state)
- Tier 1 graph provider exists: `satnet.network.hypatia_adapter.HypatiaAdapter` supports `calculate_isls()` and `get_graph_at_step(t)`.
- Current engine is effectively `t=0`: `satnet.simulation.engine.SimulationEngine` loads `get_graph_at_step(0)` into a static snapshot.
- “Toy world” still exists: `satnet.network.topology` and `satnet.simulation.monte_carlo.py` depend on it.

## v1 Target State (definition of done)
- No `src/` module imports `satnet.network.topology`.
- Temporal rollout produces per-step metrics:
  - `gcc_frac[t] = |GCC(G_t)| / |V_t|`
  - `partitioned[t] = gcc_frac[t] < gcc_threshold`
- Per-run aggregate labels exist (examples):
  - `gcc_frac_min`, `gcc_frac_mean`
  - `partition_fraction`, `partition_any`, `max_partition_streak`
- `scripts/` workflows use the Tier 1 dataset generator.
- Rerunning with the same config + seed reproduces identical outputs.

## Architecture (v1 core + v2+ extension points)

```mermaid
graph TD
    A[HypatiaAdapter] -->|Precompute| B[TLEs + ISL cache]
    B -->|Yields G_t| C[Temporal Runner]

    C -->|Loop t=0..T| D{Time Step Loop}

    D --> E[Failure Injection]
    E -->|Mutates| F[Effective Graph G'_t]

    F --> G[Labeler (pure)]
    G --> H[Run Summary + Step Metrics]
    H --> I[Dataset Writer (schema validation)]

    %% v2+ extension
    J[Ground Stations] -.-> K[Visibility Provider]
    K -.-> G
    L[Correlated/Probabilistic FailureProcess] -.-> E
    M[Traffic/Routing] -.-> G
```

Notes:
- In v1, “Temporal Runner” can be either a dedicated rollout module or a refactored `SimulationEngine.run()`; the steps below pick a rollout module first to minimize blast radius.

---

# v1 Implementation Plan (Atomic Steps)

## Step 0 — Create a tracking issue + establish conventions
**Goal**: prevent scope creep and lock reproducibility semantics.

- Create a tracking epic: “Tier1-only temporal GCC refactor (v1)”.
- Adopt dataset identifiers:
  - `dataset_version = "tier1_temporal_connectivity_v1"`
  - `schema_version = 1`
- Adopt naming:
  - “run” = one constellation + one failure realization evaluated over time
  - “step” = one time index `t`

**Acceptance check**:
- This document clearly states v1 scope + v2+ non-goals.

**Stop point**:
- No code changes yet.

---

## Step 1 — Create pure labeling/metrics module (no Hypatia dependency)
**Goal**: make connectivity labeling testable in isolation.

- Add new module: `src/satnet/metrics/labels.py`

v1 required functions (pure):
- `compute_num_components(G: nx.Graph) -> int`
- `compute_gcc_size(G: nx.Graph) -> int`
- `compute_gcc_frac(G: nx.Graph) -> float` (handle empty graph)
- `compute_partitioned(gcc_frac: float, threshold: float) -> int`
- `aggregate_partition_streaks(partitioned: list[int]) -> int`

**Acceptance checks**:
- Unit tests validate:
  - Empty graph handling
  - Connected graph → `gcc_frac == 1`
  - Two-component graph → `gcc_frac < 1`

**Stop point**:
- Tests run fast; no Hypatia used.

---

## Step 2 — Define rollout config + results contract (dataclasses)
**Goal**: establish a stable API for temporal evaluation before changing callers.

- Add `src/satnet/simulation/tier1_rollout.py` with:
  - `Tier1RolloutConfig`
    - constellation: `num_planes`, `sats_per_plane`, `inclination_deg`, `altitude_km`, `phasing_factor`
    - time: `duration_minutes`, `step_seconds`
    - ISL: `max_isl_distance_km` (optional)
    - labeling: `gcc_threshold` (default e.g. `0.8`)
    - failures (v1): `node_failure_prob`, `edge_failure_prob`, `seed`
  - `Tier1RolloutStep`
    - `t`, `num_nodes`, `num_edges`
    - `num_components`, `gcc_size`, `gcc_frac`, `partitioned`
  - `Tier1RolloutSummary`
    - `gcc_frac_min`, `gcc_frac_mean`
    - `partition_fraction`, `partition_any`, `max_partition_streak`
    - metadata: `num_steps`, `schema_version`, `dataset_version`, `config_hash`

**Acceptance checks**:
- A smoke test can instantiate these dataclasses without importing toy topology.

---

## Step 3 — Add a graph iterator to HypatiaAdapter (optional but recommended)
**Goal**: simplify time-loop logic and avoid repeated call patterns.

- Add `iter_graphs(start_t, end_t, step_seconds)` (or equivalent) to `HypatiaAdapter`.
- Ensure it reuses existing caches (`_isl_data`) and does not re-parse TLEs per step.

**Acceptance checks**:
- A tiny rollout can iterate graphs without exceptions.

---

## Step 4 — Implement the temporal Tier 1 rollout runner (Hypatia integration)
**Goal**: generate `Tier1RolloutStep[]` from Hypatia time steps and aggregate labels.

In `tier1_rollout.py`, implement:
- `run_tier1_rollout(cfg) -> (steps: list[Tier1RolloutStep], summary: Tier1RolloutSummary)`

v1 algorithm:
- Construct `HypatiaAdapter` from constellation params.
- `adapter.generate_tles()`
- `adapter.calculate_isls(duration_minutes=..., step_seconds=..., max_isl_distance_km=...)`
- Sample failures once per run from base graph:
  - `G0 = adapter.get_graph_at_step(0)`
  - sample node failures and edge failures with `seed`
- For each time step `t`:
  - get `G_t`
  - apply persistent node failures
  - apply edge failures only if the edge exists at `t`
  - compute GCC/component metrics
  - build `Tier1RolloutStep`
- Aggregate into `Tier1RolloutSummary`.

**Important v1 assumption**:
- Edge failures are sampled from edges present at `t=0`. Edges that appear later are not eligible to fail in v1.

**Acceptance checks**:
- Short run completes and returns:
  - `len(steps) == num_steps`
  - `0 <= gcc_frac <= 1`
  - summary fields are finite

**Stop point**:
- Demo a single rollout producing a non-trivial `gcc_frac[t]` curve.

---

## Step 5 — Refactor Monte Carlo to become Tier 1 temporal dataset generator
**Goal**: make Tier 1 temporal connectivity the canonical dataset generator.

- Rewrite `src/satnet/simulation/monte_carlo.py` to generate Tier 1 rollouts.

Recommended API:
- `Tier1MonteCarloConfig` (num_runs, sampling ranges, time params, failure params, seed, output controls)
- `generate_tier1_temporal_dataset(cfg) -> (runs_rows, steps_rows)`

**Acceptance checks**:
- Module imports do not reference `satnet.network.topology`.
- Local small sample works.

---

## Step 6 — Remove toy topology generator from `src/`
**Goal**: prevent regressions back to toy data.

- Delete or relocate: `src/satnet/network/topology.py`.
- If preserved, move to tests-only location and ensure nothing in `src/` imports it.

**Acceptance checks**:
- No references to `satnet.network.topology` in `src/` or `scripts/`.

**Stop point**:
- Verify toy topology is removed from production.

---

## Step 7 — Update scripts to use Tier 1 temporal dataset generator
**Goal**: make runnable workflows produce Tier 1-only data.

Update (at minimum):
- `scripts/export_failure_dataset.py`
- `scripts/failure_sweep.py`
- `scripts/export_design_dataset.py`

**Acceptance checks**:
- Outputs land under `data/`.
- Outputs include both:
  - runs table (one row per run)
  - steps table (one row per step)

---

## Step 8 — Define and enforce the v1 dataset schema
**Goal**: prevent silent schema drift.

- Add `docs/datasets/tier1_temporal_connectivity_v1_schema.md` defining required columns.
- Add schema validation that asserts required fields exist before writing.

**Acceptance checks**:
- Writer refuses to write if required fields are missing.

**Stop point**:
- Verify schema is stable and reproducible.

---

## Step 9 — Refactor SimulationEngine to stop being a static `t=0` holder
**Goal**: eliminate the conceptual trap at the engine level.

- Update `src/satnet/simulation/engine.py` to expose time steps (choose one):
  - passthrough `get_graph_at_step(t)` API
  - or a generator `iter_graphs()`

**Acceptance checks**:
- No other module relies on `engine.network_graph` being a single snapshot.

---

## Step 10 — Update risk model training to match Tier 1 temporal labels
**Goal**: remove training pipelines that rely on toy/leaky columns.

- Update `src/satnet/models/risk_model.py`:
  - define v1 feature columns (design-time inputs only)
  - define v1 labels (`partition_any` or `partition_fraction > X`)

**Acceptance checks**:
- Training can load the v1 runs table without referencing removed toy columns.

---

## Step 11 — Add regression tests for the Tier 1-only contract
**Goal**: make it hard to reintroduce toy topology or regress temporal outputs.

- Add tests:
  - `test_no_toy_imports`
  - `test_rollout_shapes`

**Acceptance checks**:
- Tests run fast and pass locally/CI.

---

# Suggested Execution Order (Junior Engineer)
Execute in this order:
1. Step 1 (pure metrics)
2. Step 2 (contracts)
3. Step 4 (rollout runner)
4. Step 5 (Tier1 Monte Carlo)
5. Step 6 (remove toy)
6. Step 7 (update scripts)
7. Step 8 (schema doc + validation)
8. Step 9 (engine cleanup)
9. Step 10 (training updates)
10. Step 11 (regression tests)

Rationale:
- Contracts + pure labeling first prevents thrash when Hypatia integration changes.

---

# v2+ Roadmap (Do not mix into v1 PRs)

## Phase 2 — Ground stations / gateways (service availability)
**Prerequisite**: v1 temporal GCC is stable and schema is locked.

Atomic steps:
- Add `GroundStation` model module (dataclass: lat/lon/min_elevation).
- Add visibility provider:
  - use Hypatia positions per step + elevation geometry
  - yield `visible_sats: Set[sat_id]` per step
- Extend labeler with gateway-aware reachability metrics:
  - `reachable_sat_frac[t]` via multi-source BFS from `visible_sats`
  - aggregates: mean/min/streak-based reachability

Acceptance checks:
- Visibility outputs are deterministic and validated with small geometry tests.

## Phase 3 — FailureProcess interface + correlated failures + probabilistic ISLs
Atomic steps:
- Introduce `FailureProcess` interface with persistent state (node status) and per-step events.
- Implement correlated regimes (e.g., plane-level common-cause failures).
- Implement probabilistic ISL availability (margin → `P_drop`) as transient per-step failures.

Acceptance checks:
- RNG control produces reproducible outcomes.
- Unit tests cover persistence vs transient semantics.

## Phase 4 — Dataset factory + ML readiness hardening
Atomic steps:
- Extend export schema to include:
  - `design_family_id` for grouped splitting
  - stronger provenance (git SHA, engine version)
- Add grouped-splitting utilities and leakage checks.

Acceptance checks:
- Running twice with the same seed produces bit-identical rows.

## Phase 5 — Traffic / routing labels
Atomic steps:
- Implement a routing model (or interface) that consumes `G'_t`.
- Add throughput/latency/flow feasibility metrics.
- Add temporal aggregates.

Acceptance checks:
- Metrics are stable on small synthetic graphs and realistic on Tier 1 rollouts.

---

# Risk Register (v1)
- Risk: time-varying graphs never partition (labels all zeros)
  - Mitigation: include persistent node failures; test smaller constellations.
- Risk: runtime too slow
  - Mitigation: start with small constellation/time; add caching only after correctness.
- Risk: silent schema drift
  - Mitigation: explicit schema doc + runtime validation.

---

# References
- `docs/gpt5.2 refactor plan - tier1 temporal gcc.md`
- `docs/refactor_plans/gemini_refactor_plan.md`
- `Tier1_TimeVarying_Connectivity_Implementation_Plan.md`
- `Tier1_Correlated_Failure_Models_Implementation_Plan.md`
- `Tier1_Probabilistic_ISL_Availability_Implementation_Plan.md`
- `Tier1_Labels_Implementation_Plan.md`
- `research-plane2.md`
