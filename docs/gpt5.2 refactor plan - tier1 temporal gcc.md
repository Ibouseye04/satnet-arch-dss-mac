# gpt5.2 refactor plan — Tier 1-only temporal simulation + satellite-only GCC labels (v1)

## Goal
Refactor the repo to **eliminate the “Two Worlds” problem** by making the **Tier 1 (Hypatia-based) temporal pipeline** the *only* supported simulation/data path in `src/`.

Phase v1 focuses on:
- **Time-stepped simulation loop** (multiple `t` steps, not `t=0` snapshot)
- **Satellite-only connectivity labels** based on **GCC (Giant Connected Component) fraction**
- A **canonical dataset export contract** suitable for dissertation use

## Non-goals (explicitly deferred)
Do not implement in this refactor phase:
- Ground stations / gateway service availability labels
- Probabilistic ISL availability models (weather, pointing, etc.)
- Correlated failure regimes
- Traffic / routing / throughput labels

These will be added only after v1 temporal GCC is correct and stable.

## Guiding principles (must follow)
- **Tier 1 only in `src/`**: remove toy topology generation from production code.
- **Determinism**: every dataset row must be reproducible from config + seed.
- **Separation of concerns**:
  - HypatiaAdapter: generates time-varying topology
  - Failure sampling: generates persistent failure sets for a run
  - Labeling: pure functions from time-series connectivity → label aggregates
  - Export: schema-stable writer
- **Atomic steps**: each step in this doc must be a small PR that compiles and has a verification check.

## Current state (baseline)
- Tier 1 graph provider exists: `satnet.network.hypatia_adapter.HypatiaAdapter` supports `calculate_isls()` and `get_graph_at_step(t)`.
- Current engine only uses `t=0`: `satnet.simulation.engine.SimulationEngine` loads `get_graph_at_step(0)` into `self.network_graph`.
- “Toy world” still exists: `satnet.network.topology.generate_topology()` and `satnet.simulation.monte_carlo.py` depend on it.

## Target state (definition of done)
- No `src/` module imports `satnet.network.topology` (toy generator removed from production).
- `satnet.simulation.monte_carlo` is rewritten to use Hypatia-based temporal graphs.
- Dataset generation produces time-series connectivity metrics and aggregate labels:
  - `gcc_frac[t] = |GCC(G_t)| / |V_t|`
  - `partitioned[t] = gcc_frac[t] < gcc_threshold`
  - `partition_fraction`, `partition_any`, `max_partition_streak`, etc.
- Scripts in `scripts/` use the Tier 1 dataset generator.
- A junior engineer can rerun a dataset job and reproduce identical outputs.

---

# Implementation plan (atomic steps)

## Step 0 — Create a tracking issue + establish conventions
**Goal**: Prevent scope creep and enforce reproducibility.

- **Create** a single tracking issue/epic: “Tier1-only temporal GCC refactor (v1)”.
- **Adopt** a dataset version string: `tier1_temporal_connectivity_v1`.
- **Adopt** a schema version integer: `schema_version=1`.
- **Adopt** naming:
  - “run” = one constellation + one failure realization evaluated over time
  - “step” = one time index `t`

**Acceptance check**:
- README (or this doc) clearly states v1 scope and non-goals.

**Stop point**: Do not touch code yet.

---

## Step 1 — Add a Tier 1 rollout config + result contract (new module)
**Goal**: Establish a stable API for temporal evaluation before changing callers.

- **Add file**: `src/satnet/simulation/tier1_rollout.py`

Minimum required contents:
- `Tier1RolloutConfig` dataclass:
  - constellation parameters: `num_planes`, `sats_per_plane`, `inclination_deg`, `altitude_km`, `phasing_factor`
  - time parameters: `duration_minutes`, `step_seconds`
  - ISL parameters: `max_isl_distance_km` (optional)
  - labeling parameters: `gcc_threshold` (default e.g. `0.8`)
  - failure parameters (v1): `node_failure_prob`, `edge_failure_prob`, `seed`
- `Tier1RolloutStep` dataclass:
  - `t`, `num_nodes`, `num_edges`
  - `num_components`, `gcc_size`, `gcc_frac`
  - `partitioned` (computed using `gcc_threshold`)
- `Tier1RolloutSummary` dataclass:
  - aggregate statistics over steps: `gcc_frac_min`, `gcc_frac_mean`, `partition_fraction`, `partition_any`, `max_partition_streak`
  - metadata: `num_steps`, `schema_version`, `dataset_version`, config hash

**Notes/constraints**:
- Keep labeling math **pure**: all aggregation functions should accept lists of scalars and return scalars.

**Acceptance check**:
- A tiny unit test (or a doctest-style smoke test) can instantiate the dataclasses without importing toy topology.

---

## Step 2 — Implement GCC metrics as pure functions
**Goal**: Make connectivity labeling testable without Hypatia.

- **Add file**: `src/satnet/simulation/connectivity_metrics.py`

Functions (pure):
- `compute_gcc_size(G: nx.Graph) -> int`
- `compute_num_components(G: nx.Graph) -> int`
- `compute_gcc_frac(G: nx.Graph) -> float` (handle `|V|=0`)
- `compute_partitioned(gcc_frac: float, threshold: float) -> int`
- `aggregate_partition_streaks(partitioned_series: list[int]) -> int` (max streak)
- `aggregate_rollout(steps: list[Tier1RolloutStep]) -> Tier1RolloutSummary`

**Acceptance checks**:
- Add a small test graph to verify:
  - empty graph handling
  - connected graph: `gcc_frac=1`
  - two-component graph: `gcc_frac < 1`

**Stop point**:
- No Hypatia integration yet; tests must run fast.

---

## Step 3 — Implement the temporal Tier 1 rollout runner (Hypatia integration)
**Goal**: Actually generate `Tier1RolloutStep[]` from Hypatia time steps.

In `src/satnet/simulation/tier1_rollout.py`:

- Implement `run_tier1_rollout(cfg: Tier1RolloutConfig) -> tuple[list[Tier1RolloutStep], Tier1RolloutSummary]`.

Algorithm (v1):
1. Construct `HypatiaAdapter` from constellation params.
2. Call `adapter.generate_tles()`.
3. Call `adapter.calculate_isls(duration_minutes=..., step_seconds=..., max_isl_distance_km=...)`.
4. Choose a **base graph** for failure sampling (v1): `G0 = adapter.get_graph_at_step(0)`.
5. Sample failures once per run using existing failure utilities:
   - `FailureConfig(node_failure_prob, edge_failure_prob, seed)`
   - `failures = sample_failures(G0, fail_cfg)`
6. For each time step `t`:
   - `G_t = adapter.get_graph_at_step(t)`
   - Apply the *same* failures to `G_t` (v1 semantics):
     - remove failed nodes
     - remove failed edges if they exist at `t`
   - Compute GCC + components metrics
   - Build `Tier1RolloutStep`
7. Aggregate to `Tier1RolloutSummary`.

**Important v1 assumption**:
- Edge failures are sampled from edges present at `t=0`. Edges that appear later are not eligible to fail in v1.

**Acceptance checks**:
- A short run (small constellation, short duration) completes and returns:
  - `len(steps) == num_steps`
  - `0 <= gcc_frac <= 1`
  - summary fields are finite

**Performance guardrails**:
- Start with small defaults for local tests (e.g. 4 planes × 6 sats, duration 5 minutes).

---

## Step 4 — Refactor `SimulationEngine` to stop being a static graph holder
**Goal**: Remove the conceptual trap where Tier1 is computed but only `t=0` is used.

- **Edit file**: `src/satnet/simulation/engine.py`

Minimal changes (v1):
- Replace `self.network_graph = adapter.get_graph_at_step(0)` usage.
- Introduce an API that exposes time steps (pick one):
  - Option A: `get_graph_at_step(t)` passthrough
  - Option B: `iter_graphs()` generator

Keep backward compatibility only if absolutely required by scripts.

**Acceptance checks**:
- Existing `scripts/simulate.py` still runs (or is updated to call the new rollout runner).
- No other module relies on `engine.network_graph` being a single snapshot.

---

## Step 5 — Replace `satnet.simulation.monte_carlo.py` with Tier 1 Monte Carlo
**Goal**: Kill the “toy data” path and make Tier1 temporal connectivity the canonical dataset generator.

- **Rewrite file**: `src/satnet/simulation/monte_carlo.py`

Design constraints:
- Preserve a small number of script-friendly entry points, but update semantics.
- Do not keep the old ring topology generator in `src/`.

Recommended API in `monte_carlo.py`:
- `Tier1MonteCarloConfig` dataclass:
  - `num_runs`
  - sampling ranges for constellation params (planes, sats/plane, inclination)
  - time params
  - failure params
  - `seed`
  - export controls (output path, parquet/csv)
- `Tier1RunRow` dataclass (one row per run):
  - config fields
  - summary label fields (from `Tier1RolloutSummary`)
- `Tier1StepRow` dataclass (one row per step):
  - `run_id`, `t`, gcc metrics
- `generate_tier1_temporal_dataset(cfg) -> (runs: list[Tier1RunRow], steps: list[Tier1StepRow])`

**Acceptance checks**:
- Module imports do not reference `satnet.network.topology`.
- Dataset generation works on a small sample size locally.

---

## Step 6 — Remove toy topology generator from `src/`
**Goal**: Ensure production can’t regress back to toy data.

- **Delete or relocate**: `src/satnet/network/topology.py`

Recommended approach:
- Move toy logic to `tests/legacy_topology_sanity.py` (or similar) only if needed for unit testing patterns.
- If moved, ensure:
  - it’s not importable from `src/satnet/...`
  - it’s not used by any script

**Acceptance checks**:
- Ripgrep: no references to `satnet.network.topology` in `src/` or `scripts/`.
- Unit tests (if present) still run.

---

## Step 7 — Update scripts to the Tier1 temporal dataset generator
**Goal**: Make all runnable workflows produce Tier1-only data.

Update these scripts:
- `scripts/export_failure_dataset.py`
- `scripts/failure_sweep.py`

Required changes:
- Replace imports from old Monte Carlo API with new Tier1 dataset API.
- Fix output directory casing consistency (`data/` vs `Data/`).
- Ensure scripts write both:
  - `runs` table
  - `steps` table (or time-series file)

**Acceptance checks**:
- Running the scripts produces files under `data/`.
- The output includes time-series metrics (not only post-failure snapshot counts).

---

## Step 8 — Define the v1 dataset schema (write it down + enforce it)
**Goal**: Prevent silent schema drift.

- **Add file**: `docs/datasets/tier1_temporal_connectivity_v1_schema.md`

Schema recommendation (minimum viable):

### runs table (1 row per run)
- `dataset_version` (string)
- `schema_version` (int)
- `run_id` (int)
- `seed` (int)
- constellation: `num_planes`, `sats_per_plane`, `inclination_deg`, `altitude_km`, `phasing_factor`
- time: `duration_minutes`, `step_seconds`, `num_steps`
- failures (v1): `node_failure_prob`, `edge_failure_prob`
- labels:
  - `gcc_threshold`
  - `gcc_frac_min`, `gcc_frac_mean`
  - `partition_fraction`, `partition_any`, `max_partition_streak`
- reproducibility:
  - `config_hash` (string)
  - `engine_version` (string or git SHA)

### steps table (1 row per (run, t))
- `run_id` (int)
- `t` (int)
- `num_nodes`, `num_edges`
- `num_components`, `gcc_size`, `gcc_frac`
- `partitioned` (0/1)

**Enforcement**:
- Add a schema validation function that asserts required columns exist before writing.

**Acceptance checks**:
- The writer refuses to write if required fields are missing.

---

## Step 9 — Update risk model training to match Tier1 temporal labels
**Goal**: Remove training pipelines that rely on toy/leaky columns.

- **Edit file**: `src/satnet/models/risk_model.py`

Changes:
- Add a new `TIER1_TEMPORAL_FEATURE_COLUMNS` for v1:
  - Keep only design-time inputs:
    - constellation params
    - failure probabilities
    - time params (optional)
- Add a new label column definition:
  - default label: `partition_any` OR `partition_fraction > X`

**Acceptance checks**:
- Training script can load v1 runs table and train without referencing removed toy columns.

---

## Step 10 — Add regression tests for the Tier1-only contract
**Goal**: Make it hard to reintroduce toy topology.

Add tests:
- `test_no_toy_imports`: fails if `satnet.network.topology` is importable from `src`.
- `test_rollout_shapes`: run a tiny rollout and validate:
  - `num_steps` correct
  - `gcc_frac` bounds
  - summary aggregates stable

**Acceptance checks**:
- Tests run fast and pass in CI/local.

---

# Suggested execution order for a junior engineer
Execute steps in this exact order:
1. Step 1 (contracts)
2. Step 2 (pure GCC metrics)
3. Step 3 (Hypatia temporal runner)
4. Step 5 (rewrite Monte Carlo)
5. Step 6 (remove toy)
6. Step 7 (update scripts)
7. Step 8 (schema doc + validation)
8. Step 4 (engine cleanup)
9. Step 9 (training updates)
10. Step 10 (regression tests)

Rationale:
- Establishing contracts + pure labeling first prevents thrash when Hypatia integration changes.

---

# Risk register (what can go wrong + mitigation)
- **Risk: time-varying graphs never partition** (labels all zeros)
  - **Mitigation**: include persistent node failures (v1) and sample smaller constellations.
- **Risk: runtime too slow**
  - **Mitigation**: start with small constellation/time; add caching later only if needed.
- **Risk: silent schema drift**
  - **Mitigation**: explicit schema doc + runtime validation.

---

# “Stop points” (when to pause and request review)
- After Step 3: demo a single rollout producing a non-trivial `gcc_frac[t]` curve.
- After Step 6: verify toy topology removed from production.
- After Step 8: verify dataset schema is stable and reproducible.

---

# Notes for follow-on phases (v2+)
Once v1 is stable:
- Add gateways + reachability labels (service availability)
- Add probabilistic ISL availability
- Add correlated failure regimes
- Add routing/traffic-derived labels
