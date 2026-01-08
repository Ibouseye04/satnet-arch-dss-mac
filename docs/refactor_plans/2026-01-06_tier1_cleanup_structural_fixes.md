# Tier 1 Cleanup Implementation Plan — Structural Fixes from Code Review (Phase 1: Satellite-only)

**Date:** 2026-01-06  
**Status:** Draft (implementation plan only; no code changes in this doc)  

## Purpose

Create an executable, Tier-1-only cleanup plan that:

- Removes/quarantines **legacy/non–Tier 1 entrypoints** that reintroduce static `t=0` thinking.
- Closes the **structural correctness gaps** identified in the codebase audit (determinism, dataset contracts, physics fidelity boundaries).
- Preserves **Phase 1 scope**: satellite-to-satellite connectivity only (no gateways/ground stations).

This plan is designed to be executed as **small, verifiable PRs**.

## Constraints (must follow)

- **Tier 1 only in `src/`:** no toy topology generators or static-only simulation APIs in production paths.
- **Temporal evaluation is mandatory:** all Tier 1 metrics/labels must be computed over `t = 0..T` (not just `t=0`).
- **Determinism is law:** reproduce results from `config_hash + seed + epoch + code version`.
- **Labeling is pure:** metrics/labels must be functions of graph state only (no knowledge of failure parameters).
- **Execution reality:** heavy dataset generation and training runs occur on a tunneled remote machine; local work should focus on structure, correctness, and fast unit tests.

## Ground-truth repo snapshot (relevant paths)

### Tier 1 canonical pipeline (keep; harden)

- `src/satnet/network/hypatia_adapter.py`
  - Physics/topology provider.
  - Uses SGP4 when available; falls back if `sgp4` missing.
  - Implements **simplified TEME→ECEF** as a GMST Z-rotation.
  - Contains link budget constants (optical 1550nm, RF Ka-band).

- `src/satnet/simulation/tier1_rollout.py`
  - Tier 1 temporal rollout contract + runner.
  - Uses `DEFAULT_EPOCH_ISO = "2000-01-01T12:00:00+00:00"` (J2000.0) and passes `epoch` to `HypatiaAdapter`.
  - Samples **persistent failures once at `t=0`**.
  - Samples **edge failures from `G0.edges()` only**.
  - `config_hash()` currently truncates SHA256 to 16 hex chars.

- `src/satnet/simulation/monte_carlo.py`
  - Tier 1 temporal dataset generator (runs table + steps table).

- `src/satnet/metrics/labels.py`
  - Pure connectivity metrics (GCC, components, partition flags).

- Tier 1 scripts:
  - `scripts/export_design_dataset.py`
  - `scripts/export_failure_dataset.py`
  - `scripts/failure_sweep.py`

- ML modules consuming Tier 1 outputs:
  - `src/satnet/models/risk_model.py` (Tier 1 v1 functions exist, but legacy columns also remain)
  - `src/satnet/models/gnn_dataset.py` (regenerates graphs on-the-fly; currently mismatched vs rollout semantics)
  - `src/satnet/models/gnn_model.py` (GCLSTM model)
  - `scripts/train_gnn_model.py`
  - `scripts/train_design_risk_model.py`

### Legacy/cleanup targets (remove or quarantine)

- `src/satnet/simulation/engine.py`
  - `SimulationEngine.run()` uses `graph_at_t0` and placeholder “fake load” logic.
  - Instantiates `HypatiaAdapter` **without an explicit epoch**.
  - Exposes deprecated snapshot aliases (`network_graph`, `get_graph`).

- `scripts/simulate.py`
  - Imports `satnet.simulation.engine.run_simulation` (legacy snapshot path).

- `src/satnet/simulation/failures.py`
  - Generic i.i.d. failures used by legacy engine (separate semantics from `tier1_rollout`).

**Note on “simulationPy()”:** no `simulationPy()` symbol appears in the current repo. The legacy behavior you likely mean is the `SimulationEngine.run()` + `scripts/simulate.py` entrypoint.

## Structural issues to fix (from audit)

### H1) Determinism leak: `HypatiaAdapter` default epoch + missing epoch in callers

- **Observed in:**
  - `WalkerDeltaConfig.epoch = datetime.utcnow()` default.
  - `src/satnet/models/gnn_dataset.py` constructs `HypatiaAdapter(...)` without `epoch`.
  - `src/satnet/simulation/engine.py` constructs `HypatiaAdapter(...)` without `epoch`.

- **Risk:** “same seed/config” does not reproduce across runs/machines.

### H2) GNN dataset regeneration contract is underspecified / inconsistent

- **Observed mismatches:**
  - Epoch not passed (can fall back to wall-clock time).
  - Failure realization not applied when regenerating graphs.
  - Seed read but unused for reconstruction.
  - Per-run duration/step not necessarily consumed.

- **Risk:** ML labels (`partition_any`) may not correspond to the regenerated input sequences.

### M1) Legacy static-snapshot engine remains as an attractive footgun

- `SimulationEngine.run()` still performs static `t=0` computation with toy placeholders.
- `scripts/simulate.py` still invokes it.

### M2) Link budget constants are uncited; one parameter is likely inappropriate for ISLs

- Link budget constants are plausible but lack citations.
- `RF_RAIN_MARGIN_DB` is a red flag for space-to-space ISLs (rain attenuation is an Earth-space path concern).

### M3) Environment-dependent physics fidelity (optional dependency fallback)

- If `sgp4` is missing, the adapter falls back to simplified Keplerian.
- No pinned/locked “core simulation” dependency set exists.

### M4) Simplified TEME→ECEF transform

- Implemented as GMST Z-rotation; not a full TEME→ITRS transform.
- **Phase 1 note (satellite-only):** our ISL topology checks are based on inter-satellite distance and Earth-center line-of-sight with a spherical Earth model. These are invariant under any rigid rotation of the entire constellation, so a higher-fidelity TEME→ITRS/ITRF transform is **not expected** to change Phase 1 ISL edge existence.
- **Phase 2 note (deferred):** a rigorous TEME→ITRS/ITRF transform becomes critical once we introduce any Earth-fixed quantity (e.g., ground station visibility / Earth rotation / geodetic computations used by labels).

### M5) Persistent edge failures sampled from `t=0` edge set only

- Edges not present at `t=0` are implicitly immune.
- Semantics need to be made explicit and defendable.

### Additional cleanup targets (code-accuracy issues)

- `src/satnet/models/gnn_dataset.py` docstring claims “Target Model: EvolveGCN-O”, but `src/satnet/models/gnn_model.py` implements **GCLSTM**.
- `Tier1RolloutConfig.config_hash()` truncates SHA256 → collision risk for long-lived datasets.

## Definition of Done (Tier1-only, Phase 1)

- No production workflow uses `SimulationEngine.run()` or `scripts/simulate.py`.
- All `HypatiaAdapter` instantiations in `src/` require explicit `epoch` (or explicitly consume `DEFAULT_EPOCH_ISO`).
- Tier 1 dataset generation and ML dataset reconstruction share a single, explicit contract:
  - time semantics (`t=0..T`),
  - epoch,
  - failure realization,
  - schema versioning.
- Physics-dependency behavior is explicit:
  - Tier 1 mode fails fast without required libs, or
  - Tier 1 mode records the fallback engine used and is treated as a different dataset/version.
- Coordinate transform fidelity is either upgraded or bounded by a documented validation/sensitivity protocol.
- Tests exist that prevent regressions to snapshot-only or nondeterministic behavior.

---

# Implementation Plan (Atomic PRs)

## Step 0 — Baseline guardrails (fast checks)

- **Action:** Add/extend tests that assert:
  - no `datetime.utcnow()` default epoch is reachable from Tier 1 entrypoints,
  - no code path used by Tier 1 scripts calls `get_graph_at_step(0)` and stops.
  - satellite-only geometry invariance sanity check: `_check_line_of_sight(r1, r2)` is unchanged under a shared rigid rotation applied to both endpoints (guards against introducing non-invariant Earth-fixed logic in Phase 1).

- **Acceptance:** `pytest` runs quickly locally and fails loudly on regressions.

## Step 1 — Remove/quarantine the legacy snapshot engine entrypoint

- **Action (recommended):**
  - Remove `scripts/simulate.py` (or replace it with a Tier 1 rollout demo that uses `run_tier1_rollout`).
  - Quarantine `src/satnet/simulation/engine.py` into an explicit legacy namespace (e.g., `satnet.legacy.*`) or delete it if it is not used.

- **Acceptance:**
  - No `scripts/` entrypoint imports `satnet.simulation.engine`.
  - Grep-based check: no new imports of legacy engine are introduced.

## Step 2 — Enforce the epoch contract everywhere

- **Action:** Make “epoch is required” a Tier 1 invariant.
  - Option A (strict): remove default `datetime.utcnow()` in `WalkerDeltaConfig` and require explicit epoch.
  - Option B (safe default): default to J2000.0 (`DEFAULT_EPOCH_ISO`) at the adapter boundary.

- **Update call sites:**
  - `src/satnet/simulation/engine.py` (if kept)
  - `src/satnet/models/gnn_dataset.py`
  - Any scripts/tests that instantiate `HypatiaAdapter`

- **Acceptance:**
  - Determinism test: same config+seed+epoch yields identical graph statistics across runs.

## Step 3 — Define a single “graph reconstruction contract” for ML

- **Action:** Decide and document how ML inputs are reconstructed.

  **Contract requirements (Phase 1):**
  - Exact epoch used (`epoch_iso`).
  - Time parameters per run (`duration_minutes`, `step_seconds`).
  - Failure realization applied to graphs (not just failure probabilities).

 - **Implementation direction:**
  - Extend dataset export to persist failure masks/lists per run (preferred), or
  - Persist enough determinism inputs to regenerate the exact same failures (risky if topology varies).

  **Repo note:** the Tier 1 runs table already contains the key ingredients (`epoch_iso`, `seed`, `failed_nodes_json`, `failed_edges_json`). The remaining work is to enforce their presence in the schema and ensure `src/satnet/models/gnn_dataset.py` consumes them during graph regeneration.

- **Acceptance:**
  - A unit test can take one exported run row and regenerate a temporal sequence whose aggregate label matches (`partition_any`).

## Step 4 — Make failure semantics explicit (especially edge failures)

- **Action:** Choose and document what “edge failure” means in a time-varying graph:
  - **Hardware pair failure:** (u,v) link terminals fail and the edge can never appear.
  - **Time-specific link outage:** edges fail per step with some process.

 - **If hardware-pair semantics:** sample from a stable candidate set (e.g., the deterministic +Grid neighbor pairs implied by the adapter), not `G0.edges()`.
 - **If time-specific semantics:** introduce per-step outages (still Phase 1 satellite-only).

- **Acceptance:**
  - Dataset schema clearly encodes the semantics.
  - Edge-failure sampling is not biased by `t=0` visibility.

## Step 5 — Physics fidelity boundaries: coordinate transforms + link budget citations

 - **Action (coordinate transforms):**
  - Phase 1: treat transform fidelity primarily as a documentation + validation item (the satellite-only LOS/distance checks are rotation-invariant).
  - Phase 2: adopt a higher-fidelity TEME→ITRS/ITRF transform before introducing any ground-station/earth-fixed logic.

 - **Action (link budgets):**
  - Move optical/RF constants to a cited configuration layer.
  - Remove/disable rain margin for ISLs (or justify with citations if retained).

- **Acceptance:**
  - A `docs/` page exists listing all constants with citations.

## Step 6 — Dependency pinning strategy for Tier 1 reproducibility

- **Action:** Add a “core simulation” dependency spec and a lock mechanism.
  - At minimum: pin versions for `sgp4`, `numpy`, `networkx`.
  - Ensure Tier 1 mode fails fast if physics deps are missing.

- **Acceptance:**
  - A fresh environment can be constructed deterministically.
  - Tier 1 runs cannot silently downgrade physics without being detected.

## Step 7 — Clean up ML naming + model truthfulness

- **Action:** Align docs and code:
  - `gnn_dataset.py` docstring should match the implemented model (GCLSTM) or the model should be implemented as claimed.

- **Acceptance:**
  - Repo text is consistent about which temporal GNN is used.

## Step 8 — Fix `config_hash` collision risk

- **Action:** Increase `config_hash()` length (full SHA256 or at least 32 hex chars).
  - If this breaks downstream consumers, bump `schema_version` / `dataset_version` and provide migration notes.

- **Acceptance:**
  - Dataset IDs are stable and collision-resistant.

---

# Notes / Non-goals (Phase 1)

- No gateways / no ground stations / no Earth rotation modeling.
- No routing/traffic labels.
- No correlated failures beyond documenting the interface; implement later once the v1 temporal GCC pipeline is locked.
