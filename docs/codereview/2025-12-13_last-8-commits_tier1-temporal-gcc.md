# Code Review — Last 8 Commits (Tier1 Temporal GCC v1)

**Date:** 2025-12-13  
**Scope:** Review of the last 8 commits on `main`, focusing on adherence to:
- `docs/refactor_plans/unified_refactor_plan_tier1_temporal_gcc.md`
- `AGENTS.md`
- Beads intent (epic + steps in `.beads/issues.jsonl`)

## Commits reviewed (oldest → newest)
- `2b460d6` Add Tier1 correlated failure models and dynamic graph feature engineering implementation plans
- `44a0586` Add unified Tier1 temporal simulation refactor plan with v1 scope (satellite-only GCC) and v2+ roadmap
- `f454610` Add Beads AI-native issue tracking system with config and documentation
- `3b2655e` Add pure connectivity metrics module with labeling functions and comprehensive unit tests
- `72995ed` Add Tier1 temporal rollout configuration and results contracts with comprehensive unit tests
- `bcd3d7f` Add `iter_graphs` generator and convenience properties to `HypatiaAdapter` with comprehensive unit tests
- `c67c1ca` feat: Complete Tier 1 temporal GCC refactor (v1)
- `308c9d7` moved research into research folder

## Intended v1 acceptance criteria (from plan + AGENTS)
- **Tier 1 only in `src/`**: no `satnet.network.topology` imports; no toy topology generator in production.
- **Temporal by default**: evaluate connectivity over `t=0..T`, not `t=0` snapshots.
- **Determinism**: reruns with same config + seed produce identical outputs.
- **Metrics are pure** (`src/satnet/metrics/labels.py`): labels derive only from graph state.
- **Satellite-only**: no ground stations/gateway logic in v1.
- **Schema stability + validation**: writer refuses to write on missing required fields.

## What looks good / aligned
- **Toy topology removed from `src/`**
  - `src/satnet/network/topology.py` deleted; regression test `tests/test_tier1_contract.py` asserts no imports and file deletion.
- **Pure metrics module exists and is clean**
  - `src/satnet/metrics/labels.py` is stateless, graph-only, and unit-tested.
- **Temporal rollout exists and is wired end-to-end**
  - `run_tier1_rollout()` drives a time loop via `HypatiaAdapter.iter_graphs()` and computes per-step GCC metrics.
- **Beads mapping matches the plan’s atomic steps**
  - `.beads/issues.jsonl` encodes Step 1..11 as tasks and matches the unified plan structure.

---

# Findings

## High severity

### 1) Determinism is violated by default epoch (`datetime.utcnow()`)
- **Where**
  - `src/satnet/network/hypatia_adapter.py`: `WalkerDeltaConfig.epoch` defaults to `datetime.utcnow()`.
- **Why this violates the plan**
  - The unified plan and `AGENTS.md` require **reproducibility from config + seed**.
  - Current epoch is **time-of-run dependent**, so the same config/seed run at different wall-clock times can yield different TLE epochs → different propagated positions → different ISLs/graphs → different GCC traces.
- **Why tests didn’t catch it**
  - Determinism tests call the rollout twice in the same process; the epoch is created once per adapter instantiation and will likely be stable over those two calls.
- **Recommended fix**
  - Add an explicit `epoch` to `Tier1RolloutConfig` (and include it in `config_hash`).
  - Default it to a **fixed constant** (timezone-aware) rather than wall-clock now.

### 2) Time-step contract mismatch (`num_steps` semantics inconsistent)
- **Where**
  - `Tier1RolloutConfig.num_steps` returns `duration_seconds // step_seconds`.
  - `HypatiaAdapter.calculate_isls()` uses `duration_seconds // step_seconds + 1`.
  - Tests expect inclusive step indexing (`t=0..T`) in some places (e.g., “3 minutes @ 60s = 4 steps”).
- **Why this violates the plan**
  - Plan states **`t=0..T`**, which implies inclusive endpoints.
  - Contract confusion will leak into dataset sizing, validation, and downstream feature engineering.
- **Symptoms / risk**
  - `Tier1RolloutConfig.num_steps` is currently **not equal** to `len(steps)` returned by `run_tier1_rollout()`.
- **Recommended fix**
  - Decide and document the canonical convention (strongly recommend inclusive `0..T`):
    - `num_steps = duration_seconds // step_seconds + 1`
  - Update:
    - `Tier1RolloutConfig.num_steps`
    - Unit tests that encode the old convention
    - Any code that assumes `duration_minutes * 60 / step_seconds` without `+1`

### 3) Schema validation exists but is not enforced at write-time
- **Where**
  - `src/satnet/simulation/monte_carlo.py` defines `validate_runs_schema()` and `validate_steps_schema()`.
  - `scripts/export_*_dataset.py` writes CSV directly and does **not** call validation.
- **Why this violates the plan**
  - Step 8 requires: **“Writer refuses to write if required fields are missing.”**
  - Right now, the schema checks are **optional utilities**, not part of the export pipeline.
- **Recommended fix**
  - Introduce a single export function (e.g., `write_tier1_dataset(...)`) that:
    - converts rows → dicts
    - runs schema validation
    - then writes output
  - Ensure scripts call this writer (or call validation explicitly before writing).

## Medium severity

### 4) `SimulationEngine` still centers a static `t=0` snapshot (conceptual trap remains)
- **Where**
  - `src/satnet/simulation/engine.py`:
    - `run()` uses `self.network_graph` (which is `t=0`).
    - `network_graph` property remains and is presented as “backward compatibility.”
- **Why this conflicts with AGENTS**
  - `AGENTS.md`: **NO STATIC SNAPSHOTS** (“Do not just load `get_graph_at_step(0)` and stop.”)
- **Recommended fix**
  - If `SimulationEngine` remains, make the temporal API the default and demote `t=0` access:
    - rename `network_graph` to `graph_at_t0` and/or clearly deprecate
    - refactor `run()` into a time loop or remove it if Tier1 rollout is the canonical path

### 5) Production modules contain runtime `sys.path` patching and verbose `print()`
- **Where**
  - `src/satnet/simulation/engine.py`: modifies `sys.path` at import time.
  - `src/satnet/network/hypatia_adapter.py`: prints warnings/status; writes files into temp output dirs.
- **Why this is an issue (Tier1 standards)**
  - Import-time path mutations are brittle and can break packaging/testing.
  - Unstructured prints complicate reproducible runs and downstream tooling.
- **Recommended fix**
  - Move script-run convenience path patches into `scripts/` only.
  - Prefer logging (or at least a controlled verbosity flag) for Tier1 runs.

### 6) v1 output format is CSV, while plan language suggests a canonical dataset writer (often Parquet)
- **Where**
  - `scripts/export_design_dataset.py`, `scripts/export_failure_dataset.py` write CSV.
  - Schema doc implies DataFrame/Parquet flows.
- **Why this matters**
  - CSV is fine for quick iteration, but schema stability + reproducibility workflows (hashing, types) are typically more robust in Parquet.
- **Recommendation**
  - Either:
    - explicitly bless CSV for v1 and document it, or
    - switch to Parquet in the canonical writer (while still optionally emitting CSV for convenience).

## Low severity / polish

### 7) `tier1_rollout.py` module docstring contradicts its contents
- **Where**
  - `src/satnet/simulation/tier1_rollout.py` claims “No Hypatia imports in this module”, but `run_tier1_rollout()` imports HypatiaAdapter (inside the function).
- **Recommendation**
  - Align docstring with reality (contracts + runner live in this module) or split contracts vs runner into separate modules.

### 8) `config_hash` truncation increases collision risk
- **Where**
  - `Tier1RolloutConfig.config_hash()` truncates SHA256 to 16 hex chars.
- **Recommendation**
  - Consider keeping the full hash (or at least 32 chars) for long-lived datasets.

---

# Suggested follow-up actions (actionable)
- **[Reproducibility]** Add deterministic, explicit `epoch` into rollout config; include in hash and output metadata.
- **[Time semantics]** Unify step count convention across adapter/config/tests (`t=0..T` inclusive).
- **[Schema]** Make schema validation mandatory in the export path (single writer).
- **[Engine]** Reduce or remove the `SimulationEngine` “t=0 default” trap; keep temporal-first APIs.

---

# Notes about Beads status
Beads issues for Steps 1–11 are currently marked **closed**. Several acceptance checks are only partially met (notably determinism via epoch and schema enforcement at write-time). Consider reopening/creating new Beads issues for these gaps so the repo’s issue state matches engineering reality.
