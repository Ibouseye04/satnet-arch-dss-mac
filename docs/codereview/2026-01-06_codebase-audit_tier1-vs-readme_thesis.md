# Codebase Audit — Tier 1 Temporal Connectivity vs `README_THESIS.md`

**Date:** 2026-01-06  
**Scope:** Static audit of Tier 1 temporal connectivity pipeline + dataset/model interfaces.  
**Primary reference:** `README_THESIS.md` (“Tier 1” requirements + reproducibility guarantees).  

## Methodology & constraints

- Static code reading + targeted repository search.
- No heavy local execution (design dataset generation runs elsewhere).
- Phase 1 scope: satellite-to-satellite only (no gateways/ground stations).

## Modules reviewed

- `src/satnet/network/hypatia_adapter.py`
- `src/satnet/simulation/tier1_rollout.py`
- `src/satnet/simulation/monte_carlo.py`
- `scripts/export_design_dataset.py`
- `src/satnet/models/gnn_dataset.py`
- `src/satnet/models/risk_model.py`
- `src/satnet/simulation/engine.py`

## Executive summary

- The **Tier 1 temporal rollout path** largely matches the thesis intent:
  - Temporal iteration over `t=0..T`.
  - Persistent failures sampled once at `t=0` and applied across time.
  - Metrics/labels computed via **pure functions** over graph state.
  - Dataset export enforces schema validation.

- Two **high-severity deviations** undermine “defense-ready determinism” claims:
  - `HypatiaAdapter` has a default epoch of `datetime.utcnow()` (nondeterministic) when callers forget to pass `epoch`.
  - At least two non-rollout callers (`gnn_dataset.py`, `engine.py`) instantiate `HypatiaAdapter` without `epoch`.

- The legacy `SimulationEngine.run()` remains a **static `t=0` snapshot** path with placeholder logic. It is documented as deprecated, but its presence is an architectural hazard if reused.

## Crosswalk: thesis guarantees vs implementation

### 1) Temporal, not static

- **Rollout path:** `run_tier1_rollout()` performs temporal evaluation across steps.
- **Legacy path:** `SimulationEngine.run()` is a static snapshot.

### 2) Determinism (seed + stable epoch + config hash)

- **Rollout path:** explicit `seed` and explicit epoch passed to `HypatiaAdapter`.
- **Other entry points:** `epoch` omission falls back to `datetime.utcnow()`.

### 3) “No leaky labels”

- `src/satnet/metrics/labels.py` implements pure, stateless graph metrics.

### 4) Dataset schema/versioning

- `write_tier1_dataset_csv(...)` validates schemas before writing.

---

# Findings (prioritized)

## High severity

### H1) Determinism leak: `HypatiaAdapter` default epoch + missing epoch in callers

- **Where:**
  - `src/satnet/network/hypatia_adapter.py` (default `epoch=datetime.utcnow()`)
  - `src/satnet/models/gnn_dataset.py` (constructs adapter without `epoch`)
  - `src/satnet/simulation/engine.py` (constructs adapter without `epoch`)

- **Why it matters:** violates `README_THESIS.md` determinism guarantees (stable epoch + explicit seed).

- **Recommended fix:** require an explicit epoch in all adapter entry points (or change the adapter’s default to the project’s `DEFAULT_EPOCH_ISO`).

### H2) GNN dataset regeneration contract is underspecified

- **Observation:** graphs are regenerated on-the-fly from run metadata, but the code path does not clearly guarantee that graph regeneration (including failures) matches the rollout’s label-generation semantics.

 - **Concrete mismatches observed (current `SatNetTemporalDataset` behavior):**
   - **Epoch not applied:** the dataset does not pass `epoch` / `epoch_iso` into `HypatiaAdapter`, so it can silently fall back to `datetime.utcnow()`.
   - **Failure realization not applied:** the dataset does not apply the run’s sampled persistent failures (nodes/edges) when regenerating graphs.
   - **Failure probabilities not represented:** the dataset does not include `node_failure_prob` / `edge_failure_prob` as graph features, yet the label `partition_any` depends on those probabilities in the rollout pipeline.
   - **Timing mismatch risk:** the dataset uses constructor-level `duration_minutes` / `step_seconds` instead of consuming per-run columns (when present).
   - **Seed read but unused:** `seed = row.get("seed", None)` is read but not used to reconstruct anything.

- **Risk:** label/feature mismatch can silently invalidate ML results.

- **Recommended fix:** document and enforce a single “graph reconstruction contract” keyed by `config_hash` + `seed` + `epoch` + failure realization identifiers.

## Medium severity

### M1) `SimulationEngine` is a legacy static-snapshot API with placeholder logic

- **Observation:** `SimulationEngine.run()` assigns “fake” loads and uses `graph_at_t0`.

- **Risk:** accidental reuse reintroduces Tier 0 thinking (static snapshot / toy behavior) into Tier 1 pipelines.

- **Recommended fix:** quarantine or clearly namespace as legacy, and ensure Tier 1 scripts never import it.

### M2) Link budget defaults lack citations and include at least one physically questionable parameter

- **Observation:** optical/RF constants are plausible but uncited; RF includes a “rain margin” despite ISLs being space-to-space.

- **Risk:** makes “physics layer” claims hard to defend academically.

- **Recommended fix:** make parameters explicit config with citations to standards or peer-reviewed sources, and separate ISL vs ground-link margin models.

### M3) Environment-dependent physics fidelity (optional dependency fallback)

- **Observation:** `HypatiaAdapter` attempts to use SGP4/satgenpy when available, but contains a fallback to a simplified Keplerian model if `sgp4` is not installed.

- **Risk:** results can change across environments (e.g., local dev vs remote runner) even with the same `seed` and config.

- **Recommended fix:** pin and validate the physics dependencies for Tier 1 runs (fail fast if `sgp4`/`satgenpy` are missing in Tier 1 mode).

 - **Additional note:** the repo does not currently include a pinned/locked “core simulation” dependency spec (e.g., `requirements.txt`/`pyproject.toml` + lockfile). This increases the chance that Tier 1 runs silently differ across machines.

### M4) Coordinate frame transform is a simplified TEME→ECEF approximation

- **Observation:** `HypatiaAdapter` implements TEME→ECEF as a **simple Z-axis rotation** by GMST (IAU 1982 approximation).

- **Risk:** this may be “good enough” for many metrics, but it is not a full TEME→ITRS/ITRF transform. Near **Earth-obscuration grazing thresholds** (LOS edge flip boundary), small frame/time errors can change link existence decisions.

- **Recommended fix:** either:
  - adopt a rigorous transform library (e.g., `astropy` TEME→ITRS), or
  - keep the approximation but add a **sensitivity/validation protocol** that quantifies LOS/edge mismatch rates vs a higher-fidelity reference.

### M5) Persistent edge failures are sampled from `t=0` edges only

- **Observation:** `run_tier1_rollout()` samples persistent edge failures from the **edge set of `G0 = get_graph_at_step(0)` only**, then applies removals at later steps *only if* the failed edge exists at that step.

- **Risk:** if the topology is time-varying (edges appear/disappear due to geometry or link budget), then edges that do not exist at `t=0` are implicitly **immune** to persistent failures. This can bias failure realism and undermine claims about outage modeling.

- **Recommended fix:** define what “edge failure” means (pairwise terminal/hardware vs a time-specific link instance) and then sample failures from a stable superset:
  - union of edges across all steps, or
  - a static “potential link” set (e.g., +Grid neighbor pairs), or
  - move to per-step transient outages (with explicit semantics and labels).

## Low severity

### L1) Risk model module still references legacy ground-station features

- **Observation:** `risk_model.py` retains “legacy” feature columns including `num_ground_stations`.

- **Risk:** confusion with Phase 1 scope.

- **Recommended fix:** keep legacy behind an explicit “legacy” API boundary and ensure Tier 1 v1 training functions are the default.

---

# Recommended remediation checklist (actionable)

- **[H1: determinism / epoch]** Require an explicit `epoch` in *all* `HypatiaAdapter` construction sites (rollout, GNN dataset regeneration, any legacy engines). Prefer defaulting at the call site to `DEFAULT_EPOCH_ISO` (J2000.0) rather than allowing `datetime.utcnow()` to enter the config.

- **[H2: dataset/label contract]** Define and enforce a single Tier 1 “graph reconstruction contract” for ML that guarantees the same:
  - `epoch`
  - time-step semantics (`t=0..T` inclusive vs exclusive)
  - persistent failure realization (nodes/edges)

  Either:
  - persist failure masks/lists per run (recommended), or
  - persist enough identifiers/entropy to re-generate the exact same failure realization.

- **[M1: legacy engine trap]** Quarantine `SimulationEngine` as a legacy API (e.g., explicit legacy namespace/module name) so Tier 1 scripts/models cannot accidentally depend on `graph_at_t0` behavior.

- **[M3: physics dependency pinning]** Pin and validate physics dependencies for Tier 1 execution (fail fast if SGP4/satgenpy are missing) to avoid environment-dependent fallbacks.

- **[M2: link budget citations]** Move RF/optical link budget constants into an explicit, cited configuration (and separate ISL vs ground-link margins) so “physics layer” claims are defendable in writing.

- **[M4: frame transform fidelity]** Either adopt a rigorous TEME→ITRS transform or add a sensitivity protocol to quantify LOS/edge mismatch near grazing thresholds.

- **[M5: failure semantics]** Document the intended semantics of “persistent edge failure” and adjust sampling so it does not depend on `t=0` edge presence (or explicitly defend why it should).

---

# Literature grounding (open-access)

## Why “temporal, not static” is a defensible requirement

- **Time-varying behavior is the core research object, not a nuisance detail.** Hypatia’s IMC’20 abstract explicitly frames “high-velocity orbital motion” as a defining property that the simulator must incorporate:

  > “To enable research in this exciting space, we present Hypatia, a framework for simulating and visualizing the network behavior of these constellations by incorporating their unique characteristics, such as high-velocity orbital motion.”
  > 
  > — Kassing et al., *Exploring the “Internet from space” with Hypatia* (IMC 2020)

- **LEO networks change frequently at the ISL level.** `xeoverse` motivates simulation fidelity requirements by pointing out the mismatch with static terrestrial assumptions:

  > “the dynamic nature of LEO satellite networks, with their high-speed movement and frequent changes in Inter-Satellite Links (ISLs), poses a challenge to the existing terrestrial Internet protocols and algorithms, which were designed for a static infrastructure.”
  > 
  > — `xeoverse` (2024)

- **Precomputing time-indexed state is a common pattern in LEO simulators.** Hypatia’s project documentation describes the approach succinctly:

  > “Hypatia is a low earth orbit (LEO) satellite network simulation framework. It pre-calculates network state over time …”
  > 
  > — Hypatia README

- **ISLs are frequently modeled as optical.** The Westphal survey notes the bent-pipe → ISL evolution and explicitly calls out free-space optics for inter-satellite links:

  > “Inter-satellite links use free-space optics to connect fast moving object 2,000 miles apart.”
  > 
  > — Westphal et al. (2023)

These support `README_THESIS.md`’s requirement that Tier 1 evaluation must iterate over `t=0..T` rather than running a static snapshot.

## Why determinism (fixed epoch) is part of “defense-ready” rigor

- **Reproducibility depends on fully specified inputs.** The trace-driven Hypatia workflow emphasizes that trace artifacts can be reused *only* when the simulation inputs are unchanged:

  > “These Trace Files can be archived, since, as long as no input parameters of the simulation change, they can be used for multiple experiments in the emulation environment.”
  > 
  > — Ottens et al., *Trace-driven Path Emulation of Satellite Networks using Hypatia* (2025)

  In practice, allowing `epoch` to default to wall-clock time makes “input parameters” differ across runs even when `seed` is fixed.

This supports treating the `datetime.utcnow()` default epoch path as a **Tier 1 correctness bug**, not merely a “nice-to-have”.

## Background scaling pressure (why performance design matters)

- **Scaling is constrained by physical space and demand imbalance (cyber-physical limits).** The HotNets’24 position paper frames LEO scalability as bounded by physical constraints:

  > “the LEO network’s sustainable expansion is constrained by its harsh, crowded, and imbalanced physical environment.”
  > 
  > — Chen et al., *Unraveling Physical Space Limits for LEO Network Scalability* (HotNets 2024)

This supports proactively auditing SGP4 usage patterns, caching strategy, and graph construction overhead in Tier 1 Monte Carlo loops.

## References (open access)

- Kassing et al. (2020). *Exploring the “Internet from space” with Hypatia.*
  - PDF (author-hosted): https://bdebopam.github.io/papers/imc2020-hypatia.pdf

- Chen et al. (2024). *Unraveling Physical Space Limits for LEO Network Scalability.*
  - PDF (HotNets 2024): https://conferences.sigcomm.org/hotnets/2024/papers/hotnets24-211.pdf

- Westphal et al. (2023). *LEO Satellite Networking Relaunched: Survey and Current Research Challenges.*
  - https://ar5iv.labs.arxiv.org/html/2310.07646
  - https://arxiv.org/abs/2310.07646

- `xeoverse` (2024). *xeoverse: A Real-time Simulation Platform for Large LEO Satellite Mega-Constellations.*
  - https://arxiv.org/abs/2406.11366
  - https://arxiv.org/html/2406.11366

- Ottens et al. (2025). *Trace-driven Path Emulation of Satellite Networks using Hypatia.*
  - https://arxiv.org/abs/2510.27027
  - https://arxiv.org/html/2510.27027v1

- Hypatia project README (software documentation; referenced by the IMC’20 paper):
  - https://raw.githubusercontent.com/snkas/hypatia/master/README.md
