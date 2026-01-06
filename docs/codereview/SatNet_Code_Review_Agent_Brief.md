# SatNet Architecture DSS — Code Review + Agent Brief (Dissertation-Defensible)

**Scope**: repo `satnet-arch-dss-mac` (Python). Focus is (a) Tier-1 temporal constellation rollout + Monte Carlo dataset generation, (b) ML training (design-space model + temporal GNN), and (c) reproducibility/defensibility gaps.

**Why this doc exists**: you can hand this to a coding agent and get *atomic*, verifiable changes that (1) fix correctness/reproducibility issues, (2) align implementation to your thesis slides, and (3) reduce “gotchas” you’d get grilled on in a dissertation defense.

---

## 0) Thesis → Implementation traceability (what you claim vs what the code actually does)

Your PPT describes a pipeline that: (i) builds Walker-Delta constellations and propagates satellites using **SGP4/WGS72** (Tier-1), (ii) constructs time-indexed ISL graphs with Earth-obscuration + link-budget viability (optical 1550nm and RF 28GHz with rain margin), then (iii) runs Monte Carlo over design/failure assumptions and trains models (LightGBM + temporal GNN) and validates results. fileciteturn2file4L10-L13 fileciteturn2file1L2-L4

### Where repo matches the PPT (good news)
- **Tier-1 temporal rollout exists** (`src/satnet/simulation/tier1_rollout.py`) and creates time-indexed graphs, computes GCC metrics, and writes a Tier-1 dataset via Monte Carlo (`src/satnet/simulation/monte_carlo.py`). fileciteturn2file4L11-L13
- The adapter uses **TLEs + (intended) SGP4** and includes a fallback Keplerian mode (matches your slide narrative). fileciteturn2file1L2-L2

### Where repo diverges from the PPT (defense-risk)
1. **SGP4 is not in the pinned environment** (and PyG isn’t either). In a clean environment, Tier-1 “preferred mode” won’t run as described. This is the #1 “reproducibility” landmine for your committee. (SGP4 is a standard orbit propagator and you explicitly claim it.) citeturn0search0turn7search3
2. **Your ML stack doesn’t line up cleanly**:
   - PPT calls out **LightGBM** for the design-space model; repo has `risk_model.py` (LogReg/RandomForest) and training scripts that still reference old files/columns (stale). fileciteturn2file4L11-L12 citeturn9search0turn9search1
   - Temporal model: code implements **GCLSTM** (via torch_geometric_temporal), while docs mention EvolveGCN(-O) in places. Pick one and defend it with citations. citeturn1search1turn8search48
3. **Determinism mismatch**: Tier-1 rollout passes an explicit epoch (good), but the GNN dataset re-simulates graphs with a default “utcnow” epoch, so training can silently mismatch labels/data. That’s a correctness issue, not just style.

---

## 1) Build/Run health check (what currently breaks on a fresh machine)

### Observed breakages in this environment
- `pytest` fails immediately because imports can’t resolve `satnet` (no packaging / no pythonpath config).
- `sgp4` isn’t installed (HypatiaAdapter will raise on import).
- `torch_geometric` / `torch_geometric_temporal` aren’t installed, so GNN training scripts will not run.

### Dissertation-level fix requirement
You need a reproducible environment spec and an executable “one command” pipeline (even if it takes hours), otherwise any results can be challenged as non-repeatable.

**Primary references**:
- SGP4 algorithm + standard implementation background: Vallado et al. citeturn0search0turn0search1
- PyTorch Geometric official install instructions (to avoid “works on my laptop”): citeturn6search0

---

## 2) Tier-1 simulation code review (correctness, modeling assumptions, and defense points)

### 2.1 Orbit propagation + coordinate frames
**What code does**:
- Generates synthetic TLEs.
- Propagates in TEME (SGP4), then converts TEME → “ECEF-ish” via a simple Earth-rotation `Rz(theta)` transform.

**Defense risk**:
- A simple Z-rotation is *not* the full TEME→ITRS/ECEF transformation; rigorous conversion uses Earth orientation parameters, precession/nutation, etc. If your claims depend on geometric line-of-sight near grazing thresholds, this can matter.

**Actionable fix** (two tiers):
- **Tier A (thesis-grade, credible)**: use `astropy.coordinates` with TEME/ITRS transforms and time scales. Cite astropy’s frame definitions and transformation machinery. citeturn7search0turn7search1
- **Tier B (fast, but must be justified)**: keep the simplified transform but add a justification + sensitivity analysis (quantify how much the simplification changes link existence counts vs full transform).

### 2.2 Geometry (Earth-obscuration / LOS)
The current approach checks Earth intersection via closest approach / clearance logic. That’s fine as a first-order LOS gate, but you need to explicitly document:
- Earth radius constant used
- “grazing height” / clearance threshold definition
- numerical stability near tangency

**Research grounding**: LOS constraints and orbit propagation are standard in sat-network simulation; if you cite Hypatia-style emulation/simulation approaches, explicitly scope your physics fidelity. citeturn4search1turn4search4turn1search4

### 2.3 Link budget modeling (optical + RF)
**Big conceptual issue**: you currently add a “rain margin” to Ka-band viability checks, but **rain attenuation is an Earth–space phenomenon**; it does not apply to *space-to-space ISLs*. If Tier-1 is ISL-only, rain margin should be zero or removed. If you’re modeling a ground segment later, rain margin belongs on GSLs (ground↔sat).

**Fix**:
- Split link budget into:
  - `ISLLinkBudget` (vacuum, no rain)
  - `GSLLinkBudget` (Earth-space; rain attenuation etc)
- Cite ITU-R P.618 for Earth-space rain attenuation modeling when you implement GSLs. citeturn2search3

### 2.4 Routing + congestion
PPT mentions routing and congestion datasets; Tier-1 currently focuses on **connectivity/partition metrics** (GCC frac, num components). That’s defensible if you clearly scope Tier-1 as *topological robustness*, but then you must avoid claims about throughput/congestion until you implement a traffic model.

**Grounding**:
- GCC/LCC as robustness indicator is common in network robustness literature (and is easy to defend). citeturn9search4turn9search10

---

## 3) Dataset + Monte Carlo (schema, leakage, and reproducibility)

### 3.1 What’s good already
- Monte Carlo writes a clean per-run row with config parameters, seeds, failure rates, and aggregate metrics.

### 3.2 Major issues to fix
1. **Schema versioning is not “defense-grade” yet**
   - You need explicit: `schema_version`, `code_commit`, `dependency_lock_hash`, `epoch_iso`, and `sim_engine_version`.
2. **Failure sampling bias**
   - Edge failures are sampled only from edges present at `t=0` (“persistent failures sampled from t=0 edges only”). That can bias results in dynamic graphs where edges appear/disappear. fileciteturn2file4L11-L12
   - Fix: sample failures from the union of all edges across time, OR sample per-timestep stochastic outages (and label accordingly).
3. **Determinism**
   - Ensure *every* random draw derives from a single `run_seed` and that `epoch` is stored + passed everywhere.

---

## 4) ML code review (what’s wrong + what to change)

### 4.1 Design-space model (LightGBM claim)
PPT says “Design and Train LightGBM.” fileciteturn2file4L11-L12

**Current repo**:
- `risk_model.py` is a mixed bag: legacy columns, inconsistent paths, and multiple model types.
- `scripts/train_risk_model.py` looks stale (file paths don’t match current dataset output).

**Fix path (defensible)**:
- Implement `scripts/train_design_model.py` that:
  - Loads the Tier-1 dataset CSV written by Monte Carlo (single source of truth)
  - Splits train/test with a fixed seed + logs the split
  - Trains LightGBM (or explain why you chose a different model)
  - Reports AUC/PR-AUC + calibration curve (because you’re producing a “risk score”)
- Reference LightGBM paper for algorithm choice. citeturn9search1

### 4.2 Temporal GNN (GCLSTM / EvolveGCN alignment)
- Your code uses a GCLSTM-style temporal model, which is a known approach for spatiotemporal/dynamic graph modeling. citeturn1search1
- If you want to match “EvolveGCN-O” language, either:
  1) actually implement EvolveGCN-O and cite the AAAI paper, or
  2) update thesis text to GCLSTM and cite Seo et al.

**Critical correctness bug**:
- `SatNetTemporalDataset` re-simulates the topology with a default epoch (utcnow), so it can drift from the dataset labels generated with a fixed epoch. This can invalidate results.

**Fix**:
- Stop “re-simulating on the fly” inside `__getitem__`.
- Precompute and cache graph sequences (edge_index + node features) keyed by `config_hash + epoch_iso + step_s` and store them under `data/processed/`.
- `__getitem__` should read precomputed tensors.

### 4.3 Environment + dependency reality
PyG is nontrivial to install; you must include exact install steps (CUDA/CPU) and versions in a lockfile. citeturn6search0

---

## 5) Atomic implementation plan for a coding agent (do these in order)

Each step includes: **Goal**, **Files**, **Done when**, **Notes**.

### Step 1 — Make the repo importable + tests runnable
- **Goal**: `pytest` runs.
- **Files**:
  - Add `pyproject.toml` (src-layout package) *or* add `pytest.ini` with `pythonpath=src`.
  - Add `src/satnet/py.typed` if you’re doing mypy later.
- **Done when**: `pytest -q` passes for existing tests.

### Step 2 — Create a reproducible environment spec
- **Goal**: a fresh machine can run Tier-1 simulation.
- **Files**:
  - `requirements.txt` (core sim): `numpy`, `networkx`, `sgp4`, `pydantic`, `typer`, `pandas`…
  - `requirements-ml.txt`: torch + PyG + torch_geometric_temporal (+ documented install command).
  - Optional: `uv.lock` or `poetry.lock`.
- **Done when**: `python -c "import sgp4; import satnet"` works; `scripts/generate_tier1_dataset.py` runs.
- **Notes**: cite SGP4 + PyG install docs. citeturn0search0turn6search0turn7search3

### Step 3 — Remove non-source artifacts + add hygiene
- **Goal**: repo is clean and reviewable.
- **Files**:
  - Delete committed `__pycache__/*.pyc`
  - Add `.gitignore` entries for `__pycache__/`, `*.pyc`, `.venv/`, `data/processed/`, `runs/`
- **Done when**: `git status` is clean after running code/tests.

### Step 4 — Make determinism a first-class contract
- **Goal**: the same config produces identical dataset outputs.
- **Changes**:
  - Add `RunMetadata` struct with `run_seed`, `epoch_iso`, `config_hash`, `code_commit`.
  - Pass `epoch` explicitly into `HypatiaAdapter` everywhere (including the GNN dataset).
  - Use a single RNG (e.g., `numpy.random.Generator(PCG64(run_seed))`) and derive all random values from it.
- **Done when**: re-running `generate_tier1_temporal_dataset(..., run_seed=X)` yields identical CSV (byte-for-byte).

### Step 5 — Fix link-budget conceptual split (ISL vs GSL)
- **Goal**: no “rain margin” in ISLs; rain attenuation only in GSLs.
- **Files**: `src/satnet/network/link_budget.py` (or wherever implemented), plus configs.
- **Done when**:
  - ISL tests assert rain isn’t applied.
  - GSL path exists behind a feature flag.
- **Citation**: ITU-R P.618 for Earth-space rain attenuation. citeturn2search3

### Step 6 — Upgrade TEME→ECEF transformation (or justify simplification)
- **Option A (recommended)**: use `astropy` transforms for TEME→ITRS.
- **Option B**: keep current math but add a sensitivity study script.
- **Done when**: you can cite and defend the chosen approach. citeturn7search0turn7search1

### Step 7 — Make Tier-1 dataset schema explicit + enforceable
- **Goal**: schema is validated at write time and on load.
- **Files**:
  - Add `src/satnet/data/schema.py` with `pydantic` models.
  - Add `tests/test_dataset_schema.py`.
- **Done when**: loading any Tier-1 CSV validates; missing columns fail fast.

### Step 8 — Fix the design-space model to match the thesis (LightGBM)
- **Goal**: reproducible LightGBM baseline for design-space risk.
- **Files**:
  - `scripts/train_design_model.py`
  - `src/satnet/models/design_model.py`
- **Done when**: script emits metrics + model artifact + config hash; results reproducible. citeturn9search1

### Step 9 — Fix temporal GNN training to use cached, aligned graphs
- **Goal**: no on-the-fly re-simulation inside Dataset.
- **Files**:
  - `src/satnet/models/gnn_dataset.py`
  - new `scripts/precompute_temporal_graphs.py`
- **Done when**: training script runs entirely from cached tensors and logs config/epoch.

### Step 10 — Validation harness (committee-proof)
- **Goal**: show correctness, not just code running.
- **What to include**:
  - Unit tests for LOS gating edge cases
  - Cross-check vs an external emulator/simulator for a tiny constellation (Hypatia / LeoEM / StarryNet) on 1-2 metrics.
- **Grounding**: satellite network emulation/simulation tools and surveys can justify your baseline choices. citeturn4search1turn4search4turn1search4turn3search1

---

## 6) Research questions a “computer-use agent” should answer (to tighten the dissertation)

These are explicitly the “go research and bring back citations + numbers” items.

1. **Realistic failure models**:
   - What are empirically reported failure rates for LEO satellites and ISLs (hardware, pointing, software)?
   - Are outages best modeled as persistent, transient, or correlated events (solar storms, debris avoidance maneuvers, common-mode software bugs)?
2. **ISL link budget realism**:
   - Typical terminal parameters for 1550nm optical ISLs (aperture sizes, pointing loss, receiver sensitivity) and for Ka-band ISLs.
   - When does Doppler/pointing/beam divergence dominate vs FSPL?
3. **Frame transformation fidelity**:
   - Quantify how much simplified TEME→ECEF changes LOS decisions at your chosen time-step and altitudes.
4. **Metric validity**:
   - For a DSS aimed at EM decisions, is GCC fraction sufficient, or do you need latency stretch / path diversity / min-cut metrics?
   - Find papers using GCC/LCC for comm network robustness and justify your label choice. citeturn9search4turn9search10
5. **Model choice justification**:
   - Compare GCLSTM vs EvolveGCN(-O) on dynamic graph classification and pick based on fit + interpretability. citeturn1search1turn8search48
6. **External tool alignment**:
   - Identify which external simulator best matches your “Tier-2+” ambitions (routing + congestion + ground segment): Hypatia vs others; cite survey + emulator docs. citeturn4search1turn4search4turn1search4turn3search1

---

## 7) Quick “committee defense” bullets (use these in writing)

- **Reproducibility**: every run is keyed by `config_hash + code_commit + dependency_lock_hash + epoch_iso + run_seed`.
- **Scope control**: Tier-1 measures *topological robustness* (GCC/LCC) rather than throughput; routing/traffic is reserved for Tier-2.
- **Physics fidelity**: SGP4/WGS72 is the Tier-1 propagator; frame transforms are either astropy-backed or sensitivity-tested.
- **Model justification**: LightGBM baseline for interpretable design-space risk; temporal GNN for dynamic topology with citations.

---

## Appendix A — Minimal “golden commands” (after fixes)

```bash
# 1) create env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) run tests
pytest -q

# 3) generate Tier-1 dataset
python scripts/generate_tier1_dataset.py --runs 200 --out data/tier1_v1.csv --seed 123 --epoch 2026-01-01T00:00:00Z

# 4) train design model
python scripts/train_design_model.py --data data/tier1_v1.csv --seed 123 --out runs/design_lgbm/

# 5) precompute graphs + train GNN
pip install -r requirements-ml.txt
python scripts/precompute_temporal_graphs.py --data data/tier1_v1.csv --out data/processed/
python scripts/train_gnn_model.py --data data/tier1_v1.csv --processed data/processed/ --seed 123
```

(These commands are targets; the coding agent should implement the scripts/flags to make them real.)

