# Agent Handoff Prompt — SatNet Tier 1 Continuation

**Date:** 2026-01-07  
**Repo:** `satnet-arch-dss-mac`  
**Context:** Tier 1 cleanup refactor is COMPLETE. Next phase is ML pipeline alignment.

---

## What Was Completed (Tier 1 Cleanup — Steps 0-8)

All items from `docs/refactor_plans/2026-01-06_tier1_cleanup_structural_fixes.md` are done:

| Step | Description | Status |
|------|-------------|--------|
| 0 | Baseline guardrail tests | ✅ |
| 1 | Quarantine legacy snapshot engine | ✅ |
| 2 | Enforce epoch contract | ✅ |
| 3 | Graph reconstruction contract for ML | ✅ |
| 4 | Make failure semantics explicit | ✅ |
| 5 | Physics fidelity boundaries | ✅ |
| 6 | Dependency pinning | ✅ |
| 7 | ML naming cleanup (GCLSTM) | ✅ |
| 8 | Config hash collision fix (64-char SHA256) | ✅ |

### Key Files Modified
- `src/satnet/simulation/tier1_rollout.py` — Added `Tier1FailureRealization`, failure semantics docs, full SHA256 hash
- `src/satnet/simulation/monte_carlo.py` — Added `failed_nodes_json`, `failed_edges_json` to schema
- `src/satnet/models/gnn_dataset.py` — Now applies failure realization when regenerating graphs
- `src/satnet/network/hypatia_adapter.py` — Rain margin disabled for ISLs
- `pyproject.toml` — Created with pinned dependencies
- `docs/physics_constants.md` — Created with cited link budget parameters
- `tests/test_tier1_guardrails.py` — 28 guardrail tests covering all contracts

### Test Status
All 28 guardrail tests pass on remote Windows machine.

---

## What Remains (from `docs/codereview/SatNet_Code_Review_Agent_Brief.md`)

### Priority 1: LightGBM Design Model (Step 8 of Brief)
**Problem:** Thesis/PPT claims LightGBM for design-space model, but `risk_model.py` uses LogReg/RandomForest.

**Action:**
- Create `scripts/train_design_model.py` that trains LightGBM on Tier 1 dataset
- Or update thesis to say RandomForest (less work, but weaker claim)

### Priority 2: Graph Caching for GNN (Step 9 of Brief)
**Problem:** `SatNetTemporalDataset` regenerates graphs on-the-fly in `__getitem__`. This is slow and was a correctness risk (now fixed with explicit epoch + failures).

**Action:**
- Create `scripts/precompute_temporal_graphs.py` to cache graph tensors
- Update `gnn_dataset.py` to load from cache instead of regenerating

### Priority 3: External Validation (Step 10 of Brief)
**Problem:** No cross-validation against external simulator (Hypatia, etc.)

**Action:**
- Add a small validation script comparing ISL counts or GCC metrics against Hypatia output for a reference constellation

---

## Key Constraints (from AGENTS.md)

1. **NO TOY TOPOLOGY** — Never use `satnet.network.topology` (quarantined)
2. **TEMPORAL EVALUATION** — Always iterate `t=0..T`, never just `t=0`
3. **NO GROUND STATIONS (Phase 1)** — Satellite-to-satellite only
4. **DETERMINISM** — All runs reproducible from `config_hash + seed + epoch`
5. **User runs heavy scripts on remote Windows** — Don't run locally, just validate syntax

---

## Environment Setup (Remote Windows)

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Install package (editable)
pip install -e .

# Run tests
pytest tests/test_tier1_guardrails.py -v
```

---

## Reference Documents

- **Refactor plan (DONE):** `docs/refactor_plans/2026-01-06_tier1_cleanup_structural_fixes.md`
- **Code audit:** `docs/codereview/2026-01-06_codebase-audit_tier1-vs-readme_thesis.md`
- **Agent brief (remaining work):** `docs/codereview/SatNet_Code_Review_Agent_Brief.md`
- **Physics constants:** `docs/physics_constants.md`
- **Agent rules:** `AGENTS.md`

---

## Suggested Next Steps

1. **If continuing ML alignment:**
   - Implement LightGBM training script
   - Add graph caching for GNN dataset

2. **If starting new feature work:**
   - Review `docs/research/` for Phase 2 plans (ground stations, correlated failures)

3. **If validating current state:**
   - Run full test suite: `pytest -v`
   - Generate a small dataset: `python scripts/export_design_dataset.py`
