# Stage A Training Plan v1 Narrow Independent Acceptance Gate

**Verdict: STAGE A TRAINING PLAN V1 BLOCKED — TARGETED REMEDIATION REQUIRED**

This gate is read-only validation of the frozen plan at implementation HEAD `7607eb88156862706e74710fc8da42db66abc862`. It did not train RF or TGNN models, instantiate a training optimizer, create checkpoints or fitted preprocessing artifacts, calculate normalization statistics, open test targets, or modify accepted datasets/evidence.

## Results

- All 13 declared plan artifact SHA-256 values match.
- The plan inventory is self-excluding and binds the other 12 files.
- RF and TGNN accepted dataset gate identities and supplied manifest/report hashes remain accepted and unmodified.
- Common split, model coverage, target aliases, class weights, search arithmetic, TGNN contract, classification policy, reproducibility rules, and final-test seal pass the checks recorded in the JSON reports.
- RF budget: 24 candidates + 5 final fits = 29 fits per task; 58 fits total; `29 * 2 * 400 = 23,200` maximum trees.
- TGNN budget: 4 candidates + 5 final runs = 9 runs per task; 18 runs total; `18 * 60 = 1,080` maximum run-epochs.
- Expected training class weights: class 0 `350 / (2 * 49) = 3.5714285714285716`; class 1 `350 / (2 * 301) = 0.5813953488372093`.
- Runtime package versions match exactly. RF imports and no-fit estimator constructor checks pass. TGNN import is currently blocked by an incompatible `torch_sparse` native extension and requires environment preparation before TGNN authorization.

## Targeted plan blockers

1. Explicitly freeze every RF estimator parameter, including the omitted parameters identified in `search_budget_verification.json` and `n_jobs=1`.
2. Define the exact RF `max_features` simplicity order for `sqrt` and `0.75`.
3. Add exact run-level and design-level RMSE, MAE, and R2 formulas to the regression policy.
4. Define the exact preprocessing artifact serialization and canonical SHA-256 hashing procedure.

These are recorded only; this gate does not revise the plan. The existing unrelated worktree changes were preserved.

## Gate artifacts

- `training_plan_gate_report.json`
- `runtime_compatibility_report.json`
- `search_budget_verification.json`
- `metric_and_selection_verification.json`
- `preprocessing_and_test_seal_verification.json`
- `artifact_inventory.json` — self-excluding inventory for the six files above
- `README.md`
