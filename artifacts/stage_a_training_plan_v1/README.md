# Stage A Training Plan v1

**Verdict: STAGE A TRAINING PLAN V1 REMEDIATED — READY FOR TARGETED INDEPENDENT RE-GATE**

This committed artifact freezes only the Stage A training, validation, model-selection, preprocessing, reproducibility, compute-budget, and one-time final-test protocol. It does not train an RF or TGNN model, create checkpoints, calculate fitted normalization statistics, open or materialize test targets, modify an accepted dataset/evidence artifact, or perform the independent training-plan gate.

## Remediation provenance

- Original training-plan HEAD: `7607eb88156862706e74710fc8da42db66abc862`
- Blocked independent gate commit: `67e2633e10d942165b64008570e3f7cb1396e0f6`
- Blocked gate report SHA-256: `75560841527dae6c3526bafc1c41d8adbfcf9e00f491a9d05a819f39e3b409df`
- Remediated defects:
  1. Every parameter from installed scikit-learn 1.9.0 `get_params(deep=False)` is explicit in all 48 RF candidate estimator dictionaries, including `n_jobs: 1`.
  2. RF simplicity uses a complete deterministic lexicographic key, with `sqrt` ranked simpler than `0.75` and explicit `None` handling.
  3. Regression run-level, design-level, residual, and design-aware bootstrap formulas are exact and frozen.
  4. The future TGNN preprocessing statistics schema, canonical JSON serialization, float64 hexadecimal statistics, SHA-256 byte domain, and exact byte count are frozen.
- All other frozen policies remain unchanged. Accepted datasets, dataset gates, and the prior blocked gate artifacts are not modified.

## Authoritative accepted inputs

- RF dataset manifest SHA-256: `3fae8566a2194f6a2410e8173bd4619955631226a6a213fffe22769bf74dfb16`
- RF independent gate report SHA-256: `9f47a7a1beeceef5907025f93930ef1b15c1976ac5af049abd3474678dc81414`
- RF gate commit: `5ca5d984645466321a8733c59a2dd5a879d8ce50`
- TGNN dataset manifest SHA-256: `6023bcde5f62d1724485789ed2876ac84f0619839b6de1bf8072067f92df252e`
- TGNN independent gate report SHA-256: `8d095dbd3b184c8ba6d00d474cf75bb921060034bc45258f3e9aca3bfb4afe6a`
- TGNN gate commit: `ea32faab3945f24e5ed234947b03cda56dbdd95f`
- Both dataset statuses: **accepted**

Both accepted datasets use 70 train designs / 350 runs, 15 validation designs / 75 runs, and 15 sealed test designs / 75 runs. Five realizations per design remain together. Classification is `partition_any` from `overall_threshold_breach_any`; regression is `gcc_frac_min` from `space_gcc_fraction_original_min`.

## Frozen model list

1. RF classification — `RandomForestClassifier`
2. RF regression — `RandomForestRegressor`
3. TGNN classification — accepted contracted one-layer GCLSTM, Chebyshev `K=2`
4. TGNN regression — accepted contracted one-layer GCLSTM, Chebyshev `K=2`

## Frozen protocol highlights

- Classification class weights use training labels only: `w_c = N_train / (2 * n_train_c)`.
- Classification selection: weighted validation log loss, macro-F1, balanced accuracy, class-0 recall/specificity, simpler model.
- Classification threshold: exactly `0.5`; no threshold optimization or calibration.
- Regression selection: validation RMSE, MAE, R², simpler model.
- Regression formulas use `error = y_predicted - y_actual`; a zero R² denominator is recorded as undefined/null, never zero. Residual standard deviation uses population denominator `N`.
- Regression uncertainty: design-aware bootstrap, 10,000 replicates, seed `63002`, 95% percentile intervals; all five realizations remain together for every selected design, and defined/undefined R² replicate counts are reported.
- RF search: 24 deterministic candidates per task, 48 complete executable parameter dictionaries total, one worker, five final seeds.
- RF simplicity key: `(n_estimators, max_depth_is_none, max_depth_value_if_not_none_else_0, -min_samples_leaf, max_features_simplicity_rank, candidate_index)`, ascending; `sqrt` rank `0`, `0.75` rank `1`, and `max_depth=None` follows finite depths.
- TGNN search: 4 deterministic candidates per task, one worker, 60-epoch maximum, patience 12, five final seeds.
- TGNN input contract: combined satellite-ground graph, 14 accepted node features, 10 accepted edge attributes, variable node counts, and exact zero-edge snapshots. The contracted GCLSTM consumes connectivity and preserves but does not reduce `edge_attr`.
- Preprocessing: no RF scaling; future training-only standardization of six applicable physical ground-node coordinate features; binary, one-hot, identifier-index, and status fields remain unscaled; no target transformation. Future statistics use float64 `float.hex()` strings and canonical JSON bytes hashed with SHA-256.
- Reproducibility: Python 3.11.9, Torch 2.12.1, PyG 2.8.0, PyG Temporal 0.56.2, scikit-learn 1.9.0, CPU-only, deterministic algorithms, one worker, no shuffle, and fixed seed schedules.
- Test opening: one-time authorization after all four identities and fitted weights are frozen; evaluate all four once; no changes after results open.

## Files

- `stage_a_training_plan.json` — top-level freeze, remediation provenance, invariants, and execution order
- `common_split_usage_policy.json` — split and sealed-test rules
- `classification_metric_and_imbalance_policy.json` — imbalance, threshold, metrics, and uncertainty
- `regression_metric_policy.json` — exact regression metrics, aggregation, residuals, and bootstrap
- `rf_training_specification.json` — exact 12 predictors, complete RF parameter dictionaries, search, seeds, and serialization
- `tgnn_training_specification.json` — exact accepted graph/model/training contract and search
- `preprocessing_specification.json` — future statistics artifact schema, canonical serialization, hashing, training-only preprocessing, and sentinel handling
- `reproducibility_and_seed_schedule.json` — versions, deterministic settings, ordering, and capture
- `model_selection_policy.json` — unambiguous selection hierarchies and complete RF simplicity key
- `final_test_opening_protocol.json` — one-time test authorization and post-opening lock
- `compute_budget.json` — exact candidate, epoch, seed, timeout, and storage budgets
- `artifact_inventory.json` — SHA-256 inventory of this directory, excluding the inventory itself

The inventory is intentionally self-excluding. This directory is the only intended scope of the plan commit; unrelated pre-existing worktree changes are not part of it.
