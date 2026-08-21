# Stage A RF Regression Training v1 Authorization

**Verdict: STAGE A RF REGRESSION TRAINING AUTHORIZED — PREFLIGHT PASSED — READY TO EXECUTE**

This authorization binds one deterministic `RandomForestRegressor` campaign to the accepted remediated Stage A training plan and accepted RF dataset. It authorizes preparation and one future execution only. The PowerShell execution command is deliberately not run by this task.

## Controls

- Exactly 24 complete `RandomForestRegressor` constructor dictionaries from the frozen scikit-learn 1.9.0 specification; all 24 instantiated in constructor-only preflight without `fit()`.
- Candidate fitting, when later executed, uses training rows only. Validation is used only for candidate selection and reporting. No train/validation merge is permitted.
- Exactly the frozen 12 predictors and order; `criterion="squared_error"`, `n_jobs=1`, no scaling, no target normalization, and no target transformation.
- Candidate seeds are `41000 + candidate_index`; final seeds are `51001` through `51005`.
- Validation reporting includes run-level metrics, five-realization design means, residual population standard deviation, and design-aware bootstrap intervals with seed `63002` and 10,000 replicates.
- Undefined R² remains null; only defined R² bootstrap replicates are used for R² percentiles, with valid and undefined counts reported.
- The sealed test index and all test target sources are prohibited and were not opened by preflight.
- The output root and lock were absent at preflight; the command refuses an existing root or stale lock.
- The command verifies the exact activation HEAD, activation branch, clean activation worktree, and committed runner identity before executing exactly once.
- The command preserves the Python process exit code, writes a timestamped transcript outside the output root, and omits null, empty, or whitespace-only optional arguments.

## Required future command

Run the exact PowerShell command returned by the authorization response. Replace no arguments. The command must be executed against the clean activation worktree and exact activation HEAD recorded in `runtime_plan.json`.

## Artifacts

- `active_authorization.json` — semantic authorization scope and immutable input identities.
- `runtime_plan.json` — one-time runtime controls, identity checks, output-root and transcript policy.
- `preflight_certificate.json` — read-only preflight evidence and non-execution confirmations.
- `rf_regression_training_command.ps1` — fail-closed PowerShell activation wrapper.
- `authorization_inventory.json` — self-excluding exact-file inventory.
- `README.md` — this operational summary.
