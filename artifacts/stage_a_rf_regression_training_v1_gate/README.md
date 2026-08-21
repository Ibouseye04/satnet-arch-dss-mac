# Stage A RF Regression Training v1 Independent Gate

Mode: read-only completed-training independent acceptance gate.

Verdict: **STAGE A RF REGRESSION TRAINING V1 ACCEPTED — READY FOR TGNN CLASSIFICATION TRAINING AUTHORIZATION**

The gate loaded the declared completed-training artifacts, the frozen validation CSV, and the five frozen RandomForestRegressor joblib models. It did not open the sealed test index or any test target source, call `fit()`, rerun candidate search, write predictions or metrics into the training output root, retrain or replace models, train either TGNN, or modify unrelated worktree files.

Candidate ranking was recomputed from the frozen candidate result table using validation RMSE ascending, validation MAE ascending, validation R2 descending, and the frozen RF simplicity key. Candidate 8 was independently confirmed as the selected candidate. All five final models were loaded without fitting and independently evaluated on the frozen validation split.

The gate recomputed run-level and design-level metrics, residual population standard deviations, and the 10,000-replicate design-aware bootstrap with seed 63002. Undefined R2 values remain null and are excluded only from R2 percentile calculation.

Gate artifacts are listed in `artifact_inventory.json`; that inventory is self-excluding.
