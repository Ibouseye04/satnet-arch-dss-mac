# Tonight Checklist

## Completed

- [x] Fetch origin and audit repository state.
- [x] Map physics, simulation, metrics, RF, and GNN modules.
- [x] Add RF run-level train/validation/test splitting.
- [x] Add explicit RF precision, recall, F1, confusion matrix, MAE, RMSE, and R2 reporting.
- [x] Add RF `feature_importance.csv`, `predictions.csv`, metrics JSON, and optional plots.
- [x] Add dataset `--smoke`.
- [x] Add RF `--smoke`.
- [x] Add GNN `--smoke`, `--subset`, validation split, and validation-based checkpoint selection.
- [x] Generate smoke Tier 1 dataset.
- [x] Run RF classification smoke.
- [x] Run RF regression smoke.
- [x] Generate target summary and correlation CSVs.
- [x] Run non-long tests: `160 passed, 1 skipped`.
- [x] Run Tier 1 guardrails: `28 passed`.
- [x] Run Tier 1 contract: `8 passed`.

## Verify Before Meeting

- [ ] Confirm current branch with `git branch --show-current`.
- [ ] Decide whether to keep generated `artifacts/smoke/` files in the working tree.
- [ ] Open `artifacts/smoke/rf/design_risk_model_tier1_metrics.json`.
- [ ] Open `artifacts/smoke/rf/design_risk_model_tier1_confusion_matrix.png`.
- [ ] Open `artifacts/smoke/rf/rf_gcc_frac_min_metrics.json`.
- [ ] Show `data/tier1_design_runs.csv` columns if asked.
- [ ] Be ready to explain why GNN training is dependency-blocked locally.

## Demo Commands

```bash
python scripts/export_design_dataset.py --smoke --seed 42
python scripts/train_design_risk_model.py --smoke --target-name partition_any --seed 42
python scripts/train_design_risk_model.py --smoke --target-name gcc_frac_min --seed 42
python tools/analyze_dataset_targets.py data/tier1_design_runs.csv
```

## Do Not Spend Time Tonight

- [ ] Do not add ground stations.
- [ ] Do not redesign failure models.
- [ ] Do not hand-tune model hyperparameters on the test split.
- [ ] Do not claim GNN results until the PyG temporal dependency is fixed.
