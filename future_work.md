# Future Work

## High Priority

- Fix the GNN dependency path for `torch-geometric-temporal` and `torch-sparse`.
- Run the GNN smoke test after dependencies are stable.
- Add `phasing_factor` to dataset export and RF feature selection if it will vary experimentally.
- Run a larger Tier 1 dataset, at least hundreds of runs, using fixed seeds.
- Save a canonical split manifest so RF and GNN compare on identical run IDs.

## Medium Priority

- Add design-only RF mode for predicting architecture risk without failure probability inputs.
- Add calibration plots and ROC/PR curves for `partition_any`.
- Add confidence intervals or repeated-seed evaluation for RF metrics.
- Add Parquet export once CSV schema is stable.
- Add a dependency installation note for Apple Silicon PyG temporal packages.

## Low Priority

- Improve console formatting for target summaries.
- Add a richer feature set for GNN nodes, such as altitude/inclination/design constants.
- Add optional cache telemetry to GNN metrics JSON.
- Add Makefile targets for regression RF smoke and target audit.

## Technical Debt

- Legacy RF helpers in `src/satnet/models/risk_model.py` still support pre-Tier-1 schemas and should remain quarantined or be retired.
- `docs/codemap_data_to_decision_pipeline.md` contains stale references to legacy simulation modules.
- Current `.venv` is Python 3.14, which conflicts with the project dependency pins; use Python 3.11 or 3.12.
- `torch_geometric_temporal` installation should be documented with exact wheel/source strategy.

## Future Experiments

- Compare RF and GNN on `partition_any`, `gcc_frac_min`, and `partition_fraction`.
- Compare full feature RF vs design-only RF.
- Sweep failure rates and constellation sizes to find balanced regimes.
- Evaluate proxy ranking quality between `gcc_frac_min` and `partition_fraction`.
- Test sensitivity to timestep size and simulation horizon.

## Dissertation Improvements

- Formalize the data leakage argument with a run-level split diagram.
- Define the Phase 1 truth-label scope: satellite-only temporal GCC resilience.
- Document v1 failure assumptions and the planned v2 correlated/probabilistic failure model.
- Include reproducibility tables listing seed, config hash, schema version, and git SHA.
- Separate "demo smoke metrics" from "research-quality experiment metrics" in advisor slides.
