# Advisor Questions

1. Why use Random Forest?
   Random Forest is a fast, interpretable baseline over run-level design features. It gives feature importances and a sanity-check benchmark before using a higher-variance GNN.

2. Why use a Temporal GNN?
   Resilience is temporal. The GNN consumes `G0 -> G1 -> ... -> GT`, so it can learn topology evolution instead of only tabular summaries.

3. What prevents timestep leakage?
   RF trains on one row per run. GNN dataset index `i` is one complete run sequence, and the training script splits complete run indices.

4. Why split by run instead of timestep?
   Timesteps from the same run share constellation design, seed, failure realization, and label. Splitting timesteps would leak near-identical samples across train/test.

5. Are labels leaky?
   The core labels are pure GCC metrics from graph state only. Failure probabilities are inputs for a stated-scenario risk task, but failure outcomes are not RF features.

6. What are the RF inputs?
   `num_planes`, `sats_per_plane`, `total_satellites`, `inclination_deg`, `altitude_km`, `node_failure_prob`, `edge_failure_prob`, `duration_minutes`, `step_seconds`.

7. What RF feature is missing?
   `phasing_factor` exists in rollout config but is not currently sampled/exported in the run-level dataset.

8. What are the targets?
   `partition_any`, `partition_fraction`, `gcc_frac_min`, `gcc_frac_mean`, and `max_partition_streak`.

9. Is risk tier implemented?
   Risk binning exists as a separate metrics module, but `risk_tier` is not currently a canonical training target.

10. What does RMSE measure?
    RMSE is the square root of mean squared prediction error. It is in the target's units and penalizes large errors more than MAE.

11. What does R2 tell us?
    R2 estimates how much target variance the model explains relative to predicting the mean. Negative R2 means worse than that baseline.

12. What makes a good confusion matrix?
    For this task, both robust and partitioned classes should have nonzero support, and false negatives on partitioned runs matter most operationally.

13. What artifacts are produced?
    Runs/steps CSVs, RF model checkpoints, metrics JSON, predictions CSV, feature importance CSV, confusion matrix PNG, prediction-vs-actual PNG, and experiment logs.

14. How do you know the model is improving?
    Compare validation metrics across runs and hold test metrics for final reporting. The current smoke run only proves execution, not convergence.

15. Why is the GNN currently blocked locally?
    `torch_geometric_temporal` is missing, and its `torch-sparse` dependency did not install cleanly in the local Python 3.12 test environment.

16. Is the physics layer Tier 1?
    Yes. It uses Walker Delta configuration, TLE generation, SGP4 when available, Earth obscuration, and optical/RF link budgets.

17. Are ground stations modeled?
    No. Phase 1 is satellite-to-satellite only by design.

18. How is reproducibility handled?
    Runs include `seed`, `config_hash`, `epoch_iso`, schema version, dataset version, and failure realization JSON for graph reconstruction.

19. What did the smoke test prove?
    Dataset generation, RF classification, RF regression, target analysis, metrics export, prediction export, feature export, and plots execute end to end.

20. What should happen next?
    Stabilize the GNN environment, add phasing-factor sampling/export, run larger datasets, and compare RF vs GNN on fixed run-level splits.
