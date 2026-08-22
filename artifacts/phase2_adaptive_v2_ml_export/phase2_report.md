# SATNET Adaptive-v2 Phase 2 ML Export

## Final status

**PASS.** This phase materialized RF tabular datasets and space-only temporal TGNN graph sequences from persisted accepted adaptive-v2 artifacts. No simulation, replay, training, fitting, tuning, model selection, held-out evaluation, robustness run, external validation, or checkpoint rebinding was performed.

## Lineage and execution

- Dedicated worktree/branch: `C:\Users\johns\Developer\satnet-arch-dss-mac-adaptive-v2` / `experiment/final-integrated-dataset-10k-adaptive-v2`
- Accepted Phase-1 scientific source SHA: `346b3ff1670237645acf4836283adbbdc359093a`
- Phase-2 exporter tooling SHA: `767b9161f0301236153551a2319dd570d7dd342f`
- Canonical Phase-1 report: `C:\Users\johns\external\satnet-10k-final-production-v2-adaptive\phase1_report.json`
- Phase-1 report SHA-256: `e3177222eacd046b1f78c21e19e17e896620e3a919979366436b138046761ccd`
- Adaptive contract specification: `23c5fffc10849c3bc3ea027251ac3e5ad4c96f0eea85edf1e8deab079cb0871e`
- Adaptive contract bundle: `da3c73711b1d60635afcceee8bda0a60d1379e492e1ad0d588d5a0c25e10abe3`
- Output root: `C:\Users\johns\external\satnet-10k-final-ml-datasets-v2-adaptive`
- Output inventory bundle/content hash: `1399f12e6ed7e159076028bc01971cf2274924fc55b43cca0a44219db8c978d1`

## Gates

- Source runs/designs/realizations: `10000 / 2000 / 5`
- Splits: train `7000 / 1400`, validation `1500 / 300`, test `1500 / 300` runs/designs
- Topology: `('grid_adaptive', 1, 1, 'persistent_temporal_union_edges_v1'): 10000`
- Grid-fixed source contamination: `0`
- Missing/duplicate run IDs: `0 / 0`
- Design cross-split leakage: `0`
- Source/export target mismatches: `0`
- NaN/Inf/out-of-range/illegal values: `0 / 0 / 0 / 0`
- Historical fixed ML root unchanged: `true`

## Task outputs

RF exports have one row per run. Space features are `num_planes`, `sats_per_plane`, `altitude_km`, `inclination_deg`, `satellite_node_failure_probability`, and `satellite_edge_failure_probability`. Integrated RF adds `civilian_count`, `government_count`, `military_count`, and `ground_station_failure_probability`.

- `rf_space_classification`: target `space_threshold_breach_any`, 10,000 rows
- `rf_space_regression`: target `space_gcc_fraction_original_min`, 10,000 rows
- `rf_integrated_classification`: target `overall_threshold_breach_any`, 10,000 rows
- `rf_integrated_regression`: targets `failure_adjusted_overall_service_fraction_mean` and `failure_adjusted_overall_service_fraction_min`, 10,000 rows

TGNN is space-only, one complete run per sample, 11 ordered timesteps at 60-second cadence, node dimension 3, and edge dimension 4. There are 10,000 sequences shared by the two target manifests.

## Observed class distributions

- Space classification: overall false/true `1443/8557`; train `950/6050`; validation `266/1234`; test `227/1273`.
- Integrated classification: overall false/true `62/9938`; train `37/6963`; validation `15/1485`; test `10/1490`.

The corrected integrated distribution is reported without threshold changes, balancing, resampling, synthesis, or dropping examples.

## Graph parity

Representative persisted adaptive source graphs covered P01, P02, P03, P04, P05 and train/validation/test. Run 9 (`D0001-R04`, validation) proves adaptive edge `(0,15)` (`inter_plane`) at timestep 0: it exists in the persisted source graph and in the serialized TGNN representation. Source and serialized sequence identity, 11-step order, node/edge counts, and 3/4 feature dimensions were checked.

The complete machine-readable manifest is retained externally at `C:\Users\johns\external\satnet-10k-final-ml-datasets-v2-adaptive\phase2_manifest.json`; the tracked summary is `phase2_manifest.json` in this directory. The historical fixed exporter remains the default fixed profile and rejects adaptive source roots, while the adaptive entrypoint requires the accepted adaptive source/replay roots and adaptive contract identity.
