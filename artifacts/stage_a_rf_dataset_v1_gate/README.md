# Stage A RF Dataset v1 Independent Gate

Verdict: **STAGE A RF DATASET V1 ACCEPTED — READY FOR TGNN DATASET CONSTRUCTION**

This directory contains the narrow independent, read-only acceptance gate for `artifacts/stage_a_rf_dataset_v1`. The gate did not rebuild the RF dataset, inspect test targets, build a TGNN dataset, train/tune a model, or modify production, replay, acceptance, freeze, or unrelated worktree files.

- Independent gate implementation HEAD: `c6c76cfdbcae4ba278a51d0d75d2ac3ae7fbdc3a`
- Dataset implementation HEAD: `c6c76cfdbcae4ba278a51d0d75d2ac3ae7fbdc3a`
- Dataset root: `C:\Users\johns\Developer\satnet-arch-dss-mac\artifacts\stage_a_rf_dataset_v1`
- Freeze manifest: `C:\Users\johns\Developer\satnet-arch-dss-mac\artifacts\stage_a_accepted_production_freeze_v1\accepted_production_evidence_freeze_manifest.json`
- Freeze manifest SHA-256: `b2d1fbd9510d3d828fe051b4da04f6747088ae00d78d65e0a9851dc85559844e`

## Bound dataset artifacts

| Artifact | SHA-256 |
|---|---|
| `rf_train.csv` | `b5098f3cafbebef97e71425a79bd75bc0625d63ca3578e2d7d6b7bbc57b1aa4b9` |
| `rf_validation.csv` | `376d615efbf8a538456975fb3f932d83e7b9bab254066fb4bfcc08aa3cb40c6d` |
| `rf_test_sealed_index.csv` | `40e9120ed3b735bf94711cce7d44952680a10733f623245c17960c7f8e923bc2` |
| `rf_feature_schema.json` | `5452b4d578948c2d0a1835967a768c217f584d453babd884ed62335e23a9ca39` |
| `rf_dataset_manifest.json` | `3fae8566a2194f6a2410e8173bd4619955631226a6a213fffe22769bf74dfb16` |
| `rf_provenance_manifest.jsonl` | `54bb8b61dcecd43b293a84dbf1ff8af2c226781cdccd7f350794b7f79ca74b8d` |
| `rf_leakage_exclusion_report.json` | `196d31ee9e892e8fff16991b27e69fc59fb90cd6b7e9baafcb71c6cccf818f88` |
| `rf_construction_report.json` | `ae9b41ef55a74e3f0ea8bd02051c82169a15ecece71d9e645617cdfecb295d95` |

## Acceptance summary

- Split counts: train 70 designs / 350 rows; validation 15 designs / 75 rows; sealed test index 15 designs / 75 rows.
- Five realizations per design; zero design overlap; zero run overlap; zero duplicate run keys; zero duplicate design/realization pairs.
- Predictors: exactly 12, ordered identically in train and validation, complete numeric, and leakage-free.
- Classification target: `partition_any` ← `overall_threshold_breach_any`.
- Regression target: `gcc_frac_min` ← `space_gcc_fraction_original_min`. The independent contract/source trace confirms this is the space-only original-denominator GCC minimum, not the overall/integrated service minimum; no additional transformation, thresholding, aggregation, or semantic change was introduced.
- Target quality and distributions are recorded in `rf_dataset_gate_report.json`. Test targets were not inspected.
- Provenance: all 425 train/validation rows resolve to frozen source results, scientific inventories, and canonical target artifacts; all 75 test rows are provenance-only.
- Test seal: sealed; no target values materialized; outcome fields unused.

See `rf_target_mapping_verification.json` for the critical scientific mapping check and `rf_schema_verification.json` for the independent column/leakage check.
