# Stage A RF Classification Training v1 Corrected Gate

This directory is the targeted reporting-and-acceptance correction for the blocked Stage A RF classification gate. It independently reads the frozen validation prediction records and frozen JSON evidence, verifies candidate 20, the five frozen model identities, run-level metrics, design aggregation, and the design-level bootstrap policy.

No model was loaded for inference. No training, fitting, preprocessing fitting, candidate search, threshold optimization, calibration fitting, validation prediction generation, model write, test-data access, RF regression change, or TGNN change occurred. The completed training-output root was byte-identical before and after this read-only verification.

The corrected bootstrap uses seed 63001, 10,000 replicates, 15 design draws per replicate with replacement, all five realizations retained per selected design occurrence, explicit confusion-matrix labels `[0, 1]`, percentile bounds 2.5 and 97.5, and deterministic duplicate occurrence identities. Undefined metrics are excluded from percentile calculations and represented with JSON `null` bounds and explicit valid/undefined counts.

The blocked historical gate commit `4efc799dc6263f71c55cd9395a0047abdbabf48a` remains unchanged and is superseded for acceptance. The only corrected bootstrap mismatch is class-0 recall/specificity: its independently recomputed bounds are `null` with 0 valid and 10,000 undefined replicates for every final seed. The defective historical representation is preserved as evidence and marked reporting-only.

Verdict: **STAGE A RF CLASSIFICATION TRAINING V1 ACCEPTED — BOOTSTRAP REPORTING DEFECT CORRECTED — READY FOR FOUR-MODEL SEALED TEST AUTHORIZATION**
