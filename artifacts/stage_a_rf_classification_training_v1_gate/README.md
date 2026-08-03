# Stage A RF Classification Training v1 Independent Gate

Mode: read-only completed-training acceptance gate.

Verdict: **STAGE A RF CLASSIFICATION TRAINING V1 BLOCKED — TARGETED REMEDIATION REQUIRED**

The verifier loaded only the frozen training and validation CSVs and the five frozen RandomForestClassifier joblib files. It did not open or materialize test targets, call `fit()`, modify training artifacts, alter selection, train RF regression, or train either TGNN.

The source training artifact inventory was checked against the declared 15 exact-file identities, and pre/post SHA-256 snapshots were compared. Candidate selection was recomputed from the frozen hierarchy and the complete candidate dictionaries. Validation probabilities, run metrics, design aggregation, and the 10,000 design-level bootstrap replicates were independently recomputed.

Bootstrap warnings are acceptable only when undefined metrics remain null with zero valid replicates. See `bootstrap_verification.json` and the gate report for the concrete result.

Gate artifact files are listed in `artifact_inventory.json`; that inventory is self-excluding.
