# Stage A RF Classification Training v1 Bootstrap Reporting Remediation

This record corrects the frozen bootstrap reporting artifact only. It regenerates the design-level, 10,000-replicate bootstrap from the existing validation prediction records; it does not fit, search, predict, retrain, access test targets, or modify model files.

The corrected class-0 specificity and ROC-AUC intervals are explicitly undefined when a replicate lacks the required class support. Undefined values are represented as null with valid and undefined replicate counts. Ordinary validation artifacts, candidate 20, threshold 0.5, and all five model identities remain unchanged. No independent re-gate was performed.

Verdict: **STAGE A RF CLASSIFICATION BOOTSTRAP REPORTING REMEDIATED — READY FOR TARGETED RE-GATE**
