# Stage A TGNN Classification Training v1 Acceptance Gate

## Verdict

**STAGE A TGNN CLASSIFICATION TRAINING V1 ACCEPTED — READY FOR TGNN REGRESSION TRAINING AUTHORIZATION**

This is a narrow independent, read-only completed-training gate. The completed output root was not modified. The verifier loaded frozen checkpoints on CPU and recomputed validation inference only; it did not train, fit, construct an optimizer, step an optimizer, backpropagate, refit preprocessing, regenerate predictions, rerun bootstrap, access test indexes/targets, or write checkpoints.

## Results

- Final output tree: 22 files, 3,755,617 bytes, SHA-256 `125829be00d8ea670c1a0ec8ae4602b9399c16d88c83802c83c628123b2691a0`.
- Candidate search: four candidates, seeds 61000–61003; candidate 3 selected independently with best epoch 38 and exact 51-row candidate-3 history identity.
- Checkpoints: selected candidate 3 and final seeds 62001–62005 match their declared byte identities and CPU model contracts.
- Validation: 75 unique records, 15 designs, five realizations per design; frozen ensemble predictions reproduce under deterministic CPU inference.
- Metrics: run-level, design-level, and 10,000-replicate design bootstrap results reproduce; undefined metrics are null and valid numeric zero is retained.
- Recovery: selected checkpoint restoration, superseded checkpoint preservation, frozen prediction preservation, and reporting-only recovery manifest are verified.
- Test seal: zero test indexes/targets opened; no training or model/checkpoint writes occurred during this gate.

See the adjacent JSON evidence files for exact identities and recomputation results.
