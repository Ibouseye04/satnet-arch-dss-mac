# Stage A TGNN Regression Candidate-1 Checkpoint-Mismatch Assessment

This directory records an independent, read-only assessment of the completed candidate-1 reconstruction attempt.

## Verdict

**STAGE A TGNN REGRESSION CANDIDATE-1 MISMATCH ASSESSED — SERIALIZATION-ONLY REMEDIATION REQUIRED**

Classification: `SERIALIZATION_ONLY`.

The reconstructed history exactly matches the preserved candidate-1 rows. The reconstructed checkpoint loads on CPU, contains candidate 1 / seed 61001 / epoch 36 / hidden dimension 32 / learning rate 0.003, has finite tensors, and carries the expected stored best RMSE. The expected original checkpoint bytes are not present in the failed output because the original runner later overwrote `checkpoints/candidate_1.pt` during the final-seed loop with seed 62005.

## Read-only boundaries

No training epoch, optimizer instantiation, backward call, optimizer step, preprocessing refit, new prediction generation, bootstrap, reporting finalization, checkpoint rewrite, or output cleanup was performed. No Python diagnostic opened dataset index or target artifacts. No existing failed or reconstruction output was modified.

## Key identities

- Reconstruction root: `C:\Users\johns\satnet-stage-a-tgnn-regression-candidate1-reconstruction-v1`
- Reconstructed checkpoint SHA-256: `f12b4002201aeee3e6cb1bb347ec4a7480a72eb9891e0f6014e188c4208800b2`
- Expected checkpoint SHA-256: `89455aeca56ff169f9dbe4d7d8a22ae08339ab60a7e7ef392262ebe11bcabda1`
- History SHA-256: `c6e37e8cb7bd1d31b0056093bc16a10506ee22c0c3361d15bb4ff1d5dccd0dcd`
- Failed-output tree SHA-256: `141b5ac617337b168dadbd9e2a9579e5e9ba35877feedf7d01f713088101e29c`

## Narrow remediation recommendation

Prevent the final-seed loop from writing through the candidate checkpoint basename. Run final-seed training with no candidate output root, or save each final seed directly to a distinct final-seed path. Preserve the candidate checkpoint before final-seed execution. This assessment made no code changes.

## Artifact index

- `checkpoint_mismatch_assessment.json`: verdict, scope, evidence, and safety assertions.
- `reconstruction_state_inventory.json`: reconstruction-root inventory and failure boundary.
- `history_comparison.json`: exact epoch-history comparison and numeric differences.
- `checkpoint_semantic_identity.json`: independently calculated checkpoint semantic identities.
- `metric_consistency_verification.json`: stored epoch-36 metric consistency.
- `serialization_contract_comparison.json`: original/reconstruction save-contract comparison.
- `failed_output_preservation.json`: failed-output identity before and after assessment.
- `artifact_inventory.json`: hashes and sizes of assessment artifacts, excluding itself.
