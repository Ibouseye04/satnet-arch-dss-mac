# Stage A TGNN Classification Candidate-3 Reconstruction Gate v1

## Verdict

**STAGE A TGNN CLASSIFICATION CANDIDATE-3 RECONSTRUCTION ACCEPTED — READY FOR REPORTING FINALIZATION**

This is a narrow independent acceptance gate. It is a read-only identity and
metadata verification of the isolated reconstructed candidate-3 checkpoint.
No training, fitting, optimizer construction or stepping, backward pass,
prediction generation, checkpoint write, or reporting finalization was run.
No test index or test target was opened.

## Scope and inputs

- Reconstruction root: `C:\Users\johns\satnet-stage-a-tgnn-classification-candidate3-reconstruction-v1`
- Source output root, read-only: `C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1`
- Recovery assessment commit: `f9522904987a4ee1587c817e11716806c77c4fb3`
- Reconstruction authorization head: `53c624d6ac172b811aa0bae5a650b1b5da7cd0e7`
- Dedicated read-only checkpoint runtime: Python 3.11.9, torch 2.12.1+cpu

## Results

- Checkpoint: 470,803 bytes; SHA-256
  `e94c94bac6b6f0bf89d25e878871df9f1d4261dd98f8d796ab5206abf349302a`; exact
  identity restored.
- Checkpoint metadata: candidate 3, seed 61003, hidden dimension 64,
  learning rate 0.003, best epoch 38, weighted validation log loss
  `0.06203101027124465`.
- Model contract: GCLSTM input dimension 14, hidden dimension 64,
  Chebyshev K=2, complete 22-key model state, and two-logit classification
  head. The expected Adam optimizer state is complete for all 22 parameters.
- Epoch history: 51 rows in deterministic epoch order 0 through 50, best epoch
  38, exact SHA-256
  `730e39eb125f4d1d8f62fafb7889f7731b5b462d85016c45dbfb0ca5d5d396b0`. The
  complete candidate-3 sequence matches the preserved failed-execution history.
- Frozen preprocessing: 2,663 bytes; exact SHA-256
  `446c91cb7d793e1ae6c598e9e880663acdb1ac237d1bd3f7372eb65750acef94`; its
  manifest binding matches the checkpoint and reconstruction result.
- Source preservation: observed tree SHA-256
  `0d4b7f46c7b662d20b19419a2e7f04d27417d73071cbe904a4fa7f5af4883d7a`, equal to
  the expected preserved hash. All five final-seed checkpoints and
  `validation_predictions.jsonl` remain unchanged.
- Result JSON: all required metadata, exact hashes, and false safety flags
  (`predictions_written`, `reporting_finalization_run`, `test_access`) pass.

## Evidence files

- `candidate3_reconstruction_gate_report.json` — aggregate verdict and checks
- `checkpoint_identity_verification.json` — exact bytes/hash, CPU metadata,
  architecture, model state, and optimizer state
- `epoch_history_verification.json` — exact history hash, ordering, and source
  candidate-3 sequence comparison
- `preprocessing_binding_verification.json` — frozen artifact and checkpoint/result
  binding
- `source_output_preservation.json` — source tree and preserved output identities
- `artifact_inventory.json` — self-excluding committed gate inventory

The artifact inventory records SHA-256 values for the other six gate artifacts;
it excludes itself by design.
