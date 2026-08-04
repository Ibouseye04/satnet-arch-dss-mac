# Stage A TGNN Classification Selected Checkpoint Recovery Assessment v1

This directory records a narrow, read-only diagnosis of the failed Stage A TGNN classification execution at:

`C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1`

## Verdict

**STAGE A TGNN CLASSIFICATION SELECTED CHECKPOINT RECOVERABLE — READY FOR TARGETED RECONSTRUCTION AUTHORIZATION**

This is an assessment verdict only. No reconstruction was authorized or run.

## Root cause

The committed runner's `train_configuration` function writes `candidate_<candidate_index>.pt`. The search loop first invokes it for candidate 3 (`seed=61003`), then calculates and records the expected selected-checkpoint SHA. The final-seed loop reuses the selected configuration with seeds 62001–62005 and invokes the same function with the same output root. Because `candidate_index` remains 3, each final-seed call rewrites `candidate_3.pt`; the final seed 62005 call is last. The separate `final_seed_62005.pt` is then saved from the same returned checkpoint dictionary.

Candidate 0, candidate 1, and candidate 2 retain their search metadata. The current candidate 3 is valid final-seed-62005 data, not a corrupt file.

## Semantic comparison

`candidate_3.pt` and `final_seed_62005.pt` are not byte-identical, but are exactly identical in model state, optimizer state, metadata excluding container fields, preprocessing binding, tensor keys, shapes, dtypes, and values. Their Torch archive prefixes differ (`candidate_3/` versus `final_seed_62005/`), explaining the byte difference.

The expected original SHA-256 was not found as a checkpoint copy, and no checkpoint with the original candidate-3 metadata (`candidate_index=3`, `seed=61003`, `best_epoch=38`, `hidden_dim=64`, `learning_rate=0.003`) was found in the bounded search.

## Reconstruction feasibility

A candidate-3-only deterministic reconstruction is technically feasible using the frozen index hashes, preprocessing binding, candidate parameters, seed, ordering, deterministic CPU/Torch settings, loss weights, early-stopping rule, stored history, runtime package identities, and training-tooling identity. It must use a new recovery root and compare the stored epoch history and validation metric before accepting the checkpoint. Canonical semantic state hashing is the reliable comparison; exact Torch file SHA is conditional on retaining the same serialization basename and runtime.

## Safety and preservation

- No training epoch ran.
- No optimizer was instantiated and no backward pass occurred.
- No predictions were generated.
- No checkpoint was rewritten or resaved in the training output root.
- The finalize-reporting command was not executed.
- No test index or test target was opened.
- The completed output root remained byte-identical with tree hash `0d4b7f46c7b662d20b19419a2e7f04d27417d73071cbe904a4fa7f5af4883d7a`.
- The pre-existing unrelated worktree changes were not modified.

## Evidence files

- `selected_checkpoint_recovery_assessment.json` — aggregate assessment and verdict
- `checkpoint_overwrite_root_cause.json` — source locations and execution-stage ordering
- `checkpoint_semantic_comparison.json` — CPU semantic comparison and canonical hashes
- `original_checkpoint_search_report.json` — bounded original-copy search
- `deterministic_reconstruction_feasibility.json` — frozen-evidence binding and feasibility
- `torch_save_byte_determinism_report.json` — isolated synthetic serialization test
- `preserved_output_identity.json` — output identities and preservation assertions
- `artifact_inventory.json` — self-excluding assessment inventory
- `torch_save_probe_*.pt` — synthetic probe files retained only in this assessment directory

The assessment was committed separately from unrelated worktree changes.
