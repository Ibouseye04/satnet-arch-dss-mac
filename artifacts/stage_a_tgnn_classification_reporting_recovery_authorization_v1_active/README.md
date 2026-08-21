# Stage A TGNN Classification Reporting Recovery Authorization v1

This authorization records a read-only salvage assessment for the completed
Stage A TGNN classification execution at:

`C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1`

## Verdict

**STAGE A TGNN CLASSIFICATION RECOVERY BLOCKED — INSUFFICIENT COMPLETED TRAINING EVIDENCE**

Training and frozen validation prediction generation completed. The execution
failed while serializing `validation_design_metrics.json` because design-level
aggregation produced a single observed class and a non-finite ROC-AUC reached
canonical JSON serialization with `allow_nan=false`.

A second preservation defect prevents authorization: the selected candidate is
candidate 3, but `selected_checkpoint_identity.json` declares SHA-256
`e94c94bac6b6f0bf89d25e878871df9f1d4261dd98f8d796ab5206abf349302a` while the
file currently at `checkpoints/candidate_3.pt` hashes to
`08d83176f09b88c701ba64ccc233a7ff93621d2a3fbfa7f6a4ec94500ce961a3` and has
final-seed 62005 metadata. The selected search checkpoint therefore cannot be
preserved byte-for-byte by a reporting-only operation. No recovery was run.

## Evidence

- Candidate training runs completed: 4.
- Final-seed training runs completed: 5.
- Final checkpoints present: 5; total checkpoints present: 9.
- Selected candidate: index 3, hidden dimension 64, learning rate 0.003,
  search seed 61003, best epoch 38.
- Preprocessing, candidate histories, selected configuration, validation
  metrics, and 75 frozen validation predictions are present and hashed in
  `salvage_assessment.json`.
- No training lock remains.
- No test target, sealed test index, dataset root, or sequence array was read.
- Existing output tree hash before remediation:
  `0d4b7f46c7b662d20b19419a2e7f04d27417d73071cbe904a4fa7f5af4883d7a`.

## Recovery implementation

The finalize-only implementation is committed in the activation worktree at
HEAD `7b7a198de1884034318486e9f92ed208590ba505`:

- `scripts/finalize_stage_a_tgnn_classification_reporting.py`
- `scripts/tgnn_classification_finalize_reporting_command.ps1`

It requires the existing output root, campaign hash, complete pre-recovery tree
hash, exact activation HEAD, clean worktree, dedicated Python executable, and
an explicit `--finalize-reporting-only` guard. It never contains a training
loop, optimizer step, backward pass, preprocessing fit, candidate search, or
checkpoint write. It refuses finalization when the selected checkpoint identity
is inconsistent, so the command is intentionally blocked for this output root.

The reporting path uses explicit confusion-matrix labels `[0, 1]`, represents
undefined metrics as JSON null, preserves valid numeric zero, records bootstrap
valid/undefined replicate counts, and recursively rejects non-finite values
before canonical JSON serialization with `allow_nan=false`.

## Exact command (will refuse on the current evidence)

```powershell
& 'C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1-activation\scripts\tgnn_classification_finalize_reporting_command.ps1' `
  -RepoRoot 'C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1-activation' `
  -OutputRoot 'C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1' `
  -ExpectedCampaignHash 'fbba2688e3b33838331aea216bcd48f5ab7aa070eb22677b6920f1cd9de0ec8f' `
  -ExpectedPreRecoveryTreeHash '0d4b7f46c7b662d20b19419a2e7f04d27417d73071cbe904a4fa7f5af4883d7a' `
  -ExpectedActivationHead '7b7a198de1884034318486e9f92ed208590ba505' `
  -PythonExecutable 'C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe'
```

Use `-PreflightOnly` to run the read-only certificate check. Do not remove or
replace the selected checkpoint to bypass the block.
