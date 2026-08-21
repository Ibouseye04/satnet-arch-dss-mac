# Stage A TGNN Classification Reporting Finalization Authorization v1

## Verdict

**STAGE A TGNN CLASSIFICATION REPORTING FINALIZATION AUTHORIZED — READY TO EXECUTE ONCE**

This authorization prepares a reporting-finalization-only recovery for the completed Stage A TGNN classification campaign. Finalization was **not executed** while preparing this authorization.

## Bound execution state

- Activation branch: `activate/stage-a-tgnn-classification-training-v1`
- Activation worktree: `C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1-activation`
- Activation HEAD: `32a0244bbb8c8a3955988b3c7be4f0128807377a`
- Dedicated Python: `C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe`
- Campaign hash: `fbba2688e3b33838331aea216bcd48f5ab7aa070eb22677b6920f1cd9de0ec8f`
- Failed-output pre-finalization tree hash: `0d4b7f46c7b662d20b19419a2e7f04d27417d73071cbe904a4fa7f5af4883d7a`
- Pre-finalization file count/bytes: `16` / `3742138`
- Accepted reconstruction gate commit: `1bb496e`
- Accepted reconstructed checkpoint SHA-256: `e94c94bac6b6f0bf89d25e878871df9f1d4261dd98f8d796ab5206abf349302a`
- Current superseded `candidate_3.pt` SHA-256: `08d83176f09b88c701ba64ccc233a7ff93621d2a3fbfa7f6a4ec94500ce961a3`

The five final-seed checkpoints are frozen and individually bound by the authorization and preflight certificate. Only `checkpoints\candidate_3.pt` may be atomically replaced. The current file must first be preserved externally as:

`C:\Users\johns\satnet-stage-a-tgnn-classification-reporting-finalization-v1-recovery\superseded_checkpoints\candidate_3_overwritten_by_final_seed_62005.pt`

## Reporting policy

- Existing 75 validation predictions are reused; no predictions are regenerated.
- 15 designs are formed deterministically from five realizations each.
- Run labels are 2 class 0 / 73 class 1; design labels are 0 class 0 / 15 class 1.
- Confusion matrices always use labels `[0, 1]`.
- Undefined ROC-AUC and class-0 recall/specificity are JSON `null`; valid numeric zero remains numeric zero.
- Bootstrap uses 10,000 design-level replicates with seed 63001 and percentile bounds 2.5/97.5.
- Every staged JSON is recursively checked for non-finite values and serialized with `allow_nan=false`.
- The staged publication allowlist is exactly the six reporting artifacts named in the authorization.
- `artifact_inventory.json` is generated and published last.

## Safety boundary

The dedicated command requires the existing output root, expected campaign and tree hashes, accepted checkpoint identity, reconstruction gate identity, exact activation HEAD, clean activation worktree, dedicated Python executable, absent recovery root, absent recovery lock, and explicit finalize-reporting-only mode. The normal training wrapper remains unable to execute against the existing output root.

The reporting-only source contains no training epoch, optimizer construction or step, backward pass, preprocessing fit, candidate search, final-seed training, test access, or validation prediction generation. Preflight performed no training, checkpoint write, prediction generation, or finalization.

## Exact one-time PowerShell finalization command

Do not execute this command as part of authorization preparation. It is the one-time future finalization command:

```powershell
& 'C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1-activation\scripts\tgnn_classification_finalize_reporting_command.ps1' `
  -RepoRoot 'C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1-activation' `
  -OutputRoot 'C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1' `
  -ReconstructedRoot 'C:\Users\johns\satnet-stage-a-tgnn-classification-candidate3-reconstruction-v1' `
  -ReconstructionGateRoot 'C:\Users\johns\Developer\satnet-arch-dss-mac\artifacts\stage_a_tgnn_classification_candidate3_reconstruction_gate_v1' `
  -RecoveryRoot 'C:\Users\johns\satnet-stage-a-tgnn-classification-reporting-finalization-v1-recovery' `
  -ExpectedCampaignHash 'fbba2688e3b33838331aea216bcd48f5ab7aa070eb22677b6920f1cd9de0ec8f' `
  -ExpectedPreFinalizationTreeHash '0d4b7f46c7b662d20b19419a2e7f04d27417d73071cbe904a4fa7f5af4883d7a' `
  -ExpectedActivationHead '32a0244bbb8c8a3955988b3c7be4f0128807377a' `
  -AcceptedCheckpointSha256 'e94c94bac6b6f0bf89d25e878871df9f1d4261dd98f8d796ab5206abf349302a' `
  -AcceptedReconstructionGateCommit '1bb496e' `
  -PythonExecutable 'C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe'
```

Use `-PreflightOnly` only for a future read-only recheck. Do not use a normal training command and do not open test indexes or targets.
