# Stage A TGNN Regression Dual-Replacement Resume Authorization v1

Status: active prepared; recovery was not executed during authorization.

**STAGE A TGNN REGRESSION DUAL-REPLACEMENT RESUME AUTHORIZED — READY TO FINALIZE ONCE**

This package corrects only the existing Stage A TGNN Regression reporting-recovery publication defect. It does not rerun training, reconstruction, inference, prediction generation, bootstrap or scientific metrics, checkpoint serialization, or staged-artifact generation. It does not delete or recreate the recovery root and does not read test indexes or targets.

## Exact policy

Only `checkpoints/candidate_1.pt` and `selected_checkpoint_identity.json` receive bound replacement semantics. Each staged payload must have its exact replacement SHA. An already-published replacement is accepted idempotently; otherwise the public path must have its exact historical SHA. Any other SHA blocks. Replacement is atomic and verified after publication.

All other staged reports are create-or-idempotent only: absent files publish, exact bytes are preserved, and conflicting bytes block. The planned final-tree check is unconditional. The preserved superseded checkpoint remains byte-identical.

Bound identities:

- Candidate historical `3e28e10ddc024b21b57cbc986367d0393742ddc387f523eb1ddb45cfba213ef2`; replacement `f12b4002201aeee3e6cb1bb347ec4a7480a72eb9891e0f6014e188c4208800b2`.
- Selected identity historical `87d6b2eab05129485ca167314407758c6f850d502a0fc80e1e97fce0d9eb136a`; replacement `bed3df81e0be80e5f33de9a5b29fcca260522912ae6322ab4d2cce0c4c720282`.
- Staged selected identity preserves declared SHA `89455aeca56ff169f9dbe4d7d8a22ae08339ab60a7e7ef392262ebe11bcabda1`, recovered SHA `f12b4002201aeee3e6cb1bb347ec4a7480a72eb9891e0f6014e188c4208800b2`, classification `SERIALIZATION_ONLY`, and all required no-rerun flags.

## Verified state

- Public: 17 files, 2,058,047 bytes, `141b5ac617337b168dadbd9e2a9579e5e9ba35877feedf7d01f713088101e29c`.
- Reconstruction: 2 files, 166,184 bytes, `29a3f93735fc9b4d51a21f9e25257cbfd96068bae252b4b8675e4282541c441a`.
- Recovery: 8 files, 310,165 bytes, `31991cea4521d761671955cce5a73bd4ee41cdf59921b90875b5f767c6ecec5c`.
- Stale lock: 11 bytes, `9e84dfc48bf4b92c032a07bdfa760ff22f2e63be15e5ca58ef6c5fc69b906889`, `pid=23620`, owner not running; completion marker absent.
- Corrected final: 22 files, 2,070,248 bytes, `95134dab5669ee4bc9515777d19f48449186324977d17981be856fde533e6111`.
- Final seed 62005: `2570f45ccd441c60b1f2bed6abf9959317c5b29be976c296eac87e110bc3df83`.

## Activation and tests

- Branch: `activate/stage-a-tgnn-regression-dual-replacement-resume-v1`
- Worktree: `C:\Users\johns\satnet-stage-a-tgnn-regression-dual-replacement-resume-v1-activation`
- Activation HEAD: `a1df92bb2f4ba5f51f358a296321d2846cd6d0eb`
- Recovery tooling commit: `a1df92bb2f4ba5f51f358a296321d2846cd6d0eb`
- Recovery script SHA-256: `90722b91d85fafe0fda1de284b63a5f20b86174dd017dea5f78d64be94973d9d`
- Source tooling commit: `3c388d353939178b2c4cf7843f2b50a39fffe61d`
- Source script SHA-256: `6c30d1c4111d0565dad63d14544925d7da06f6ddfa3ebaf45741adfb1c7900ff`
- Focused dual-replacement tests: 9 passed.

## Exact one-time finalization command

Run only after explicit authorization; do not add `-PreflightOnly`:

```powershell
& 'C:\Users\johns\Developer\satnet-arch-dss-mac\artifacts\stage_a_tgnn_regression_dual_replacement_resume_authorization_v1_active\tgnn_regression_dual_replacement_resume_command.ps1' -RepoRoot 'C:\Users\johns\satnet-stage-a-tgnn-regression-dual-replacement-resume-v1-activation' -OutputRoot 'C:\Users\johns\satnet-stage-a-tgnn-regression-training-v1' -ReconstructionRoot 'C:\Users\johns\satnet-stage-a-tgnn-regression-candidate1-reconstruction-v1' -RecoveryRoot 'C:\Users\johns\satnet-stage-a-tgnn-regression-reporting-recovery-v2' -AssessmentRoot 'C:\Users\johns\Developer\satnet-arch-dss-mac\artifacts\stage_a_tgnn_regression_candidate1_checkpoint_mismatch_assessment_v1' -ExpectedActivationHead 'a1df92bb2f4ba5f51f358a296321d2846cd6d0eb' -ExpectedSourceToolingCommit '3c388d353939178b2c4cf7843f2b50a39fffe61d' -ExpectedSourceScriptSha256 '6c30d1c4111d0565dad63d14544925d7da06f6ddfa3ebaf45741adfb1c7900ff' -ExpectedRecoveryToolingCommit 'a1df92bb2f4ba5f51f358a296321d2846cd6d0eb' -ExpectedRecoveryScriptSha256 '90722b91d85fafe0fda1de284b63a5f20b86174dd017dea5f78d64be94973d9d' -ExpectedPublicTreeSha256 '141b5ac617337b168dadbd9e2a9579e5e9ba35877feedf7d01f713088101e29c' -ExpectedPublicFileCount 17 -ExpectedPublicBytes 2058047 -ExpectedPublicCandidateSha256 '3e28e10ddc024b21b57cbc986367d0393742ddc387f523eb1ddb45cfba213ef2' -ExpectedPublicSelectedIdentitySha256 '87d6b2eab05129485ca167314407758c6f850d502a0fc80e1e97fce0d9eb136a' -ExpectedReconstructionTreeSha256 '29a3f93735fc9b4d51a21f9e25257cbfd96068bae252b4b8675e4282541c441a' -ExpectedReconstructionFileCount 2 -ExpectedReconstructionBytes 166184 -ExpectedReconstructedCandidateSha256 'f12b4002201aeee3e6cb1bb347ec4a7480a72eb9891e0f6014e188c4208800b2' -ExpectedPartialRecoveryTreeSha256 '31991cea4521d761671955cce5a73bd4ee41cdf59921b90875b5f767c6ecec5c' -ExpectedPartialRecoveryFileCount 8 -ExpectedPartialRecoveryBytes 310165 -ExpectedPartialLockSha256 '9e84dfc48bf4b92c032a07bdfa760ff22f2e63be15e5ca58ef6c5fc69b906889' -ExpectedPartialLockPid 23620 -ExpectedStagedSelectedIdentitySha256 'bed3df81e0be80e5f33de9a5b29fcca260522912ae6322ab4d2cce0c4c720282' -RecoveryTimestamp '2026-08-04T18:49:31.526Z' -PlannedFinalTreeSha256 '95134dab5669ee4bc9515777d19f48449186324977d17981be856fde533e6111' -PlannedFinalFileCount 22 -PlannedFinalBytes 2070248 -PythonExecutable 'C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe' -TranscriptDirectory 'C:\Users\johns\satnet-stage-a-tgnn-regression-dual-replacement-resume-v1-transcripts'
```

The wrapper refuses a completed recovery, removes only the exact stale lock after the Python preflight confirms its owner is not running, preserves the recovery tool exit code, and writes transcripts outside governed roots.
