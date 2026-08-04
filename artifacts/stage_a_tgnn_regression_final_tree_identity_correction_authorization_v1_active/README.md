# Stage A TGNN Regression Final-Tree Identity Correction Authorization v1

Status: active prepared; recovery not executed during authorization.

This bundle corrects only the planned final public tree identity for the existing replacement-resume partial state. It preserves all staged recovery artifacts byte-for-byte.

Root cause: `reporting_recovery_manifest.json` was planned with `recovery_tooling_sha=f783e61879140faf4b5c4b7c18a63728ddc67f57`, but the existing staged manifest is byte-bound to `recovery_tooling_sha=ac6aed5751f740bfee17f9cf85983709d994bf78`. The self-excluding `artifact_inventory.json` changes derivatively because it binds the manifest SHA.

Corrected final identity: 22 files, 2,070,248 bytes, tree SHA-256 `95134dab5669ee4bc9515777d19f48449186324977d17981be856fde533e6111`.

Exact one-time corrected resume command:

```powershell
& 'C:\Users\johns\Developer\satnet-arch-dss-mac\artifacts\stage_a_tgnn_regression_final_tree_identity_correction_authorization_v1_active\tgnn_regression_corrected_resume_command.ps1' -RepoRoot 'C:\Users\johns\satnet-stage-a-tgnn-regression-serialization-replacement-resume-v1-activation' -OutputRoot 'C:\Users\johns\satnet-stage-a-tgnn-regression-training-v1' -ReconstructionRoot 'C:\Users\johns\satnet-stage-a-tgnn-regression-candidate1-reconstruction-v1' -RecoveryRoot 'C:\Users\johns\satnet-stage-a-tgnn-regression-reporting-recovery-v2' -AssessmentRoot 'C:\Users\johns\Developer\satnet-arch-dss-mac\artifacts\stage_a_tgnn_regression_candidate1_checkpoint_mismatch_assessment_v1' -ExpectedActivationHead 'c44d1d4fa527be93c1e057df019fc4a9dda2d08c' -ExpectedSourceToolingCommit '3c388d353939178b2c4cf7843f2b50a39fffe61d' -ExpectedSourceScriptSha256 '6c30d1c4111d0565dad63d14544925d7da06f6ddfa3ebaf45741adfb1c7900ff' -ExpectedRecoveryToolingCommit 'c44d1d4fa527be93c1e057df019fc4a9dda2d08c' -ExpectedRecoveryScriptSha256 '9cd05a3300c50cb655bd4be02339184f6ed3830af9f2c403c0e024bf68ac9890' -ExpectedPublicTreeSha256 '141b5ac617337b168dadbd9e2a9579e5e9ba35877feedf7d01f713088101e29c' -ExpectedPublicFileCount 17 -ExpectedPublicBytes 2058047 -ExpectedPublicCandidateSha256 '3e28e10ddc024b21b57cbc986367d0393742ddc387f523eb1ddb45cfba213ef2' -ExpectedReconstructionTreeSha256 '29a3f93735fc9b4d51a21f9e25257cbfd96068bae252b4b8675e4282541c441a' -ExpectedReconstructionFileCount 2 -ExpectedReconstructionBytes 166184 -ExpectedReconstructedCandidateSha256 'f12b4002201aeee3e6cb1bb347ec4a7480a72eb9891e0f6014e188c4208800b2' -ExpectedPartialRecoveryTreeSha256 '31991cea4521d761671955cce5a73bd4ee41cdf59921b90875b5f767c6ecec5c' -ExpectedPartialRecoveryFileCount 8 -ExpectedPartialRecoveryBytes 310165 -ExpectedPartialLockSha256 '9e84dfc48bf4b92c032a07bdfa760ff22f2e63be15e5ca58ef6c5fc69b906889' -ExpectedPartialLockPid 23620 -RecoveryTimestamp '2026-08-04T18:49:31.526Z' -PlannedFinalTreeSha256 '95134dab5669ee4bc9515777d19f48449186324977d17981be856fde533e6111' -PlannedFinalFileCount 22 -PlannedFinalBytes 2070248 -PythonExecutable 'C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe' -TranscriptDirectory 'C:\Users\johns\satnet-stage-a-tgnn-regression-reporting-recovery-v2-corrected-final-tree-transcripts'
```

Do not run this command more than once. Do not run it without explicit user authorization.
