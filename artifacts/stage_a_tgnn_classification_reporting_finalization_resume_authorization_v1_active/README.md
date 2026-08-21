# Stage A TGNN Classification Reporting Finalization Resume Authorization v1

## Verdict

**STAGE A TGNN CLASSIFICATION REPORTING FINALIZATION RESUME AUTHORIZED — STAGED OUTPUTS PRESERVED — READY TO PUBLISH ONCE**

Publication was **not executed** while preparing this authorization.

## Root cause of the prior failure

The prior command computed `post_finalization_tree_hash` from the 20-entry destination plan containing the current public files with `checkpoints/candidate_3.pt` replaced plus the four base reports. That definition intentionally excluded both self-referential reports.

The line-570 verification filtered out only `artifact_inventory.json`; it retained `reporting_recovery_manifest.json`, producing a 21-entry tree. The first positional mismatch was index 13: the expected `selected_checkpoint_identity.json` entry was displaced by the unexpected destination entry `reporting_recovery_manifest.json`. The 20-entry hash was `ad01a0101ba48bf1d4ed60026cd07eff5c6077c411850866fa30ee748251e4ab`; the incorrect 21-entry hash was `91b78ca6bb941f7f2eb6c1812f98811efa6491d3b30025fa2ceac8c69a765ba3`.

This was not caused by source-versus-destination identities, relative-versus-absolute normalization, inventory sequencing, newline handling, byte counts, or an incorrect artifact-inventory self-exclusion policy. The staged inventory already has correct destination identities and includes the manifest while excluding itself. No staged artifact required correction.

The corrected implementation excludes both self-referential reports from the manifest plan recheck and uses the same canonical absolute destination path, Windows backslash separator, preserved case, ordinal path ordering, UTF-8 encoding, LF line separators, decimal byte counts, and lowercase SHA-256 representation for current, planned, and actual tree hashes.

## Bound state

- Activation branch: `activate/stage-a-tgnn-classification-training-v1`
- Activation HEAD: `7e28c1118ef1a7404a11d9f8b0d7c3b94187e7ae`
- Corrected finalization implementation: `scripts/finalize_stage_a_tgnn_classification_reporting.py`
- Resume implementation SHA-256: `e91da7f344e1841b1c79c39428194cd2005ae2e810b20fa6ef538fad754da797`
- Public pre-finalization tree: `0d4b7f46c7b662d20b19419a2e7f04d27417d73071cbe904a4fa7f5af4883d7a` (16 files, 3,742,138 bytes)
- Recovery tree: `1b05bbb88acd0fa3da58040e49bd9dd91470b20b149de27ae26edaf3ce4086d5` (8 files, 955,085 bytes)
- Staged accepted checkpoint: `e94c94bac6b6f0bf89d25e878871df9f1d4261dd98f8d796ab5206abf349302a`
- Preserved superseded checkpoint: `08d83176f09b88c701ba64ccc233a7ff93621d2a3fbfa7f6a4ec94500ce961a3`
- Corrected complete planned post-publication tree: `125829be00d8ea670c1a0ec8ae4602b9399c16d88c83802c83c628123b2691a0` (22 files, 3,755,617 bytes)
- Manifest plan excluding both self-referential reports: `ad01a0101ba48bf1d4ed60026cd07eff5c6077c411850866fa30ee748251e4ab`
- Artifact inventory plan excluding itself: `91b78ca6bb941f7f2eb6c1812f98811efa6491d3b30025fa2ceac8c69a765ba3`

All six staged reports parse, recursively contain only finite values, serialize with `allow_nan=false`, contain no test results, and bind to the existing 75 frozen validation predictions. No report was regenerated and no inventory or manifest correction was necessary.

## Exact one-time resume-publication command

Do not execute this command as part of authorization preparation. Run it once only after reviewing this authorization and the read-only preflight certificate:

```powershell
& 'C:\Users\johns\Developer\satnet-arch-dss-mac\artifacts\stage_a_tgnn_classification_reporting_finalization_resume_authorization_v1_active\tgnn_classification_resume_reporting_publication_command.ps1' `
  -RepoRoot 'C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1-activation' `
  -OutputRoot 'C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1' `
  -RecoveryRoot 'C:\Users\johns\satnet-stage-a-tgnn-classification-reporting-finalization-v1-recovery' `
  -ExpectedCampaignHash 'fbba2688e3b33838331aea216bcd48f5ab7aa070eb22677b6920f1cd9de0ec8f' `
  -ExpectedPreFinalizationTreeHash '0d4b7f46c7b662d20b19419a2e7f04d27417d73071cbe904a4fa7f5af4883d7a' `
  -ExpectedRecoveryTreeHash '1b05bbb88acd0fa3da58040e49bd9dd91470b20b149de27ae26edaf3ce4086d5' `
  -ExpectedPlannedPostPublicationTreeHash '125829be00d8ea670c1a0ec8ae4602b9399c16d88c83802c83c628123b2691a0' `
  -ExpectedActivationHead '7e28c1118ef1a7404a11d9f8b0d7c3b94187e7ae' `
  -AcceptedCheckpointSha256 'e94c94bac6b6f0bf89d25e878871df9f1d4261dd98f8d796ab5206abf349302a' `
  -SupersededCheckpointSha256 '08d83176f09b88c701ba64ccc233a7ff93621d2a3fbfa7f6a4ec94500ce961a3' `
  -PythonExecutable 'C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe'
```

Use the same command with `-PreflightOnly` for a future read-only check. The command refuses any changed public or recovery identity, changed staged artifact, changed preserved checkpoint, wrong activation HEAD, dirty worktree, existing resume lock, or completion marker. It publishes only the six allowlisted reports and atomically replaces only `checkpoints\candidate_3.pt`; it leaves every other public file unchanged and records pre/post tree hashes in the external completion marker. Process exit status is returned unchanged.

## Safety confirmation

- Training, optimizer construction, optimizer steps, backward passes, preprocessing fitting, prediction generation, bootstrap reruns, and final-seed reporting reruns are unreachable.
- No test index, target, or test source was opened.
- No public file was written during authorization or preflight.
- The activation worktree is clean.
