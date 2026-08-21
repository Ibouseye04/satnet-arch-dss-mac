# Stage A TGNN Classification Windows Resume Authorization v1

## Verdict

**STAGE A TGNN CLASSIFICATION WINDOWS RESUME PUBLICATION AUTHORIZED — TEMPORARY CHECKPOINT VERIFIED — READY TO PUBLISH ONCE**

Publication was **not executed** during preparation of this authorization. The only runtime invocation was the read-only `-PreflightOnly` path, including an isolated scratch-directory atomic-copy probe.

## Root-cause correction

The failed implementation left the descriptor returned by `mkstemp()` open, reopened the temporary file read-only, and attempted `fsync()` on that handle. On Windows this produced `OSError: [Errno 9] Bad file descriptor`; cleanup then ran while a handle remained open and produced `WinError 32`.

The corrected `copy_to_atomic_destination()` in `scripts/resume_stage_a_tgnn_classification_reporting.py` closes the `mkstemp()` descriptor, opens source and destination with nested context managers, copies bytes, flushes, and calls `os.fsync(destination_handle.fileno())` while the writable destination handle is open. Both context managers exit before `os.replace()`, and cleanup is attempted only after that scope has exited. The isolated preflight observed the order `destination_handle_opened`, `fsync_succeeded`, `destination_handle_closed`, `replace_succeeded` with no temporary residue.

## Bound state

- Activation branch: `activate/stage-a-tgnn-classification-training-v1`
- Activation HEAD: `ead97ac4582b8517dff818a4ce4368ff9b18667a`
- Corrected resume-tooling SHA-256: `7a04a12ea7e9311fc7f16354409115917ff47ad94b30f2708b3fd1c40b16b458`
- Corrected PowerShell command SHA-256: `63a95004af376ee2f7b042e988770df9910138b09b51f18f89acca1a26c459ef`
- Public governed tree excluding the exact residue: `0d4b7f46c7b662d20b19419a2e7f04d27417d73071cbe904a4fa7f5af4883d7a` (16 files, 3,742,138 bytes)
- Recovery tree: `1b05bbb88acd0fa3da58040e49bd9dd91470b20b149de27ae26edaf3ce4086d5` (8 files, 955,085 bytes)
- Planned completed public tree: `125829be00d8ea670c1a0ec8ae4602b9399c16d88c83802c83c628123b2691a0` (22 files, 3,755,617 bytes)

## Verified residue

The public checkpoint residue is the only file matching `.candidate_3.pt.*.resume.tmp`:

`C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1\checkpoints\.candidate_3.pt.atp6bm7w.resume.tmp`

It is exactly 470,803 bytes with SHA-256 `e94c94bac6b6f0bf89d25e878871df9f1d4261dd98f8d796ab5206abf349302a`. Publication uses this exact file directly with `os.replace()`; it is not copied again. The accepted staged checkpoint in the recovery root remains unchanged as evidence.

All six staged reports were verified by byte count and SHA-256, parse successfully, contain only finite JSON values, and are the only report sources allowlisted for publication. No report is regenerated.

## Exact one-time publication command

Do **not** execute this command during authorization review. Run it once only after approval:

```powershell
& 'C:\Users\johns\Developer\satnet-arch-dss-mac\artifacts\stage_a_tgnn_classification_reporting_finalization_windows_resume_authorization_v1_active\tgnn_classification_windows_resume_publication_command.ps1' `
  -RepoRoot 'C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1-activation' `
  -OutputRoot 'C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1' `
  -RecoveryRoot 'C:\Users\johns\satnet-stage-a-tgnn-classification-reporting-finalization-v1-recovery' `
  -ExpectedCampaignHash 'fbba2688e3b33838331aea216bcd48f5ab7aa070eb22677b6920f1cd9de0ec8f' `
  -ExpectedPreFinalizationTreeHash '0d4b7f46c7b662d20b19419a2e7f04d27417d73071cbe904a4fa7f5af4883d7a' `
  -ExpectedRecoveryTreeHash '1b05bbb88acd0fa3da58040e49bd9dd91470b20b149de27ae26edaf3ce4086d5' `
  -ExpectedPlannedPostPublicationTreeHash '125829be00d8ea670c1a0ec8ae4602b9399c16d88c83802c83c628123b2691a0' `
  -ExpectedActivationHead 'ead97ac4582b8517dff818a4ce4368ff9b18667a' `
  -AcceptedCheckpointSha256 'e94c94bac6b6f0bf89d25e878871df9f1d4261dd98f8d796ab5206abf349302a' `
  -SupersededCheckpointSha256 '08d83176f09b88c701ba64ccc233a7ff93621d2a3fbfa7f6a4ec94500ce961a3' `
  -PythonExecutable 'C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe'
```

The same command with `-PreflightOnly` performs a read-only recheck. The publication path refuses changed public or recovery identities, changed staged artifacts, a missing or altered residue, an existing lock, an existing completion marker, a wrong activation HEAD, or a dirty activation worktree.

## Safety confirmations

- No publication occurred.
- No training, prediction generation, bootstrap rerun, report regeneration, preprocessing refit, or final-seed recomputation occurred.
- No test index, test target, or test source was opened.
- The preflight wrote only to an isolated temporary scratch directory; it wrote no public or recovery output.
- Final publication is restricted to the direct checkpoint replacement and the six existing staged report files.
- Final verification requires candidate SHA-256 `e94c94bac6b6f0bf89d25e878871df9f1d4261dd98f8d796ab5206abf349302a`, no residue, six matching reports, finite JSON, 22 files, 3,755,617 bytes, and the planned completed public tree hash.
