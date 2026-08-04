# Stage A TGNN Regression Training v1 Authorization

Status: **AUTHORIZED — READY TO EXECUTE**.

This authorization binds one future CPU-only execution of the contracted GCLSTM TGNN regression task for `gcc_frac_min` (`space_gcc_fraction_original_min`). Preparation is read-only/constructor-only with respect to accepted training inputs. No training epoch, optimizer, fitted preprocessing statistic, checkpoint, validation prediction, validation metric, or test target was created during authorization.

## Bound inputs

- Training plan HEAD: `a5e224f354f4ac2b3950da87d52f5cdc722f560d`
- Training-plan gate: `4f9e5e3a2534fc9dbe1e8fb9613cbc041d03e489`
- Dataset manifest: `6023bcde5f62d1724485789ed2876ac84f0619839b6de1bf8072067f92df252e`
- Dataset implementation HEAD: `f9cd85bbf762dce673a5877e080e675bf8721290`
- Dataset gate: `ea32faab3945f24e5ed234947b03cda56dbdd95f`
- Runtime gate: `dcabfba20fb6e9aaa71e9c174b6f9fdfe7b70c51`
- Dedicated Python: `C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe`

Train and validation indexes are bound by SHA-256 and are the only indexes opened by preflight. The sealed test index and all test targets remain prohibited.

## Frozen execution

The four candidates are `(hidden_dim, learning_rate, seed)` = `(32,0.001,61000)`, `(32,0.003,61001)`, `(64,0.001,61002)`, and `(64,0.003,61003)`. Final repeated seeds are `62001`–`62005`. The loss is exactly `torch.nn.SmoothL1Loss(beta=1.0)` on the untransformed scalar target. Candidate selection is validation RMSE ascending, validation MAE ascending, validation R² descending, frozen architecture simplicity (hidden dimension 32 before 64), then candidate index ascending.

Early stopping is executable and frozen: validation RMSE, lower is better, minimum delta `0.0`, strict improvement (`metric < best_metric - minimum_delta`; equality is not improvement), first eligible epoch `0`, patience `12`, maximum `60` epochs, earliest qualifying epoch for ties, and restoration of the exact model and optimizer state captured at the best validation RMSE epoch.

Preprocessing is fitted only during future execution from training graph nodes. The six physical ground-station features are standardized with population standard deviation; satellite sentinel zeros are excluded and remain exactly zero. No preprocessing statistic is calculated by this authorization.

## Activation

- Branch: `activate/stage-a-tgnn-regression-training-v1`
- Worktree: `C:\Users\johns\satnet-stage-a-tgnn-regression-training-v1-activation`
- Activation HEAD: `d9cb035abff3668ccdba80039523110043ecf6ac`
- Training-tooling SHA: `d9cb035abff3668ccdba80039523110043ecf6ac`
- Output root: `C:\Users\johns\satnet-stage-a-tgnn-regression-training-v1` (must be absent)

The PowerShell wrapper verifies activation HEAD, branch, clean worktree, source equivalence, dedicated runtime identity, train/validation index hashes, output-root absence, and stale-lock absence. It invokes the dedicated Python executable once, preserves its exit code, writes a timestamped transcript outside the output root, omits null/empty/whitespace optional arguments, and refuses reruns after output creation.

## Exact execution command

```powershell
& 'C:\Users\johns\Developer\satnet-arch-dss-mac\artifacts\stage_a_tgnn_regression_training_authorization_v1_active\tgnn_regression_training_command.ps1' -RepoRoot 'C:\Users\johns\satnet-stage-a-tgnn-regression-training-v1-activation' -DatasetRoot 'C:\Users\johns\Developer\satnet-arch-dss-mac\artifacts\stage_a_tgnn_dataset_v1' -OutputRoot 'C:\Users\johns\satnet-stage-a-tgnn-regression-training-v1' -ExpectedActivationHead 'd9cb035abff3668ccdba80039523110043ecf6ac' -CommittedTrainingToolingSha 'd9cb035abff3668ccdba80039523110043ecf6ac' -PythonExecutable 'C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe'
```

Do not execute that command as part of authorization.

## Clarification and verdict

The committed TGNN classification candidate-selection artifact contains candidate-3 history SHA-256 `730e39eb125f4d1d8f62fafb7889f7731b5b462d85016c45dbfb0ca5d5d396b0` and full history SHA-256 `37c356d10dd115a9114a3fb4a418c04a2ab8106c4b7c971bcfb94736c5b24f43`; the malformed response line is treated as a transcription error. Completed RF and TGNN classification gates and outputs remain bound and unchanged.

**STAGE A TGNN REGRESSION TRAINING AUTHORIZED — PREFLIGHT PASSED — READY TO EXECUTE**
