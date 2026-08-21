# Stage A TGNN Classification Training v1 Authorization

**Verdict: STAGE A TGNN CLASSIFICATION TRAINING AUTHORIZED — PREFLIGHT PASSED — READY TO EXECUTE**

This directory authorizes exactly one future CPU-only TGNN classification execution. The authorized target is `partition_any` from `overall_threshold_breach_any`, using the accepted 70-design/350-sequence train split and 15-design/75-sequence validation split. The sealed 15-design/75-sequence test split is prohibited.

## Preflight seal

- The preflight was read-only and constructor-only.
- No training epoch, optimizer, loss evaluation, prediction, metric, checkpoint, or fitted preprocessing statistic was created.
- No sequence array was loaded. Only the train/validation index records and frozen graph/node/edge schemas were read.
- The sealed test index and all test target sources were not opened.
- Four candidates instantiated with hidden dimensions 32/32/64/64, Chebyshev K=2, and seeds 61000–61003.
- The dedicated runtime imported successfully and the synthetic exact zero-edge `[2, 0]` constructor/runtime check passed.
- Both accepted RF model gates were checked without reading or rerunning RF outputs.
- The project `.venv`, accepted datasets, gates, plans, runtime evidence, RF outputs, and production evidence were not modified.

See `preflight_certificate.json` for the machine-readable evidence.

## Frozen execution behavior

The future command must use the dedicated Python executable, verify the activation HEAD and branch, require a clean activation worktree, verify the runner source-equivalence SHA, recheck train/validation index hashes, reject an existing output root or lock, and execute exactly once. The wrapper preserves the process exit code, captures a timestamped transcript outside the output root, and omits null/empty/whitespace-only optional arguments.

The contracted architecture is one GCLSTM layer with input dimension 14, K=2, no dropout, final-timestep global mean pooling, and a two-logit linear head. The ten edge attributes remain preserved and validated but are not falsely injected as an edge weight into the contracted GCLSTM. Training uses weighted `CrossEntropyLoss` with class weights `[3.5714285714285716, 0.5813953488372093]`, indexed by the actual target class.

Early stopping uses strictly improving training-weighted validation log loss (`minimum_delta=0.0`; equality is not improvement), first eligible epoch 0, patience 12 completed non-improving epochs, earliest epoch on ties, and exact best-checkpoint restoration before final validation reporting. Threshold is fixed at 0.5 with no calibration or threshold search.

## Exact future PowerShell command

Run exactly:

```powershell
& 'C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1-activation\artifacts\stage_a_tgnn_classification_training_authorization_v1_active\tgnn_classification_training_command.ps1' -RepoRoot 'C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1-activation' -DatasetRoot 'C:\Users\johns\Developer\satnet-arch-dss-mac\artifacts\stage_a_tgnn_dataset_v1' -OutputRoot 'C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1' -ExpectedActivationHead 'fa314c07cdb3c60186365888e9c726a0e4a567e3' -CommittedTrainingToolingSha 'fa314c07cdb3c60186365888e9c726a0e4a567e3' -PythonExecutable 'C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe'
```

Do not add a command-line override or replace the identity arguments.

## Required future output

The runner is frozen to produce the execution/environment manifests, training-only preprocessing statistics, candidate search and epoch histories, selected configuration/checkpoint identity, validation metrics/predictions/design metrics/bootstrap intervals, five final checkpoints, final-seed results, report, self-excluding inventory, and the external command transcript. It never opens or materializes test targets.
