# Stage A TGNN CPU Runtime v1 Independent Acceptance Gate

**Verdict: STAGE A TGNN CPU RUNTIME V1 ACCEPTED — READY FOR TGNN CLASSIFICATION TRAINING AUTHORIZATION**

This is a narrow, independent environment-validation gate. It did not train a model, calculate preprocessing statistics, open any train/validation/test sequence or index artifact, create a checkpoint, instantiate an optimizer, calculate a loss, run a backward pass, modify the dedicated runtime, modify the project `.venv`, reinstall packages, or modify accepted datasets, gates, plans, RF outputs, or unrelated files.

## Identity

- Implementation commit: `f102111`
- Dedicated Python: `C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe`
- Dedicated environment: `C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv`
- Gate artifact directory: `artifacts/stage_a_tgnn_cpu_runtime_v1_gate/`

## Results

- Evidence identity: PASS. All eight declared preparation artifacts matched their declared SHA-256 values; the prior inventory was complete and self-excluding; all eight files were committed in `f102111`; that commit contains only the prior runtime-evidence directory.
- Version compatibility: PASS. Exact logical versions matched the frozen plan. Full local build versions were retained in the package report.
- CPU and ABI: PASS. `torch 2.12.1+cpu`, `torch_sparse 0.6.18+pt212cpu`, and `torch_scatter 2.1.2+pt212cpu` satisfy the base-version, CPU-only, PyTorch 2.12 target, native-operation, and no-CUDA criteria. `pip check` passed.
- Native imports: PASS. `torch`, PyG, PyG Temporal, `torch_sparse`, `torch_scatter`, and `GCLSTM` imported. Resolved CPU `.pyd` paths and hashes are recorded in `native_extension_import_verification.json`.
- Determinism: PASS. CPU tensor operations, deterministic algorithms with `warn_only=False`, one intra-op thread, and one inter-op thread succeeded.
- GCLSTM contract: PASS for `(in_channels=14, out_channels=32, K=2)` and `(in_channels=14, out_channels=64, K=2)`; constructor signature matched the frozen plan.
- Structural smoke: PASS for synthetic non-empty input and exact zero-edge `edge_index` shape `[2, 0]`; output shapes were `[5, 32]` for hidden and cell states. No placeholder edge, self-loop, or adapter was used.
- Serialization reader: PASS. `scripts.build_stage_a_tgnn_dataset.load_sequence` was imported only and never called. Audited restricted opens were zero.
- Isolation: PASS. The dedicated environment is outside the repository; project and dedicated environment file counts were unchanged; no install/download command was run; pre-existing worktree changes were preserved.

The reader import emitted the source project's existing `sgp4 not available. Canonical SGP4 propagation is unavailable.` observation while no propagation was invoked. This did not affect the frozen TGNN package check, which passed in the dedicated environment with repository source metadata excluded from `pip check`.

## Required artifacts

- `tgnn_cpu_runtime_gate_report.json`
- `package_version_and_abi_verification.json`
- `native_extension_import_verification.json`
- `gclstm_runtime_verification.json`
- `zero_edge_runtime_verification.json`
- `environment_isolation_and_test_seal.json`
- `artifact_inventory.json` (self-excluding)
- `README.md`

The self-excluding artifact inventory records exact byte lengths and SHA-256 values for the seven other gate artifacts.
