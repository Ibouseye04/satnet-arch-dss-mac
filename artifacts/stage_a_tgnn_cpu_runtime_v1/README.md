# Stage A TGNN CPU Runtime Environment v1

**Verdict: STAGE A TGNN CPU RUNTIME V1 PREPARED — READY FOR NARROW INDEPENDENT RUNTIME GATE**

This artifact bundle records environment preparation only. It does not perform an independent runtime gate.

## Dedicated environment

- Environment root: `C:\Users\johns\satnet-stage-a-tgnn-runtime-v1`
- Virtual environment: `C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv`
- Python executable: `C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe`
- Activation: `& 'C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\Activate.ps1'`
- Platform: Windows AMD64, CPython 3.11.9

## Frozen packages

| Package | Installed distribution |
|---|---:|
| Python | 3.11.9 |
| PyTorch | 2.12.1+cpu |
| PyTorch Geometric | 2.8.0 |
| PyTorch Geometric Temporal | 0.56.2 |
| torch_sparse | 0.6.18+pt212cpu |
| torch_scatter | 2.1.2+pt212cpu |
| scikit-learn | 1.9.0 |
| joblib | 1.5.3 |
| NumPy | 1.26.4 |
| pandas | 2.3.3 |
| SciPy | 1.17.1 |

The local `+cpu` and `+pt212cpu` suffixes identify the CPU-native wheel builds; the frozen base versions match exactly. Source identities, wheel filenames, and wheel SHA-256 values are in `installed_package_inventory.json`.

## Validation results

- `torch`, `torch_geometric`, `torch_geometric_temporal`, `torch_sparse`, and `torch_scatter` imports: PASS.
- `GCLSTM` import: PASS; the former `WinError 127` failure is absent.
- `torch_sparse` native result: PASS; `_version_cpu.pyd` loaded, no CUDA extension.
- `torch_scatter` native result: PASS; `_version_cpu.pyd` loaded, no CUDA extension.
- CUDA status: `torch.cuda.is_available()` is `false`; `torch.version.cuda` is `null`; no NVIDIA distributions or CUDA runtime DLLs are installed.
- CPU device, deterministic algorithms (`warn_only=False`), one intra-op thread, and one inter-op thread: PASS.
- Contracted GCLSTM constructor `(in_channels=14, out_channels=32, K=2)`: PASS.
- Hidden-dimension 64 constructor variant: PASS.
- Synthetic non-empty structural GCLSTM call: PASS.
- Synthetic exact zero-edge shape `[2, 0]`: PASS directly; no runtime adapter is required.
- Accepted serialization reader import: PASS (`scripts.build_stage_a_tgnn_dataset.load_sequence`); no sequence artifact was opened.

The synthetic smoke test used only a 5-node, 14-feature matrix and synthetic edge indexes. It performed no optimization, loss calculation, backward pass, checkpoint creation, or production-data access.

## Scope protections

- No train, validation, or test graph sequence was loaded.
- No preprocessing statistics were calculated.
- No model was trained and no checkpoint was created.
- Accepted datasets, gates, training plans, and RF outputs were not modified.
- The existing project environment `C:\Users\johns\Developer\satnet-arch-dss-mac\.venv` was not used or modified.
- The pre-existing unrelated worktree changes were not touched.

## Evidence files

- `runtime_environment_manifest.json`
- `native_extension_verification.json`
- `gclstm_import_and_constructor_report.json`
- `synthetic_zero_edge_smoke_report.json`
- `installed_package_inventory.json`
- `environment_setup_commands.ps1`
- `artifact_inventory.json`

Artifact SHA-256 values are recorded in `artifact_inventory.json`.
