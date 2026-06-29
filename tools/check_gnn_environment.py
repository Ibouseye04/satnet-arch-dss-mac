#!/usr/bin/env python3
"""Check whether the local machine is ready for Temporal GNN training.

This script is intentionally lightweight and safe to run before attempting a
GNN smoke/full training run. It checks Python, optional ML imports, CUDA/GPU
visibility, and whether the current SatelliteGNN can be instantiated.

Usage:
    python tools/check_gnn_environment.py
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def import_optional(module_name: str) -> ModuleType | None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        print(f"[FAIL] import {module_name}: {type(exc).__name__}: {exc}")
        return None

    version = getattr(module, "__version__", "unknown")
    print(f"[ OK ] import {module_name}: version={version}")
    return module


def check_python() -> bool:
    header("Python")
    print(f"Executable: {sys.executable}")
    print(f"Version:    {sys.version.split()[0]}")
    print(f"Platform:   {platform.platform()}")
    supported = (3, 11) <= sys.version_info[:2] <= (3, 12)
    if supported:
        print("[ OK ] Python version is in the recommended 3.11-3.12 range.")
    else:
        print("[WARN] Recommended Python version is 3.11 or 3.12 for this repo.")
    return supported


def check_imports() -> dict[str, ModuleType | None]:
    header("Required GNN Imports")
    modules = {
        "torch": import_optional("torch"),
        "torch_geometric": import_optional("torch_geometric"),
        "torch_geometric_temporal": import_optional("torch_geometric_temporal"),
    }
    return modules


def check_cuda(torch_module: ModuleType | None) -> bool:
    header("CUDA / GPU")
    if torch_module is None:
        print("[FAIL] torch is unavailable, so CUDA cannot be checked.")
        return False

    print(f"torch.version.cuda: {getattr(torch_module.version, 'cuda', None)}")
    cuda_available = bool(torch_module.cuda.is_available())
    print(f"torch.cuda.is_available(): {cuda_available}")

    if not cuda_available:
        print("[WARN] CUDA is not visible to PyTorch. GNN can still run on CPU, but slower.")
        return False

    device_count = torch_module.cuda.device_count()
    print(f"CUDA devices: {device_count}")
    for idx in range(device_count):
        props = torch_module.cuda.get_device_properties(idx)
        total_gib = props.total_memory / (1024 ** 3)
        print(f"  [{idx}] {props.name} | VRAM={total_gib:.2f} GiB")

    print("[ OK ] PyTorch can see a CUDA GPU.")
    return True


def check_model_instantiation() -> bool:
    header("SatelliteGNN Smoke Import")
    try:
        import torch
        from torch_geometric.data import Data
        from satnet.models.gnn_model import SatelliteGNN

        model = SatelliteGNN(node_features=3, hidden_channels=8, out_channels=2)
        sequence = [
            Data(
                x=torch.randn(6, 3),
                edge_index=torch.tensor(
                    [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]],
                    dtype=torch.long,
                ),
            ),
            Data(
                x=torch.randn(6, 3),
                edge_index=torch.tensor(
                    [[0, 2, 4, 1, 3, 5], [2, 4, 0, 3, 5, 1]],
                    dtype=torch.long,
                ),
            ),
        ]
        out = model(sequence)
    except Exception as exc:
        print(f"[FAIL] Could not instantiate/run SatelliteGNN: {type(exc).__name__}: {exc}")
        return False

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[ OK ] SatelliteGNN forward pass succeeded.")
    print(f"Model parameters: {param_count}")
    print(f"Output shape: {tuple(out.shape)}")
    return True


def check_dataset_presence() -> bool:
    header("Dataset Presence")
    runs_csv = PROJECT_ROOT / "data" / "tier1_design_runs.csv"
    steps_csv = PROJECT_ROOT / "data" / "tier1_design_steps.csv"
    ok = True
    for path in [runs_csv, steps_csv]:
        if path.exists():
            print(f"[ OK ] {path}")
        else:
            print(f"[WARN] Missing {path}")
            ok = False
    if not ok:
        print("Run: python scripts/export_design_dataset.py --smoke --seed 42")
    return ok


def main() -> None:
    python_ok = check_python()
    modules = check_imports()
    cuda_ok = check_cuda(modules["torch"])
    model_ok = check_model_instantiation()
    dataset_ok = check_dataset_presence()

    header("Readiness Summary")
    import_ok = all(modules[name] is not None for name in modules)
    if import_ok and model_ok:
        print("[ OK ] Temporal GNN code path can import and execute a tiny forward pass.")
    else:
        print("[FAIL] Temporal GNN dependency stack is not ready.")

    if cuda_ok:
        print("[ OK ] GPU acceleration is available for training.")
    else:
        print("[WARN] GPU acceleration is unavailable; training will use CPU if requested.")

    if dataset_ok:
        print("[ OK ] Smoke dataset files are present.")
    else:
        print("[WARN] Dataset files are missing.")

    if python_ok and import_ok and model_ok and dataset_ok:
        print("\nNext smoke command:")
        print("python scripts/train_gnn_model.py --smoke --target-name partition_any --device auto")
    else:
        print("\nResolve the FAIL/WARN items above before treating GNN training as ready.")


if __name__ == "__main__":
    main()
