from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def initialize_determinism(seed: int) -> dict[str, Any]:
    """Initialize only a frozen, explicitly supplied seed; never generate one."""
    if seed not in {42, 123, 456, 789, 2026}:
        raise ValueError(f"Seed {seed} is not in the frozen training seed registry")
    random.seed(seed)
    np.random.seed(seed)
    state: dict[str, Any] = {"python_random": seed, "numpy": seed, "seed": seed}
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        state.update({"torch": seed, "torch_cuda": bool(torch.cuda.is_available()), "torch_deterministic_algorithms": True})
    except ImportError:
        state["torch"] = "not_imported"
    os.environ["PYTHONHASHSEED"] = str(seed)
    return state
