# utils/seed.py

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_global_seed(seed: int) -> None:
    """
    Set global seeds for reproducibility across Python, NumPy, and PyTorch.

    This helper is safe even if PyTorch is not installed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Make CuDNN deterministic when possible.
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False