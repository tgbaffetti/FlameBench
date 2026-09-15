# seed_utils.py

import os
import random
from typing import Optional


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """
    Set random seeds for Python, NumPy, PyTorch, and optionally other common libraries.

    Args:
        seed: The seed value to use.
        deterministic: If True, configures PyTorch for more deterministic behavior.
            This can reduce performance and may raise errors for some operations.

    Returns:
        The seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    except ImportError:
        pass

    return seed


def init_weights(module) -> None:
    """
    Initialize weights of a PyTorch module deterministically.
    Call this after seed_everything() and after building the model.

    Conv2d / ConvTranspose2d  → Kaiming uniform (default PyTorch, made explicit)
    Linear                    → Xavier uniform
    LayerNorm / BatchNorm     → weight=1, bias=0
    """
    import torch.nn as nn

    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)