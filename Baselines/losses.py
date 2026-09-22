"""Pointwise error functions shared by compressors and forecasters (mean reduction)."""
from torch import nn

ERRORS = {"mse": nn.functional.mse_loss, "mae": nn.functional.l1_loss,
          "huber": nn.functional.huber_loss, "smooth_l1": nn.functional.smooth_l1_loss}


def error_function(name):
    if name not in ERRORS:
        raise ValueError(f"loss must be one of {sorted(ERRORS)}")
    return ERRORS[name]
