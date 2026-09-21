"""DeepONet on the unstructured mesh (Lu et al. 2021, arXiv:1910.03193).

Branch: field history sampled at fixed random sensor cells + the phi window.
Trunk: normalized (x, z) cell-center coordinates -> p basis functions per query.
Output: next scaled snapshot, one coefficient set per field (flattened to the
identity-compressor latent). Use with compressor "identity".
"""
import numpy as np
import torch
from torch import nn
from .DLModel import DLModel


def mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.GELU()]
    return nn.Sequential(*layers[:-1])


class DeepONetNetwork(nn.Module):
    def __init__(self, rank, Nx=4, Ni=4, coordinates=None, fields=11, sensors=2048,
                 p=64, hidden=256, branch_layers=3, trunk_layers=3, **kwargs):
        super().__init__()
        if coordinates is None:
            raise ValueError("DeepONet requires the coordinates file (see DataProcessing.grid)")
        xyz = np.load(coordinates, allow_pickle=False) if isinstance(coordinates, str) else np.asarray(coordinates)
        cells = rank // fields
        if fields * cells != rank or len(xyz) != cells:
            raise ValueError(f"rank {rank} incompatible with {fields} fields x {len(xyz)} cells")
        xz = xyz[:, [0, 2]].astype(np.float32)  # planar mesh: Y is constant
        xz = (xz - xz.min(axis=0)) / (xz.max(axis=0) - xz.min(axis=0))
        self.register_buffer("coords", torch.from_numpy(xz))
        self.register_buffer("sensor_idx", torch.randperm(cells)[:sensors])
        self.fields, self.cells, self.p = fields, cells, p
        self.branch = mlp([(Nx + 1) * fields * len(self.sensor_idx) + Ni + 2]
                          + [hidden] * branch_layers + [fields * p])
        self.trunk = mlp([2] + [hidden] * trunk_layers + [p])
        self.bias = nn.Parameter(torch.zeros(fields))

    def forward(self, history, forcing):
        batch = history.shape[0]
        x = history.reshape(batch, history.shape[1], self.fields, self.cells)
        sensed = x[..., self.sensor_idx].reshape(batch, -1)
        coefficients = self.branch(torch.cat((sensed, forcing - 1), dim=1))
        coefficients = coefficients.reshape(batch, self.fields, self.p)
        basis = self.trunk(self.coords)
        out = torch.einsum("bfp,cp->bfc", coefficients, basis) + self.bias[None, :, None]
        return out.reshape(batch, self.fields * self.cells)


class DeepONet(DLModel):
    hyperparams = {"lr": {"value": 1e-3, "type": "float", "low": 1e-5, "high": 3e-3, "log": True},
                   "p": {"value": 64, "type": "categorical", "choices": [32, 64, 128]},
                   "hidden": {"value": 256, "type": "categorical", "choices": [128, 256, 512]}}

    def __init__(self, rank, **kwargs):
        super().__init__(DeepONetNetwork(rank, **kwargs), **kwargs)
