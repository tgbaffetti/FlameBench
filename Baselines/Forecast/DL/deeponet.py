"""DeepONet on the mesh cells (Lu et al. 2021, arXiv:1910.03193).

Branch: the state window sampled at fixed random sensor cells plus the forcing window.
Trunk: normalized cell-center coordinates -> p basis functions. The output is the next latent
of the identity compressor (fields x cells, absolute state: on the cell benchmark the
unrolled-trained absolute variant was the operator configuration that survived full rollouts,
and rollout training arrives through the dataset horizon and rollout_weight).
"""
import numpy as np
import torch
from torch import nn
from .DLModel import DLModel
from .mesh import cell_coordinates


def mlp(sizes):
    layers = []
    for before, after in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(before, after), nn.GELU()]
    return nn.Sequential(*layers[:-1])


class DeepONetNetwork(nn.Module):
    def __init__(self, rank, Nx, Ni, coordinates, fields, sensors, p, hidden, branch_layers, trunk_layers):
        super().__init__()
        cells = rank // fields
        if fields * cells != rank or len(coordinates) != cells:
            raise ValueError(f"rank {rank} incompatible with {fields} fields x {len(coordinates)} cells")
        self.Nx, self.Ni, self.fields, self.cells, self.p = Nx, Ni, fields, cells, p
        self.register_buffer("coords", torch.as_tensor(coordinates))
        self.register_buffer("sensor_idx", torch.randperm(cells)[:min(sensors, cells)])
        self.branch = mlp([(Nx + 1) * fields * len(self.sensor_idx) + Ni + 1]
                          + [hidden] * branch_layers + [fields * p])
        self.trunk = mlp([2] + [hidden] * trunk_layers + [p])
        self.bias = nn.Parameter(torch.zeros(fields))

    def forward(self, states, forcing):
        batch = states.shape[0]
        x = states[:, -self.Nx - 1:].reshape(batch, self.Nx + 1, self.fields, self.cells)
        sensed = x[..., self.sensor_idx].reshape(batch, -1)
        coefficients = self.branch(torch.cat((sensed, forcing[:, -self.Ni - 1:]), dim=1))
        basis = self.trunk(self.coords)
        out = torch.einsum("bfp,cp->bfc", coefficients.reshape(batch, self.fields, self.p), basis)
        return (out + self.bias[None, :, None]).reshape(batch, self.fields * self.cells)


class DeepONet(DLModel):
    """grid: path to the ImageGrid .npz; fields: field count (latent rank = fields x cells)."""
    name = "deeponet"
    needs_grid = True  # grid and fields default from the dataset metadata (see train_forecaster).
    hyperparameters_ranges = {**{k: v for k, v in DLModel.hyperparameters_ranges.items() if k != "dropout"},
                              "sensors": {"type": "categorical", "choices": [512, 1024, 2048]},
                              "p": {"type": "categorical", "choices": [32, 64, 128]},
                              "hidden": {"type": "categorical", "choices": [128, 256]}}

    def __init__(self, input_size=None, output_size=None, Nx=9, Ni=0, grid=None, fields=None,
                 sensors=2048, p=64, hidden=256, branch_layers=3, trunk_layers=3, **training):
        if fields is None:
            raise ValueError("DeepONet requires the field count (fields)")
        network = DeepONetNetwork(output_size, Nx, Ni, cell_coordinates(grid), fields,
                                  sensors, p, hidden, branch_layers, trunk_layers)
        super().__init__(network, Nx, Ni, **training)
