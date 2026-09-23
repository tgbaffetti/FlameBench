"""Transolver on the mesh cells (Wu et al. 2024, arXiv:2402.02366).

Physics attention: cells are softly assigned to M learned slices, attention runs over the M
slice tokens (linear in cells) and the result is broadcast back. Simplification vs the paper:
one slice assignment shared by all heads. The forcing window enters as per-point channels.
The output is the next latent of the identity compressor (fields x cells, absolute state: the
only Transolver configuration that survived full rollouts on the cell benchmark was
unrolled-trained absolute; rollout training arrives through the dataset horizon and
rollout_weight).
"""
import torch
from torch import nn
from .DLModel import DLModel
from .mesh import cell_coordinates


class PhysicsAttention(nn.Module):
    def __init__(self, dim, heads=4, slices=32):
        super().__init__()
        self.assign = nn.Linear(dim, slices)
        self.value = nn.Linear(dim, dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        weights = torch.softmax(self.assign(x), dim=-1)              # (batch, cells, slices)
        mass = weights.sum(dim=1, keepdim=True).transpose(1, 2)      # (batch, slices, 1)
        tokens = torch.bmm(weights.transpose(1, 2), self.value(x)) / (mass + 1e-8)
        tokens, _ = self.attention(tokens, tokens, tokens)
        return self.out(torch.bmm(weights, tokens))                  # (batch, cells, dim)


class Block(nn.Module):
    def __init__(self, dim, heads, slices, mlp_ratio=2):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = PhysicsAttention(dim, heads, slices)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(),
                                 nn.Linear(mlp_ratio * dim, dim))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TransolverNetwork(nn.Module):
    def __init__(self, rank, Nx, Ni, coordinates, fields, dim, depth, heads, slices, mlp_ratio):
        super().__init__()
        cells = rank // fields
        if fields * cells != rank or len(coordinates) != cells:
            raise ValueError(f"rank {rank} incompatible with {fields} fields x {len(coordinates)} cells")
        self.Nx, self.Ni, self.fields, self.cells = Nx, Ni, fields, cells
        self.register_buffer("coords", torch.as_tensor(coordinates))
        self.embed = nn.Linear((Nx + 1) * fields + 2 + Ni + 1, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, slices, mlp_ratio) for _ in range(depth))
        self.head = nn.Linear(dim, fields)

    def forward(self, states, forcing):
        batch = states.shape[0]
        x = states[:, -self.Nx - 1:].reshape(batch, self.Nx + 1, self.fields, self.cells)
        x = x.permute(0, 3, 1, 2).reshape(batch, self.cells, -1)     # (batch, cells, window * fields)
        coords = self.coords.unsqueeze(0).expand(batch, -1, -1)
        phi = forcing[:, -self.Ni - 1:].unsqueeze(1).expand(-1, self.cells, -1)
        x = self.embed(torch.cat((x, coords, phi), dim=-1))
        for block in self.blocks:
            x = block(x)
        return self.head(x).transpose(1, 2).reshape(batch, self.fields * self.cells)


class Transolver(DLModel):
    """grid: path to the ImageGrid .npz; fields: field count (latent rank = fields x cells)."""
    name = "transolver"
    hyperparameters_ranges = {**{k: v for k, v in DLModel.hyperparameters_ranges.items() if k != "dropout"},
                              "dim": {"type": "categorical", "choices": [64, 128, 256]},
                              "slices": {"type": "categorical", "choices": [16, 32, 64]}}

    def __init__(self, input_size=None, output_size=None, Nx=9, Ni=0, grid=None, fields=None,
                 dim=128, depth=4, heads=4, slices=32, mlp_ratio=2, **training):
        if fields is None:
            raise ValueError("Transolver requires the field count (fields)")
        network = TransolverNetwork(output_size, Nx, Ni, cell_coordinates(grid), fields,
                                    dim, depth, heads, slices, mlp_ratio)
        super().__init__(network, Nx, Ni, **training)
