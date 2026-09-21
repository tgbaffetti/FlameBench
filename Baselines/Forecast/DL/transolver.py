"""Transolver on the unstructured mesh (Wu et al. 2024, arXiv:2402.02366).

Physics attention: cells are softly assigned to M learned slices, attention runs
over the M slice tokens (linear in cells), and the result is broadcast back.
Simplification vs the paper: one slice assignment shared by all heads.
phi(t) enters as per-point input channels, one per tap of the forcing window.
Use with compressor "identity"; output is the next scaled snapshot.
"""
import numpy as np
import torch
from torch import nn
from .DLModel import DLModel


class PhysicsAttention(nn.Module):
    def __init__(self, dim, heads=4, slices=32):
        super().__init__()
        self.assign = nn.Linear(dim, slices)
        self.value = nn.Linear(dim, dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        weights = torch.softmax(self.assign(x), dim=-1)              # (B, C, M)
        mass = weights.sum(dim=1, keepdim=True).transpose(1, 2)      # (B, M, 1)
        tokens = torch.bmm(weights.transpose(1, 2), self.value(x)) / (mass + 1e-8)
        tokens, _ = self.attention(tokens, tokens, tokens)
        return self.out(torch.bmm(weights, tokens))                  # (B, C, dim)


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
    def __init__(self, rank, Nx=4, Ni=4, coordinates=None, fields=11, dim=128,
                 depth=4, heads=4, slices=32, mlp_ratio=2, **kwargs):
        super().__init__()
        if coordinates is None:
            raise ValueError("Transolver requires the coordinates file (see DataProcessing.grid)")
        xyz = np.load(coordinates, allow_pickle=False) if isinstance(coordinates, str) else np.asarray(coordinates)
        cells = rank // fields
        if fields * cells != rank or len(xyz) != cells:
            raise ValueError(f"rank {rank} incompatible with {fields} fields x {len(xyz)} cells")
        xz = xyz[:, [0, 2]].astype(np.float32)  # planar mesh: Y is constant
        xz = (xz - xz.min(axis=0)) / (xz.max(axis=0) - xz.min(axis=0))
        self.register_buffer("coords", torch.from_numpy(xz))
        self.fields, self.cells = fields, cells
        self.embed = nn.Linear((Nx + 1) * fields + 2 + Ni + 2, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, slices, mlp_ratio) for _ in range(depth))
        self.head = nn.Linear(dim, fields)

    def forward(self, history, forcing):
        batch = history.shape[0]
        x = history.reshape(batch, history.shape[1], self.fields, self.cells)
        x = x.permute(0, 3, 1, 2).reshape(batch, self.cells, -1)     # (B, C, H*F)
        coords = self.coords.unsqueeze(0).expand(batch, -1, -1)
        phi = (forcing - 1).unsqueeze(1).expand(-1, self.cells, -1)
        x = self.embed(torch.cat((x, coords, phi), dim=-1))
        for block in self.blocks:
            x = block(x)
        return self.head(x).transpose(1, 2).reshape(batch, self.fields * self.cells)


class Transolver(DLModel):
    hyperparams = {"lr": {"value": 1e-3, "type": "float", "low": 1e-5, "high": 3e-3, "log": True},
                   "dim": {"value": 128, "type": "categorical", "choices": [64, 128, 256]},
                   "slices": {"value": 32, "type": "categorical", "choices": [16, 32, 64]}}

    def __init__(self, rank, **kwargs):
        super().__init__(TransolverNetwork(rank, **kwargs), **kwargs)
