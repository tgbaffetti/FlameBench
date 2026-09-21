"""MeshGraphNets on the cell-adjacency graph (Pfaff et al. 2021, arXiv:2010.03409).

Encode-process-decode with residual message passing over face-sharing cells.
phi(t) enters as global node channels (one per tap of the forcing window).
Prediction target is the next scaled snapshot (absolute, matching the pipeline
convention); the paper's delta target + noise injection are protocol ablations.
Use with compressor "identity".
"""
import numpy as np
import torch
from torch import nn
from .DLModel import DLModel


def mlp(sizes, norm=True):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.GELU()]
    layers = layers[:-1]
    if norm:
        layers.append(nn.LayerNorm(sizes[-1]))
    return nn.Sequential(*layers)


class GraphBlock(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.edge_mlp = mlp([3 * dim, hidden, dim])
        self.node_mlp = mlp([2 * dim, hidden, dim])

    def forward(self, nodes, edges, senders, receivers):
        edges = edges + self.edge_mlp(
            torch.cat((edges, nodes[:, senders], nodes[:, receivers]), dim=-1))
        incoming = torch.zeros_like(nodes).index_add_(1, receivers, edges)
        nodes = nodes + self.node_mlp(torch.cat((nodes, incoming), dim=-1))
        return nodes, edges


class MGNNetwork(nn.Module):
    def __init__(self, rank, Nx=4, Ni=4, coordinates=None, edges=None, fields=11,
                 dim=128, hidden=128, passes=8, **kwargs):
        super().__init__()
        if coordinates is None or edges is None:
            raise ValueError("MeshGraphNets requires coordinates and edges files (see DataProcessing.grid)")
        xyz = np.load(coordinates, allow_pickle=False) if isinstance(coordinates, str) else np.asarray(coordinates)
        adjacency = np.load(edges, allow_pickle=False) if isinstance(edges, str) else np.asarray(edges)
        cells = rank // fields
        if fields * cells != rank or len(xyz) != cells:
            raise ValueError(f"rank {rank} incompatible with {fields} fields x {len(xyz)} cells")
        if adjacency.ndim != 2 or adjacency.shape[0] != 2 or adjacency.max() >= cells:
            raise ValueError("edges must be (2, E) indices into cells")
        xz = xyz[:, [0, 2]].astype(np.float32)  # planar mesh: Y is constant
        xz = (xz - xz.min(axis=0)) / (xz.max(axis=0) - xz.min(axis=0))
        relative = xz[adjacency[1]] - xz[adjacency[0]]
        distance = np.linalg.norm(relative, axis=1, keepdims=True)
        self.register_buffer("coords", torch.from_numpy(xz))
        self.register_buffer("senders", torch.from_numpy(adjacency[0]))
        self.register_buffer("receivers", torch.from_numpy(adjacency[1]))
        self.register_buffer("edge_attr", torch.from_numpy(
            np.concatenate((relative, distance), axis=1).astype(np.float32) / max(distance.max(), 1e-12)))
        self.fields, self.cells = fields, cells
        self.node_encoder = mlp([(Nx + 1) * fields + 2 + Ni + 2, hidden, dim])
        self.edge_encoder = mlp([3, hidden, dim])
        self.blocks = nn.ModuleList(GraphBlock(dim, hidden) for _ in range(passes))
        self.decoder = mlp([dim, hidden, fields], norm=False)

    def forward(self, history, forcing):
        batch = history.shape[0]
        x = history.reshape(batch, history.shape[1], self.fields, self.cells)
        x = x.permute(0, 3, 1, 2).reshape(batch, self.cells, -1)
        coords = self.coords.unsqueeze(0).expand(batch, -1, -1)
        phi = (forcing - 1).unsqueeze(1).expand(-1, self.cells, -1)
        nodes = self.node_encoder(torch.cat((x, coords, phi), dim=-1))
        edges = self.edge_encoder(self.edge_attr).unsqueeze(0).expand(batch, -1, -1)
        for block in self.blocks:
            nodes, edges = block(nodes, edges, self.senders, self.receivers)
        return self.decoder(nodes).transpose(1, 2).reshape(batch, self.fields * self.cells)


class MeshGraphNet(DLModel):
    hyperparams = {"lr": {"value": 1e-3, "type": "float", "low": 1e-5, "high": 3e-3, "log": True},
                   "dim": {"value": 128, "type": "categorical", "choices": [64, 128]},
                   "passes": {"value": 8, "type": "categorical", "choices": [4, 8, 12]}}

    def __init__(self, rank, **kwargs):
        super().__init__(MGNNetwork(rank, **kwargs), **kwargs)
