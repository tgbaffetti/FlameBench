import torch
from torch import nn
from .DLModel import DLModel


class RecurrentNetwork(nn.Module):
    """GRU/LSTM over the rows [x(s), phi(s+1) - 1]; an MLP head maps the last state to the increment.

    hidden: state size; layers: stacked recurrent layers; dropout: between recurrent layers
    (ignored by torch when layers == 1).
    """
    def __init__(self, rank, hidden=64, layers=2, dropout=0.0, kind="gru"):
        super().__init__()
        cls = nn.GRU if kind == "gru" else nn.LSTM
        self.core = cls(rank + 1, hidden, num_layers=layers, dropout=dropout if layers > 1 else 0.0,
                        batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, rank))

    def forward(self, states, forcing):
        outputs, _ = self.core(torch.cat((states, forcing[..., None]), dim=-1))
        return states[:, -1] + self.head(outputs[:, -1])


class TransformerNetwork(nn.Module):
    """Transformer encoder over the rows [x(s), phi(s+1) - 1]; the last token predicts the increment.

    length: number of rows, max(Nx, Ni) + 1; hidden: token size; layers: encoder layers; heads:
    attention heads (must divide hidden); feedforward: MLP size inside each layer (default
    4 * hidden); dropout: inside each layer.
    """
    def __init__(self, rank, length, hidden=64, layers=2, heads=4, feedforward=None, dropout=0.0):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.embed = nn.Linear(rank + 1, hidden)
        self.position = nn.Parameter(torch.zeros(1, length, hidden))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, feedforward or 4 * hidden, dropout=dropout,
                                           batch_first=True, activation="gelu")
        self.core = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(hidden, rank)

    def forward(self, states, forcing):
        tokens = self.embed(torch.cat((states, forcing[..., None]), dim=-1)) + self.position
        return states[:, -1] + self.head(self.core(tokens)[:, -1])


class GRU(DLModel):
    """training: DLModel keywords (device, lr, epochs, patience, frozen_epochs, rollout_*, loss)."""
    name = "gru"
    def __init__(self, rank, Nx=9, Ni=0, hidden=64, layers=2, dropout=0.0, **training):
        super().__init__(RecurrentNetwork(rank, hidden, layers, dropout, "gru"), Nx, Ni, **training)


class LSTM(DLModel):
    name = "lstm"
    def __init__(self, rank, Nx=9, Ni=0, hidden=64, layers=2, dropout=0.0, **training):
        super().__init__(RecurrentNetwork(rank, hidden, layers, dropout, "lstm"), Nx, Ni, **training)


class Transformer(DLModel):
    name = "transformer"
    hyperparams = {**DLModel.hyperparams, "heads": {"type": "categorical", "choices": [2, 4, 8]}}

    def __init__(self, rank, Nx=9, Ni=0, hidden=64, layers=2, heads=4, feedforward=None, dropout=0.0, **training):
        network = TransformerNetwork(rank, max(Nx, Ni) + 1, hidden, layers, heads, feedforward, dropout)
        super().__init__(network, Nx, Ni, **training)
