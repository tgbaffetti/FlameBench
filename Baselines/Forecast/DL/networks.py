import torch
from torch import nn
from .DLModel import DLModel


def forecaster_inputs(history, forcing, Nx, Ni):
    """Align states and forcing on times t-max(Nx,Ni), ..., t, t+1.

    Channels: latent state, phi-1, state-present flag, forcing-present flag.
    Missing values are zero-filled and flagged, never treated as observations.
    """
    if history.ndim != 3 or history.shape[1] != Nx + 1:
        raise ValueError("Expected history shape (batch, Nx+1, rank)")
    if forcing.shape != (history.shape[0], Ni + 2):
        raise ValueError("Expected forcing shape (batch, Ni+2)")
    batch, _, rank = history.shape
    length = max(Nx, Ni) + 2
    inputs = history.new_zeros(batch, length, rank + 3)
    state_start = length - Nx - 2
    forcing_start = length - Ni - 2
    inputs[:, state_start:-1, :rank] = history
    inputs[:, state_start:-1, rank + 1] = 1
    inputs[:, forcing_start:, rank] = forcing - 1
    inputs[:, forcing_start:, rank + 2] = 1
    return inputs


class RecurrentNetwork(nn.Module):
    def __init__(self, rank, Nx=9, Ni=0, hidden=64, layers=2, kind="gru", **kwargs):
        super().__init__()
        self.Nx, self.Ni = Nx, Ni
        cls = nn.GRU if kind == "gru" else nn.LSTM
        self.core = cls(rank + 3, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, rank))

    def forward(self, history, forcing):
        inputs = forecaster_inputs(history, forcing, self.Nx, self.Ni)
        states, _ = self.core(inputs)
        return history[:, -1] + self.head(states[:, -1])


class TransformerNetwork(nn.Module):
    def __init__(self, rank, Nx=9, Ni=0, hidden=64, layers=2, heads=4, **kwargs):
        super().__init__()
        self.Nx, self.Ni = Nx, Ni
        self.embed = nn.Linear(rank + 3, hidden)
        self.position = nn.Parameter(torch.zeros(1, max(Nx, Ni) + 2, hidden))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.0,
                                            batch_first=True, activation="gelu")
        self.core = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(hidden, rank)

    def forward(self, history, forcing):
        inputs = forecaster_inputs(history, forcing, self.Nx, self.Ni)
        tokens = self.embed(inputs) + self.position
        output = self.core(tokens)[:, -1]
        return history[:, -1] + self.head(output)


class GRU(DLModel):
    name = "gru"
    def __init__(self, rank, **kwargs):
        super().__init__(RecurrentNetwork(rank, kind="gru", **kwargs), **kwargs)


class LSTM(DLModel):
    name = "lstm"
    def __init__(self, rank, **kwargs):
        super().__init__(RecurrentNetwork(rank, kind="lstm", **kwargs), **kwargs)


class Transformer(DLModel):
    name = "transformer"
    def __init__(self, rank, Nx=9, **kwargs):
        super().__init__(TransformerNetwork(rank, Nx=Nx, **kwargs), **kwargs)
