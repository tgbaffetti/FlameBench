"""Neural forecasters with configurable layers.

Every network reads the rows [x(s), phi(s+1) - 1] of a ForecasterDataset window, shape
(batch, length, input_size), and returns the increment of the state, shape (batch, output_size):
x(t+1) = x(t) + increment. input_size is the state size plus one (the forcing column) and
output_size is the state size. The layers form one nn.Sequential, so printing a forecaster's
network shows each layer in order.

GRU, LSTM and CNN take a list with one width per layer (hiddens or channels). Each layer is
followed by normalization (None, "layer" or "batch"), activation (None, "relu", "gelu", "silu"
or "tanh") and dropout. input_normalization (None, "layer" or "batch") normalizes the input rows
before the first layer; "batch" standardizes each column (latent mode or forcing) with batch
statistics, which suits raw POD latents whose modes differ in scale. The optimizer and
weight_decay are DLModel keywords.
"""
import torch
from torch import nn
from .DLModel import DLModel

ACTIVATIONS = {None: nn.Identity, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}


class Increment(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, states, forcing):
        return states[:, -1] + self.layers(torch.cat((states, forcing[..., None]), dim=-1))


class Normalization(nn.Module):
    """None, "layer" or "batch" normalization of the features of (batch, length, features)."""
    def __init__(self, kind, size):
        super().__init__()
        if kind not in {None, "layer", "batch"}:
            raise ValueError(f"Unknown normalization: {kind}")
        self.kind = kind
        if kind == "layer":
            self.norm = nn.LayerNorm(size)
        elif kind == "batch":
            self.norm = nn.BatchNorm1d(size)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        if self.kind == "batch":  # BatchNorm1d expects (batch, features, length).
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x)


class Recurrent(nn.Module):
    """One nn.GRU or nn.LSTM layer that returns its outputs only, (batch, length, directions * size)."""
    def __init__(self, layer_class, input_size, size, bidirectional):
        super().__init__()
        self.layer = layer_class(input_size, size, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        return self.layer(x)[0]


class Convolution(nn.Module):
    """nn.Conv1d over the rows of (batch, length, channels), with valid padding."""
    def __init__(self, input_size, size, kernel_size):
        super().__init__()
        self.layer = nn.Conv1d(input_size, size, kernel_size)

    def forward(self, x):
        return self.layer(x.transpose(1, 2)).transpose(1, 2)


class Position(nn.Module):
    """Learned position embedding added to the tokens, so attention knows the row order."""
    def __init__(self, length, size):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, length, size) * 0.02)

    def forward(self, tokens):
        return tokens + self.embedding


class LastRow(nn.Module):
    """Features of the last row. With backward_from (bidirectional layers), the features from that
    index on come from the backward direction and are read at the first row instead, where it has
    seen the whole window; both parts are concatenated into one vector."""
    def __init__(self, backward_from=None):
        super().__init__()
        self.backward_from = backward_from

    def forward(self, x):
        if self.backward_from is None:
            return x[:, -1]
        return torch.cat((x[:, -1, :self.backward_from], x[:, 0, self.backward_from:]), dim=-1)


def block(layer, size, normalization, activation, dropout):
    return [layer, Normalization(normalization, size), ACTIVATIONS[activation](), nn.Dropout(dropout)]


class GRU(DLModel):
    """hiddens: hidden size of each recurrent layer; bidirectional: each layer also reads the
    window backward. A linear layer maps the last row to the increment.
    training: DLModel keywords (device, lr, optimizer, weight_decay, epochs, ...)."""
    name = "gru"
    layer_class = nn.GRU
    hyperparameters_ranges = {**DLModel.hyperparameters_ranges,
                              "hiddens": {"type": "categorical", "choices": [[16], [32], [64], [32, 32]]},
                              "normalization": {"type": "categorical", "choices": [None, "layer"]}}

    def __init__(self, input_size, output_size, Nx=9, Ni=0, hiddens=(64, 64), bidirectional=False,
                 normalization=None, activation=None, dropout=0.0, input_normalization=None, **training):
        directions = 2 if bidirectional else 1
        layers, size = [Normalization(input_normalization, input_size)], input_size
        for hidden in hiddens:
            recurrent = Recurrent(self.layer_class, size, hidden, bidirectional)
            size = directions * hidden
            layers += block(recurrent, size, normalization, activation, dropout)
        last = LastRow(hiddens[-1] if bidirectional else None)
        super().__init__(Increment(*layers, last, nn.Linear(size, output_size)), Nx, Ni, **training)


class LSTM(GRU):
    name = "lstm"
    layer_class = nn.LSTM


class CNN(DLModel):
    """channels: output channels of each 1D convolution over the rows. Valid padding: each
    convolution removes kernel_size - 1 rows. The remaining rows are flattened and a linear layer
    maps them to the increment."""
    name = "cnn"
    hyperparameters_ranges = {**DLModel.hyperparameters_ranges,
                              "channels": {"type": "categorical", "choices": [[16], [32], [16, 16], [32, 32]]},
                              "kernel_size": {"type": "categorical", "choices": [3, 5]},
                              "activation": {"type": "categorical", "choices": ["relu", "gelu", "silu"]},
                              "normalization": {"type": "categorical", "choices": [None, "layer", "batch"]}}

    def __init__(self, input_size, output_size, Nx=9, Ni=0, channels=(32, 32), kernel_size=3, normalization=None,
                 activation="relu", dropout=0.0, input_normalization=None, **training):
        rows = max(Nx, Ni) + 1 - len(channels) * (kernel_size - 1)
        if rows < 1:
            raise ValueError(f"A window of {max(Nx, Ni) + 1} rows is too short for {len(channels)} "
                             f"convolutions of kernel_size {kernel_size}; increase Nx or Ni")
        layers, size = [Normalization(input_normalization, input_size)], input_size
        for width in channels:
            layers += block(Convolution(size, width, kernel_size), width, normalization, activation, dropout)
            size = width
        super().__init__(Increment(*layers, nn.Flatten(), nn.Linear(rows * size, output_size)), Nx, Ni, **training)


class Transformer(DLModel):
    """A linear layer embeds each row into a token of size hidden, a learned position embedding
    gives the row order, then torch's nn.TransformerEncoder; a linear layer maps the last token to
    the increment. hidden, heads, layers, feedforward (default 4 * hidden), dropout and activation
    ("relu" or "gelu") are those of nn.TransformerEncoderLayer."""
    name = "transformer"
    hyperparameters_ranges = {**DLModel.hyperparameters_ranges,
                              "hidden": {"type": "categorical", "choices": [16, 32, 64]},
                              "layers": {"type": "int", "low": 1, "high": 2},
                              "heads": {"type": "categorical", "choices": [2, 4]},
                              "activation": {"type": "categorical", "choices": ["relu", "gelu"]}}

    def __init__(self, input_size, output_size, Nx=9, Ni=0, hidden=64, layers=2, heads=4, feedforward=None,
                 activation="gelu", dropout=0.0, input_normalization=None, **training):
        layer = nn.TransformerEncoderLayer(hidden, heads, feedforward or 4 * hidden, dropout, activation,
                                           batch_first=True)
        network = Increment(Normalization(input_normalization, input_size), nn.Linear(input_size, hidden), Position(max(Nx, Ni) + 1, hidden),
                            nn.TransformerEncoder(layer, layers, enable_nested_tensor=False), LastRow(),
                            nn.Linear(hidden, output_size))
        super().__init__(network, Nx, Ni, **training)
