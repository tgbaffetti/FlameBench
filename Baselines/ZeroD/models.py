"""0-D flame response: a window of phi'(t) taps -> normalized heat release q'(t)/q0.

The combustion-community baseline (MLP flame response, PI-LSTM, dual-path FTF
models): no field state at all, the flame is a nonlinear filter of the forcing.
"""
import torch
from torch import nn


class ZeroDMLP(nn.Module):
    def __init__(self, window, hidden=128, layers=3, **kwargs):
        super().__init__()
        sizes = [window] + [hidden] * layers
        blocks = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            blocks += [nn.Linear(a, b), nn.GELU()]
        self.net = nn.Sequential(*blocks, nn.Linear(sizes[-1], 1))

    def forward(self, phi_window):
        return self.net(phi_window).squeeze(-1)


class ZeroDGRU(nn.Module):
    def __init__(self, window, hidden=64, layers=2, **kwargs):
        super().__init__()
        self.rnn = nn.GRU(1, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, phi_window):
        output, _ = self.rnn(phi_window.unsqueeze(-1))
        return self.head(output[:, -1]).squeeze(-1)
