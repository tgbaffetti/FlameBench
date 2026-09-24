"""Fourier neural operator FNO-2D (Li et al., ICLR 2021, arXiv:2010.08895) on the image grid.

As the paper's Navier-Stokes FNO-2D: the last Nx + 1 states are input channels, together with
the (x, z) coordinates; a pointwise linear lifting P; Fourier layers
v <- activation(K v + W v), where K multiplies the lowest modes[0] x modes[1] frequencies of
the 2-D FFT by learned complex weights and W is pointwise linear (no activation after the last
layer); a pointwise projection Q to the fields. The output is the next state itself (absolute,
as in the paper), trained recurrently over the dataset horizon with rollout_weight, as the paper
unrolls its Navier-Stokes model in training.

Configuration: layers Fourier layers, all of `width` channels (the lifting maps the inputs to
width), as in the paper; no layer changes the grid size. modes (rows, columns) and the
activation are shared by all layers; projection gives one width per hidden layer of Q. The
defaults are the paper's: four layers of width 32 (20 in its NS runs), modes 12, projection
(128,), GELU, coordinates on, padding 9 (the official code's value for non-periodic problems).

Adaptations to this dataset, none of them architectural:
  - the Ni + 1 forcing values enter as extra input channels, constant over the grid;
  - the identity-compressor latent (fields x cells) is scattered into the ImageGrid image, zero
    outside the mask, and the output is read back at the cells only;
  - the domain is not periodic, so after lifting it is zero-padded by `padding` pixels on the
    far side of each axis and cropped before the projection, as the official code does for
    non-periodic problems (Darcy flow).
The body (lifting, padding, Fourier layers, crop, projection) is one nn.Sequential, so printing
the network shows each layer in order.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .DLModel import DLModel
from .networks import ACTIVATIONS


class SpectralConv2d(nn.Module):
    """K: FFT, keep the lowest modes (both signs along the first axis), multiply by complex
    weights mixing the channels, inverse FFT."""

    def __init__(self, before, after, modes):
        super().__init__()
        self.after, self.modes = after, tuple(modes)
        scale = 1 / (before * after)
        shape = (before, after, *self.modes)
        self.weights1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    def forward(self, x):
        batch, _, height, width = x.shape
        m1, m2 = self.modes
        spectrum = torch.fft.rfft2(x)
        out = torch.zeros(batch, self.after, height, width // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", spectrum[:, :, :m1, :m2], self.weights1)
        out[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", spectrum[:, :, -m1:, :m2], self.weights2)
        return torch.fft.irfft2(out, s=(height, width))

    def extra_repr(self):
        return f"{self.weights1.shape[0]}, {self.after}, modes={self.modes}"


class FourierLayer(nn.Module):
    """K v + W v: spectral convolution plus pointwise (1 x 1) convolution."""

    def __init__(self, before, after, modes):
        super().__init__()
        self.spectral = SpectralConv2d(before, after, modes)
        self.pointwise = nn.Conv2d(before, after, 1)

    def forward(self, x):
        return self.spectral(x) + self.pointwise(x)


class Pad(nn.Module):
    """Zero padding on the far side of both axes (the domain is not periodic)."""

    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, [0, self.padding, 0, self.padding])

    def extra_repr(self):
        return f"padding={self.padding}"


class Crop(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.height, self.width = height, width

    def forward(self, x):
        return x[..., :self.height, :self.width]

    def extra_repr(self):
        return f"{self.height}x{self.width}"


class FNONetwork(nn.Module):
    def __init__(self, rank, Nx, Ni, grid, layers, width, modes, projection, padding, activation, coordinates):
        super().__init__()
        with np.load(grid) as mapping:
            rows, columns, mask = mapping["rows"], mapping["columns"], mapping["mask"]
            x, z = mapping["x"].astype(np.float32), mapping["z"].astype(np.float32)
        cells = len(rows)
        fields = rank // cells
        if fields < 1 or fields * cells != rank:
            raise ValueError(f"rank {rank} incompatible with the {cells} cells of the grid")
        if layers < 1 or width < 1 or min(projection, default=1) < 1:
            raise ValueError("Require layers >= 1, width >= 1 and positive projection widths")
        if padding < 0 or activation not in ACTIVATIONS:
            raise ValueError("Require padding >= 0 and an activation of networks.ACTIVATIONS")
        height, image_width = mask.shape
        if len(modes) != 2 or not (1 <= modes[0] <= (height + padding) // 2
                                   and 1 <= modes[1] <= (image_width + padding) // 2 + 1):
            raise ValueError(f"modes {modes} do not fit the padded {height + padding}x{image_width + padding} grid")
        self.Nx, self.Ni, self.fields, self.cells, self.shape = Nx, Ni, fields, cells, (height, image_width)
        self.register_buffer("rows", torch.as_tensor(rows, dtype=torch.long))
        self.register_buffer("columns", torch.as_tensor(columns, dtype=torch.long))
        grid_coordinates = None
        if coordinates:  # Physical pixel coordinates, each axis normalized to [0, 1].
            span = lambda v: (v - v.min()) / (v.max() - v.min() if v.max() > v.min() else 1)
            grid_z, grid_x = np.meshgrid(span(z), span(x), indexing="ij")
            grid_coordinates = torch.as_tensor(np.stack((grid_x, grid_z))[None])
        self.register_buffer("coordinates", grid_coordinates)
        inputs = (Nx + 1) * fields + (Ni + 1) + (2 if coordinates else 0)
        body = [nn.Conv2d(inputs, width, 1), Pad(padding)]
        for _ in range(layers):
            body += [FourierLayer(width, width, modes), ACTIVATIONS[activation]()]
        body = body[:-1] + [Crop(height, image_width)]  # No activation after the last Fourier layer.
        sizes = (width, *projection)
        for before, after in zip(sizes[:-1], sizes[1:]):
            body += [nn.Conv2d(before, after, 1), ACTIVATIONS[activation]()]
        self.body = nn.Sequential(*body, nn.Conv2d(sizes[-1], fields, 1))

    def forward(self, states, forcing):
        batch = states.shape[0]
        height, width = self.shape
        history = states[:, -self.Nx - 1:].reshape(batch, (self.Nx + 1) * self.fields, self.cells)
        image = history.new_zeros(batch, history.shape[1], height, width)
        image[:, :, self.rows, self.columns] = history
        inputs = [image, forcing[:, -self.Ni - 1:, None, None].expand(-1, -1, height, width)]
        if self.coordinates is not None:
            inputs.append(self.coordinates.expand(batch, -1, -1, -1))
        out = self.body(torch.cat(inputs, dim=1))
        return out[:, :, self.rows, self.columns].reshape(batch, self.fields * self.cells)


class FNO(DLModel):
    """grid: path to the ImageGrid .npz (the field count is the latent rank over its cell count);
    layers: number of Fourier layers; width: their channels; modes: kept frequencies along the
    rows (z) and the columns (x); projection: one width per hidden layer of Q; padding: zero
    padding of the non-periodic domain; activation: a networks.ACTIVATIONS key; coordinates:
    append the (x, z) channels. training: DLModel keywords (device, lr, optimizer, ...)."""
    name = "fno"
    hyperparameters_ranges = {**{k: v for k, v in DLModel.hyperparameters_ranges.items() if k != "dropout"},
                              "layers": {"type": "int", "low": 2, "high": 4},
                              "width": {"type": "categorical", "choices": [20, 32, 64]},
                              "modes": {"type": "categorical", "choices": [[8, 8], [12, 12], [16, 16]]}}
    # Backpropagation through the unrolled full-field model: the paper unrolls 10 steps (NS);
    # longer horizons multiply activation memory.
    dataset_ranges = {"horizon": {"type": "categorical", "choices": [1, 5, 10]}}

    def __init__(self, input_size=None, output_size=None, Nx=9, Ni=0, grid=None, layers=4, width=32,
                 modes=(12, 12), projection=(128,), padding=9, activation="gelu", coordinates=True, **training):
        if grid is None:
            raise ValueError("FNO requires the grid mapping (grid: ImageGrid .npz)")
        network = FNONetwork(output_size, Nx, Ni, grid, layers, width, tuple(modes), tuple(projection),
                             padding, activation, coordinates)
        super().__init__(network, Nx, Ni, **training)
