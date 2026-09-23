"""No compression: the latent of a frame is the vector of its valid pixels.

Used by the constant baseline, so that it repeats the exact initial field instead of its
reconstruction by some compressor. rank is the number of valid values (fields x cells).
"""
import numpy as np
from .Compressor import Compressor


class Identity(Compressor):
    name = "identity"
    deterministic = True

    def __init__(self, device=None):
        self.device = device or "cpu"

    def fit(self, dataset, **kwargs):
        self.shape = tuple(dataset.field_shape)
        self.grid_indices = dataset.grid_indices
        self.rank = self.shape[0] * len(self.grid_indices[0])
        return self

    def encode(self, frames):
        rows, columns = self.grid_indices
        return np.asarray(frames, dtype=np.float32)[..., rows, columns].reshape(len(frames), -1)

    def decode(self, latent):
        rows, columns = self.grid_indices
        images = np.zeros((len(latent), *self.shape), dtype=np.float32)
        images[..., rows, columns] = np.asarray(latent).reshape(len(latent), self.shape[0], len(rows))
        return images
