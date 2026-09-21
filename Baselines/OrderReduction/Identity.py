"""No-op compressor: flattens snapshots so field-level models reuse the latent pipeline."""
import numpy as np
from .Compressor import Compressor


class Identity(Compressor):
    name = "identity"

    def __init__(self, **kwargs):
        pass

    def fit(self, dataset, scaler, **kwargs):
        self.shape = tuple(dataset.field_shape)
        self.rank = int(np.prod(self.shape))
        return self

    def encode(self, frames):
        return np.asarray(frames, dtype=np.float32).reshape(len(frames), -1)

    def decode(self, latent):
        return np.asarray(latent, dtype=np.float32).reshape(len(latent), *self.shape)
