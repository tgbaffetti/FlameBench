"""Centered randomized POD using the legacy PODReducer algorithm."""
import numpy as np
from sklearn.utils.extmath import randomized_svd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from ..Compressor import Compressor


class POD(Compressor):
    """Legacy randomized SVD, fitted on the full training matrix in memory.

    Each image is first flattened to the vector of its valid pixels, in original cell order.
    Frames arrive scaled from the dataset, with the scaler shared by all compressors. rank is
    fixed, so there is nothing to tune. device is accepted for a uniform build and not used.
    """
    name = "pod"

    def __init__(self, rank=16, batch_size=64, device=None):
        if rank < 1 or batch_size < 1:
            raise ValueError("Require rank >= 1 and batch_size >= 1")
        self.rank, self.batch_size = rank, batch_size

    def vectors(self, frames):
        rows, columns = self.grid_indices
        return frames[..., rows, columns].reshape(len(frames), -1)

    def fit(self, dataset, **kwargs):
        self.shape = tuple(dataset.field_shape)
        self.grid_indices = dataset.grid_indices
        snapshots = np.concatenate([self.vectors(frames.numpy())
                                    for frames in tqdm(DataLoader(dataset, batch_size=self.batch_size), desc="POD snapshots")])
        X = snapshots.T
        self.mean = X.mean(axis=1, keepdims=True)
        X_c = X - self.mean
        self.rank = min(self.rank, *X_c.shape)
        self.U_r, self.singular_values, _ = randomized_svd(
            X_c, n_components=self.rank, n_oversamples=20, n_iter=7, random_state=42)
        return self

    def encode(self, frames):
        return ((self.vectors(frames) - self.mean.T) @ self.U_r).astype(np.float32)

    def decode(self, latent):
        vectors = (latent @ self.U_r.T + self.mean.T).astype(np.float32)
        rows, columns = self.grid_indices
        images = np.zeros((len(latent), *self.shape), dtype=np.float32)
        images[..., rows, columns] = vectors.reshape(len(latent), self.shape[0], len(rows))
        return images
