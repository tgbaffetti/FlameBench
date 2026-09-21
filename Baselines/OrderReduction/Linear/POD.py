"""Centered randomized POD using the legacy PODReducer algorithm."""
import numpy as np
from sklearn.utils.extmath import randomized_svd
from ..Compressor import Compressor


class POD(Compressor):
    """Legacy randomized SVD, fitted on the full training matrix in memory.

    Image inputs are gathered into original cell order before decomposition.
    Scaling remains external and shared with the other benchmark compressors.
    """
    name = "pod"

    def __init__(self, rank=16, batch_size=64):
        if rank < 1 or batch_size < 1:
            raise ValueError("Require rank >= 1 and batch_size >= 1")
        self.rank, self.batch_size = rank, batch_size

    def vectors(self, frames):
        if self.grid_indices is not None:
            rows, columns = self.grid_indices
            frames = frames[..., rows, columns]
        return frames.reshape(len(frames), -1)

    def fit(self, dataset, scaler, **kwargs):
        self.shape = tuple(dataset.field_shape)
        self.grid_indices = dataset.grid_indices
        snapshots = np.concatenate([
            self.vectors(scaler.transform(frames))
            for frames in dataset.snapshot_batches(self.batch_size)
        ])
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
        if self.grid_indices is None:
            return vectors.reshape(len(latent), *self.shape)
        rows, columns = self.grid_indices
        images = np.zeros((len(latent), *self.shape), dtype=np.float32)
        images[..., rows, columns] = vectors.reshape(len(latent), self.shape[0], len(rows))
        return images
