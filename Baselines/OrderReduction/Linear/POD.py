import numpy as np
from sklearn.decomposition import IncrementalPCA
from ..Compressor import Compressor


class POD(Compressor):
    """Centered Euclidean POD approximated by incremental truncated SVD.

    Memory scales with (batch_size + rank) * fields * cells, not trajectory length.
    This is not a volume-weighted POD and is not an exact batch SVD.
    """
    name = "pod"

    def __init__(self, rank=16, batch_size=64):
        if rank < 1 or batch_size < rank:
            raise ValueError("Require batch_size >= rank >= 1")
        self.rank, self.batch_size = rank, batch_size

    def fit(self, dataset, scaler, **kwargs):
        self.shape = tuple(dataset.field_shape)
        self.pca = IncrementalPCA(n_components=self.rank)
        pending = None
        # Keep one batch pending so the final short batch is included, never dropped.
        for x in dataset.snapshot_batches(self.batch_size):
            x = scaler.transform(x).reshape(len(x), -1)
            pending = x if pending is None else np.concatenate((pending, x))
            if len(pending) >= self.batch_size + self.rank:
                self.pca.partial_fit(pending[:self.batch_size])
                pending = pending[self.batch_size:]
        if pending is None or len(pending) < self.rank:
            raise ValueError("Too few training snapshots for POD rank")
        self.pca.partial_fit(pending)
        return self

    def encode(self, frames):
        return self.pca.transform(frames.reshape(len(frames), -1)).astype(np.float32)

    def decode(self, latent):
        return self.pca.inverse_transform(latent).reshape(len(latent), *self.shape).astype(np.float32)
