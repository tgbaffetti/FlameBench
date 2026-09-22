"""Centered randomized POD using the legacy PODReducer algorithm."""
import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from DataProcessing.loading import make_loader
from ..Compressor import Compressor


class POD(Compressor):
    """Legacy randomized SVD, fitted on the full training matrix in memory.

    Each image is first flattened to the vector of its valid pixels, in original cell order.
    Frames arrive scaled from the dataset, with the scaler shared by all compressors. rank is
    fixed, so there is nothing to tune. sklearn preserves the legacy implementation; torch
    optionally fits on a GPU with the same oversampling and iteration count. The randomized
    bases can differ. Both backends store a NumPy basis and encode/decode on CPU.
    """
    name = "pod"

    def __init__(self, rank=16, batch_size=64, device=None, backend="sklearn"):
        if rank < 1 or batch_size < 1:
            raise ValueError("Require rank >= 1 and batch_size >= 1")
        self.rank, self.batch_size = rank, batch_size
        if backend not in {"sklearn", "torch"}:
            raise ValueError("POD backend must be sklearn or torch")
        self.backend, self.device = backend, device or "cpu"

    def vectors(self, frames):
        rows, columns = self.grid_indices
        return frames[..., rows, columns].reshape(len(frames), -1)

    def fit(self, dataset, loader_options=None, **kwargs):
        self.shape = tuple(dataset.field_shape)
        self.grid_indices = dataset.grid_indices
        loader = make_loader(dataset, self.batch_size, **(loader_options or {}))
        snapshots = np.concatenate([self.vectors(frames.numpy())
                                    for frames in tqdm(loader, desc="POD snapshots")])
        del loader  # Stop persistent workers before allocating the centered matrix and SVD workspace.
        X = snapshots.T
        self.mean = X.mean(axis=1, keepdims=True)
        X_c = X - self.mean
        self.rank = min(self.rank, *X_c.shape)
        with tqdm(total=1, desc=f"POD SVD ({self.backend}; no iteration callback)") as progress:
            if self.backend == "sklearn":
                self.U_r, self.singular_values, _ = randomized_svd(
                    X_c, n_components=self.rank, n_oversamples=20, n_iter=7, random_state=42)
            else:
                device = torch.device(self.device)
                devices = [device.index if device.index is not None else torch.cuda.current_device()] \
                    if device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices), torch.no_grad():
                    torch.random.default_generator.manual_seed(42)
                    if devices:
                        with torch.cuda.device(devices[0]):
                            torch.cuda.manual_seed(42)
                    matrix = torch.as_tensor(X_c, device=device)
                    U, S, _ = torch.svd_lowrank(matrix, q=min(self.rank + 20, *X_c.shape), niter=7)
                    self.U_r = U[:, :self.rank].cpu().numpy().copy()
                    self.singular_values = S[:self.rank].cpu().numpy().copy()
            progress.update()
        return self

    def encode(self, frames):
        return ((self.vectors(frames) - self.mean.T) @ self.U_r).astype(np.float32)

    def decode(self, latent):
        vectors = (latent @ self.U_r.T + self.mean.T).astype(np.float32)
        rows, columns = self.grid_indices
        images = np.zeros((len(latent), *self.shape), dtype=np.float32)
        images[..., rows, columns] = vectors.reshape(len(latent), self.shape[0], len(rows))
        return images
