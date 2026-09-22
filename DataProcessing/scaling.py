"""Streaming per-field statistics of training images, over valid pixels only."""
import numpy as np
from tqdm.auto import tqdm


class FeatureScaler:
    def __init__(self, mask):
        self.mask = mask

    def fit(self, batches):
        count, mean, m2 = 0, None, None
        for x in tqdm(batches, desc="Scaler fit"):
            x = np.asarray(x, dtype=np.float64)
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite training data")
            x = x[..., self.mask]  # (batch, field, valid pixels)
            n = x.shape[0] * x.shape[2]
            mu = x.mean(axis=(0, 2))
            ss = ((x - mu[None, :, None]) ** 2).sum(axis=(0, 2))
            if mean is None:
                count, mean, m2 = n, mu, ss
            else:
                delta = mu - mean
                m2 += ss + delta ** 2 * count * n / (count + n)
                mean += delta * n / (count + n)
                count += n
        if mean is None:
            raise ValueError("No training snapshots")
        std = np.sqrt(m2 / count)
        self.mean = mean.astype(np.float32)
        self.scale = np.where(std > 0, std, 1).astype(np.float32)
        return self

    def transform(self, x):
        result = (x - self.mean[:, None, None]) / self.scale[:, None, None]
        result[..., ~self.mask] = 0
        return result

    def inverse(self, x):
        result = x * self.scale[:, None, None] + self.mean[:, None, None]
        result[..., ~self.mask] = 0
        return result
