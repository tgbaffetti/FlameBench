"""Streaming feature statistics fitted strictly on training snapshots."""
import numpy as np


class FeatureScaler:
    def __init__(self, mask=None):
        self.mask = mask

    def fit(self, batches):
        count, mean, m2 = 0, None, None
        for x in batches:
            x = np.asarray(x, dtype=np.float64)
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite training data")
            if x.ndim == 4:
                if self.mask is None:
                    raise ValueError("Image normalization requires a valid-cell mask")
                x = x[..., self.mask]
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
        shape = (len(self.mean),) + (1,) * (x.ndim - 2)
        result = (x - self.mean.reshape(shape)) / self.scale.reshape(shape)
        if self.mask is not None:
            result[..., ~self.mask] = 0
        return result

    def inverse(self, x):
        shape = (len(self.mean),) + (1,) * (x.ndim - 2)
        result = x * self.scale.reshape(shape) + self.mean.reshape(shape)
        if self.mask is not None:
            result[..., ~self.mask] = 0
        return result
