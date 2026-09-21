import numpy as np
from ..Model import Model


class ARX(Model):
    """Ridge autoregression with independent state and prescribed inlet-forcing histories."""
    name = "arx"
    hyperparams = {"alpha": {"value": 1e-4, "type": "float", "low": 1e-8, "high": 1.0, "log": True}}

    def __init__(self, alpha=1e-4, **kwargs):
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        self.alpha = alpha

    @staticmethod
    def features(history, forcing):
        return np.column_stack((history.reshape(len(history), -1), forcing - 1,
                                np.ones(len(history))))

    def fit(self, training, validation=None, **kwargs):
        gram = rhs = None
        for history, forcing, target in training:
            x = self.features(np.asarray(history), np.asarray(forcing)).astype(np.float64)
            y = np.asarray(target, dtype=np.float64)
            if gram is None:
                gram, rhs = x.T @ x, x.T @ y
            else:
                gram += x.T @ x
                rhs += x.T @ y
        if gram is None:
            raise ValueError("Empty training loader")
        penalty = np.eye(len(gram)) * self.alpha
        penalty[-1, -1] = 0
        self.weights = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
        return self

    def predict(self, history, forcing):
        return (self.features(history, forcing) @ self.weights).astype(np.float32)


class NARX(ARX):
    """Polynomial NARX: linear terms, squares, and state x forcing bilinear couplings.

    The bilinear block is the control-affine structure (z' = A z + B(z) phi) of the
    forced-ROM literature; ridge fit identical to ARX.
    """
    name = "narx"

    @staticmethod
    def features(history, forcing):
        state = history.reshape(len(history), -1)
        phi = forcing - 1
        linear = np.column_stack((state, phi))
        cross = (state[:, :, None] * phi[:, None, :]).reshape(len(state), -1)
        return np.column_stack((linear, np.square(linear), cross, np.ones(len(state))))


class Constant(Model):
    name = "constant"

    def __init__(self, **kwargs):
        pass

    def fit(self, training, validation=None, **kwargs):
        return self

    def predict(self, history, forcing):
        return np.asarray(history[:, -1]).copy()
