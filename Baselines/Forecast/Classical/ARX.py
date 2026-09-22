import numpy as np
from ..Model import Model


class ARX(Model):
    """Ridge autoregression of the increment: x(t+1) = x(t) + W [x(t-Nx..t), phi(t+1-Ni..t+1) - 1, 1].

    Fitting the increment makes the ridge penalty shrink toward the constant forecast, not toward
    zero. Only the last Nx + 1 state rows and Ni + 1 forcing rows enter, so the zero padding of a
    ForecasterDataset sample never becomes a feature.
    """
    name = "arx"
    hyperparams = {"alpha": {"type": "float", "low": 1e-8, "high": 1.0, "log": True}}

    def __init__(self, rank=None, Nx=9, Ni=0, device="cpu", alpha=1e-4):
        # rank and device are part of the shared constructor; ARX infers the rank from data.
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        self.Nx, self.Ni, self.alpha = Nx, Ni, alpha

    def features(self, states, forcing):
        states, forcing = np.asarray(states), np.asarray(forcing)
        return np.column_stack((states[:, -self.Nx - 1:].reshape(len(states), -1), forcing[:, -self.Ni - 1:],
                                np.ones(len(states))))

    def fit(self, training, validation=None, **kwargs):
        gram = rhs = None
        for batch in training:
            states, target = np.asarray(batch["states"]), np.asarray(batch["target"])
            if target.shape[1] != 1:
                raise ValueError("ARX fits one-step targets; use horizon=1")
            x = self.features(states, batch["forcing"][:, :states.shape[1]]).astype(np.float64)
            y = target[:, 0].astype(np.float64) - states[:, -1].astype(np.float64)
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

    def predict(self, states, forcing):
        return (states[:, -1] + self.features(states, forcing) @ self.weights).astype(np.float32)


class Constant(Model):
    name = "constant"

    def __init__(self, rank=None, Nx=9, Ni=0, device="cpu"):
        pass

    def fit(self, training, validation=None, **kwargs):
        return self

    def predict(self, states, forcing):
        return np.asarray(states[:, -1]).copy()
