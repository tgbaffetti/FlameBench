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
        penalty = np.diag(self.penalties(len(gram)))
        self.weights = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
        return self

    def penalties(self, columns):
        """Per-feature ridge weights; the trailing bias column is never penalized."""
        values = np.full(columns, float(self.alpha))
        values[-1] = 0.0
        return values

    def predict(self, history, forcing):
        return (self.features(history, forcing) @ self.weights).astype(np.float32)


class NARX(ARX):
    """Control-affine NARX: linear terms plus state x forcing bilinear couplings.

    z' = A z + b phi + (z x phi) couplings — the control-affine structure of the
    forced-ROM literature. Linear in the state for fixed phi, so rollouts stay
    bounded far more readily than with polynomial state terms (a squares variant
    diverged in validation rollout).

    cross_alpha penalizes the bilinear block separately (default: same as alpha).
    The bilinear gain scales with the forcing amplitude, so a model fitted on the
    training sweeps (phi' up to 0.4) can be stable there and still diverge on the
    0.5-amplitude test cases; validation on the sweeps cannot see this coming.
    """
    name = "narx"
    hyperparams = dict(ARX.hyperparams,
                       cross_alpha={"value": None, "type": "float", "low": 1e-4, "high": 1e4, "log": True})

    def __init__(self, alpha=1e-4, cross_alpha=None, **kwargs):
        super().__init__(alpha=alpha, **kwargs)
        if cross_alpha is not None and cross_alpha < 0:
            raise ValueError("cross_alpha must be nonnegative")
        self.cross_alpha = cross_alpha

    @staticmethod
    def features(history, forcing):
        state = history.reshape(len(history), -1)
        phi = forcing - 1
        cross = (state[:, :, None] * phi[:, None, :]).reshape(len(state), -1)
        return np.column_stack((state, phi, cross, np.ones(len(state))))

    def penalties(self, columns):
        values = super().penalties(columns)
        if self.cross_alpha is None:
            return values
        # Feature layout: [state | phi | cross | bias]; the cross block is what
        # the separate penalty targets, so it is located from the two edges.
        state_and_phi = self.state_width + self.forcing_width
        values[state_and_phi:-1] = float(self.cross_alpha)
        return values

    def fit(self, training, validation=None, **kwargs):
        for history, forcing, _ in training:
            self.state_width = int(np.asarray(history).reshape(len(history), -1).shape[1])
            self.forcing_width = int(np.asarray(forcing).shape[1])
            break
        else:
            raise ValueError("Empty training loader")
        return super().fit(training, validation, **kwargs)


class Constant(Model):
    name = "constant"

    def __init__(self, **kwargs):
        pass

    def fit(self, training, validation=None, **kwargs):
        return self

    def predict(self, history, forcing):
        return np.asarray(history[:, -1]).copy()
