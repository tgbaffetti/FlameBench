import numpy as np
from tqdm import tqdm
from ..Model import Model


class ARX(Model):
    """Ridge autoregression of the increment: x(t+1) = x(t) + W [x(t-Nx..t), phi(t+1-Ni..t+1) - 1, 1].

    Fitting the increment makes the ridge penalty shrink toward the constant forecast, not toward
    zero. The penalty of each weight is alpha times the energy of its feature (the diagonal of
    the Gram matrix), as a ridge on standardized features: alpha is dimensionless, and latent
    columns of large scale and forcing columns of small scale are penalized alike. Only the last Nx + 1 state rows and Ni + 1 forcing rows enter, so the zero padding of a
    ForecasterDataset sample never becomes a feature.
    """
    name = "arx"
    deterministic = True
    hyperparameters_ranges = {"alpha": {"type": "float", "low": 1e-8, "high": 1.0, "log": True}}
    dataset_ranges = {"horizon": 1}  # The closed-form fit is one-step.

    def __init__(self, input_size=None, output_size=None, Nx=9, Ni=0, device="cpu", alpha=1e-4):
        # The sizes and device are part of the shared constructor; ARX infers the sizes from data.
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        self.Nx, self.Ni, self.alpha = Nx, Ni, alpha

    def features(self, states, forcing):
        states, forcing = np.asarray(states), np.asarray(forcing)
        return np.column_stack((states[:, -self.Nx - 1:].reshape(len(states), -1), forcing[:, -self.Ni - 1:],
                                np.ones(len(states))))

    def fit(self, training, validation=None, **kwargs):
        gram = rhs = None
        for batch in tqdm(training, desc=f"{type(self).__name__} fit"):
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
        penalty = np.diag(np.diag(gram) * self.penalties(len(gram)))
        self.weights = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
        return self

    def penalties(self, columns):
        """Dimensionless ridge weight of each feature, times its Gram energy in fit; the
        constant is not penalized."""
        values = np.full(columns, float(self.alpha))
        values[-1] = 0.0
        return values

    def predict(self, states, forcing):
        return (states[:, -1] + self.features(states, forcing) @ self.weights).astype(np.float32)


class NARX(ARX):
    """Control-affine NARX: the ARX increment fit plus state x forcing bilinear couplings.

    x(t+1) = x(t) + W [x(t-Nx..t), phi - 1, (x kron (phi - 1)), 1]: linear in the state for
    fixed phi, the control-affine structure of the forced-ROM literature (polynomial state
    terms diverged in rollouts on the cell benchmark). The bilinear gain scales with the
    forcing amplitude, so a model stable on the training amplitudes can diverge beyond them;
    cross_alpha (default: alpha) penalizes only the bilinear block, energy-scaled like alpha,
    trading forcing response for rollout stability.
    """
    name = "narx"
    # The 2026-09-23 smoke runs needed alpha=30, cross_alpha=3e4 to stay finite over 2000-step
    # rollouts (amplitude fragility of the bilinear term), so the search must reach well past that.
    hyperparameters_ranges = {"alpha": {"type": "float", "low": 1e-8, "high": 1e3, "log": True},
                              "cross_alpha": {"type": "float", "low": 1e-8, "high": 1e6, "log": True}}
    dataset_ranges = {"horizon": 1}  # The closed-form fit is one-step.

    def __init__(self, input_size=None, output_size=None, Nx=9, Ni=0, device="cpu", alpha=1e-4, cross_alpha=None):
        super().__init__(input_size, output_size, Nx, Ni, device, alpha)
        if cross_alpha is not None and cross_alpha < 0:
            raise ValueError("cross_alpha must be nonnegative")
        self.cross_alpha = cross_alpha

    def features(self, states, forcing):
        states, forcing = np.asarray(states), np.asarray(forcing)
        recent_states = states[:, -self.Nx - 1:].reshape(len(states), -1)
        recent_forcing = forcing[:, -self.Ni - 1:]
        self.linear_width = recent_states.shape[1] + recent_forcing.shape[1]
        cross = (recent_states[:, :, None] * recent_forcing[:, None, :]).reshape(len(states), -1)
        return np.column_stack((recent_states, recent_forcing, cross, np.ones(len(states))))

    def penalties(self, columns):
        values = super().penalties(columns)
        if self.cross_alpha is not None:
            # Layout: [states | forcing | cross | 1]; linear_width was set by features in fit.
            values[self.linear_width:-1] = float(self.cross_alpha)
        return values


class Constant(Model):
    """Repeats the last state: with the identity compressor, the initial field forever."""
    name = "constant"
    dataset_ranges = {"horizon": 1}
    trainable = False
    deterministic = True

    def __init__(self, input_size=None, output_size=None, Nx=9, Ni=0, device="cpu"):
        self.Nx, self.Ni = Nx, Ni

    def fit(self, training, validation=None, **kwargs):
        return self

    def predict(self, states, forcing):
        return np.asarray(states[:, -1]).copy()
