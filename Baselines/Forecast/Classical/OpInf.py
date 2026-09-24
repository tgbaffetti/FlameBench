"""Operator Inference (Peherstorfer & Willcox 2016), with the regularization of McQuarrie, Huang
& Willcox 2021 (a single-injector combustion ROM).

On POD coordinates x, OpInf fits by least squares the operators that a Galerkin projection of a
quadratic PDE would give:

    dx/dt = c + A x + H (x kron x) + B u

Here in discrete time, on the increment of one data step:

    x(t+1) = x(t) + c + A x(t) + H (x(t) kron x(t)) + B u,    u = phi(t+1) - 1

This is continuous-time OpInf with forward-difference derivatives and explicit Euler at the
data step, with dt absorbed into the operators. H keeps only the r(r+1)/2 unique products
x_i x_j (i <= j). The model is Markovian (Nx = Ni = 0), and one-step and closed form like ARX.

Tikhonov regularization, as in McQuarrie et al.: alpha on c, A and B, quadratic_alpha on H, on
the plain operator entries (not scaled by the feature energy as ARX does). The least-squares
term is the mean over training samples, so the alphas do not depend on the number of windows.

The problem is solved by QR of the Tikhonov-augmented data matrix [D Y; Gamma 0], not by the
normal equations: those square the condition number of D (about 1e7 here, so about 1e14), and
lost most digits of the solution at large quadratic_alpha. The QR is streamed: rows are folded
into a triangular factor, so memory stays at (features + outputs)^2 like a Gram matrix.
"""
import numpy as np
import scipy.linalg
from tqdm import tqdm
from .ARX import ARX


def fold(triangle, rows):
    """Triangular factor R of [triangle; rows]: R^T R = triangle^T triangle + rows^T rows."""
    stacked = rows if triangle is None else np.vstack((triangle, rows))
    factor = scipy.linalg.qr(stacked, mode="r", check_finite=False)[0]
    return factor[:min(stacked.shape)]


class OpInf(ARX):
    name = "opinf"
    # Ranges from POD rank 32 on the sweeps (coefficient std up to ~70, so quadratic features
    # reach ~5e3): quadratic_alpha below ~1e6 diverged in 150-step rollouts; large values turn
    # the quadratic term off, leaving the linear model.
    hyperparameters_ranges = {"alpha": {"type": "float", "low": 1e-8, "high": 1e4, "log": True},
                              "quadratic_alpha": {"type": "float", "low": 1e3, "high": 1e12, "log": True}}
    dataset_ranges = {"Nx": 0, "Ni": 0, "horizon": 1}  # Markovian, closed-form one-step fit.

    def __init__(self, input_size=None, output_size=None, Nx=0, Ni=0, device="cpu", alpha=1e-2,
                 quadratic_alpha=None):
        if Nx != 0 or Ni != 0:
            raise ValueError("OpInf is Markovian: use Nx=0 and Ni=0")
        super().__init__(input_size, output_size, Nx, Ni, device, alpha)
        if quadratic_alpha is not None and quadratic_alpha < 0:
            raise ValueError("quadratic_alpha must be nonnegative")
        self.quadratic_alpha = alpha if quadratic_alpha is None else quadratic_alpha

    def features(self, states, forcing):
        """[x, u, unique x_i x_j, 1] rows; the constant column carries c."""
        x = np.asarray(states, dtype=np.float64)[:, -1]  # Products in float64: float32 loses digits.
        u = np.asarray(forcing, dtype=np.float64)[:, -1:]
        rows, columns = np.triu_indices(x.shape[1])
        self.linear_width = x.shape[1] + 1
        return np.column_stack((x, u, x[:, rows] * x[:, columns], np.ones(len(x))))

    def penalties(self, columns):
        """Layout [x | u | quadratic | 1]: alpha on A, B and c; quadratic_alpha on H."""
        values = np.full(columns, float(self.alpha))
        values[self.linear_width:-1] = float(self.quadratic_alpha)
        return values

    def fit(self, training, validation=None, **kwargs):
        """Minimize mean ||x(t+1) - x(t) - W f||^2 + sum_j penalty_j ||W_j||^2 by streamed QR."""
        triangle, buffer, buffered, samples = None, [], 0, 0
        for batch in tqdm(training, desc=f"{type(self).__name__} fit"):
            states, target = np.asarray(batch["states"]), np.asarray(batch["target"])
            if target.shape[1] != 1:
                raise ValueError("OpInf fits one-step targets; use horizon=1")
            x = self.features(states, batch["forcing"][:, :states.shape[1]]).astype(np.float64)
            y = target[:, 0].astype(np.float64) - states[:, -1].astype(np.float64)
            buffer.append(np.hstack((x, y)))
            buffered += len(x)
            samples += len(x)
            if buffered >= buffer[0].shape[1]:  # Fold once the rows fill a square block.
                triangle, buffer, buffered = fold(triangle, np.vstack(buffer)), [], 0
        if samples == 0:
            raise ValueError("Empty training loader")
        if buffer:
            triangle = fold(triangle, np.vstack(buffer))
        features = x.shape[1]
        penalties = self.penalties(features)
        regularization = np.zeros((features, triangle.shape[1]))
        regularization[:, :features] = np.diag(np.sqrt(samples * penalties))
        triangle = fold(triangle, regularization)
        r11, r12 = triangle[:features, :features], triangle[:features, features:]
        if (penalties > 0).all():
            # The penalty rows bound the smallest singular value of R11 away from zero.
            self.weights = scipy.linalg.solve_triangular(r11, r12, check_finite=False)
        else:
            # A zero penalty leaves R11 singular when features are dependent (or outnumber the
            # samples): minimum-norm solution, with numpy's rank cutoff. LAPACK's default (eps)
            # keeps rounding-level pivots and returned weights of norm ~1e10 instead of ~0.1.
            cutoff = np.finfo(np.float64).eps * max(samples, features)
            self.weights = scipy.linalg.lstsq(r11, r12, cond=cutoff, lapack_driver="gelsy",
                                              check_finite=False)[0]
        return self
