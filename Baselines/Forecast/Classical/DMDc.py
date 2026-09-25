"""DMD with control (Proctor, Brunton & Kutz 2016) on POD coordinates: ARX with Nx = Ni = 0.

x(t+1) = x(t) + W [x(t), phi(t+1) - 1, 1], i.e. x(t+1) = A x(t) + B u + c: the DMDc operator
pair (A, B) fitted by least squares in the POD subspace, as ARX fits it. Differences from the
textbook algorithm, all inherited from ARX: an affine term c (the POD is centred on the training
mean, not on the steady state), the dimensionless ridge alpha (tuned; 1e-8 is plain least
squares), and truncation of the state basis only (the POD rank), not of the SVD of the
augmented [X; U] data. A separate name keeps its runs apart from ARX in W&B and the results.

ExactDMDc is the algorithm of the paper itself (Section 3.2, unknown B), for the identity
compressor: it reads the full scaled field and does its own SVDs, with no POD before it.
"""
import numpy as np
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from ..Model import Model
from .ARX import ARX


class DMDc(ARX):
    name = "dmdc"
    dataset_ranges = {"Nx": 0, "Ni": 0, "horizon": 1}

    def __init__(self, input_size=None, output_size=None, Nx=0, Ni=0, device="cpu", alpha=1e-4):
        if Nx != 0 or Ni != 0:
            raise ValueError("DMDc is Markovian: use Nx=0 and Ni=0 (ARX adds delays)")
        super().__init__(input_size, output_size, Nx, Ni, device, alpha)


class ExactDMDc(Model):
    """DMDc as Proctor, Brunton & Kutz (2016), Section 3.2, on the full field.

    With snapshots as columns, X the states, X' the next states and U the forcing
    u = phi(t+1) - 1:
      1. truncated SVD of Omega = [X; U] = U~ S~ V~^T at rank p, with U~ = [U~1; U~2] split into
         its state and forcing rows;
      2. truncated SVD of X' = U^ S^ V^^T at rank r;
      3. A~ = U^^T X' V~ S~^-1 U~1^T U^ and B~ = U^^T X' V~ S~^-1 U~2^T;
      4. x~(t+1) = A~ x~(t) + B~ u, with x~ = U^^T x and x = U^ x~.
    No centering, regularization or affine term, as in the paper: the mean field is carried by
    modes with eigenvalue near 1. The fields arrive scaled per field like every model's input;
    unscaled, the heat release would hold nearly all the energy of the SVDs. Both SVDs are
    randomized (fixed seed, as POD), so the fit is deterministic.

    Memory: the training states and next states are held in RAM as float32, twice the size of the
    encoded training windows.
    """
    name = "dmdc_exact"
    deterministic = True
    # p >= r: the paper truncates the input space at least as far as the output space.
    hyperparameters_ranges = {"p": {"type": "int", "low": 8, "high": 256, "log": True},
                              "r": {"type": "int", "low": 8, "high": 128, "log": True}}
    dataset_ranges = {"Nx": 0, "Ni": 0, "horizon": 1}

    def __init__(self, input_size=None, output_size=None, Nx=0, Ni=0, device="cpu", p=64, r=32):
        if Nx != 0 or Ni != 0:
            raise ValueError("DMDc is Markovian: use Nx=0 and Ni=0")
        if not 1 <= r <= p:
            raise ValueError("Require 1 <= r <= p")
        self.Nx, self.Ni, self.p, self.r = Nx, Ni, p, r

    def fit(self, training, validation=None, **kwargs):
        size, row = len(training.dataset), 0
        omega = following = None
        for batch in tqdm(training, desc=f"{type(self).__name__} fit"):
            states, target = np.asarray(batch["states"]), np.asarray(batch["target"])
            if target.shape[1] != 1:
                raise ValueError("DMDc fits one-step targets; use horizon=1")
            if omega is None:
                n = states.shape[-1]
                omega = np.empty((size, n + 1), dtype=np.float32)
                following = np.empty((size, n), dtype=np.float32)
            stop = row + len(states)
            omega[row:stop, :n] = states[:, -1]
            omega[row:stop, n] = np.asarray(batch["forcing"])[:, states.shape[1] - 1]
            following[row:stop] = target[:, 0]
            row = stop
        if omega is None:
            raise ValueError("Empty training loader")
        omega, following = omega[:row], following[:row]
        # Rows are snapshots here, so each SVD is the transpose of the paper's: omega = V~ S~ U~^T.
        right, values, left = randomized_svd(omega, min(self.p, *omega.shape), n_oversamples=20, n_iter=7,
                                             random_state=42)
        self.basis = randomized_svd(following, min(self.r, *following.shape), n_oversamples=20, n_iter=7,
                                    random_state=42)[2].T  # U^
        core = ((following @ self.basis).T.astype(np.float64) @ right) / values  # U^^T X' V~ S~^-1
        self.A = core @ (left[:, :n] @ self.basis)
        self.B = core @ left[:, n:]
        return self

    def predict(self, states, forcing):
        reduced = np.asarray(states, dtype=np.float32)[:, -1] @ self.basis
        following = reduced @ self.A.T + np.asarray(forcing, dtype=np.float64)[:, -1:] @ self.B.T
        return (following.astype(np.float32) @ self.basis.T).astype(np.float32)
