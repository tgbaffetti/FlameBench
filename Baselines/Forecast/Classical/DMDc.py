"""DMD with control (Proctor, Brunton & Kutz 2016) on POD coordinates: ARX with Nx = Ni = 0.

x(t+1) = x(t) + W [x(t), phi(t+1) - 1, 1], i.e. x(t+1) = A x(t) + B u + c: the DMDc operator
pair (A, B) fitted by least squares in the POD subspace, as ARX fits it. Differences from the
textbook algorithm, all inherited from ARX: an affine term c (the POD is centred on the training
mean, not on the steady state), the dimensionless ridge alpha (tuned; 1e-8 is plain least
squares), and truncation of the state basis only (the POD rank), not of the SVD of the
augmented [X; U] data. A separate name keeps its runs apart from ARX in W&B and the results.
"""
from .ARX import ARX


class DMDc(ARX):
    name = "dmdc"
    dataset_ranges = {"Nx": 0, "Ni": 0, "horizon": 1}

    def __init__(self, input_size=None, output_size=None, Nx=0, Ni=0, device="cpu", alpha=1e-4):
        if Nx != 0 or Ni != 0:
            raise ValueError("DMDc is Markovian: use Nx=0 and Ni=0 (ARX adds delays)")
        super().__init__(input_size, output_size, Nx, Ni, device, alpha)
