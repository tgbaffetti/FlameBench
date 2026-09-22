"""Physical-unit metrics; evaluation statistics never feed training."""
import numpy as np
from scipy.ndimage import gaussian_filter


class FieldMetrics:
    def __init__(self, fields):
        self.fields = list(fields)
        self.count = 0
        self.mean = np.zeros(len(fields), dtype=np.float64)
        self.m2 = np.zeros(len(fields), dtype=np.float64)
        self.sse = np.zeros(len(fields), dtype=np.float64)

    def update(self, predicted, reference):
        predicted, reference = np.asarray(predicted, dtype=np.float64), np.asarray(reference, dtype=np.float64)
        if predicted.shape != reference.shape or reference.ndim != 2:
            raise ValueError("Expected matching (field, cell) arrays")
        if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
            raise FloatingPointError("Nonfinite rollout/reference: metric undefined")
        n = reference.shape[1]
        mu = reference.mean(axis=1)
        delta = mu - self.mean
        self.m2 += np.square(reference-mu[:, None]).sum(axis=1) + delta**2*self.count*n/(self.count+n)
        self.mean += delta*n/(self.count+n)
        self.count += n
        self.sse += np.square(predicted-reference).sum(axis=1)

    def result(self):
        std = np.sqrt(self.m2/self.count)
        rmse = np.sqrt(self.sse/self.count)
        # Constant reference fields have undefined std-normalized error, not zero error.
        nrmse = [float(a/b) if b > 0 else None for a, b in zip(rmse, std)]
        return {"field_nrmse": dict(zip(self.fields, nrmse)),
                "field_rmse": dict(zip(self.fields, map(float, rmse))),
                "mean_nrmse": float(np.mean(nrmse)) if all(x is not None for x in nrmse) else None,
                "undefined_fields": [f for f, value in zip(self.fields, nrmse) if value is None]}


class FieldSSIM:
    """Mean structural similarity of each field over the frames of a rollout.

    SSIM of Wang et al. (2004): Gaussian window with sigma 1.5, K1 = 0.01, K2 = 0.03. data_range
    is the range of each reference field over the whole case. Invalid pixels are set to zero in
    both images and the SSIM map is averaged over valid pixels only.
    """
    def __init__(self, fields, mask, data_range):
        self.fields, self.mask = list(fields), np.asarray(mask, dtype=bool)
        data_range = np.asarray(data_range, dtype=np.float64)[:, None, None]
        self.c1, self.c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
        self.total = np.zeros(len(fields), dtype=np.float64)
        self.count = 0

    def update(self, predicted, reference):
        """predicted, reference: (field, height, width) images."""
        def blur(image):
            return gaussian_filter(image, 1.5, truncate=3.5, axes=(1, 2))
        p = np.where(self.mask, predicted, 0).astype(np.float64)
        r = np.where(self.mask, reference, 0).astype(np.float64)
        mean_p, mean_r = blur(p), blur(r)
        var_p, var_r = blur(p * p) - mean_p ** 2, blur(r * r) - mean_r ** 2
        covariance = blur(p * r) - mean_p * mean_r
        ssim = ((2 * mean_p * mean_r + self.c1) * (2 * covariance + self.c2)
                / ((mean_p ** 2 + mean_r ** 2 + self.c1) * (var_p + var_r + self.c2)))
        self.total += ssim[:, self.mask].mean(axis=1)
        self.count += 1

    def result(self):
        ssim = self.total / self.count
        return {"field_ssim": dict(zip(self.fields, map(float, ssim))), "mean_ssim": float(ssim.mean())}


def relative_l2(predicted, reference):
    denom = np.linalg.norm(reference)
    return float(np.linalg.norm(np.asarray(predicted)-reference)/denom) if denom > 0 else None


def harmonic(signal, time, frequency):
    omega = 2*np.pi*frequency*np.asarray(time)
    design = np.column_stack((np.cos(omega), np.sin(omega), np.ones(len(omega))))
    coefficients = np.linalg.lstsq(design, signal, rcond=None)[0]
    return coefficients[0] - 1j*coefficients[1]


def gain_phase(predicted_q, reference_q, phi, time, frequency, q0, start=0.5):
    time = np.asarray(time)
    # Exclude final endpoint to avoid double-counting a period boundary.
    mask = (time >= start) & (time < time[-1])
    if mask.sum() < 4 or (time[mask][-1]-time[mask][0]) * frequency < 1:
        raise ValueError("Insufficient sinusoidal evaluation window")
    if q0 == 0:
        return {"status": "undefined_zero_Q0"}
    input_h = harmonic(np.asarray(phi)[mask]-1, time[mask], frequency)
    if abs(input_h) < 1e-12:
        return {"status": "undefined_zero_forcing"}
    ref_h = harmonic((np.asarray(reference_q)[mask]-q0)/q0, time[mask], frequency)/input_h
    pred_h = harmonic((np.asarray(predicted_q)[mask]-q0)/q0, time[mask], frequency)/input_h
    defined = abs(ref_h) > 1e-12
    return {"status": "finite_window_estimate_periodicity_not_certified", "window_start_s": start,
            "reference_gain": float(abs(ref_h)), "predicted_gain": float(abs(pred_h)),
            "relative_gain_error": float(abs(abs(pred_h)-abs(ref_h))/abs(ref_h)) if defined else None,
            "phase_error_deg": float(np.degrees(np.angle(pred_h/ref_h))) if defined and abs(pred_h)>1e-12 else None}
