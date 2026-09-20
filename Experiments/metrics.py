"""Physical-unit metrics; evaluation statistics never feed training."""
import numpy as np


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
