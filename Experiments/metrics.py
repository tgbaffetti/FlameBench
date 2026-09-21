"""Physical-unit metrics; evaluation statistics never feed training."""
from bisect import bisect_left
import numpy as np


class FieldMetrics:
    """Streaming nRMSE per field, pooled and per rollout-horizon bin.

    update() is called once per rollout step, in order; bin_edges (in steps,
    1-based, inclusive) split the horizon, e.g. (10, 100, 1000) gives bins
    1-10, 11-100, 101-1000, 1001+. Normalization uses the pooled reference
    std so bins are comparable with each other and with the pooled value.
    """

    def __init__(self, fields, bin_edges=(10, 100, 1000)):
        if list(bin_edges) != sorted(set(bin_edges)) or (bin_edges and bin_edges[0] < 1):
            raise ValueError("bin_edges must be strictly increasing and >= 1")
        self.fields = list(fields)
        self.bin_edges = tuple(bin_edges)
        lows = (1,) + tuple(e + 1 for e in self.bin_edges)
        highs = self.bin_edges + (None,)
        self.bin_labels = [f"{lo}-{hi}" if hi else f"{lo}+" for lo, hi in zip(lows, highs)]
        self.step = 0
        self.count = 0
        self.mean = np.zeros(len(fields), dtype=np.float64)
        self.m2 = np.zeros(len(fields), dtype=np.float64)
        self.bin_sse = np.zeros((len(self.bin_labels), len(fields)), dtype=np.float64)
        self.bin_count = np.zeros(len(self.bin_labels), dtype=np.int64)

    @property
    def sse(self):
        return self.bin_sse.sum(axis=0)

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
        self.step += 1
        b = bisect_left(self.bin_edges, self.step)
        self.bin_sse[b] += np.square(predicted-reference).sum(axis=1)
        self.bin_count[b] += n

    @staticmethod
    def _nrmse(rmse, std):
        # Constant reference fields have undefined std-normalized error, not zero error.
        return [float(a/b) if b > 0 else None for a, b in zip(rmse, std)]

    def result(self):
        std = np.sqrt(self.m2/self.count)
        nrmse = self._nrmse(np.sqrt(self.sse/self.count), std)
        horizon = {}
        for label, sse, count in zip(self.bin_labels, self.bin_sse, self.bin_count):
            if count == 0:
                continue
            binned = self._nrmse(np.sqrt(sse/count), std)
            horizon[label] = {"field_nrmse": dict(zip(self.fields, binned)),
                              "mean_nrmse": float(np.mean(binned)) if all(x is not None for x in binned) else None,
                              "steps": int(count/(self.count/self.step))}
        return {"field_nrmse": dict(zip(self.fields, nrmse)),
                "field_rmse": dict(zip(self.fields, map(float, np.sqrt(self.sse/self.count)))),
                "mean_nrmse": float(np.mean(nrmse)) if all(x is not None for x in nrmse) else None,
                "horizon_nrmse": horizon,
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
