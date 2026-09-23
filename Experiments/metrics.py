"""Physical-unit metrics; evaluation statistics never feed training."""
import numpy as np
import torch


class FieldMetrics:
    """Streaming nRMSE per field, pooled and per rollout-horizon bin.

    Frames arrive in rollout order; bin_edges (1-based steps, inclusive) split the horizon,
    e.g. (10, 100, 1000) gives bins 1-10, 11-100, 101-1000, 1001+. Bins are normalized by the
    pooled reference std, so they are comparable with each other and with the pooled value.
    """

    def __init__(self, fields, bin_edges=(10, 100, 1000)):
        if list(bin_edges) != sorted(set(bin_edges)) or (bin_edges and bin_edges[0] < 1):
            raise ValueError("bin_edges must be strictly increasing and >= 1")
        self.fields = list(fields)
        self.bin_edges = np.asarray(bin_edges)
        lows, highs = (1, *(edge + 1 for edge in bin_edges)), (*bin_edges, None)
        self.bin_labels = [f"{lo}-{hi}" if hi else f"{lo}+" for lo, hi in zip(lows, highs)]
        self.count = 0
        self.frames = 0
        self.mean = np.zeros(len(fields), dtype=np.float64)
        self.m2 = np.zeros(len(fields), dtype=np.float64)
        self.sse = np.zeros(len(fields), dtype=np.float64)
        self.bin_sse = np.zeros((len(self.bin_labels), len(fields)), dtype=np.float64)
        self.bin_frames = np.zeros(len(self.bin_labels), dtype=np.int64)

    def update(self, predicted, reference):
        """One frame: (field, cell) arrays."""
        self.update_batch(np.asarray(predicted)[None], np.asarray(reference)[None])

    def update_batch(self, predicted, reference):
        """Frames: (frame, field, cell) arrays; the same statistics as one update per frame."""
        predicted, reference = np.asarray(predicted, dtype=np.float64), np.asarray(reference, dtype=np.float64)
        if predicted.shape != reference.shape or reference.ndim != 3:
            raise ValueError("Expected matching (frame, field, cell) arrays")
        if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
            raise FloatingPointError("Nonfinite rollout/reference: metric undefined")
        n = reference.shape[0] * reference.shape[2]
        mu = reference.mean(axis=(0, 2))
        delta = mu - self.mean
        self.m2 += np.square(reference - mu[None, :, None]).sum(axis=(0, 2)) + delta**2*self.count*n/(self.count+n)
        self.mean += delta*n/(self.count+n)
        self.count += n
        per_frame = np.square(predicted-reference).sum(axis=2)
        self.sse += per_frame.sum(axis=0)
        steps = self.frames + 1 + np.arange(len(reference))  # 1-based rollout steps
        bins = np.searchsorted(self.bin_edges, steps, side="left")
        np.add.at(self.bin_sse, bins, per_frame)
        np.add.at(self.bin_frames, bins, 1)
        self.frames += len(reference)

    @staticmethod
    def _nrmse(rmse, std):
        # Constant reference fields have undefined std-normalized error, not zero error.
        return [float(a/b) if b > 0 else None for a, b in zip(rmse, std)]

    def result(self):
        std = np.sqrt(self.m2/self.count)
        nrmse = self._nrmse(np.sqrt(self.sse/self.count), std)
        cells = self.count // self.frames
        horizon = {}
        for label, sse, frames in zip(self.bin_labels, self.bin_sse, self.bin_frames):
            if frames == 0:
                continue
            binned = self._nrmse(np.sqrt(sse / (frames * cells)), std)
            horizon[label] = {"field_nrmse": dict(zip(self.fields, binned)), "frames": int(frames),
                              "mean_nrmse": float(np.mean(binned)) if all(x is not None for x in binned) else None}
        return {"field_nrmse": dict(zip(self.fields, nrmse)),
                "field_rmse": dict(zip(self.fields, map(float, np.sqrt(self.sse/self.count)))),
                "mean_nrmse": float(np.mean(nrmse)) if all(x is not None for x in nrmse) else None,
                "horizon_nrmse": horizon,
                "undefined_fields": [f for f, value in zip(self.fields, nrmse) if value is None]}


class FieldSSIM:
    """Mean structural similarity of each field over the frames of a rollout.

    SSIM of Wang et al. (2004): Gaussian window with sigma 1.5, K1 = 0.01, K2 = 0.03. data_range
    is the range of each reference field over the whole case. Invalid pixels are set to zero in
    both images and the SSIM map is averaged over valid pixels only. The blur is that of
    scipy.ndimage.gaussian_filter(sigma=1.5, truncate=3.5), whose default border mode repeats
    the edge pixels mirrored, computed in float64 torch on `device` for a batch of frames at once.
    """
    SIGMA, TRUNCATE = 1.5, 3.5

    def __init__(self, fields, mask, data_range, device="cpu"):
        self.fields, self.mask = list(fields), np.asarray(mask, dtype=bool)
        self.device = torch.device(device)
        data_range = torch.as_tensor(np.asarray(data_range, dtype=np.float64), device=self.device)[:, None, None]
        self.c1, self.c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
        self.valid = torch.as_tensor(self.mask, device=self.device)
        radius = int(self.TRUNCATE * self.SIGMA + 0.5)
        x = torch.arange(-radius, radius + 1, dtype=torch.float64, device=self.device)
        kernel = torch.exp(-0.5 * (x / self.SIGMA) ** 2)
        self.kernel, self.radius = kernel / kernel.sum(), radius
        self.total = np.zeros(len(fields), dtype=np.float64)
        self.count = 0

    def blur(self, images):
        """Separable Gaussian blur of (batch, height, width), mirrored ("reflect") borders."""
        r = self.radius
        for dim in (1, 2):
            # scipy "reflect": the image repeats mirrored, period 2 * size, also for sizes below r.
            size = images.shape[dim]
            index = torch.arange(-r, size + r, device=self.device) % (2 * size)
            index = torch.where(index < size, index, 2 * size - 1 - index)
            images = images.index_select(dim, index).movedim(dim, -1)
            shape = images.shape
            images = torch.nn.functional.conv1d(images.reshape(-1, 1, shape[-1]), self.kernel.view(1, 1, -1))
            images = images.reshape(*shape[:-1], -1).movedim(-1, dim)
        return images

    def update(self, predicted, reference):
        """One frame: (field, height, width) images."""
        self.update_batch(np.asarray(predicted)[None], np.asarray(reference)[None])

    def update_batch(self, predicted, reference):
        """Frames: (frame, field, height, width) images."""
        frames, fields = predicted.shape[:2]
        p = torch.as_tensor(np.asarray(predicted), device=self.device).double() * self.valid
        r = torch.as_tensor(np.asarray(reference), device=self.device).double() * self.valid
        p, r = p.flatten(0, 1), r.flatten(0, 1)  # (frame * field, height, width)
        mean_p, mean_r = self.blur(p), self.blur(r)
        var_p, var_r = self.blur(p * p) - mean_p ** 2, self.blur(r * r) - mean_r ** 2
        covariance = self.blur(p * r) - mean_p * mean_r
        c1, c2 = self.c1.repeat(frames, 1, 1), self.c2.repeat(frames, 1, 1)
        ssim = ((2 * mean_p * mean_r + c1) * (2 * covariance + c2)
                / ((mean_p ** 2 + mean_r ** 2 + c1) * (var_p + var_r + c2)))
        per_frame = ssim[:, self.valid].mean(dim=1).view(frames, fields)
        self.total += per_frame.sum(dim=0).cpu().numpy()
        self.count += frames

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
