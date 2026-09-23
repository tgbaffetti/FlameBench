"""Validation errors for model selection, and test rollouts with streaming field metrics."""
import time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from DataProcessing.loading import make_loader
from DataProcessing.Dataset import keep_recent
from .metrics import FieldMetrics, FieldSSIM, relative_l2, gain_phase
from utils import write_json


def forcing_window(phi, target, length, Ni):
    """Forcing rows phi(s+1) - 1 for predicting x(target): phi(target-length+1 .. target) - 1.

    Forcing before time zero equals one (deviation zero); rows older than the last Ni + 1 are zero.
    """
    start = target - length + 1
    values = np.array(phi[max(start, 0):target + 1], dtype=np.float32) - 1
    values = np.concatenate((np.zeros(max(-start, 0), dtype=np.float32), values))
    return keep_recent(values[None], Ni + 1)


def reconstruction_error(compressor, dataset, batch_size=16, loader_options=None):
    """MSE of encode-decode on a scaled CompressorDataset, valid pixels only."""
    sse = count = 0
    for frames in tqdm(make_loader(dataset, batch_size, **(loader_options or {})), desc="Reconstruction error"):
        frames = frames.numpy()
        error = (compressor.decode(compressor.encode(frames)) - frames)[..., dataset.mask]
        sse += float(np.square(error, dtype=np.float64).sum())
        count += error.size
    return sse / count


def validation_error(forecaster, compressor, dataset, batch_size=16, loader_options=None, per_step=False):
    """MSE of recursive forecasts on a scaled image ForecasterDataset, valid pixels only.

    Image windows are encoded before rollout. Latent windows from a compressed ForecasterDataset
    are used directly, but still score against the corresponding scaled image targets. Use stride =
    horizon for image validation so every frame is compared once. A diverged forecast scores
    infinity. With per_step, return (MSE, list of the MSE at each rollout step; None if diverged).
    """
    sse = count = 0
    step_sse, step_count = np.zeros(dataset.horizon), np.zeros(dataset.horizon)
    latent_dataset = getattr(dataset, "latents", None) is not None
    loader = make_loader(dataset, batch_size, **(loader_options or {}))
    for batch_index, batch in enumerate(tqdm(loader, desc="Validation error")):
        frames, forcing, target = (batch[key].numpy() for key in ("states", "forcing", "target"))
        size, length = frames.shape[:2]
        if latent_dataset:
            states = keep_recent(frames, dataset.Nx + 1)
            indices = range(batch_index * batch_size, batch_index * batch_size + size)
            target_fields = np.stack([dataset.target_frames(index) for index in indices])
        else:
            states = compressor.encode(frames.reshape(size * length, *frames.shape[2:])).reshape(size, length, -1)
            states = keep_recent(states, dataset.Nx + 1)
            target_fields = target
        for step in range(target.shape[1]):
            predicted = forecaster.predict(states, keep_recent(forcing[:, step:step + length], dataset.Ni + 1))
            if not np.isfinite(predicted).all():
                return (float("inf"), None) if per_step else float("inf")
            error = (compressor.decode(predicted) - target_fields[:, step])[..., dataset.mask]
            squares = float(np.square(error, dtype=np.float64).sum())
            sse += squares
            count += error.size
            step_sse[step] += squares
            step_count[step] += error.size
            states = keep_recent(np.concatenate((states[:, 1:], predicted[:, None]), axis=1), dataset.Nx + 1)
    return (sse / count, list(step_sse / step_count)) if per_step else sse / count


def summarize(results, cases):
    """The few numbers that rank runs (Test_Summary/...), from the per-case test results of evaluate.

    nrmse and ssim average mean_nrmse and mean_ssim over the test cases; nrmse_worst is the worst
    case. heat_release_l2 averages the relative L2 error of integrated Q. Sine
    cases are grouped by forcing frequency: gain_error_<f>hz averages the relative gain error and
    phase_error_<f>hz the absolute phase error in degrees, because models can be right at one
    frequency and wrong at another. Missing values (e.g. heat release disabled) are skipped.
    """
    def mean(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None

    nrmse = [r["mean_nrmse"] for r in results.values() if r.get("mean_nrmse") is not None]
    summary = {"Test_Summary/nrmse": mean(nrmse),
               "Test_Summary/nrmse_worst": max(nrmse) if nrmse else None,
               # Diverged cases drop out of the means; count them so they cannot vanish silently.
               "Test_Summary/diverged_cases": sum(1 for r in results.values() if r.get("status") == "diverged"),
               "Test_Summary/ssim": mean(r.get("mean_ssim") for r in results.values()),
               "Test_Summary/heat_release_l2": mean(r.get("heat_release_relative_l2") for r in results.values()),
               "Test_Summary/seconds_per_step": mean(r.get("seconds_per_step") for r in results.values())}
    frequencies = {case["name"]: case["frequency_hz"] for case in cases if case.get("waveform") == "sine"}
    for frequency in sorted(set(frequencies.values())):
        gain_phase = [results[name].get("gain_phase", {}) for name, f in frequencies.items()
                      if f == frequency and name in results]
        phase = [g.get("phase_error_deg") for g in gain_phase]
        summary[f"Test_Summary/gain_error_{frequency:g}hz"] = mean(g.get("relative_gain_error") for g in gain_phase)
        summary[f"Test_Summary/phase_error_{frequency:g}hz"] = mean(abs(p) for p in phase if p is not None)
    return summary


def synchronize(model):
    device = getattr(model, "device", None)
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


METRIC_BATCH = 128  # Test frames per metric batch: batched SSIM (on the GPU when present) is far faster.


def evaluate(model, dataset, compressor, scaler, directory, heat_release=True,
             initialization="steady", gain_phase_start=0.5, save_predictions=False,
             restart_every=None):
    """restart_every=k re-encodes the observed history every k steps (k=1 is the pure one-step
    protocol); None rolls out freely. Restarted protocols write metrics_restart<k>.json and
    suffixed Q/prediction files, so free-rollout results are never overwritten. A case whose
    rollout goes nonfinite is recorded (status "diverged", diverged_at_step, the metrics
    accumulated so far) and evaluation continues with the next case."""
    if restart_every is not None and restart_every < 1:
        raise ValueError("restart_every must be a positive integer or None")
    suffix = f"_restart{restart_every}" if restart_every else ""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = dataset.metadata
    if initialization == "steady" and metadata.get("initial_snapshot_is_steady") is not True:
        raise ValueError("Confirm initial_snapshot_is_steady=true in the metadata before steady-start evaluation")
    if initialization not in {"steady", "observed_history"}:
        raise ValueError("Unknown initialization protocol")
    def cells(frame):
        rows, columns = dataset.grid_indices
        return frame[..., rows, columns]

    volumes = None
    if heat_release:
        if dataset.volumes is None:
            raise ValueError("Integrated Q requires physical cell volumes; run prepare with the grid or disable heat_release")
        volumes = dataset.volumes.astype(np.float64)
        q_index = metadata["fields"].index("mix:Q")
    results = {}
    length = dataset.length
    for case, lo, hi in dataset.segments:
        x, phi = dataset.arrays(case)
        first = lo + (1 if initialization == "steady" else length)
        if initialization == "steady":
            initial = np.repeat(np.array(x[lo:lo+1]), length, axis=0)
        else:
            initial = np.array(x[first-length:first])
        states = keep_recent(compressor.encode(scaler.transform(initial))[None], dataset.Nx + 1)
        # Warm up transition and decoder. Timing excludes disk reads and metrics.
        forcing = forcing_window(phi, first, length, dataset.Ni)
        warmup_latent = model.predict(states, forcing)
        scaler.inverse(compressor.decode(warmup_latent))
        synchronize(model)
        metrics, elapsed = FieldMetrics(metadata["fields"]), 0.0
        # SSIM data range: range of each reference field over the case, read in chunks.
        low = high = None
        for start in range(lo, hi, 256):
            chunk = cells(np.asarray(x[start:min(start + 256, hi)]))
            low = chunk.min(axis=(0, 2)) if low is None else np.minimum(low, chunk.min(axis=(0, 2)))
            high = chunk.max(axis=(0, 2)) if high is None else np.maximum(high, chunk.max(axis=(0, 2)))
        ssim = FieldSSIM(metadata["fields"], dataset.mask, high - low,
                         device="cuda" if torch.cuda.is_available() else "cpu")
        q_ref, q_pred = [], []
        buffer = []  # (predicted, reference) frames; metrics run in batches of METRIC_BATCH frames

        def score(frames):
            predicted, reference = (np.stack(side) for side in zip(*frames))
            predicted_cells, reference_cells = cells(predicted), cells(reference)
            metrics.update_batch(predicted_cells, reference_cells)
            ssim.update_batch(predicted, reference)
            if volumes is not None:
                q_pred.extend(predicted_cells[:, q_index].astype(np.float64) @ volumes)
                q_ref.extend(reference_cells[:, q_index].astype(np.float64) @ volumes)
            frames.clear()

        output = None
        if save_predictions:
            output = np.lib.format.open_memmap(directory / f"{case['name']}_predictions{suffix}.npy", mode="w+",
                       dtype="float32", shape=(hi-first, *dataset.field_shape))
        diverged_at = None
        for k in tqdm(range(first, hi), desc=f"Test {case['name']}"):
            if restart_every and k > first and (k - first) % restart_every == 0 and k - length >= lo:
                states = keep_recent(compressor.encode(scaler.transform(np.array(x[k-length:k])))[None],
                                     dataset.Nx + 1)
            forcing = forcing_window(phi, k, length, dataset.Ni)
            synchronize(model)
            begin = time.perf_counter()
            z = model.predict(states, forcing)
            # Divergence is a reportable outcome, not an abort: the latent check runs before
            # decoding because decoders may reject nonfinite input.
            if not np.isfinite(z).all():
                diverged_at = k - first
                buffer.clear()
                break
            predicted = scaler.inverse(compressor.decode(z))[0]
            synchronize(model)
            elapsed += time.perf_counter()-begin
            buffer.append((predicted, np.array(x[k])))
            if len(buffer) == METRIC_BATCH:
                try:
                    score(buffer)
                except FloatingPointError:  # Nonfinite past the decoder; step known to batch precision.
                    diverged_at = k - first
                    buffer.clear()
                    break
            if output is not None:
                output[k-first] = predicted
            states = keep_recent(np.concatenate((states[:, 1:], z[:, None]), axis=1), dataset.Nx + 1)
        if buffer:
            try:
                score(buffer)
            except FloatingPointError:
                diverged_at = hi - first
                buffer.clear()
        if output is not None:
            output.flush()
            del output
        if diverged_at is not None:
            result = metrics.result() if metrics.frames else {}
            if ssim.count:
                result.update(ssim.result())
            result.update({"status": "diverged", "diverged_at_step": diverged_at,
                           "first_predicted_index": first, "restart_every": restart_every})
            results[case["name"]] = result
            write_json(directory / f"metrics{suffix}.json", results)
            continue
        result = {**metrics.result(), **ssim.result()}
        result.update({"status": "completed", "restart_every": restart_every,
                       "forecast_steps": hi-first, "first_predicted_index": first,
                       "inference_seconds": elapsed, "seconds_per_step": elapsed/(hi-first),
                       "timing_scope": "latent transition + field decoding + inverse scaling; excludes initial encoding, IO, metrics",
                       "initialization": initialization, "Nx": dataset.Nx, "Ni": dataset.Ni})
        if volumes is not None:
            q_pred, q_ref = [float(q) for q in q_pred], [float(q) for q in q_ref]
            result["heat_release_relative_l2"] = relative_l2(q_pred, q_ref)
            times = np.arange(first, hi)*metadata["dt"]
            np.savez(directory / f"{case['name']}_Q{suffix}.npz", time=times, reference=q_ref, predicted=q_pred)
            if case["waveform"] == "sine":
                q0 = float(np.dot(np.asarray(cells(x[0])[q_index], dtype=np.float64), volumes))
                result["gain_phase"] = gain_phase(q_pred, q_ref, phi[first:hi], times,
                       case["frequency_hz"], q0, gain_phase_start)
        else:
            result["heat_release_status"] = "explicitly_disabled"
        results[case["name"]] = result
        write_json(directory / f"metrics{suffix}.json", results)
    return results
