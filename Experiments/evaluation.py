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


def validation_error(forecaster, compressor, dataset, batch_size=16, loader_options=None):
    """MSE of recursive forecasts on a scaled image ForecasterDataset, valid pixels only.

    Image windows are encoded before rollout. Latent windows from a compressed ForecasterDataset
    are used directly, but still score against the corresponding scaled image targets. Use stride =
    horizon for image validation so every frame is compared once. A diverged forecast scores
    infinity.
    """
    sse = count = 0
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
                return float("inf")
            error = (compressor.decode(predicted) - target_fields[:, step])[..., dataset.mask]
            sse += float(np.square(error, dtype=np.float64).sum())
            count += error.size
            states = keep_recent(np.concatenate((states[:, 1:], predicted[:, None]), axis=1), dataset.Nx + 1)
    return sse / count


def synchronize(model):
    device = getattr(model, "device", None)
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(model, dataset, compressor, scaler, directory, heat_release=True,
             initialization="steady", gain_phase_start=0.5, save_predictions=False):
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
        ssim = FieldSSIM(metadata["fields"], dataset.mask, high - low)
        q_ref, q_pred = [], []
        output = None
        if save_predictions:
            output = np.lib.format.open_memmap(directory / f"{case['name']}_predictions.npy", mode="w+",
                       dtype="float32", shape=(hi-first, *dataset.field_shape))
        for k in tqdm(range(first, hi), desc=f"Test {case['name']}"):
            forcing = forcing_window(phi, k, length, dataset.Ni)
            synchronize(model)
            begin = time.perf_counter()
            z = model.predict(states, forcing)
            predicted = scaler.inverse(compressor.decode(z))[0]
            synchronize(model)
            elapsed += time.perf_counter()-begin
            reference = np.array(x[k])
            metrics.update(cells(predicted), cells(reference))
            ssim.update(predicted, reference)
            if output is not None:
                output[k-first] = predicted
            if volumes is not None:
                q_pred.append(float(np.dot(cells(predicted)[q_index].astype(np.float64), volumes)))
                q_ref.append(float(np.dot(cells(reference)[q_index].astype(np.float64), volumes)))
            states = keep_recent(np.concatenate((states[:, 1:], z[:, None]), axis=1), dataset.Nx + 1)
        if output is not None:
            output.flush()
            del output
        result = {**metrics.result(), **ssim.result()}
        result.update({"forecast_steps": hi-first, "first_predicted_index": first,
                       "inference_seconds": elapsed, "seconds_per_step": elapsed/(hi-first),
                       "timing_scope": "latent transition + field decoding + inverse scaling; excludes initial encoding, IO, metrics",
                       "initialization": initialization, "Nx": dataset.Nx, "Ni": dataset.Ni})
        if volumes is not None:
            result["heat_release_relative_l2"] = relative_l2(q_pred, q_ref)
            times = np.arange(first, hi)*metadata["dt"]
            np.savez(directory / f"{case['name']}_Q.npz", time=times, reference=q_ref, predicted=q_pred)
            if case["waveform"] == "sine":
                q0 = float(np.dot(np.asarray(cells(x[0])[q_index], dtype=np.float64), volumes))
                result["gain_phase"] = gain_phase(q_pred, q_ref, phi[first:hi], times,
                       case["frequency_hz"], q0, gain_phase_start)
        else:
            result["heat_release_status"] = "explicitly_disabled"
        results[case["name"]] = result
        write_json(directory / "metrics.json", results)
    return results
