"""Recursive rollout from one observed initial state, with streaming field metrics."""
import time
from pathlib import Path
import numpy as np
import torch
from .metrics import FieldMetrics, relative_l2, gain_phase
from utils import write_json


def forcing_window(phi, target, Ni):
    """phi(target-Ni-1 : target), inclusive; pre-zero forcing equals one."""
    start = target - Ni - 1
    values = np.array(phi[max(start, 0):target+1], dtype=np.float32)
    if start < 0:
        values = np.concatenate((np.ones(-start, dtype=np.float32), values))
    return values[None]


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
    volumes = None
    if heat_release:
        if not metadata.get("cell_volumes"):
            raise ValueError("Integrated Q requires physical cell volumes; set cell_volumes or explicitly disable heat_release")
        volumes = np.load(metadata["cell_volumes"], allow_pickle=False)
        if volumes.shape != (dataset.field_shape[-1],) or not np.isfinite(volumes).all() or np.any(volumes <= 0):
            raise ValueError("Cell volumes must be finite, positive, and aligned with cells")
        q_index = metadata["fields"].index("mix:Q")
    results = {}
    for case, lo, hi in dataset.segments():
        x, phi = dataset.arrays(case)
        first = lo + (1 if initialization == "steady" else dataset.context)
        if initialization == "steady":
            initial = np.repeat(np.array(x[lo:lo+1]), dataset.history, axis=0)
        else:
            initial = np.array(x[first-dataset.history:first])
        history = compressor.encode(scaler.transform(initial))[None]
        # Warm up transition and decoder. Timing excludes disk reads and metrics.
        forcing = forcing_window(phi, first, dataset.Ni)
        warmup_latent = model.predict(history, forcing)
        scaler.inverse(compressor.decode(warmup_latent))
        synchronize(model)
        metrics, elapsed = FieldMetrics(metadata["fields"]), 0.0
        q_ref, q_pred = [], []
        output = None
        if save_predictions:
            output = np.lib.format.open_memmap(directory / f"{case['name']}_predictions.npy", mode="w+",
                       dtype="float32", shape=(hi-first, *dataset.field_shape))
        for k in range(first, hi):
            forcing = forcing_window(phi, k, dataset.Ni)
            synchronize(model)
            begin = time.perf_counter()
            z = model.predict(history, forcing)
            predicted = scaler.inverse(compressor.decode(z))[0]
            synchronize(model)
            elapsed += time.perf_counter()-begin
            reference = np.array(x[k])
            metrics.update(predicted, reference)
            if output is not None:
                output[k-first] = predicted
            if volumes is not None:
                q_pred.append(float(np.dot(predicted[q_index].astype(np.float64), volumes)))
                q_ref.append(float(np.dot(reference[q_index].astype(np.float64), volumes)))
            history = np.concatenate((history[:, 1:], z[:, None]), axis=1)
        if output is not None:
            output.flush()
            del output
        result = metrics.result()
        result.update({"forecast_steps": hi-first, "first_predicted_index": first,
                       "inference_seconds": elapsed, "seconds_per_step": elapsed/(hi-first),
                       "timing_scope": "latent transition + field decoding + inverse scaling; excludes initial encoding, IO, metrics",
                       "initialization": initialization, "Nx": dataset.Nx, "Ni": dataset.Ni})
        if volumes is not None:
            result["heat_release_relative_l2"] = relative_l2(q_pred, q_ref)
            times = np.arange(first, hi)*metadata["dt"]
            np.savez(directory / f"{case['name']}_Q.npz", time=times, reference=q_ref, predicted=q_pred)
            if case["waveform"] == "sine":
                q0 = float(np.dot(np.asarray(x[0, q_index], dtype=np.float64), volumes))
                result["gain_phase"] = gain_phase(q_pred, q_ref, phi[first:hi], times,
                       case["frequency_hz"], q0, gain_phase_start)
        else:
            result["heat_release_status"] = "explicitly_disabled"
        results[case["name"]] = result
        write_json(directory / "metrics.json", results)
    return results
