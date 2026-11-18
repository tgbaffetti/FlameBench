"""Inference time per rollout step of every benchmark model, measured on one machine.

The stored seconds_per_step of the paper runs are not comparable: the seeds of a pair were tested
three at a time on one shared GPU ("parallel": 3). This script re-runs each model's test exactly
as Experiments/run.py test() does (same model.pkl, metadata and evaluation settings, evaluate()
unchanged), one model at a time.

Nothing is written next to the runs: the run directories are only read (config.json,
model.pkl, metadata.resolved.json, metrics.json), there is no W&B logging, and every output goes
to <repo>/timing/: the replayed test of each run in timing/<name>/<run>_<seed>/ and the summary
in timing/<name>.json.

Timed scope (as in evaluate): latent transition + field decoding + inverse scaling, with CUDA
synchronized before and after; initial encoding, data reads and metrics are excluded. Every step
of every completed test case is pooled: median, standard deviation, mean, step count. Closed-form
forecasters and POD are numpy and run on the CPU whatever --device says; --device moves only
torch models (autoencoders, neural forecasters, FNO). The replayed epsilon_Q is compared with the
run's stored metrics.json, to confirm the replay is the paper's model.

The runs file lists one seed directory per line (blank lines and # comments are ignored). Run it
alone on an idle GPU, e.g.
  CUDA_VISIBLE_DEVICES=2 python -m Experiments.timing --runs-file timing_runs.txt --list
  CUDA_VISIBLE_DEVICES=2 python -m Experiments.timing --runs-file timing_runs.txt --device cuda --threads 4
  python -m Experiments.timing --runs-file timing_runs.txt --device cpu --threads 4 --name timing_cpu
"""
import argparse
import json
import os
import pickle
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from DataProcessing.Dataset import Dataset, ForecasterDataset
from DataProcessing.metadata import load_metadata
from Experiments.run import to_device
from utils import seed_everything

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "timing"


def read_runs(path):
    """{pair: seed directory} from a runs file; the pair is read from the run's config."""
    runs = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        directory = Path(line)
        config = json.loads((directory / "config.json").read_text())
        pair = f"{config['compressor']['name']}_{config['forecaster']['name']}"
        if pair in runs:
            raise ValueError(f"{pair} listed twice: {runs[pair]} and {directory}")
        runs[pair] = directory
    return runs


def other_gpu_jobs():
    """Compute processes, other than this one, on the GPU this process uses (the first of
    CUDA_VISIBLE_DEVICES, all GPUs if it is unset); None if nvidia-smi is unavailable."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    command = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
    if visible:
        command += ["-i", visible]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [int(pid) for pid in output.split() if pid.isdigit() and int(pid) != os.getpid()]


def time_run(directory, device, threads, out):
    """Re-run the test of one seed directory into out; return its pooled step times and a summary."""
    config = json.loads((directory / "config.json").read_text())
    Dataset.in_memory = config.get("in_memory", False)  # as Experiments/run.py main
    seed_everything(config.get("seed", 42))
    torch.set_num_threads(threads or config.get("cpu_threads", 4))
    with (directory / "model.pkl").open("rb") as file:
        scaler, compressor, model = pickle.load(file)
    device = device or config.get("device", "cpu")
    to_device(model, device)
    to_device(compressor, device)
    dataset = ForecasterDataset(load_metadata(directory / "metadata.resolved.json"), "test", Nx=model.Nx, Ni=model.Ni)
    settings = {**config.get("evaluation", {}), "save_predictions": False}
    scratch = out / f"{directory.parent.name}_{directory.name}"
    begin = time.perf_counter()
    results = model.test(dataset, compressor, scaler, scratch, **settings)
    wall = time.perf_counter() - begin
    stored = json.loads((directory / "metrics.json").read_text())
    steps, per_case, eps_difference = [], {}, 0.0
    for name, result in results.items():
        path = scratch / f"{name}_step_seconds.npy"
        if result.get("status") != "completed" or not path.exists():
            per_case[name] = result.get("status")
            continue
        seconds = np.load(path)
        steps.append(seconds)
        per_case[name] = float(np.median(seconds))
        if stored.get(name, {}).get("heat_release_relative_l2") is not None:
            eps_difference = max(eps_difference, abs(result["heat_release_relative_l2"] - stored[name]["heat_release_relative_l2"]))
    steps = np.concatenate(steps) if steps else np.array([])
    return {"run": str(directory), "device": device, "torch_threads": torch.get_num_threads(),
            "Nx": model.Nx, "Ni": model.Ni, "rank": getattr(compressor, "rank", None),
            "steps": int(steps.size), "median_s": float(np.median(steps)) if steps.size else None,
            "std_s": float(np.std(steps)) if steps.size else None, "mean_s": float(np.mean(steps)) if steps.size else None,
            "per_case_median_s": per_case, "test_wall_s": wall,
            "max_eps_Q_difference_vs_paper": eps_difference}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-file", type=Path, required=True, help="one seed_N directory per line")
    parser.add_argument("--device", help="cuda or cpu (default: each run's config device)")
    parser.add_argument("--threads", type=int, help="torch CPU threads (default: each run's cpu_threads)")
    parser.add_argument("--name", default="timing", help="output name: timing/<name>.json and timing/<name>/")
    parser.add_argument("--list", action="store_true", help="only check and print the runs that would be timed")
    args = parser.parse_args()

    runs = read_runs(args.runs_file)
    for pair, directory in runs.items():
        missing = [f for f in ("config.json", "model.pkl", "metadata.resolved.json", "metrics.json") if not (directory / f).exists()]
        print(f"{pair:22s} {directory}" + (f"   MISSING {missing}" if missing else ""))
    print(f"{len(runs)} runs")
    if args.list:
        return
    hardware = {"gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cpu_count": os.cpu_count(), "torch": torch.__version__, "cuda": torch.version.cuda}
    print(hardware)
    out = OUT / args.name
    path = OUT / f"{args.name}.json"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for pair, directory in runs.items():
        others = other_gpu_jobs()
        if others:
            print(f"WARNING {len(others)} other process(es) on the GPU: timings of {pair} are not clean")
        print(f"== {pair}", flush=True)
        try:
            results[pair] = time_run(directory, args.device, args.threads, out)
        except Exception as error:  # one broken run must not stop the battery
            print(f"   FAILED {type(error).__name__}: {error}")
            results[pair] = {"run": str(directory), "error": f"{type(error).__name__}: {error}"}
            continue
        results[pair]["other_gpu_processes"] = len(others) if others is not None else None
        r = results[pair]
        if r["median_s"] is not None:
            print(f"   median {1e3 * r['median_s']:.3f} ms ± {1e3 * r['std_s']:.3f} (mean {1e3 * r['mean_s']:.3f}) over "
                  f"{r['steps']} steps; eps_Q vs paper max diff {r['max_eps_Q_difference_vs_paper']:.1e}", flush=True)
        path.write_text(json.dumps({"hardware": hardware, "models": results}, indent=1))

    print(f"\n{'pair':22s} {'median ms':>10s} {'std ms':>9s} {'steps':>6s} {'eps_Q diff':>11s}")
    for pair, r in sorted(results.items(), key=lambda item: item[1].get("median_s") or float("inf")):
        if r.get("median_s") is not None:
            print(f"{pair:22s} {1e3 * r['median_s']:10.3f} {1e3 * r['std_s']:9.3f} {r['steps']:6d} "
                  f"{r['max_eps_Q_difference_vs_paper']:11.1e}")
        else:
            print(f"{pair:22s} {'—':>10s}  {r.get('error') or r.get('per_case_median_s')}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
