"""Where does a model's integrated heat-release error come from? Maps of the Q-dot error.

The rollout is the paper's own code: each run is set up exactly as Experiments/run.py test()
does (seed_everything, cpu_threads, model.pkl moved to the config device, the run's resolved
metadata, the config's evaluation settings) and Experiments/evaluation.py evaluate() runs
unchanged, into a scratch folder: nothing is written into the run directory and nothing is
logged to W&B. While it runs, hooks that only observe record the latent prediction of one
snapshot per test case; it is decoded as evaluate decodes it.

Every call therefore re-scores the whole test set, and the fresh Q(t) is compared with the run's
stored <case>_Q.npz (the series logged to W&B and reported in the paper) at every step. They
match only in the environment that produced the run: run with the Python recorded by W&B
(metadata "executable") from a checkout of the run's commit; the script warns when the commit,
numpy, torch or CUDA versions differ from the run's environment.json.

Figure, one column per test case: Q-dot true; per run, Q-dot predicted, |Q-dot_pred - Q-dot_true|,
and (Q-dot_pred - Q-dot_true) * V, each cell's signed share of Q_pred - Q_true (the integrated error
behind epsilon_Q: front errors can cancel in the sum, and cell volume grows with radius).

  python figures/q_error_maps.py --run <results>/identity_fno_<ts>/seed_0 --step worst
  python figures/q_error_maps.py --run <fno seed dir> --run <pod_arx seed dir> --step 1000
--step: a snapshot index k (time k * dt), or "worst": each case's step of the largest stored
|Q_pred - Q_true| of the first run, used for every run.
"""
import argparse
import json
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import Experiments.evaluation as evaluation  # noqa: E402
from DataProcessing.Dataset import ForecasterDataset  # noqa: E402
from DataProcessing.metadata import load_metadata  # noqa: E402
from Experiments.run import to_device  # noqa: E402
from utils import seed_everything, provenance  # noqa: E402

FLAME_FRACTION = 0.01  # flame region: cells where true Q-dot exceeds 1% of its maximum at that snapshot


def check_environment(directory):
    """Warn where this process differs from the environment that ran the test."""
    recorded, now = json.loads((directory / "environment.json").read_text()), provenance()
    for key in ("commit", "numpy", "torch", "cuda"):
        if recorded.get(key) != now.get(key):
            print(f"  WARNING {key}: run {recorded.get(key)} / now {now.get(key)}")
    changed = [line for line in (recorded.get("git_status") or "").splitlines() if not line.startswith("??")]
    if changed:
        print("  WARNING the run had uncommitted changes:", "; ".join(changed))


def setup(directory, device=None, metadata_path=None):
    """Experiments/run.py test() for a saved run, without its W&B logger."""
    config = json.loads((directory / "config.json").read_text())
    seed_everything(config.get("seed", 42))
    torch.set_num_threads(config.get("cpu_threads", 4))
    with (directory / "model.pkl").open("rb") as file:
        scaler, compressor, model = pickle.load(file)
    device = device or config.get("device", "cpu")
    to_device(model, device)
    to_device(compressor, device)
    metadata = load_metadata(metadata_path or directory / "metadata.resolved.json")
    dataset = ForecasterDataset(metadata, "test", Nx=model.Nx, Ni=model.Ni)
    settings = {**config.get("evaluation", {}), "save_predictions": False}
    label = f"{config['compressor']['name']} + {config['forecaster']['name']}".replace("identity + ", "").upper()
    return scaler, compressor, model, dataset, settings, label


@contextmanager
def capture(model, dataset, steps):
    """Record model.predict's output for snapshot steps[case] while evaluate() runs unchanged.

    evaluate reads each case with dataset.arrays(case) and builds every forcing window with
    forcing_window(phi, k, ...) right before model.predict for snapshot k (the warm-up call uses
    the first k too and gives the same output). The hooks pass every value through untouched.
    """
    state, latents = {"case": None, "k": None}, {}
    arrays, predict, window = dataset.arrays, model.predict, evaluation.forcing_window

    def arrays_hook(case):
        state["case"] = case["name"]
        return arrays(case)

    def window_hook(phi, target, length, Ni):
        state["k"] = target
        return window(phi, target, length, Ni)

    def predict_hook(states, forcing):
        z = predict(states, forcing)
        if state["k"] == steps.get(state["case"]):
            latents[state["case"]] = np.array(z, copy=True)
        return z

    dataset.arrays, model.predict, evaluation.forcing_window = arrays_hook, predict_hook, window_hook
    try:
        yield latents
    finally:
        del dataset.arrays, model.predict
        evaluation.forcing_window = window


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, action="append", required=True, help="seed_N directory; repeat to compare")
    parser.add_argument("--step", default="worst", help='snapshot index, or "worst" (default)')
    parser.add_argument("--device", help="override the config device (default: as the run)")
    parser.add_argument("--metadata", help="override <run>/metadata.resolved.json (default: as the run)")
    parser.add_argument("--out", type=Path, default=ROOT / "figures" / "q_error")
    args = parser.parse_args()

    first_run = setup(args.run[0], args.device, args.metadata)
    dataset = first_run[3]
    metadata, q_index = dataset.metadata, dataset.metadata["fields"].index("mix:Q")
    rows, columns = dataset.grid_indices
    mask, volumes = dataset.mask, dataset.volumes.astype(np.float64)
    cases = list(dataset.segments)

    steps = {}
    for case, lo, hi in cases:
        if args.step == "worst":
            q = np.load(args.run[0] / f"{case['name']}_Q.npz")
            steps[case["name"]] = lo + 1 + int(np.argmax(np.abs(q["predicted"] - q["reference"])))
        else:
            steps[case["name"]] = int(args.step)
        if not lo + 1 <= steps[case["name"]] < hi:
            raise ValueError(f"step must lie in [{lo + 1}, {hi - 1}] for {case['name']}")

    truth = {case["name"]: np.asarray(np.load(case["data"], mmap_mode="r")[steps[case["name"]]], dtype=np.float64)
             [q_index][rows, columns] for case, _, _ in cases}
    results, summary, labels = {}, {name: {"step": k, "time_s": k * metadata["dt"], "runs": {}} for name, k in steps.items()}, []
    for index, directory in enumerate(args.run):
        scaler, compressor, model, run_dataset, settings, label = first_run if index == 0 else setup(directory, args.device, args.metadata)
        labels.append(label)
        print(f"== {label}: {directory}")
        check_environment(directory)
        replay_dir = args.out / "replay" / f"{directory.parent.name}_{directory.name}"
        with capture(model, run_dataset, steps) as latents:
            model.test(run_dataset, compressor, scaler, replay_dir, **settings)
        for case, lo, hi in cases:
            name = case["name"]
            stored, fresh = np.load(directory / f"{name}_Q.npz"), np.load(replay_dir / f"{name}_Q.npz")
            agreement = {"max_relative_Q_difference": float(np.max(np.abs(fresh["predicted"] - stored["predicted"])
                                                                  / np.abs(stored["predicted"]))),
                         "epsilon_Q_stored": float(np.linalg.norm(stored["predicted"] - stored["reference"]) / np.linalg.norm(stored["reference"])),
                         "epsilon_Q_replay": float(np.linalg.norm(fresh["predicted"] - fresh["reference"]) / np.linalg.norm(fresh["reference"]))}
            if not np.allclose(fresh["reference"], stored["reference"], rtol=1e-6):
                raise ValueError(f"{name}: reference Q differs from the run's; different data")
            if name not in latents:
                raise FloatingPointError(f"{name}: rollout diverged before step {steps[name]}")
            predicted = scaler.inverse(compressor.decode(latents[name]))[0][q_index][rows, columns].astype(np.float64)
            results[label, name] = predicted
            contribution = (predicted - truth[name]) * volumes
            flame = truth[name] > FLAME_FRACTION * truth[name].max()
            q_true, q_pred = float(truth[name] @ volumes), float(predicted @ volumes)
            summary[name]["Q_true"] = q_true
            summary[name]["runs"][label] = {
                "Q_pred": q_pred, "dQ": q_pred - q_true, "dQ_relative": (q_pred - q_true) / q_true,
                "dQ_flame": float(contribution[flame].sum()), "dQ_outside_flame": float(contribution[~flame].sum()),
                "dQ_positive_cells": float(contribution[contribution > 0].sum()),
                "dQ_negative_cells": float(contribution[contribution < 0].sum()), **agreement}
            print(f"  {name:13s} t={steps[name] * metadata['dt']:.4f}s dQ={100 * (q_pred - q_true) / q_true:+6.1f}% "
                  f"(flame {contribution[flame].sum():+.3g}, outside {contribution[~flame].sum():+.3g})  "
                  f"eps_Q replay {agreement['epsilon_Q_replay']:.4f} / paper {agreement['epsilon_Q_stored']:.4f}  "
                  f"max |dQ(t)| replay vs paper {agreement['max_relative_Q_difference']:.1e}")
            if agreement["max_relative_Q_difference"] > 1e-4:
                print("    WARNING the replayed rollout is not the paper's (see the environment warnings above)")

    # Shared color scales across cases (and runs), clipped at the 99.5th percentile as in Fig. 1.
    to_giga = 1e-9
    q_max = np.nanpercentile(np.concatenate(list(truth.values())), 99.5) * to_giga
    error_max = np.percentile(np.concatenate([np.abs(results[key] - truth[key[1]]) for key in results]), 99.5) * to_giga
    share_max = np.percentile(np.abs(np.concatenate([(results[key] - truth[key[1]]) * volumes for key in results])), 99.5) * 1e3

    def image(cell_values):
        picture = np.full(mask.shape, np.nan)
        picture[rows, columns] = cell_values
        return picture

    def colormap(name):
        cmap = plt.get_cmap(name).copy()
        cmap.set_bad("#e8eaed")  # outside the domain
        return cmap

    row_specs = [("true", None)] + [(kind, label) for label in labels for kind in ("pred", "abs", "share")]
    fig, axes = plt.subplots(len(row_specs), len(cases), figsize=(1.55 * len(cases) + 1.4, 2.75 * len(row_specs)),
                             squeeze=False, constrained_layout=True)
    for column, (case, _, _) in enumerate(cases):
        name = case["name"]
        axes[0, column].set_title(f"{name}\nt = {steps[name] * metadata['dt']:.3f} s", fontsize=9)
        for row, (kind, label) in enumerate(row_specs):
            ax = axes[row, column]
            if kind == "true":
                picture, cmap, low, high = image(truth[name] * to_giga), "inferno", 0, q_max
            elif kind == "pred":
                picture, cmap, low, high = image(results[label, name] * to_giga), "inferno", 0, q_max
            elif kind == "abs":
                picture, cmap, low, high = image(np.abs(results[label, name] - truth[name]) * to_giga), "Reds", 0, error_max
            else:
                picture = image((results[label, name] - truth[name]) * volumes * 1e3)
                cmap, low, high = "RdBu_r", -share_max, share_max
                ax.set_xlabel(f"ΔQ = {100 * summary[name]['runs'][label]['dQ_relative']:+.1f}%", fontsize=8)
            shown = ax.imshow(picture, origin="upper", cmap=colormap(cmap), vmin=low, vmax=high, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                ax.set_ylabel({"true": "Q̇ true", "pred": f"Q̇ {label}", "abs": f"|ΔQ̇| {label}",
                               "share": f"ΔQ̇·V {label}"}[kind], fontsize=9)
            if column == len(cases) - 1:
                fig.colorbar(shown, ax=axes[row, :], shrink=0.8,
                             label={"true": "GW/m³", "pred": "GW/m³", "abs": "GW/m³", "share": "mW per cell"}[kind])
    tag = args.step if args.step == "worst" else f"step{int(args.step):04d}"
    stem = args.out / f"q_error_{tag}_{'_vs_'.join(label.replace(' + ', '-') for label in labels)}"
    for extension in ("png", "pdf"):
        fig.savefig(f"{stem}.{extension}", dpi=200)
    np.savez_compressed(f"{stem}.npz", steps=json.dumps(steps), volumes=volumes, rows=rows, columns=columns,
                        **{f"true_{name}": values for name, values in truth.items()},
                        **{f"pred_{label}_{name}": values for (label, name), values in results.items()})
    Path(f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {stem}.png/.pdf/.npz/.json")


if __name__ == "__main__":
    main()
