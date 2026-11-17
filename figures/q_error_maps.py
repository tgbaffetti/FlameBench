"""Where does a model's integrated heat-release error come from? Maps of the Q-dot error.

Replays the test rollout of a trained run exactly as Experiments/evaluation.py does (steady
start, recursive, true forcing only), keeps the heat-release-rate field at one snapshot per test
case, and draws, one column per test case:
  - Q-dot true and predicted;
  - |Q-dot_pred - Q-dot_true|, the local error;
  - (Q-dot_pred - Q-dot_true) * V, each cell's signed share of Q_pred - Q_true, the integrated
    error behind epsilon_Q. A large local error on the thin flame front can cancel in the sum,
    and the cell volume grows with radius, so this row is what produces the integral error.
Nothing is written into the run directory. The replayed Q at each snapshot is checked against
the run's stored <case>_Q.npz; a mismatch means the replay is not the evaluated rollout.

Run from the repository root (a clone with the prepared data and the model classes the run was
trained with):
  python figures/q_error_maps.py --run <results>/identity_fno_<ts>/seed_0 --step 1000
  python figures/q_error_maps.py --run <fno seed dir> --run <pod_arx seed dir> --step worst
--step is a snapshot index k (time k * dt; 1000 = 0.5 s), or "worst": each case's step of the
largest |Q_pred - Q_true| of the first run, from its stored Q series, used for every run.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from DataProcessing.Dataset import ForecasterDataset, keep_recent  # noqa: E402
from DataProcessing.metadata import load_metadata  # noqa: E402
from Experiments.evaluation import forcing_window  # noqa: E402
from Experiments.run import to_device  # noqa: E402

FLAME_FRACTION = 0.01  # flame region: cells where true Q-dot exceeds 1% of its maximum at that snapshot


def load_run(directory, device):
    with (directory / "model.pkl").open("rb") as file:
        scaler, compressor, model = pickle.load(file)
    to_device(compressor, device)
    to_device(model, device)
    config = json.loads((directory / "config.json").read_text())
    label = f"{config['compressor']['name']} + {config['forecaster']['name']}".replace("identity + ", "")
    return scaler, compressor, model, label.upper()


def rollout(scaler, compressor, model, dataset, case, lo, last):
    """Yield (k, latent prediction of snapshot k) for k = lo+1 .. last of the steady-start test
    rollout, as Experiments/evaluation.py computes it."""
    x, phi = dataset.arrays(case)
    length = dataset.length
    initial = np.repeat(np.array(x[lo:lo + 1]), length, axis=0)
    states = keep_recent(compressor.encode(scaler.transform(initial))[None], dataset.Nx + 1)
    for k in tqdm(range(lo + 1, last + 1), desc=f"{case['name']} -> step {last}", leave=False):
        z = model.predict(states, forcing_window(phi, k, length, dataset.Ni))
        if not np.isfinite(z).all():
            raise FloatingPointError(f"{case['name']}: rollout nonfinite at step {k}")
        yield k, z
        states = keep_recent(np.concatenate((states[:, 1:], z[:, None]), axis=1), dataset.Nx + 1)


def replay(scaler, compressor, model, dataset, case, lo, step):
    """Predicted frame (physical units) at snapshot `step` of the steady-start test rollout."""
    for k, z in rollout(scaler, compressor, model, dataset, case, lo, step):
        if k == step:
            return scaler.inverse(compressor.decode(z))[0]


def trace(runs, datasets, metadata, q_index):
    """Replayed against stored Q at every step, to tell an input difference (off from step 1) from
    numerical drift (starts near float rounding and grows); plus the run's recorded metadata."""
    for (directory, scaler, compressor, model, label), dataset in zip(runs, datasets):
        print(f"== {label}: {directory}")
        recorded = json.loads((directory / "metadata.resolved.json").read_text())
        for key in ("dt", "fields", "grid_indices"):
            if recorded.get(key) != metadata.get(key):
                print(f"  metadata {key} differs: run {recorded.get(key)} / now {metadata.get(key)}")
        then = {case["name"]: case for case in recorded["cases"]}
        rows, columns = dataset.grid_indices
        volumes = dataset.volumes.astype(np.float64)
        for case, lo, hi in dataset.segments:
            for key in ("data", "phi"):
                if then.get(case["name"], {}).get(key) != case[key]:
                    print(f"  {case['name']} {key} path differs: run {then.get(case['name'], {}).get(key)} / now {case[key]}")
            q_file = directory / f"{case['name']}_Q.npz"
            newer = Path(case["phi"]).stat().st_mtime > q_file.stat().st_mtime
            stored, _, first = stored_q(directory, case, lo)
            relative = np.empty(len(stored))
            for k, z in rollout(scaler, compressor, model, dataset, case, lo, first + len(stored) - 1):
                frame = scaler.inverse(compressor.decode(z))[0]
                q = float(frame[q_index][rows, columns].astype(np.float64) @ volumes)
                relative[k - first] = abs(q - stored[k - first]) / max(abs(stored[k - first]), 1e-30)
            marks = [s for s in (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000) if s <= len(relative)]
            above = {threshold: (int(np.argmax(relative > threshold)) + 1 if (relative > threshold).any() else None)
                     for threshold in (1e-6, 1e-3, 1e-2)}
            print(f"  {case['name']:14s} phi file newer than stored Q: {newer}")
            print("    relative |Q_replay - Q_stored| at step " +
                  "  ".join(f"{s}: {relative[s - 1]:.1e}" for s in marks))
            print(f"    first step above 1e-6: {above[1e-6]}, 1e-3: {above[1e-3]}, 1e-2: {above[1e-2]}; "
                  f"max {relative.max():.1e}")


def stored_q(directory, case, lo):
    """(time index -> stored predicted Q, reference Q) of the run's evaluated rollout."""
    q = np.load(directory / f"{case['name']}_Q.npz")
    return q["predicted"], q["reference"], lo + 1  # entry i is snapshot first + i


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, action="append", required=True, help="seed_N directory; repeat to compare")
    parser.add_argument("--step", default="1000", help='snapshot index, or "worst"')
    parser.add_argument("--metadata", default="Data/metadata.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=ROOT / "figures" / "q_error")
    parser.add_argument("--trace", action="store_true",
                        help="only compare the replayed Q with the stored Q at every step, then exit")
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    q_index = metadata["fields"].index("mix:Q")
    runs = [(directory, *load_run(directory, args.device)) for directory in args.run]
    datasets = [ForecasterDataset(metadata, "test", Nx=model.Nx, Ni=model.Ni) for _, _, _, model, _ in runs]
    if args.trace:
        trace(runs, datasets, metadata, q_index)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    reference_dataset = datasets[0]
    rows, columns = reference_dataset.grid_indices
    mask, volumes = reference_dataset.mask, reference_dataset.volumes.astype(np.float64)
    cases = [(case, lo, hi) for case, lo, hi in reference_dataset.segments]

    steps = {}
    for case, lo, hi in cases:
        if args.step == "worst":
            predicted, reference, first = stored_q(args.run[0], case, lo)
            steps[case["name"]] = first + int(np.argmax(np.abs(predicted - reference)))
        else:
            steps[case["name"]] = int(args.step)
        if not lo + 1 <= steps[case["name"]] < hi:
            raise ValueError(f"step must lie in [{lo + 1}, {hi - 1}] for {case['name']}")

    def image(cell_values):
        picture = np.full(mask.shape, np.nan)
        picture[rows, columns] = cell_values
        return picture

    truth, results, summary = {}, {}, {}
    for case, lo, hi in cases:
        name = case["name"]
        k = steps[name]
        true_cells = np.asarray(reference_dataset.arrays(case)[0][k], dtype=np.float64)[q_index][rows, columns]
        truth[name] = true_cells
        flame = true_cells > FLAME_FRACTION * true_cells.max()
        q_true = float(true_cells @ volumes)
        summary[name] = {"step": k, "time_s": k * metadata["dt"], "Q_true": q_true, "runs": {}}
        for (directory, scaler, compressor, model, label), dataset in zip(runs, datasets):
            frame = replay(scaler, compressor, model, dataset, case, lo, k)
            predicted_cells = frame[q_index][rows, columns].astype(np.float64)
            contribution = (predicted_cells - true_cells) * volumes
            q_pred = float(predicted_cells @ volumes)
            stored_pred, stored_ref, first = stored_q(directory, case, lo)
            check = abs(q_pred - stored_pred[k - first]) / max(abs(stored_pred[k - first]), 1e-30)
            if abs(q_true - stored_ref[k - first]) > 1e-6 * abs(q_true):
                raise ValueError(f"{name}: reference Q differs from the run's stored series; wrong data or step")
            results[label, name] = predicted_cells
            summary[name]["runs"][label] = {
                "Q_pred": q_pred, "dQ": q_pred - q_true, "dQ_relative": (q_pred - q_true) / q_true,
                "dQ_flame": float(contribution[flame].sum()), "dQ_outside_flame": float(contribution[~flame].sum()),
                "dQ_positive_cells": float(contribution[contribution > 0].sum()),
                "dQ_negative_cells": float(contribution[contribution < 0].sum()),
                "replay_vs_stored_Q_relative": check}
            print(f"{name:14s} t={k * metadata['dt']:.4f}s {label:16s} Q_true={q_true:.4g} Q_pred={q_pred:.4g} "
                  f"dQ={100 * (q_pred - q_true) / q_true:+.1f}%  flame={contribution[flame].sum():+.3g} "
                  f"outside={contribution[~flame].sum():+.3g}  replay check {check:.1e}")
            if check > 1e-3:
                print(f"  WARNING: replayed Q differs from the stored Q by {check:.1e}; the replay is not the "
                      "evaluated rollout (different code, data or model)")

    # Shared color scales across cases (and runs), clipped at the 99.5th percentile as in Fig. 1.
    to_giga = 1e-9
    q_max = np.nanpercentile(np.concatenate(list(truth.values())), 99.5) * to_giga
    errors = [np.abs(results[key] - truth[key[1]]) * to_giga for key in results]
    error_max = np.percentile(np.concatenate(errors), 99.5)
    shares = [(results[key] - truth[key[1]]) * volumes * 1e3 for key in results]  # mW
    share_max = np.percentile(np.abs(np.concatenate(shares)), 99.5)

    def colormap(name):
        cmap = plt.get_cmap(name).copy()
        cmap.set_bad("#e8eaed")  # outside the domain
        return cmap

    labels = [label for _, _, _, _, label in runs]
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
                picture = image(np.abs(results[label, name] - truth[name]) * to_giga)
                cmap, low, high = "Reds", 0, error_max
            else:
                picture = image((results[label, name] - truth[name]) * volumes * 1e3)
                cmap, low, high = "RdBu_r", -share_max, share_max
                ax.set_xlabel(f"ΔQ = {100 * summary[name]['runs'][label]['dQ_relative']:+.1f}%", fontsize=8)
            shown = ax.imshow(picture, origin="upper", cmap=colormap(cmap), vmin=low, vmax=high, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                text = {"true": "Q̇ true", "pred": f"Q̇ {label}", "abs": f"|ΔQ̇| {label}", "share": f"ΔQ̇·V {label}"}[kind]
                ax.set_ylabel(text, fontsize=9)
            if column == len(cases) - 1:
                unit = {"true": "GW/m³", "pred": "GW/m³", "abs": "GW/m³", "share": "mW per cell"}[kind]
                fig.colorbar(shown, ax=axes[row, :], shrink=0.8, label=unit)
    tag = args.step if args.step == "worst" else f"step{int(args.step):04d}"
    stem = args.out / f"q_error_{tag}_{'_vs_'.join(label.replace(' + ', '-') for label in labels)}"
    for extension in ("png", "pdf"):
        fig.savefig(f"{stem}.{extension}", dpi=200)
    np.savez_compressed(f"{stem}.npz", steps=json.dumps(steps), volumes=volumes, rows=rows, columns=columns,
                        **{f"true_{name}": values for name, values in truth.items()},
                        **{f"pred_{label}_{name}": values for (label, name), values in results.items()})
    (Path(f"{stem}.json")).write_text(json.dumps(summary, indent=2))
    print(f"saved {stem}.png/.pdf/.npz/.json")


if __name__ == "__main__":
    main()
