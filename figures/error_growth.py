"""How the test error grows along the closed-loop rollout, for representative models.

Reads only stored test outputs, no rollout is run:
  top row     field nRMSE per rollout-horizon bin (steps 1-10, 11-100, 101-1000, 1001+), the
              horizon_nrmse of each seed's metrics.json. Every bin is normalized by the pooled
              reference std of the whole case, so bins compare with each other and with Table 4.
  bottom row  integrated heat-release error |Q_pred - Q_true| / mean(Q_true) at every step,
              as a moving RMS over WINDOW steps, from each seed's <case>_Q.npz.
Curves are the mean over seeds, bands and error bars one standard deviation (deterministic
models have one seed). The rollout starts from the steady flame (Section 5), so steps 1-10
include the start-up from a steady history.

A run is the newest Experiments/Results/<pair>_<timestamp> whose seed_0/config.json carries the
benchmark-v2 tag and is not a refit; --run <label>=<run dir> overrides one model.

  python figures/error_growth.py                       # all six cases
  python figures/error_growth.py --cases sine_f10_A05 sine_f40_A05 step_A05 --name error_growth_main
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "Experiments" / "Results"
CASES = ["sine_f10_A03", "sine_f10_A05", "sine_f40_A03", "sine_f40_A05", "step_A03", "step_A05"]
BINS = ["1-10", "11-100", "101-1000", "1001+"]
WINDOW = 50  # steps of the moving RMS of the Q error: 25 ms, one period at 40 Hz
# (label, pair prefix, color); colors as in make_q_plot.py where the model appears there
MODELS = [("POD+ARX", "pod_arx", "#eb6834"),
          ("CAE+ARX", "cae_arx", "#2a78d6"),
          ("POD+Transformer", "pod_transformer", "#eda100"),
          ("FNO", "identity_fno", "#8e44ad"),
          ("Constant", "identity_constant", "#7f7f7f")]

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6, "axes.edgecolor": "#9aa0a6",
                     "xtick.color": "#5f6368", "ytick.color": "#5f6368",
                     "axes.labelcolor": "#202124", "text.color": "#202124"})


def benchmark_run(prefix):
    """Newest tuned (benchmark-v2, non-refit) run directory of a pair."""
    for run in sorted(RESULTS.glob(f"{prefix}_2026*"), reverse=True):
        config_path = run / "seed_0" / "config.json"
        if run.name.endswith("_refit") or not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        tags = config.get("logging", {}).get("wandb", {}).get("tags", [])
        names = (config.get("compressor", {}).get("name"), config.get("forecaster", {}).get("name"))
        if "benchmark-v2" in tags and "_".join(names) == prefix:
            return run
    return None


def seeds(run):
    return [seed for seed in sorted(run.glob("seed_*")) if (seed / "metrics.json").exists()]


def moving_rms(values, window):
    """Centered moving RMS; the windows at the ends hold fewer samples, not zero padding."""
    kernel = np.ones(window)
    return np.sqrt(np.convolve(values**2, kernel, mode="same") / np.convolve(np.ones_like(values), kernel, mode="same"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", default=CASES, choices=CASES)
    parser.add_argument("--run", action="append", default=[], help="<label>=<run dir>, overrides the automatic choice")
    parser.add_argument("--name", default="error_growth", help="output file stem in figures/")
    args = parser.parse_args()
    overrides = dict(item.split("=", 1) for item in args.run)

    runs = {}
    for label, prefix, color in MODELS:
        run = Path(overrides[label]) if label in overrides else benchmark_run(prefix)
        if run is None or not seeds(run):
            print(f"{label:16s} no run found, skipped")
            continue
        runs[label] = (run, color)
        print(f"{label:16s} {run}  seeds: {[s.name for s in seeds(run)]}")

    table = {}
    fig, axes = plt.subplots(2, len(args.cases), figsize=(2.9 * len(args.cases), 5.4), squeeze=False)
    for column, case in enumerate(args.cases):
        top, bottom = axes[0, column], axes[1, column]
        for label, (run, color) in runs.items():
            per_seed, q_errors, time = [], [], None
            for seed in seeds(run):
                result = json.loads((seed / "metrics.json").read_text())[case]
                if result.get("status") != "completed":
                    print(f"  {label} {seed.name} {case}: {result.get('status')} at step {result.get('diverged_at_step')}")
                    continue
                per_seed.append([result["horizon_nrmse"][b]["mean_nrmse"] for b in BINS])
                with np.load(seed / f"{case}_Q.npz") as q:
                    time = q["time"] - q["time"][0]
                    q_errors.append(moving_rms((q["predicted"] - q["reference"]) / np.mean(q["reference"]), WINDOW))
            if not per_seed:
                continue
            per_seed = np.array(per_seed)
            mean, std = per_seed.mean(axis=0), per_seed.std(axis=0)
            table[label, case] = (mean, std)
            x = np.arange(len(BINS))
            top.errorbar(x, mean, yerr=std if len(per_seed) > 1 else None, color=color, marker="o", ms=3.5,
                         lw=1.2, capsize=2, label=label)
            q_errors = np.array(q_errors)
            q_mean = q_errors.mean(axis=0)
            bottom.plot(time, q_mean, color=color, lw=1.0, label=label)
            if len(q_errors) > 1:
                bottom.fill_between(time, q_mean - q_errors.std(axis=0), q_mean + q_errors.std(axis=0),
                                    color=color, alpha=0.15, lw=0)
        top.set_title(case, fontsize=9)
        top.set_xticks(range(len(BINS)), BINS, fontsize=7)
        top.set_yscale("log")
        bottom.set_yscale("log")
        bottom.set_xlabel("time since start (s)")
        for ax in (top, bottom):
            ax.grid(True, color="#e8eaed", linewidth=0.5)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    axes[0, 0].set_ylabel("field nRMSE")
    axes[1, 0].set_ylabel(r"$|Q_{pred}-Q_{true}|\,/\,\overline{Q}_{true}$" + f"\n(moving RMS, {WINDOW} steps)")
    axes[0, 0].set_xlabel("rollout steps", fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"{args.name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved figures/{args.name}.pdf/.png")

    print(f"\n{'model':16s} {'case':13s} " + " ".join(f"{b:>15s}" for b in BINS))
    for (label, case), (mean, std) in table.items():
        print(f"{label:16s} {case:13s} " + " ".join(f"{m:7.3f}±{s:6.3f}" for m, s in zip(mean, std)))
    out = {f"{label}|{case}": {"bins": BINS, "mean": mean.tolist(), "std": std.tolist(), "run": str(runs[label][0])}
           for (label, case), (mean, std) in table.items()}
    (ROOT / "figures" / f"{args.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
