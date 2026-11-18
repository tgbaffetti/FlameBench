"""How the test error grows along the closed-loop rollout, for representative models.

Reads only stored test outputs; no rollout is run and nothing is written next to the runs.
Both rows use the same rollout-horizon bins (the bins of Experiments/metrics.py FieldMetrics)
and the same seeds:
  top row     field nRMSE per bin: the horizon_nrmse of each seed's metrics.json. Each bin is
              normalized by the reference std of the whole case, as the pooled nRMSE of Table 4.
  bottom row  heat-release error per bin, from each seed's <case>_Q.npz:
              RMS(Q_pred - Q_true over the bin) / RMS(Q_true over the whole case). Normalized by
              the whole case like the top row, so the bins combine (frame-weighted, in
              quadrature) into the epsilon_Q of Table 5.
Markers are the mean over seeds, bars span the minimum and maximum seed (deterministic models
have one seed). In steps 1-10 every model, the constant forecast included, is dominated by a
short pressure transient of the reference data at the onset of forcing; --no-first-bin leaves
that bin out of the figure (it stays in the printed table and the JSON).

A run is the newest Experiments/Results/<pair>_<timestamp> whose seed_0/config.json carries the
benchmark-v2 tag and is not a refit; --run <label>=<run dir> overrides one model.

  python figures/error_growth.py --run "POD+ARX=<dir>" ... --name error_growth_v2
  python figures/error_growth.py --cases sine_f10_A05 sine_f40_A05 step_A03 --name error_growth_main ...
Outputs: figures/<name>.pdf, .png, .json in this checkout (existing files of that name are replaced).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "Experiments" / "Results"
# Default order fills the grid row by row: columns are the input types, rows the amplitudes.
CASES = ["sine_f10_A03", "sine_f40_A03", "step_A03", "sine_f10_A05", "sine_f40_A05", "step_A05"]
COLUMNS = 3          # cases per row of the grid
PANEL = (2.1, 1.45)  # inches per panel: the figure is drawn near its printed size (about \textwidth)
# Rollout-horizon bins of FieldMetrics: label -> 0-based slice of the forecast steps.
BINS = {"1-10": slice(0, 10), "11-100": slice(10, 100), "101-1000": slice(100, 1000), "1001+": slice(1000, None)}
# (label, pair prefix, color); colors as in make_q_plot.py where the model appears there
MODELS = [("POD+ARX", "pod_arx", "#eb6834"),
          ("CAE+ARX", "cae_arx", "#2a78d6"),
          ("POD+Transformer", "pod_transformer", "#eda100"),
          ("FNO", "identity_fno", "#8e44ad"),
          ("Constant", "identity_constant", "#7f7f7f")]

plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": "#9aa0a6",
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


def title(case):
    """'sine_f10_A05' -> 'Sine, 10 Hz, A = 0.5'; 'step_A03' -> 'Step, A = 0.3'."""
    parts = case.split("_")
    amplitude = f"A = {int(parts[-1][1:]) / 10:g}"
    if parts[0] == "sine":
        return f"Sine, {int(parts[1][1:])} Hz, {amplitude}"
    return f"{parts[0].capitalize()}, {amplitude}"


def plain_log_ticks(ax):
    """Log axis labelled with plain decimals; minor ticks unlabelled. Call after plotting: axes
    spanning under 1.6 decades get labels at 1, 2 and 5 per decade, wider ones at each decade."""
    low, high = ax.get_ylim()
    subs = (1.0, 2.0, 5.0) if np.log10(high / low) < 1.6 else (1.0,)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def seeds(run):
    return [seed for seed in sorted(run.glob("seed_*")) if (seed / "metrics.json").exists()]


def q_error_bins(path):
    """Heat-release error per horizon bin, normalized by the RMS of the true Q over the case."""
    with np.load(path) as q:
        predicted, reference = q["predicted"].astype(np.float64), q["reference"].astype(np.float64)
    scale = np.sqrt(np.mean(reference**2))
    return [float(np.sqrt(np.mean((predicted[s] - reference[s])**2)) / scale) for s in BINS.values()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", default=CASES, choices=CASES)
    parser.add_argument("--run", action="append", default=[], help="<label>=<run dir>, overrides the automatic choice")
    parser.add_argument("--name", default="error_growth", help="output file stem in figures/")
    parser.add_argument("--no-first-bin", action="store_true", help="leave steps 1-10 out of the figure")
    args = parser.parse_args()
    overrides = dict(item.split("=", 1) for item in args.run)
    shown = list(BINS)[1:] if args.no_first_bin else list(BINS)

    runs = {}
    for label, prefix, color in MODELS:
        run = Path(overrides[label]) if label in overrides else benchmark_run(prefix)
        if run is None or not seeds(run):
            print(f"{label:16s} no run found, skipped")
            continue
        runs[label] = (run, color)
        print(f"{label:16s} {run}  seeds: {[s.name for s in seeds(run)]}")

    table = {}  # (label, case) -> {"nrmse": seeds x bins, "q": seeds x bins}
    for label, (run, _) in runs.items():
        for case in args.cases:
            nrmse, q = [], []
            for seed in seeds(run):
                result = json.loads((seed / "metrics.json").read_text())[case]
                if result.get("status") != "completed":
                    print(f"  {label} {seed.name} {case}: {result.get('status')} at step {result.get('diverged_at_step')}, left out")
                    continue
                nrmse.append([result["horizon_nrmse"][b]["mean_nrmse"] for b in BINS])
                q.append(q_error_bins(seed / f"{case}_Q.npz"))
            if nrmse:
                table[label, case] = {"nrmse": np.array(nrmse), "q": np.array(q)}

    # Grid: at most COLUMNS cases per row; each case is a field-nRMSE panel directly above its
    # heat-release panel, and a spacer row separates the rows of cases, so the pairs read as units.
    ncols = min(COLUMNS, len(args.cases))
    case_rows = -(-len(args.cases) // ncols)
    spacer = 0.35  # height of the gap between rows of cases, as a fraction of a panel
    legend, title_space, bottom_space = 0.3, 0.2, 0.45  # inches: legend row, first titles, x label
    ratios = [1, 1] + [spacer, 1, 1] * (case_rows - 1)
    height = sum(ratios) * PANEL[1] + legend + title_space + bottom_space
    fig = plt.figure(figsize=(ncols * PANEL[0], height))
    grid = fig.add_gridspec(len(ratios), ncols, height_ratios=ratios, hspace=0.12, wspace=0.42,
                            left=0.6 / (ncols * PANEL[0]), right=0.99,
                            bottom=bottom_space / height, top=1 - (legend + title_space) / height)
    row_of = [3 * r for r in range(case_rows)]  # grid row of each case row's top panel
    columns = [list(BINS).index(b) for b in shown]
    x = np.arange(len(shown))
    for index in range(case_rows * ncols):
        row, col = row_of[index // ncols], index % ncols
        if index >= len(args.cases):
            continue
        top, bottom = fig.add_subplot(grid[row, col]), fig.add_subplot(grid[row + 1, col])
        case = args.cases[index]
        for ax, key in ((top, "nrmse"), (bottom, "q")):
            for label, (run, color) in runs.items():
                if (label, case) not in table:
                    continue
                values = table[label, case][key][:, columns]
                mean, low, high = values.mean(axis=0), values.min(axis=0), values.max(axis=0)
                ax.errorbar(x, mean, yerr=[mean - low, high - mean] if len(values) > 1 else None, color=color,
                            marker="o", ms=2.8, lw=1.0, elinewidth=0.8, capsize=1.5, label=label)
            ax.set_xticks(x, shown if ax is bottom else [])
            ax.set_yscale("log")
            plain_log_ticks(ax)  # after plotting: the labels depend on the data range
            ax.grid(True, color="#e8eaed", linewidth=0.5)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        top.set_title(title(case), fontsize=8)
        if index % ncols == 0:
            top.set_ylabel("field nRMSE")
            bottom.set_ylabel(r"heat-release error $\varepsilon_{\dot Q}$")
        if index // ncols == case_rows - 1 or index + ncols >= len(args.cases):
            bottom.set_xlabel("rollout steps")
    # One legend for the whole figure, in a row above the panels, so it never covers data.
    handles = [plt.Line2D([], [], color=color, marker="o", ms=2.8, lw=1.0) for _, (_, color) in runs.items()]
    fig.legend(handles, list(runs), loc="upper center", ncol=len(runs), frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"{args.name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved figures/{args.name}.pdf/.png/.json")

    def cell(values):
        return f"{values.mean():.3f} [{values.min():.3f},{values.max():.3f}]" if len(values) > 1 else f"{values[0]:.3f}"

    for key, heading in (("nrmse", "field nRMSE"), ("q", "heat-release error")):
        print(f"\n{heading}: mean [min, max] over seeds\n{'model':16s} {'case':13s} " + " ".join(f"{b:>22s}" for b in BINS))
        for (label, case), values in table.items():
            print(f"{label:16s} {case:13s} " + " ".join(f"{cell(values[key][:, i]):>22s}" for i in range(len(BINS))))
    out = {f"{label}|{case}": {"bins": list(BINS), "run": str(runs[label][0]),
                               **{key: {"mean": v.mean(axis=0).tolist(), "min": v.min(axis=0).tolist(),
                                        "max": v.max(axis=0).tolist(), "seeds": len(v)} for key, v in values.items()}}
           for (label, case), values in table.items()}
    (ROOT / "figures" / f"{args.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
