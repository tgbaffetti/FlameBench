"""Integrated heat release Q(t) of tuned models against ground truth on the test cases."""
import glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "Experiments" / "Results"
OUT = Path(__file__).resolve().parent
CASES = ["sine_f10_A03", "sine_f10_A05", "sine_f40_A03", "sine_f40_A05", "step_A03", "step_A05"]
# (pair glob, label, categorical color slot)
MODELS = [("cae_arx_2026*", "CAE+ARX", "#2a78d6"),
          ("pod_arx_2026*", "POD+ARX", "#eb6834"),
          ("pod_narx_2026*", "POD+NARX", "#1baf7a"),
          ("pod_transformer_2026*", "POD+Transformer", "#eda100")]

plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.6, "axes.edgecolor": "#9aa0a6",
                     "xtick.color": "#5f6368", "ytick.color": "#5f6368",
                     "axes.labelcolor": "#202124", "text.color": "#202124"})

def latest_tuned(pattern):
    """Newest run directory of the pair that has per-seed protocol output."""
    for run in sorted(RESULTS.glob(pattern), reverse=True):
        if (run / "seed_0").is_dir():
            return run / "seed_0"
    return None

def draw(ax, case, legend=False):
    reference = None
    for pattern, label, color in MODELS:
        seed = latest_tuned(pattern)
        if seed is None:
            continue
        q = np.load(seed / f"{case}_Q.npz")
        if reference is None:
            reference = q["reference"]
            ax.plot(q["time"], reference, color="black", lw=2.4, label="Ground truth", zorder=5)
        ax.plot(q["time"], q["predicted"], color=color, lw=1.2, label=label)
    margin = 0.4 * (reference.max() - reference.min()) or 0.4 * abs(reference.mean())
    ax.set_ylim(reference.min() - margin, reference.max() + margin)
    ax.set_xlim(0, 1)
    ax.set_title(f"Integrated Q — {case}", fontsize=10)
    ax.grid(True, color="#e8eaed", linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=3)

for case in CASES:
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    draw(ax, case, legend=True)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Integrated heat release Q (J/s)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"q_timeseries_{case}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", case)

fig, axes = plt.subplots(3, 2, figsize=(11, 9.5), sharex=True)
for k, (ax, case) in enumerate(zip(axes.flat, CASES)):
    draw(ax, case, legend=(k == 0))
    if k >= 4:
        ax.set_xlabel("Time (s)")
    if k % 2 == 0:
        ax.set_ylabel("Q (J/s)")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"q_timeseries_all.{ext}", dpi=200, bbox_inches="tight")
print("saved all-cases grid")
