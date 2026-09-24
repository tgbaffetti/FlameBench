"""Integrated heat release Q(t) of tuned models against ground truth on one test case."""
import glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "Experiments" / "Results"
OUT = Path(__file__).resolve().parent
CASE = "sine_f10_A05"
# (pair glob, label, categorical color slot)
MODELS = [("cae_arx_2026*", "CAE+ARX", "#2a78d6"),
          ("pod_arx_2026*", "POD+ARX", "#eb6834"),
          ("pod_transformer_2026*", "POD+Transformer", "#1baf7a"),
          ("pod_narx_2026*", "POD+NARX", "#eda100"),
          ("pod_gru_2026*", "POD+GRU", "#e87ba4")]

plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.6, "axes.edgecolor": "#9aa0a6",
                     "xtick.color": "#5f6368", "ytick.color": "#5f6368",
                     "axes.labelcolor": "#202124", "text.color": "#202124"})

def latest_tuned(pattern):
    """Newest run directory of the pair that has per-seed protocol output."""
    runs = sorted(RESULTS.glob(pattern), reverse=True)
    for run in runs:
        if (run / "seed_0").is_dir():
            return run / "seed_0"
    return None

fig, ax = plt.subplots(figsize=(8.6, 4.0))
reference = None
for pattern, label, color in MODELS:
    seed = latest_tuned(pattern)
    if seed is None:
        continue
    q = np.load(seed / f"{CASE}_Q.npz")
    if reference is None:
        reference = q["reference"]
        ax.plot(q["time"], reference, color="black", lw=2.6, label="Ground truth", zorder=5)
    ax.plot(q["time"], q["predicted"], color=color, lw=1.3, label=label)

margin = 0.4 * (reference.max() - reference.min())
ax.set_ylim(reference.min() - margin, reference.max() + margin)
ax.set_xlim(0, 1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Integrated heat release Q (J/s)")
ax.set_title(f"Integrated Q — {CASE}", fontsize=11)
ax.grid(True, color="#e8eaed", linewidth=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=3)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"q_timeseries_{CASE}.{ext}", dpi=200, bbox_inches="tight")
print("saved", OUT / f"q_timeseries_{CASE}.png")
