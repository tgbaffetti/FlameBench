"""Paper figures: forcing signals, field gallery of one snapshot, one field over a forcing cycle."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent.parent / "Data"
OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)
META = json.load(open(DATA / "metadata.json"))
DT = META["dt"]
LOW, HIGH = "#2a78d6", "#eb6834"  # validated categorical pair: low / high amplitude
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6, "axes.edgecolor": "#9aa0a6",
                     "xtick.color": "#5f6368", "ytick.color": "#5f6368",
                     "axes.labelcolor": "#202124", "text.color": "#202124"})

def grid_off(ax):
    ax.grid(True, color="#e8eaed", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ---------------------------------------------------------------- Fig 1: forcing
def phi(path):
    v = np.load(DATA / path)
    return np.arange(1, len(v) + 1) * DT, v

fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.2))
(a, b), (c, d) = axes

t, v = phi("Images/Training/phi_sineSweep_f1_f80_A02.npy")
a.plot(t, v, color=LOW, lw=0.7, label="A = 0.2")
t, v = phi("Images/Training/phi_sineSweep_f1_f80_A04.npy")
a.plot(t, v, color=HIGH, lw=0.7, alpha=0.75, label="A = 0.4")
a.set_title("training: sine sweep 1–80 Hz", loc="left", fontsize=9)
a.set_xlim(0, 2)

for ax, f in ((b, 10), (c, 40)):
    t, v = phi(f"Images/Test/phi_sine_f{f}_A03.npy")
    ax.plot(t, v, color=LOW, lw=1.1, label="A = 0.3")
    t, v = phi(f"Images/Test/phi_sine_f{f}_A05.npy")
    ax.plot(t, v, color=HIGH, lw=1.1, alpha=0.75, label="A = 0.5")
    ax.set_title(f"test: sine {f} Hz", loc="left", fontsize=9)
    ax.set_xlim(0, 0.3)

for amp, color, label in ((0.3, LOW, "A = 0.3"), (0.5, HIGH, "A = 0.5")):
    d.plot([-0.1, 0, 0, 1], [1, 1, 1 + amp, 1 + amp], color=color, lw=1.1,
           alpha=1.0 if amp == 0.3 else 0.75, label=label)
d.set_title("test: step at t = 0", loc="left", fontsize=9)
d.set_xlim(-0.1, 1)

for ax in axes.flat:
    grid_off(ax)
    ax.set_ylabel(r"$\varphi(t)$")
    ax.set_xlabel("t [s]")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2, borderaxespad=0)
    ax.set_ylim(0.55, 1.62)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"forcing_signals.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("forcing_signals done")

# ------------------------------------------------- Fig 2: all fields, one snapshot
case = np.load(DATA / "Images/Test/sine_f10_A05.npy", mmap_mode="r")
snap = np.array(case[400])  # t = 0.2 s, mid-cycle
fields = META["fields"]
units = META["units"]
mask = np.array(case[0, 4]) > 1.0  # invalid pixels are zero; T is always > 1 K
fig, axes = plt.subplots(1, 11, figsize=(11, 2.9))
for ax, img, name, unit in zip(axes, snap, fields, units):
    shown = np.where(mask, img, np.nan)
    title = name
    if name == "mix:Q":
        shown, title = shown / 1e9, "Q [GW/m$^3$]"
    elif name == "p":
        shown, title = shown - np.nanmean(shown), "p$'$ [Pa]"
    im = ax.imshow(shown, origin="upper", cmap="viridis", aspect="equal")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, fraction=0.05)
    lo, hi = float(np.nanmin(shown)), float(np.nanmax(shown))
    cb.set_ticks([lo, hi])
    cb.set_ticklabels(["0" if abs(v) < 1e-6 else f"{v:.2g}" for v in (lo, hi)])
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
fig.suptitle("sine 10 Hz, A = 0.5 — all fields at t = 0.2 s", fontsize=10, y=0.99)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fields_snapshot.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fields_snapshot done")

# ------------------------------------------- Fig 3: one field across a forcing cycle
qi = fields.index("mix:Q")
period = int(round(1 / 10 / DT))          # 10 Hz -> 200 steps
t0 = 400
steps = [t0 + int(round(k * period / 6)) for k in range(6)]
frames = [np.where(mask, np.array(case[s, qi]), np.nan) / 1e9 for s in steps]
vmin = 0.0
vmax = float(np.nanpercentile(np.stack(frames), 99.5))  # clip: peak Q is pointlike
fig, axes = plt.subplots(1, 6, figsize=(7.2, 2.9))
for ax, f, s in zip(axes, frames, steps):
    im = ax.imshow(f, origin="upper", cmap="inferno", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(f"t/T = {(s - t0) / period:.2f}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
cb = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.025, pad=0.02)
cb.set_label(r"Q [GW/m$^3$]", fontsize=8)
cb.ax.tick_params(labelsize=6)
cb.outline.set_visible(False)
fig.suptitle("heat release Q over one forcing period — sine 10 Hz, A = 0.5", fontsize=10)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"q_cycle.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("q_cycle done")
