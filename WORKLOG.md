# FlameBench worklog

Running log of work on branch `gianmarco`. Newest entries at the bottom.

## 2026-09-21 (morning) — repo restructure + tests
- Merged Carlo's `carlo` draft and `main` (Tommaso's `process4convolution.py`) into `gianmarco`.
- Reconstructed missing `Experiments/paths.py` from the test contract (commit c9a9147); 48/48 tests pass.
- Test data (6 cases, 18.6 GB) downloaded from SwissTransfer into `data/`.
- Literature review built: `literature/` with 46 paper cards, ranked shortlist in its README.

## 2026-09-21 (midday) — completing the dataset
- Found the gap: `Data/metadata.json` needs 8 cases but the SwissTransfer only carried the 6 test
  cases; the 2 training sweeps and `grid.vtu` were shared via Teams instead.
- Downloaded via MS Graph (device-code login, since SharePoint links are org-restricted):
  - `data/sineSweep_f1_f80_A02.npz` (6.39 GB) and `..._A04.npz` (6.48 GB) — verified byte-exact
    against SharePoint metadata; both load as `(21334 cells, 11 fields, 4001 t)`.
  - `data/grid.vtu` (3.3 MB, Tommaso's `grid_0_0 6.vtu`) — loads in pyvista, 21334 cells, matches.
- `DataProcessing/prepare.py` running over all 8 cases (CWD=`Data/`, `--source ../data`);
  output = float32 `(time, fields, cells)` memmaps in `Data/Training|Test/`.
- Installed `pyvista` in the apptainer env (`~/envs/dtd2c`), needed by `process4convolution.py`.

## 2026-09-21 — benchmark method selection (for a 9-page ICLR benchmark paper)
Principle: one representative per family, mesh-native, natural phi(t) injection. Minimal set:

| Method | Family | Status |
|---|---|---|
| Persistence + DMDc (`pod`+`arx`, n_past=1) | linear ROM | config only |
| POD/AE × LSTM/Transformer | latent ROM + sequence | already implemented |
| 0-D flame response (phi → q') | domain baseline | to implement |
| DeepONet | classic neural operator | to implement |
| Transolver | modern transformer operator | to implement |
| MeshGraphNets | GNN | to implement (next session) |

- Cut: forced-KAE, UPT, GINO, BSMS-GNN, AROMA, GNOT, RO-NORM, raw FNO/UNet (family already
  represented or not mesh-native). Single protocol ablation: pushforward/unrolled training on the
  autoregressive models.
- CNN/grid track (Tommaso's `process4convolution.py`): mesh is planar → regular (nx, nz) image via
  coordinate binning, zero-filled outside the angled side. Opens conv-AE compressor (and FNO2d/UNet
  as stretch). Script needs adapting to read the prepared float32 `(t, f, cells)` npy instead of
  raw npy. Queued after the mesh-native minimal set.

Working order (each step: implement → pytest → 2-epoch smoke run → doc):
1. DMDc + persistence configs, smoke-tested. 2. 0-D flame-response baseline (q' from `mix:Q` +
cell volumes from grid.vtu). 3. DeepONet (adds a raw-field model path in `run.py`). 4. Transolver.
5. MeshGraphNets + pushforward ablation. 6. CNN/grid track. Then big runs.
