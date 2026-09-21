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

## 2026-09-21 — step 1: linear baselines + grid-derived quantities
- Added `Experiments/Configs/pod_dmdc.json` (`pod`+`arx`, Nx=0, Ni=0 — exactly DMDc) and
  `pod_persistence.json`. Full-pipeline smoke run (fit+test on all 8 prepared cases) launched.
- New `DataProcessing/grid.py`: derives `Data/cell_volumes.npy` (abs volumes; 206 source cells
  have inverted orientation), `Data/coordinates.npy` (cell centers) and `Data/grid_indices.npz`
  (cell→pixel map, image 104×206, 99.6% coverage) from `data/grid.vtu`. Referenced from
  `Data/metadata.json`; this unlocks the q'/FTF metrics in `Experiments/evaluation.py`
  (heat_release now true in the new configs) and the DeepONet trunk + CNN track.
  Files are gitignored — regenerate with `python -m DataProcessing.grid`.
- Tests: 2 new in `tests/test_grid.py`; suite green (50 total).

## 2026-09-21 — step 2: 0-D flame-response baseline
- `DataProcessing/qprime.py` precomputes q(t) = Σ Q_c·V_c per case into `Data/Qseries/` (values
  physically sensible: q̄ ≈ 1.15 W, f10 swings ≫ f40 — low-pass FTF; steps settle high).
- `Baselines/ZeroD/` (MLP + GRU: window of phi' taps → q'/q0) and `Experiments/zerod.py` runner
  (`fit`/`test`/`run`, same run-directory layout; writes the same `metrics.json` fields and
  `<case>_Q.npz` files as `evaluation.py`, so it drops into the same results table).
- Configs `zerod_mlp.json`, `zerod_gru.json` (window 256 taps = 128 ms). Tests: 4 new
  (`tests/test_zerod.py`, incl. end-to-end learning of a synthetic linear flame response);
  suite 54/54 green.

- Smoke run on real data (`zerod_mlp`, CPU, ~40 epochs): q' rel-L2 0.0015–0.022 over the 6 test
  cases; FTF gain err 0.002–0.012 (f10), 0.08–0.13 (f40); phase err ≤ 2.3° (f10), ≤ 17° (f40).
  Strong 0-D bar — field models must beat this on q' to justify themselves.
  Run: `Experiments/Results/zerod_mlp_20260921T091428Z`.

## 2026-09-21 — smoke results: linear baselines
- DMDc (`pod_arx_20260921T090804Z`): end-to-end OK. Mean nRMSE 0.16–0.36 over the 6 test cases
  (T as low as 0.07), ~13 ms/step CPU, 4000-step steady-start rollouts stay bounded.
  (q' columns off for this run — configs now have heat_release on for future runs.)

## 2026-09-21 — step 3: DeepONet + field-model path
- `Baselines/OrderReduction/Identity.py`: no-op compressor (flatten/unflatten) — field-level
  models reuse the entire latent pipeline (run.py, validation_rollout, evaluate) unchanged.
  Cost: the latent cache in the run dir is a scaled full-field copy (~7.5 GB per run).
- `Baselines/Forecast/DL/deeponet.py`: DeepONet (branch = history at 2048 random sensor cells
  + phi window; trunk = normalized planar (x, z) cell centers → p=64 basis; per-field
  coefficients). Registered as model "deeponet" with compressor "identity";
  config `identity_deeponet.json` (Nx=4, Ni=4, batch 8, cuda).
- Tests: 4 new (`tests/test_deeponet.py`); suite 58/58 green. 2-epoch GPU smoke run launched.

## 2026-09-21 — step 4: Transolver + persistence bugfix
- `Baselines/Forecast/DL/transolver.py`: physics attention (soft assignment of cells to M=32
  learned slices, attention over slice tokens — linear in cells; simplification vs the paper:
  one slice assignment shared across heads). phi window enters as per-point channels.
  Registered as "transolver" with compressor "identity"; config `identity_transolver.json`.
  4 new tests; suite green.
- Bug found by the smoke run: `Constant` (persistence) had no `__init__`, so the run.py registry
  call crashed — persistence was never runnable end to end. Fixed + regression test; smoke rerun.
- DeepONet 2-epoch GPU smoke (`smoke_deeponet`): pipeline OK end to end — 4000-step rollouts
  finite, q'/FTF columns computed, 1.3 ms/step on the L40S. Accuracy poor as expected at
  2 epochs (mean nRMSE ~0.5); real training happens in the big runs.

## 2026-09-21 — step 5: MeshGraphNets
- `DataProcessing/grid.py` now also extracts true face-adjacency cell edges from grid.vtu
  (`Data/edges.npy`, 84,704 directed edges, ~4 per cell).
- `Baselines/Forecast/DL/meshgraphnet.py`: encode-process-decode with residual edge/node message
  passing (8 passes, dim 128); phi as global node channels; absolute scaled target (delta +
  noise injection deferred to the stabilization ablation). Registered as "meshgraphnet";
  config `identity_mgn.json` (batch 2, cuda). 4 tests incl. message-propagation check; green.

## 2026-09-21 — step 6: pushforward/unrolled-training ablation
- `LatentDataset(horizon=k)` yields per-sample forcing windows and targets for k rollout steps;
  `DLModel.compute_loss` unrolls with `unroll_grad="none"` (pushforward, detached restart —
  the default) or `"full"` (backprop through the rollout). Config: `"unroll_steps"` and
  `"unroll_grad"` under `model` in any neural config; classical models are guarded.
- 4 new tests (`tests/test_unroll.py`), no regressions elsewhere.

## 2026-09-21 — smoke results: Transolver + DMDc heat-release columns
- Transolver 2-epoch GPU smoke (`smoke_transolver`): pipeline OK, rollouts finite, 5–7 ms/step.
- DMDc re-evaluated with heat_release on (same trained run): q' rel-L2 0.03–0.21, FTF gain err
  0.50–0.89, phase err up to 31°. **First headline contrast:** the 0-D baseline beats the linear
  field ROM on q' by ~10× while DMDc's field nRMSE stays decent — fields ≠ the domain scalar.

## 2026-09-21 — smoke phase complete
- Persistence (post-fix): nRMSE 0.19–0.47, q' rel-L2 0.04–0.38. DMDc only modestly beats this
  frozen floor on fields.
- MGN 2-epoch GPU smoke (`smoke_mgn`): pipeline OK, rollouts finite, 15–21 ms/step.
- All 6 method-set entries validated end to end (66 tests green). Full training runs launched:
  DeepONet/Transolver/MGN, one per L40S; pod_lstm, pod_transformer, zerod_gru on CPU.
  Remaining queue: CNN/grid track, ae/vae compressor runs, multi-seed, big-run analysis.

## 2026-09-21 — full runs, first results
- DeepONet full run (`identity_deeponet_20260921T095631Z`, early-stopped): **worse than the
  linear baselines** — mean nRMSE 0.55–1.11, q' rel-L2 19–204. Single-step training → rollout
  drift; q' (a volume integral) amplifies field bias. Motivates the stabilization axis exactly
  as the literature predicts (MP-PDE, Unrolled-Training). Follow-up launched:
  `identity_deeponet_pf4.json` (pushforward, unroll_steps=4).
- Transolver full run: same pathology, stronger (mean nRMSE ~1.4, q' rel-L2 19–29).
  `identity_transolver_pf4.json` launched. The single-step vs pushforward pair for both
  operators is shaping into the paper's stabilization-ablation table.

## 2026-09-21 — stabilization axis extended: residual + noise injection
- DeepONet+pushforward(4) still diverged in the validation rollout (the divergence gate in
  run.py correctly refused to save it). Added the remaining standard stabilizers to `DLModel`,
  both config-driven: `residual` (network learns the state delta) and `noise_std`
  (training-time input-noise injection, the MeshGraphNets recipe). `step()` centralizes the
  transition so predict/rollout/eval all honor residual mode. 1 new test (5 in test_unroll).
- Launched `identity_deeponet_pf4res.json` (pushforward-4 + residual + noise 0.01) on GPU 0.
- pod_lstm and pod_transformer full runs finished on CPU; zerod_gru training.

## 2026-09-21 — full-run results so far
| model | field nRMSE (range over 6 test cases) | q' rel-L2 |
|---|---|---|
| DMDc | **0.16–0.36** | 0.03–0.21 |
| persistence | 0.19–0.47 | 0.04–0.38 |
| pod_transformer | 0.33–0.51 | 0.07–0.38, FTF gain err ~0.95 |
| pod_lstm | 0.39–0.81 | 0.15–0.43, FTF gain err ~0.9 |
| deeponet +pf4+residual+noise | 0.34–0.55 | 59–89 (!) |
| deeponet single-step | 0.55–1.11 | 19–204 |
| transolver single-step | ~1.4 | 19–29 |
| 0-D MLP (q' only) | — | **0.0015–0.022** |
| 0-D GRU (q' only) | — | 0.006–0.039; f40 FTF much worse than MLP (gain err ~0.65) |
- deeponet+pf4 (no residual/noise) hit the divergence gate and was refused; +residual+noise
  passes it with the best val rollout MSE so far (0.131) and competitive fields.
- **Finding:** decent field nRMSE can coexist with garbage q' — mix:Q nRMSE is 0.50 while q'
  rel-L2 is ~70: heat release concentrates at the flame front (huge σ), so a small scaled-space
  bias in Q integrates into a massive q' error. POD-decoded models are implicitly protected
  (linear projection); identity-space operators are not. Strong argument for q'/FTF as
  first-class benchmark metrics next to nRMSE.
- Nothing beats the linear ROM on fields yet; nothing approaches the 0-D baseline on q'.

## 2026-09-21 (evening) — MGN result + stabilizer sweep
- MGN full run (single-step, no stabilizers; early-stopped ~epoch 45): nRMSE 1.11–1.34,
  q' rel-L2 2.8–3.8 — drifts like the other operators, BUT best forcing response of any
  field model (FTF gain err 0.34–0.74 vs ~0.95 for latent models). Graph locality seems to
  help phi coupling.
- Launched the stabilizer recipe (residual + noise 0.01) for MGN (GPU 2) and Transolver
  (GPU 0). Transolver+pushforward still training on GPU 1.

Working order (each step: implement → pytest → 2-epoch smoke run → doc):
1. DMDc + persistence configs, smoke-tested. 2. 0-D flame-response baseline (q' from `mix:Q` +
cell volumes from grid.vtu). 3. DeepONet (adds a raw-field model path in `run.py`). 4. Transolver.
5. MeshGraphNets + pushforward ablation. 6. CNN/grid track. Then big runs.
