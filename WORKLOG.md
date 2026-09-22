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
- Transolver + residual + noise **failed**: passed the fit gate with a finite-but-absurd val
  rollout MSE (8e20), then went nonfinite in the test rollout. Two lessons: (a) residual+noise
  alone does not stabilize Transolver, (b) the divergence gate should bound the score, not just
  check finiteness (protocol TODO). Full recipe launched instead:
  `identity_transolver_pf4res.json` (pushforward-4 + residual + noise).
- New metrics live (commit earlier): horizon-binned nRMSE + `restart_every` windowed protocol;
  CPU re-evaluation pass over 5 saved models running (free rollout + one-step each).
- Transolver + pushforward (alone) finished but rolled out to garbage: nRMSE ~12.8, q' rel-L2
  2200--3200, and again a finite-but-huge val score (233) slipped the gate. Transolver is
  0-for-3 single-stabilizer recipes; the combined pf4+residual+noise run is its last shot.
- GPU 1 re-evaluation pass launched: horizon curves + one-step protocol for the three
  single-step operator runs (deeponet/transolver/mgn).
- MGN + residual + noise: **worse** than plain MGN in rollout (nRMSE ~4.3 vs ~1.2, q' in the
  hundreds) despite similar val score. Reading: delta targets integrate bias linearly over
  4000 steps; absolute targets get pulled back to the manifold. The MGN paper's recipe was
  tuned for far shorter rollouts. Launched `identity_mgn_pf4.json` (pushforward, absolute
  target) to fill the ablation grid.

## 2026-09-21 — one-step vs rollout: the headline plot
New metrics on the single-step operator runs (sine_f10_A03, mean nRMSE):
| model | one-step | rollout 1-10 | 11-100 | 101-1000 | 1001+ |
|---|---|---|---|---|---|
| Transolver | **0.018** | 0.23 | 0.41 | 1.57 | 1.26 |
| MGN | 0.020 | 0.23 | 0.36 | 0.99 | 1.30 |
| DeepONet | 0.102 | 0.27 | 0.18 | 0.53 | 0.57 |
- **Local accuracy anti-correlates with rollout stability**: the best one-step model
  (Transolver) is the worst in free rollout (70x degradation); the worst one-step model
  (DeepONet) degrades most gracefully. All three hold through ~step 100, then split.
  This inversion + the horizon bins is the paper's central stability figure.
- ROM baselines (same case): DMDc one-step 0.089, horizon flat [0.24|0.10|0.23|0.24] —
  bounded-stable at every horizon. persistence/pod_lstm/pod_transformer one-step all ~0.090:
  **the rank-16 POD compression floor dominates latent models' local error** (dynamics models
  are locally indistinguishable; identity-space operators cut below the floor: 0.018-0.027).
  Rank ablation is an obvious paper lever.
- deeponet+pf4+res+noise: one-step 0.027, rollout [0.21|0.15|0.35|0.47] — best neural field
  model on fields at every horizon, yet q' still garbage. The fields-vs-integral decoupling
  is systematic, not an artifact of bad training.
- Transolver + pf4 + residual + noise: fails too — horizon bins [0.23|0.27|10.7|7.7e6]: healthy
  to ~step 100, then explodes. **Transolver is 0-for-4 stabilizer recipes** at this rollout
  length; the same recipe rescued DeepONet. Hypothesis for the paper: DeepONet's global
  trunk-basis output regularizes rollouts where Transolver's per-point head does not.
  (Gate TODO again: val score 35.7, finite, slipped through.)

## 2026-09-22 — final benchmark matrix locked (Gianmarco's spec) + overnight runs
Matrix: {POD+ARX, POD+NARX, POD+LSTM, POD+Transformer, DeepONet, Transolver} x
{1-step eval, full-horizon AR eval, flexible k-step training selected on 500-step validation
windows}. MeshGraphNets **dropped** (its pf4 run was stopped). CAE/VAE compressors replace POD
in a second pass once Carlo's branch is fixed (his AE.py imports a missing Baselines/losses.py;
his Compressor contract also moved to image inputs — reconcile before merging; his POD is now
full randomized SVD, so POD baselines need a re-run after the merge).
- New: NARX (bilinear/control-affine ridge: linear + squares + state x phi terms, Ni=4);
  `validation_window` (windowed validation rollout = selection objective for k-step models);
  `max_steps` eval option (unused for now — full horizon = whole test trajectory; NB our
  prepared tests are 4001 steps / 2.0 s, not the 2k Tommaso mentioned — check with him).
- Carlo does NOT have k-step training on his branch (checked); ours (unroll_steps) is used.
- Overnight queues (each config: fit -> full-AR test -> one-step test, via
  Experiments/scripts/queue_runner.py): CPU: pod_arx, pod_narx. GPU0: pod_lstm_k{2,8,32},
  pod_transformer_k{2,8,32}. GPU1: identity_deeponet_k{4,16}. GPU2: identity_transolver_k{4,16}.
  Operator k-runs are plain (no residual/noise) so the k axis stays clean; epochs 50/patience 10.


## 2026-09-22 (morning) — overnight matrix results
Table below is case sine_f10_A03 (mean nRMSE unless noted); full table via scratchpad collect.py.

| run | k | 1-step | full-AR | q' relL2 | FTF gain err | horizon 1-10\|11-100\|101-1k\|1k+ |
|---|---|---|---|---|---|---|
| POD+NARX (a=10) | 1 | -- | **0.136** | **0.0287** | **0.030** | 0.24\|0.06\|0.12\|0.14 |
| POD+ARX (Nx=9) | 1 | 0.092 | 0.178 | 0.064 | 0.217 | 0.24\|0.10\|0.17\|0.17 |
| DMDc (Nx=0) | 1 | 0.089 | 0.236 | 0.142 | 0.518 | 0.24\|0.10\|0.23\|0.24 |
| persistence | 1 | 0.090 | 0.315 | 0.225 | 1.00 | 0.24\|0.15\|0.31\|0.31 |
| POD+LSTM | 1/2/8/32 | 0.092 | 0.620/0.477/0.388/**0.365** | 0.37/0.28/0.24/0.30 | ~0.92-0.98 | k32: 0.24\|0.16\|0.32\|0.40 |
| POD+Transformer | 1/2/8/32 | 0.092 | 0.376/0.374/0.391/**0.335** | 0.25/0.31/0.28/0.23 | ~0.76-0.98 | k32: 0.24\|0.13\|0.32\|0.34 |
| DeepONet | 1/16 | 0.102/0.253 | 0.545/**0.318** | 23.1/**4.38** | 0.33/0.517 | k16: 0.32\|0.18\|0.31\|0.32 |
| Transolver | 1/4 | **0.018**/0.024 | 1.411/12.78 | 19.2/2291 | 1.2/34 | k4: 0.23\|9.91\|12.9\|12.9 |
| 0-D MLP / GRU | -- | -- | -- | **0.004/0.008** | **0.002/0.007** | -- |

Findings:
1. **NARX (control-affine) is the best field model** on this case and the first with a *usable*
   FTF (gain err 0.03 vs 0.22 ARX, ~0.95 neural). The bilinear z x phi term is what lets a ROM
   respond to forcing. BUT it diverges on the two A05 (amplitude 0.5) cases and is poor on
   sine_f40_A05/f10_A05 — validation only sees sweep amplitudes 0.2/0.4, so amplitude
   extrapolation is unguarded. Wider alpha sweep (30..1000) running.
2. **k-step training monotonically helps LSTM** (0.620 -> 0.365 AR) and helps DeepONet a lot
   (0.545 -> 0.318 AR, q' 23 -> 4.4); Transformer is flat/noisy. It does NOT fix FTF gain
   (~0.95 for all latent models): stability and forcing-response are separate failures.
3. **Transolver stays broken** under every recipe (now 0-for-5). Best one-step of all models
   (0.018) and near-worst rollout — the cleanest instance of the one-step/rollout inversion.
4. POD rank-16 one-step floor (0.092) is identical for persistence/ARX/LSTM/Transformer:
   at one step the compressor, not the dynamics model, is the bottleneck.
- Protocol C for closed-form models (ARX/NARX) is read as: select ridge alpha on the 500-step
  windowed validation rollout. Sweeps running for both.
- Fixed: bounded divergence gate (rejected NARX a=0.1/1.0 at 1e24/1e14 — the old finiteness
  check would have kept them); evaluate() and test() now record a diverged case
  (status/diverged_at_step) and continue instead of aborting the run.


## 2026-09-22 (late morning) — matrix complete (POD pass), consolidated results
Case sine_f10_A03 unless noted; `--` = protocol not applicable.

| model | k | 1-step | full-AR | q' relL2 | FTF gain err | horizon 1-10\|11-100\|101-1k\|1k+ |
|---|---|---|---|---|---|---|
| POD+NARX (a=30) | 1 | 0.092 | **0.145** | **0.021** | **0.016** | 0.24\|0.06\|0.12\|0.15 |
| POD+ARX (a=1, protocol C) | 1 | 0.092 | 0.171 | 0.054 | 0.181 | 0.24\|0.10\|0.16\|0.17 |
| POD+ARX (a=1e-4) | 1 | 0.092 | 0.178 | 0.064 | 0.217 | 0.24\|0.10\|0.17\|0.17 |
| persistence | 1 | 0.090 | 0.315 | 0.225 | 1.00 | 0.24\|0.15\|0.31\|0.31 |
| POD+LSTM | 1/2/8/32 | 0.092 | 0.620/0.477/0.388/**0.365** | 0.37/0.28/0.24/0.30 | 0.92-0.98 | k32: 0.24\|0.16\|0.32\|0.40 |
| POD+Transformer | 1/2/8/32 | 0.092 | 0.376/0.374/0.391/**0.335** | 0.25/0.31/0.28/0.23 | 0.76-0.98 | k32: 0.24\|0.13\|0.32\|0.34 |
| DeepONet | 1/4/16 | 0.102/0.127/0.253 | 0.545/diverged@~500/**0.318** | 23.1/--/4.38 | 0.33/--/0.52 | k16: 0.32\|0.18\|0.31\|0.32 |
| Transolver | 1/4/16 | **0.018**/0.024/0.038 | 1.411/12.78/**0.519** | 19.2/2291/12.5 | 1.2/34/3.1 | k16: 0.22\|0.08\|0.33\|0.65 |
| 0-D MLP / GRU | -- | -- | -- | **0.004/0.008** | **0.002/0.007** | -- |

Consolidated findings (POD pass; CAE/VAE pass pending Carlo):
1. **Only the bilinear term buys forcing response.** NARX gain err 0.016 / phase 3 deg vs ARX
   0.18, and 0.76-0.98 for every neural latent model (i.e. they replay a mean cycle and ignore
   phi). This is the benchmark's central diagnostic and it separates methods that nRMSE does not.
2. **NARX is also the most fragile**: excellent on A03/f10, diverges or degrades on the
   amplitude-0.5 cases. Validation sees only sweep amplitudes 0.2/0.4, so the instability is
   invisible to model selection — an honest limitation of the standard protocol. Separate
   bilinear ridge (`cross_alpha`) implemented; sweep running.
3. **ARX (a tuned on the 500-step windowed rollout) is the only model good on all six cases**:
   AR 0.12-0.29, q' 2.7-9.3%, gain err 0.15-0.72, no divergence. The bar to beat.
4. **Large-k unrolled training is what rescues neural operators, and only at large k**:
   DeepONet 0.545 (k=1) -> diverges at ~step 500 (k=4) -> 0.318 (k=16); Transolver
   1.411 -> 12.78 -> 0.519. Non-monotonic in k, so k must be tuned, not assumed.
5. **One-step accuracy is anti-correlated with rollout quality across families**: Transolver k=1
   has the best one-step of all (0.018) and near-worst AR (1.411); DeepONet k=16 has the worst
   one-step of the operators (0.253) and the best operator AR (0.318).
6. **POD rank-16 one-step floor = 0.092**, identical for persistence/ARX/NARX/LSTM/Transformer:
   at one step the compressor is the bottleneck, not the dynamics. Motivates the rank ablation
   and Carlo's Bench1 (compressor comparison) directly from our own numbers.
7. k selected on the 500-step windowed validation is right for LSTM (k=32) and DeepONet
   (rejects k=4 at 2.6e61), but **wrong for Transformer** (picks k=8; k=32 is better on test).
- Running: NARX cross_alpha sweep (CPU); seeds 43/44 for pod_lstm_k32, pod_transformer_k32
  (GPU0) and identity_deeponet_k16 (GPU1).
- Protocol-C selection table complete (validation rollout, window=500; k=1 runs re-scored by
  replaying the pickled model on the same cached validation latents, so the same code path and
  data produce every number — they are comparable):

| model | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | picks | best on test? |
|---|---|---|---|---|---|---|---|---|
| POD+LSTM | 3828 | 2467 | -- | 2611 | -- | **2406** | k=32 | yes (AR 0.365) |
| POD+Transformer | 1173 | 1062 | -- | **1016** | -- | 1192 | k=8 | **no** (k=32 is 0.335 vs 0.391) |
| DeepONet | 0.341 | -- | 2.6e61 | -- | **0.109** | -- | k=16 | yes (AR 0.318) |
| Transolver | 2.01 | -- | 219 | -- | **0.067** | -- | k=16 | yes (AR 0.519) |

  Selection decisively rejects the catastrophic k=4 operator runs. The apparent Transformer
  inversion (picks k=8, k=32 better on test) is NOT significant: the gap is inside the 3-seed
  spread measured later the same day — see the seed-statistics entry.


## 2026-09-22 (midday) — seed robustness: the k=16 operator result is fragile
- **DeepONet k=16 fails to train on 2 of 3 seeds**: nonfinite training loss at epoch 15 (seed 43)
  and epoch 22 (seed 44); only seed 42 completed. So large-k unrolled training is unstable
  *during training*, not only at rollout time, and the headline "DeepONet k=16 AR 0.318" is a
  single-seed number until this is resolved. Testing the standard mitigation (lr 1e-3 -> 3e-4)
  over seeds 42/43/44; if it holds, the paper reports k=16 with the lower lr and states that
  the default lr diverges in 2/3 seeds.
- Every published k-step number in this benchmark needs the same treatment; single-seed
  operator results are not reportable.
- Housekeeping: reclaimed ~70 GB by deleting latent caches of finished/dead runs (they are
  regenerable and `test` does not use them). Kept the caches of the six selected configs.
  Disk 93% -> 306 GB free.


## 2026-09-22 (midday) — NARX bilinear ridge: stability bought, mostly
`cross_alpha` (separate ridge on the z x phi block, alpha fixed at 30) selected 3000 on the
500-step windowed validation. Per case, free rollout:

| case | NARX global a=30 | NARX cross_alpha=3000 | ARX a=1 |
|---|---|---|---|
| sine_f10_A03 | 0.145 (gain .016) | 0.147 (gain **.015**) | 0.171 (gain .181) |
| sine_f10_A05 | 0.885 | **0.419** (gain .316) | 0.291 (gain .146) |
| sine_f40_A03 | 2.14 | **0.146** (gain **.072**) | 0.142 (gain .72) |
| sine_f40_A05 | 2.8e21 | 1.1e6 (still diverges) | 0.196 (gain .65) |
| step_A03 | 0.135 | **0.101** | 0.122 |
| step_A05 | diverged@878 | **0.218** | 0.284 |

- Shrinking only the bilinear block fixes 2 of the 3 broken cases and *improves* the good ones;
  the highest-frequency/highest-amplitude case (f40_A05) still blows up. So the bilinear term is
  simultaneously the source of the forcing response and of the instability, and a separate
  penalty trades them off without fully resolving the hardest corner.
- Where NARX is stable it dominates ARX on FTF gain (0.015-0.32 vs 0.15-0.72) at comparable or
  better nRMSE. The paper can report NARX as "best forcing response, conditional on stability"
  with ARX as the robust baseline — an honest and interesting pairing.
- cross_alpha=30000 completes the curve: f40_A05 comes down from 1.1e6 to 11.0 (tamed, still
  unusable) while the FTF advantage is destroyed — f40_A03 gain err 0.072 -> 1.418, f10_A03
  0.015 -> 0.089. **No cross_alpha makes NARX stable on all six cases: the shrinkage that
  stabilizes the hard corner is exactly the shrinkage that removes the forcing response.**
  Validation picks 3000, the sensible operating point, and that is the number to report.
  This is a cleaner claim than a fix would have been — the bilinear gain is the mechanism for
  both effects, so they cannot be separated by regularization alone.

| cross_alpha | f10_A03 gain err | f40_A03 gain err | f40_A05 AR | cases stable |
|---|---|---|---|---|
| 30 (global) | 0.016 | 4.97 | 2.8e21 | 3/6 |
| 3000 | **0.015** | **0.072** | 1.1e6 | 5/6 |
| 30000 | 0.089 | 1.418 | 11.0 | 5/6 (f40_A05 tamed, not usable) |


## 2026-09-22 (afternoon) — 3-seed statistics for the latent k=32 models
Both train reliably on seeds 42/43/44 (unlike DeepONet k=16). Mean +/- sd over 3 seeds:

| case | LSTM k32 AR | LSTM q' | LSTM gain err | Transf. k32 AR | Transf. q' | Transf. gain err |
|---|---|---|---|---|---|---|
| sine_f10_A03 | 0.387+/-.044 | 0.266+/-.035 | 0.871+/-.15 | **0.320+/-.062** | 0.211+/-.047 | 0.846+/-.23 |
| sine_f10_A05 | 0.514+/-.036 | 0.367+/-.025 | 0.982+/-.027 | **0.439+/-.042** | 0.313+/-.040 | 0.889+/-.080 |
| sine_f40_A03 | 0.324+/-.038 | 0.143+/-.079 | 0.898+/-.028 | **0.240+/-.044** | 0.043+/-.011 | 0.832+/-.12 |
| sine_f40_A05 | 0.396+/-.043 | 0.158+/-.083 | 0.924+/-.009 | **0.286+/-.011** | 0.056+/-.010 | 0.821+/-.15 |
| step_A03 | 0.377+/-.078 | 0.174+/-.076 | -- | **0.224+/-.10** | 0.173+/-.094 | -- |
| step_A05 | 0.498+/-.059 | 0.329+/-.067 | -- | **0.371+/-.097** | 0.279+/-.11 | -- |

1. **Transformer k=32 beats LSTM k=32 on every case**, and by more than the seed spread on
   f40_A03/f40_A05. It is the better latent dynamics model here.
2. **Seed spread is large: +/-0.04-0.10 on AR and +/-0.08-0.23 on FTF gain err.** Any
   single-seed difference below ~0.1 nRMSE is noise.
3. **CORRECTION to the protocol-C claim.** I previously wrote that windowed-validation k
   selection "picks the wrong k for Transformer" (k=8 at 0.391 vs k=32 at 0.335, single seed).
   That 0.056 gap is inside the +/-0.062 seed spread, so the claim is not supported: the
   selection is not demonstrably wrong for any family. Remove it from the paper narrative and
   from the earlier selection-table commentary.
4. The "neural latent models ignore the forcing" finding **survives seeding**: gain err stays
   0.82-0.98 with modest spread for both models, versus 0.015-0.32 for NARX and 0.15-0.72 ARX.
- Still open: DeepONet k=16 at lr 3e-4 across 3 seeds (running); Transolver k=16 seeds;
  NARX/ARX are deterministic given the data so seeding is not needed for them.


Working order (each step: implement → pytest → 2-epoch smoke run → doc):
1. DMDc + persistence configs, smoke-tested. 2. 0-D flame-response baseline (q' from `mix:Q` +
cell volumes from grid.vtu). 3. DeepONet (adds a raw-field model path in `run.py`). 4. Transolver.
5. MeshGraphNets + pushforward ablation. 6. CNN/grid track. Then big runs.
