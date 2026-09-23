# FlameBench literature review

Goal: collect models (and protocol ideas) to include in FlameBench — next-step forecasting of
forced-flame CFD snapshots on an unstructured mesh (21334 cells x 11 fields), driven by phi(t).

- `SCHEMA.md` — fields used in every card
- `papers/` — one card per paper (45 cards, all read from full text; 6 journal-only papers supplied manually)
- `NEEDS_MANUAL.md` — record of the manually retrieved papers
- Raw sources/PDFs are not in the repo

## Recommended model shortlist (ranked)

Ranking considers fit to the two hard constraints — **native unstructured mesh** and a **way to inject
phi(t)** — then evidence of performance and integration effort.

| # | Model | Family | Mesh-native | phi injection | Effort | Card |
|---|---|---|---|---|---|---|
| 1 | **DMDc** (= `pod-arx`, n_past=1, linear) | linear ROM | yes (POD) | native `B·u` | config only | `1409.6358_DMDc.md` |
| 2 | **DeepONet** (branch = history+phi, trunk = xyz; +POD-trunk variant) | operator | yes | native (branch) | low | `1910.03193_DeepONet.md` |
| 3 | **Transolver / Transolver++** | transformer operator | yes | per-point channels / token | low | `2402.02366_Transolver.md` |
| 4 | **Koopman AE with inputs** (forced KAE) | latent ROM | yes | `K z + B·phi(t), phi(t+1)` | low | `2602.05416_Forced-KAE-Coastal.md`, `1712.09707_Koopman-AE.md` |
| 5 | **MeshGraphNets** (+ noise injection) | GNN | yes | global node feature | medium | `2010.03409_MeshGraphNets.md` |
| 6 | **UPT** | latent transformer operator | yes | DiT modulation (built in) | medium | `2402.12365_UPT.md` |
| 7 | **GINO** (fixes `fno3d`) | FNO on mesh | yes (GNO enc/dec) | AdaIN + Fourier features (built in) | medium | `2309.00583_GINO.md` |
| 8 | **BSMS-GNN** | multi-scale GNN | yes | global node feature | medium | `2210.02573_BSMS-GNN.md` |
| 9 | **AROMA** | latent diffusion ROM | yes | DiT conditioning | medium-high | `2406.02176_AROMA.md` |
| 10 | GNOT | transformer operator | yes | native parameter vector | low-medium | `2302.14376_GNOT.md` |
| 11 | RO-NORM (window-to-window task) | operator | yes (LBO basis) | input *is* phi(t) | medium-high | `2409.05508_RO-NORM.md` |
| — | 0-D baselines: MLP / (PI-)LSTM / Dual-Path phi → q' | scalar | n/a | input | low | `2204.05234_MLP-Flame-Response.md`, `nonarxiv_Yadav2022_PI-LSTM-Flame-Dynamics.md`, `2409.05885_Dual-Path-Flame-Response.md` |

Already in the repo and supported by the literature: `pod-transformer`, `ae-transformer`
(≈ β-VAE-Transformer / UP-dROM), `lstm` (≈ POD-LSTM for LES combustion, same n_past=10),
`pod-arx/narx`. `fno3d`/`unet`: keep only via GINO or with PDEArena-style conditioning.

## Cross-cutting recommendations for the benchmark protocol

1. **Stabilisation as a benchmark axis** (MP-PDE, Unrolled-Training, SPF, ACDM, Wu 2023):
   add `unroll_steps` + `unroll_grad ∈ {full, none}` (pushforward = 2/none), noise injection, SPF
   to `train.py`; report every model with and without them.
2. **phi injection as an ablation** (PDEArena, GINO, UPT, MP-PDE, Control-Affine, UP-dROM):
   concatenation vs FiLM/AdaGN vs control-affine `B(z)·phi` vs cross-attention.
   `pod-arx` currently uses only `phi(t+1)`, not the phi history.
3. **Splits** (Dual-Path, MLP flame response): train on sine sweeps, test on **single-frequency
   harmonics** and on unseen amplitudes — the combustion community's standard.
4. **Metrics**: per-field VRMSE (The Well); nRMSE, cRMSE (integral/conserved quantities),
   bRMSE (inlet), fRMSE bands (PDEBench); correlation time < 0.8 (UPT, PDE-Refiner); one-step
   vs windowed rollout (The Well); **integrated heat release q'(t) and FTF/FDF gain+phase**
   (flame-response papers); timings, memory (combustion review §5.2).
5. **Short vs long horizon trade-off** (Controlled-Wake paper): AEs win short-term,
   POD wins long-term — report both.
6. **Positioning**: REALM (reacting flows, static conditions, irregular meshes — closest
   benchmark), The Well / PDEBench / BLASTNet (no forced combustion). FlameBench's gap =
   **exogenously forced flame, time-varying input, unstructured mesh**.

## Card index by thread

- **Operators (thread 2):** FNO, Geo-FNO, GINO, DeepONet, Transolver, Transolver++, GNOT, OFormer, UPT, CORAL, AROMA, RO-NORM
- **GNNs (thread 3):** MeshGraphNets, BSMS-GNN, EAGLE, MP-PDE, GNS-vs-NOs, H2-Deflagration-GNN
- **ROM + sequence (thread 1):** POD-DL-ROM, β-VAE-Transformer, Koopman-AE, DMDc, Forced-KAE-Coastal, Controlled-Wake-Latent-ROM, Control-Affine-AE-ROM, UP-dROM, POD-LSTM-LES-Combustion
- **Combustion / benchmarks (thread 4):** REALM, Combustion-Surrogate-Review, Dual-Path-Flame-Response, MLP-Flame-Response, Tathawadekar2024 (turbulent flame response), Yadav2022 (PI-LSTM), Liu2024 (forced sooting flames), Hybrid-NN-PDE-Reactive-Flows, PDEBench, PDEArena, The-Well, BLASTNet2, Wu2023 (AR turbulent combustion, SimVP), NARX-Thermoacoustic-RL
- **Probabilistic / rollout (threads 5-6):** ACDM, PDE-Refiner, Stochastic-PushForward, Unrolled-Training

## Open items
- `LICENSE_TBD` in cards: code licenses not yet checked (GitHub API not queried; `gh` CLI not installed).
- Some "Demonstrated scale" entries give an order of magnitude only; check the paper tables before citing numbers.
