# A robust autoregressive long-term spatiotemporal forecasting framework for surrogate-based turbulent combustion modeling via deep learning (2023)

- **Authors / venue:** S. Wu, H. Wang, K. H. Luo (Tsinghua / Zhejiang Univ. / UCL). Energy and AI 15 (2024) 100333 (open access, CC BY).
- **Links:** https://doi.org/10.1016/j.egyai.2023.100333
- **Source read:** pdf (publisher version, supplied manually)
- **Thread:** 4 combustion / 6 rollout

## Summary
A real-time surrogate for **turbulent combustion fields with detailed chemistry**. The authors compare spatiotemporal forecasting networks — **SimVP** (encoder–translator–decoder CNN), U-Net, **ConvLSTM** (4 layers) and **FNO** — for autoregressive next-frame prediction. SimVP is the best. To counter the distribution shift in long AR rollouts, they propose **unrolled training** and **noise-injection training**. They study unroll length, noise amplitude, architecture and training-set size. Datasets: an H2 **cavity combustor** (LES 600x160, resampled to a uniform **128x64** grid; several geometries/injector positions) and the **Cabra burner** (reactingFoam LES, 9-species H2 mechanism; 64x64 planes sampled from a 3D polar grid, ~1000 snapshots). The trained model extrapolates to new cases with spatially and temporally consistent long rollouts.

## Benchmark fields
| Field | Value |
|---|---|
| Input geometry | **Regular grid** (the CFD fields are resampled to 128x64 / 64x64) |
| Exogenous conditioning | None (case parameters differ across trajectories; no time-varying input) |
| Latent reduction | None (CNN latent) |
| Prediction target | Next frame(s) from several preceding frames |
| Rollout & stability | **Unrolled training + noise injection** — the core contribution, with ablations |
| Demonstrated scale | 128x64 and 64x64 grids, multi-species fields, ~1000 snapshots per case |
| Datasets / physics | H2 cavity combustor; Cabra lifted flame in vitiated co-flow |
| Code + license | Paper CC BY; code availability not stated in the extracted text |
| Integration effort | Low for the **training tricks**; medium for SimVP (grid-only, needs interpolation like our `unet`) |

## Relevance to FlameBench
**The most direct combustion precedent for AR field forecasting with stabilisation.** Two takeaways: (1) include **SimVP** as a strong grid-based video-prediction baseline next to U-Net/FNO, if we keep a grid-interpolation track; (2) reuse its ablation design (unroll length x noise amplitude x data size) for the FlameBench stabilisation axis. Like us, it resamples unstructured CFD onto a regular grid for the CNN models, which confirms that the "interpolate for `fno3d`/`unet`" route is accepted practice.
