# Linear and nonlinear flame response prediction of turbulent flames using neural network models (2024)

- **Authors / venue:** N. Tathawadekar, A. Ösün, A. J. Eder, C. F. Silva, N. Thuerey (TU Munich). International Journal of Spray and Combustion Dynamics 16(3), 93–103 (2024).
- **Links:** https://journals.sagepub.com/doi/full/10.1177/17568277241262641
- **Source read:** pdf (publisher version, supplied manually)
- **Thread:** 4 combustion

## Summary
Extends the laminar MLP flame-response approach (2204.05234) to **turbulent flames**, where **combustion noise** makes the u' → q' mapping harder to learn. Data: **LES of an academic combustor acoustically forced with broadband signals**. Part 1: a neural network learns and **interpolates the linear flame response (FTF) across thermal conditions**. Part 2: with sufficiently large forcing amplitudes, it infers the **nonlinear response (FDF)**.

## Benchmark fields
| Field | Value |
|---|---|
| Input geometry | None — scalar u'(t) → q'(t) |
| Exogenous conditioning | Forcing signal as input; thermal condition as an extra parameter (interpolation) |
| Latent reduction | — |
| Prediction target | q'(t) → FTF/FDF |
| Rollout & stability | Evaluated through the FTF/FDF |
| Demonstrated scale | LES time series with broadband forcing |
| Datasets / physics | Turbulent academic combustor (LES) |
| Code + license | Not stated |
| Integration effort | Low as a 0-D baseline |

## Relevance to FlameBench
Shows the **forced-flame + learned-response paradigm transfers from laminar to turbulent** flames, with **interpolation across operating conditions** — a natural FlameBench extension if we add more amplitudes/conditions (e.g. train A02+A04, test A03). Also flags **combustion noise** as a key difficulty for turbulent cases; not a concern for our laminar data, but worth mentioning as future work.
