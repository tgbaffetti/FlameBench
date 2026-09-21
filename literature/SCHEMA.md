# Paper card schema

Every card in `lit_review/papers/<id>.md` uses these fields so candidate models
are comparable with each other and with the FlameBench contract: next-step
prediction of snapshots on an unstructured mesh (`Ncells=21334`, `Nf=11`) from
`n_past` snapshots + the `phi` history + `phi(t+1)`, via
`BaseForecastModel.compute_loss(x_window, phi_window)`.

```
# <Title> (<Year>)

- **Authors / venue:** ...
- **Links:** arXiv / DOI / code
- **Source read:** tex (arXiv e-print) | pdf | abstract only (-> NEEDS_MANUAL)
- **Thread:** 1 ROM+sequence | 2 neural operators | 3 GNNs | 4 combustion/benchmarks | 5 conditioning/probabilistic | 6 rollout/evaluation

## Summary
2-3 sentences: what it proposes.

## Benchmark fields
| Field | Value |
|---|---|
| Input geometry | unstructured mesh / regular grid / point cloud / graph; if grid-only, what interpolation it would need |
| Exogenous conditioning | how it ingests a scalar signal like phi(t); if none: "none - would need X" |
| Latent reduction | POD / AE / none / other |
| Prediction target | absolute state / residual (delta) / other |
| Rollout & stability | autoregressive? tricks (pushforward, noise injection, multi-step loss, refinement) |
| Demonstrated scale | nodes/cells, fields, steps vs our 21334 x 11 |
| Datasets / physics | what it was tested on |
| Code + license | URL, license |
| Integration effort | low / medium / high + why, against `compute_loss(x_window, phi_window)` |

## Relevance to FlameBench
Why include it (or not): as a baseline, as a candidate model, or only as a
methodological reference. Reusable hyperparameters/metrics.
```
