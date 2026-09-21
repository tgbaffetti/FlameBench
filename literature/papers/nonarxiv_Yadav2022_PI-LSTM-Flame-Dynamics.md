# Physics-informed recurrent neural networks for linear and nonlinear flame dynamics (2022)

- **Authors / venue:** V. Yadav, M. Casel, A. Ghani (TU Berlin). Proceedings of the Combustion Institute 39 (2023) 1597–1606.
- **Links:** https://www.sciencedirect.com/science/article/pii/S1540748922003145
- **Source read:** pdf (publisher version, supplied manually)
- **Thread:** 4 combustion / 1 sequence models

## Summary
**LSTMs** predict the heat-release-rate response of a premixed laminar flame to inlet velocity perturbations, using u' and q' time series from **DNS**. The signal length (and thus simulation cost) is comparable to classical System Identification. Purely data-driven LSTMs trained on decreasing data lengths all recover the **Flame Transfer Function (FTF)**. Adding a physical constraint — the **low-frequency limit of the FTF for perfectly premixed flames** (a PI-LSTM) — reduces the data needed. The PI-LSTM reproduces linear and nonlinear FTFs up to **50% forcing amplitude from one 100 ms simulation**.

## Benchmark fields
| Field | Value |
|---|---|
| Input geometry | None — scalar u'(t) → q'(t) |
| Exogenous conditioning | The input *is* the forcing (our phi) |
| Latent reduction | — |
| Prediction target | q'(t) |
| Rollout & stability | Time-series regression; evaluated through the FTF/FDF |
| Demonstrated scale | One 100 ms DNS run |
| Datasets / physics | Premixed laminar flame, broadband forcing |
| Code + license | Not stated |
| Integration effort | Low as a 0-D baseline |

## Relevance to FlameBench
A second **0-D baseline family** (LSTM / PI-LSTM, phi → q'), next to the MLP (Tathawadekar 2021) and Dual-Path. Two useful ideas: (1) **physics constraints as cheap priors**, e.g. the FTF low-frequency limit (|FTF| → 1 as f → 0) as a sanity metric or regulariser on the integrated heat release of our field models; (2) the **data-length study** — how much forced trajectory a model needs — is a natural FlameBench experiment (train on a truncated sine sweep).
