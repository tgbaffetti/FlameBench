# Controlling thermoacoustic instability of a laminar premixed flame with deep reinforcement learning and neural autoregressive models (2024)

- **Authors / venue:** J. C. Giraldo Delgado, K. Alhazmi, I. Gorbatenko, D. A. Lacoste, S. M. Sarathy (KAUST). Proceedings of the Combustion Institute (2024).
- **Links:** https://www.sciencedirect.com/science/article/abs/pii/S1540748924000336
- **Source read:** pdf (revised manuscript PROCI-D-23-00397 from the institutional repository, supplied manually)
- **Thread:** 4 combustion / 1 (NARX)

## Summary
Fully data-driven modelling and control of a **laminar premixed flame with a thermoacoustic instability at 166 Hz**. A **neural-network NARX model** is trained on experimental data and predicts well in closed loop. It then serves as the environment for **offline reinforcement learning**, which tunes the gain and delay of a phase-shift controller. The suggested parameters fall in the range where the instability is reduced.

## Benchmark fields
| Field | Value |
|---|---|
| Input geometry | None — scalar experimental time series (pressure, chemiluminescence) |
| Exogenous conditioning | **NARX exogenous input = actuation/control signal** |
| Latent reduction | — |
| Prediction target | Next values of the measured signals |
| Rollout & stability | Closed-loop (free-run) prediction validated, since it is used as an RL environment |
| Demonstrated scale | Experimental time series |
| Datasets / physics | Laminar premixed flame, self-excited instability |
| Code + license | Not stated |
| Integration effort | Low (0-D NARX) |

## Relevance to FlameBench
Evidence that **NARX models are the combustion community's working tool** for forced/controlled flames. That supports keeping `pod-arx`/`pod-narx` as first-class baselines and evaluating them in **free-run**, not only one-step. It also points to a downstream use case: FlameBench surrogates as **control/RL environments**.
