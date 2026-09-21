# Reconstructing soot fields in acoustically forced laminar sooting flames using physics-informed machine learning models (2024)

- **Authors / venue:** S. Liu, H. Wang, Z. Sun, K. K. Foo, G. J. Nathan, X. Dong, M. J. Evans, B. B. Dally, K. Luo, J. Fan. Proceedings of the Combustion Institute 40 (2024) 105314.
- **Links:** https://www.sciencedirect.com/science/article/abs/pii/S154074892400124X
- **Source read:** pdf (publisher version, supplied manually)
- **Thread:** 4 combustion / 2 operators

## Summary
Physics-informed ML on **acoustically forced, time-varying laminar sooting flames**, measured with planar laser diagnostics. (1) A **PINN** reconstructs velocity and temperature in regions that soot scattering makes inaccessible to lasers, constrained by momentum and energy conservation. (2) An **autoencoder-based DeepONet (AE-DeepONet)** predicts the planar **soot volume fraction** from temperature and OH images, trained with a hybrid physics-informed + data loss. It outperforms purely data-driven models.

## Benchmark fields
| Field | Value |
|---|---|
| Input geometry | Planar images (regular grid) |
| Exogenous conditioning | Acoustic forcing phase/frequency implicit in the time-varying data; no explicit forcing input to the model |
| Latent reduction | Autoencoder (AE-DeepONet) |
| Prediction target | Field reconstruction (soot from T and OH) — **not forecasting** |
| Rollout & stability | Not applicable |
| Demonstrated scale | Experimental planar fields |
| Datasets / physics | Acoustically forced laminar sooting flames (experiments) |
| Code + license | Not stated |
| Integration effort | N/A (a different task) |

## Relevance to FlameBench
**Same physical setting (forced laminar flames), different task** (cross-field reconstruction, not temporal forecasting). Cite it for the forced-flame context, and as a precedent for **AE + DeepONet** on flame fields, which supports the DeepONet-with-learned-latent variant in our shortlist. Idea for an auxiliary FlameBench task: predict hard-to-measure fields (e.g. species) from easy ones (T, velocity) on the same mesh.
