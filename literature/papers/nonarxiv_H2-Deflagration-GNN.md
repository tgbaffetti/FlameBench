# Predictions of realistic hydrogen deflagrations with graph neural networks (2026)

- **Authors / venue:** G. Covoni, V. Bisio, S. Rossin, M. Ruggiero, F. Montomoli, V. L. Tagarielli (Imperial College London / Baker Hughes / Sant'Anna Pisa). Int. J. Hydrogen Energy 225 (2026) 154343 (open access, CC BY).
- **Links:** https://doi.org/10.1016/j.ijhydene.2026.154343
- **Source read:** pdf (publisher version, supplied manually)
- **Thread:** 3 GNNs / 4 combustion

## Summary
Extends **MeshGraphNets** to coupled combustion–turbulence physics. **30 URANS simulations** (OpenFOAM, k-ω SST, premixed H2–air deflagration from ignition of gas clouds) in 3D domains with randomly shaped, positioned and oriented obstacles. The GNN predicts **increments** of pressure, temperature, velocity and extra combustion/turbulence variables between consecutive steps, on meshes of **~150k nodes** (one full graph per batch). Two sampling intervals are compared (Δt = 1 ms: ~2,100 training pairs; Δt = 0.1 ms: ~21,000). It generalises to unseen geometries and is up to **6,200x faster** than CFD. Error accumulates over the rollout, as expected; the finer Δt gives more steps and so more accumulation.

## Benchmark fields
| Field | Value |
|---|---|
| Input geometry | **Unstructured 3D CFD mesh as graph** (~150k nodes) |
| Exogenous conditioning | None time-varying; geometry and ignition set the case |
| Latent reduction | None |
| Prediction target | **Residual** (increment between consecutive steps) |
| Rollout & stability | AR rollout; error accumulation analysed vs Δt |
| Demonstrated scale | ~150k nodes, 30 simulations |
| Datasets / physics | URANS H2 deflagrations in congested industrial geometries |
| Code + license | Not stated in the extracted text |
| Integration effort | As MeshGraphNets |

## Relevance to FlameBench
**Positive evidence that MeshGraphNets works on reacting flows on unstructured meshes**, at a larger mesh (150k) than ours (21k), which counterbalances REALM's negative GNN results. It also shows the Δt trade-off (fewer, larger steps = less accumulation), which is relevant to choosing FlameBench's snapshot stride.
