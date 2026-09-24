# ROM-FlameBench

A small benchmark for forecasting forced reacting flows. Work is on branch `carlo`.
The original scripts are preserved in `legacy/`; the new pipeline does not import them.

For an implementation review, see [each method’s pseudocode](docs/pseudocode.md):
compressors, forecasters, training losses, history indexing and recursive rollout.

## Start with the notebook

Open [Walkthrough.ipynb](Walkthrough.ipynb) for a step-by-step POD–ARX example:
**data preparation → compressor → forecast → evaluation**. It uses direct calls to
the same classes as the experiment scripts, with short explanations of every stage.
Launch Jupyter from the repository root with the project environment selected:

```bash
pip install -e '.[notebook]'
jupyter lab Walkthrough.ipynb
```

Snapshot zero is confirmed to be the initial steady state. **Cell volumes remain a
TODO**: the notebook and example configs run field-only evaluation for now.

## Layout

```text
Data/
  Raw/                  original compressed archives (not committed)
  Images/
    Training/           prepared images and phi, as separate .npy files
    Test/               prepared images and phi, as separate .npy files
  Training/, Test/      preserved original cell arrays
  Metadata/             coordinates and physical cell volumes, when supplied
  metadata.json         field names, units, forcing metadata and file paths
DataProcessing/         disk-backed datasets, conversion, scaling, latent cache
Baselines/
  Forecast/
    Model.py            fit, predict, test and HPO interface
    Classical/          ARX, NARX, DMDc, Operator Inference (OpInf) and latent persistence
    DL/                 shared training loop; GRU, LSTM, CNN, Transformer; FNO-2D, DeepONet, Transolver (full field)
  OrderReduction/
    Compressor.py       fit, encode, decode interface
    Linear/             legacy randomized POD
    DL/                 dense AE/VAE, convolutional CAE, visual-attention ViTAE
Experiments/
  Configs/              plain JSON experiment settings
  Results/              per-run checkpoints, metrics and logs (not committed)
  scripts/              local multi-GPU and Slurm launch examples
utils.py                seed, JSON output and environment metadata
```

Use `fit(...)` for learning. PyTorch reserves `train(...)` for switching a neural
module between training and evaluation modes. Compressors and forecasters remain
separate: changing POD to VAE does not require another forecasting implementation.
AR/ARMA-style extensions belong in `Forecast/Classical/`; ARX is the first baseline
because inlet forcing is an explicit input.

## Install and run

Python 3.11 is recommended for the servers. Install a suitable PyTorch build first;
then install this project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e '.[test]'
python -m DataProcessing.prepare --source Data/Raw
python -m Experiments.run run --config Experiments/Configs/pod_arx.json --seeds 0 1 2
```

The CUDA 12.6 build is selected for compatibility with the GTX 1080 Ti as well as
the L40S. The CUDA number displayed by `nvidia-smi` is not an instruction to install
a CUDA 13 PyTorch build on Pascal. Verify CUDA availability in the actual server
container before launching experiments. CPU is supported; the Mac test environment
uses a different PyTorch build. The Docker image has not been tested on the servers.

Preparation reads `data.npy` inside each NPZ through a streaming extraction to disk,
then converts `(cell, field, time)` float64 to `(time, field, height, width)` float32 NPY images.
It needs disk space for the prepared dataset plus one temporarily extracted source
archive. Existing prepared files are kept unless `--overwrite` is supplied. After
changing raw files, explicitly regenerate their prepared outputs. No raw data are
deleted. DataLoader workers open read-only memory maps and copy only requested
windows. POD reads batches and concatenates the full training matrix in RAM; latent
trajectories are cached in RAM.

DataLoader options are shared across scaler fitting, compressor fitting, forecaster
training, and validation. Set `loader_options` in the notebook, or add this experiment
config section (the old `workers` key remains a fallback):

```json
"dataloader": {
  "num_workers": 2,
  "pin_memory": true,
  "persistent_workers": true,
  "prefetch_factor": 1,
  "multiprocessing_context": "spawn"
}
```

For direct calls, use `make_loader` from `DataProcessing.loading` and pass
`loader_options=...` to compressor `fit`, `reconstruction_error`, and `validation_error`.
The neural training loops use nonblocking transfers; pinned memory is useful when
moving batches to CUDA. Use `pin_memory=false` for CPU work. CAE reuses its loaders
across epochs and evaluates reconstruction on the device.

Compare 0, 2, and 4 workers on the server. Small latent batches can be faster with 0.
Prefetching queues approximately `num_workers * prefetch_factor` batches; image
windows can consume gigabytes of shared memory. Docker must provide enough `/dev/shm`.
With 0 workers, worker-only options are ignored. Spawned workers mmap the data instead
of duplicating the notebook's optional full-trajectory RAM cache. Use server-local
storage when possible. These settings are configurable, not measured speedup claims.

Fit and evaluation progress uses text tqdm without notebook widgets. Restart the
notebook kernel after updating the code. POD loading has batch progress; SVD and ARX's
linear solve show start/end bars because the libraries expose no iteration callback.

The metadata currently lists the **eight cases actually present**: two training
sweeps and six test trajectories (10/40 Hz at A=0.3/0.5, steps at A=0.3/0.5).
The previously discussed 150 Hz sweep, 10 Hz A=0.25 and PRBS are not in the supplied
files; add metadata entries when they arrive. Field indices and units are explicit.
Forcing is loaded from the supplied files, never reconstructed from a guessed law.

Before full physical evaluation, complete these metadata entries:

- `initial_snapshot_is_steady`: already `true`; the initial steady state is confirmed.
- `cell_volumes`: a relative path to a positive 1D NPY array of physical volumes in
  cell order, including the axisymmetric geometry factor. Do not substitute equal
  cell weights for missing volumes.
- `coordinates`: reserve a relative path for the cell-centre coordinates. Current
  latent baselines do not require them.

`heat_release: false` explicitly enables field-only evaluation and is currently set
in all example configs. **TODO:** add the cell-volume file, set `cell_volumes` in the
metadata, then set `heat_release: true` to enable integrated Q and gain/phase metrics. An alternative
`initialization: "observed_history"` waits for `max(Nx, Ni) + 1` context samples,
uses the latest `Nx + 1` true snapshots, and scores only the remainder; report it as a different protocol. The main physical
protocol must not silently fall back to this alternative.

## Protocol

The first 80% of each training sweep is training data; the final 20% is validation.
Windows stay entirely inside their own trajectory and partition. No snapshot is
shared across partitions. Feature scaling and POD are fitted only on training.
AE/VAE weights use training data and select reconstruction checkpoints on validation.
This simple temporal holdout changes the frequency coverage of each partition;
it is a declared benchmark choice, not a random IID split.

History lengths are independent:

- `Nx` counts snapshots strictly before the current time: the model receives
  `z[t-Nx : t+1]`, or **Nx + 1 snapshots** including the current one.
- `Ni` counts inputs strictly before the current time: the model receives
  `phi[t-Ni : t+2]`, or **Ni + 2 inputs** including current and next forcing.

`Nx=9, Ni=0` preserves the original 10-snapshot/two-input protocol. The notebook uses
`Nx=9, Ni=4` to demonstrate independent lengths. Both values may be zero. The old
`history` config key is replaced by `Nx=history-1` and `Ni=0`; start new runs when
changing input dimensions rather than resuming old checkpoints.

Training/validation samples need `max(Nx, Ni) + 1` context samples inside their own
segment. Neither history reaches across the split. Validation rollout starts after
that context and initializes from the latest `Nx + 1` observed states. ARX flattens
both histories. GRU/LSTM/Transformer receive **joint state/forcing inputs before
recurrence or attention**. The chronological sequence spans `t-max(Nx,Ni)` through
`t+1`. Each input contains the latent state, `phi-1`, and two availability flags.
Missing entries are zero-filled with their flag unset; the target-time state is
always unavailable, while `phi(t+1)` is known. The neural head uses only the output
of the sequence-processing block. The compressor still processes fields only.

This replaces the earlier head-only forcing architecture. Existing neural model
checkpoints must be retrained in a new output directory; they are not compatible
with the new input-layer dimensions.

Neural forecasters learn latent one-step MSE with gradient clipping, early stopping
and best validation one-step checkpoint selection. Final run/HPO selection reports
recursive latent validation MSE. HPO does not read test trajectories. For comparison
across different compressors/ranks, use a common physical-field validation objective;
the current HPO keeps the compressor fixed and tunes only the forecaster.

At test time, the initial steady state is repeated to fill the history. Every later
state is predicted recursively. No true future field enters the model. The initial
observed state is excluded from scores. Forcing history before time zero is padded with **1**. At and after
time zero, actual supplied forcing values are used (including the step at zero).
No forcing beyond the next prediction time enters a forecast.

All field metrics are computed after inverse scaling:

- Per-field NRMSE = RMSE / population standard deviation of that case's ground truth,
  pooling all evaluated times and cells with equal weights. Report all fields and
  their arithmetic mean. Constant reference fields give an explicit undefined value.
- Integrated heat release: `Q(t) = sum(mix:Q(t, cell) * volume(cell))` in float64.
  Report `norm(Q_pred - Q_true) / norm(Q_true)` and save both Q signals.
- Sinusoidal gain/phase: fit cosine, sine and a constant at the prescribed frequency
  to `phi - 1` and `(Q - Q0) / Q0`, with the same true initial `Q0` for both signals.
  Report relative gain error and signed wrapped phase error in degrees.
  The default window starts at 0.5 s and excludes the duplicate final endpoint.
  **These are finite-window estimates, not certified steady-periodic measurements.**
  Inspect cycle stability before claiming steady-state frequency response in the paper.
- Inference time covers latent transition, decoding and inverse scaling, with warmup
  and CUDA synchronization. It excludes disk IO, metrics and initial encoding. Use the
  same device for cross-model timing comparisons. Predictions can optionally stream
  to NPY; they are disabled by default to avoid large result directories.

POD uses the legacy `PODReducer` algorithm: temporal centering per coordinate and
`randomized_svd(n_oversamples=20, n_iter=7, random_state=42)`. Image pixels are gathered
in original cell order; empty pixels are excluded. POD loads the full training matrix into RAM,
then centers it and runs SVD, as in the legacy code. This
is randomized approximate POD, not incremental or volume-weighted POD. Feature
standardization remains shared with the other compressors; legacy range scaling
is not restored. Existing incremental-POD checkpoints require retraining.

An optional `"backend": "torch"` in the compressor config runs `torch.svd_lowrank`
on the experiment `device`. In the notebook set `pod_backend="torch"`. It uses the
same centering, seed 42, 7 iterations, and rank + 20 sampled directions (capped by
matrix dimensions). The default `"sklearn"` backend remains the legacy implementation.
Torch uses a different random draw and factorization, so compare reconstruction errors,
not signed basis entries. Both backends encode/decode with NumPy on CPU; only Torch's
SVD fit uses the GPU. Save the backend in reported experiment settings.

The current 80% training split produces roughly a 6 GB float32 matrix, plus SVD
workspace and CPU copies. Try an L40S first and measure the complete fit time. CUDA
speed and peak memory have not been measured here. PyTorch notes that low-rank SVD
is not always faster than full SVD for dense matrices; benchmark this dataset before
choosing a backend. See [torch.svd_lowrank](https://docs.pytorch.org/docs/stable/generated/torch.svd_lowrank.html)
and [DataLoader options](https://docs.pytorch.org/docs/stable/data.html).
`persistence` predicts a constant *latent* state, so includes compressor error.
All compressors read images. POD and dense AE/VAE flatten them; CAE uses spatial
convolutions and ViTAE uses attention between image patches. RAE, mesh models and
world models remain candidates in [the method plan](docs/methods.md).

## Experiments, logs and checkpoints

`output` is now the results **root**, not a single run directory. One timestamp is
created per CLI invocation and shared by all requested seeds:

```text
Experiments/Results/
  pod_gru_20260920T120000_000000Z/
    seed_0/
      config.json
      metadata.resolved.json
      environment.json
      metrics.json            # per-test-case evaluation metrics
      metrics.jsonl           # training, validation and evaluation scalars
      summary.json            # validation rollout score
      tensorboard/
      wandb/
      wandb_id.txt
      model.pkl
      preprocessing.pkl
      last.pt                 # neural forecaster checkpoint
      latent/
    seed_1/
    seed_2/
```

The name includes both compressor and forecaster. Timestamps use UTC with
microseconds. Use `--run-name` to share an explicit experiment name between jobs;
choose a new name for a different configuration.

```bash
# Fit and evaluate each seed; all three share one generated model/timestamp folder.
python -m Experiments.run run --config Experiments/Configs/pod_gru.json --seeds 0 1 2

# Fit only, using an explicit shared experiment name.
python -m Experiments.run fit --config Experiments/Configs/pod_gru.json \
  --run-name pod_gru_20260920T120000Z --seeds 0 1 2

# Evaluate or resume the exact experiment and seeds; never guess the latest run.
python -m Experiments.run test --config Experiments/Configs/pod_gru.json \
  --run-name pod_gru_20260920T120000Z --seeds 0 1 2
python -m Experiments.run fit --config Experiments/Configs/pod_gru.json \
  --run-name pod_gru_20260920T120000Z --seed 1 --resume

# Tune once (config seed), then fit the best config on the tuning split and test it, per seed.
python -m Experiments.run hpo --config Experiments/Configs/hpo_pod_arx.json --seeds 0 1 2 3 4 5 6 7 8 9
# Ablation: refit a fitted config on all training data (no validation, no early stopping).
python -m Experiments.run refit --config Experiments/Results/<run_name>/seed_0/config.json --seeds 0 1 2
tensorboard --logdir Experiments/Results
```

Every model follows the same protocol: `hpo` tunes once, then fits the best config on
the training/validation split with early stopping, once per seed (`<run_name>/seed_N`,
W&B config `stage: fit`), and tests each. Deterministic parts are not fitted again: POD is
fitted with the first seed and reused, and POD + ARX (a fully deterministic model) runs one
seed only. When the config gives no rank, stage 1 tunes an autoencoder separately at each rank
of its `rank_range` (8, 16, 32, 64) and stage 2 picks the rank on the forecast objective; POD
is fitted once at rank 128 and truncated. Stage 1 does not depend on the forecaster, so it is
cached in `<output>/hpo_cache/` and shared by every pair with the same compressor settings;
pairs running at the same time split its ranks between them. Delete `hpo_cache/` after
changing the prepared data. `refit` is an ablation (`<run_name>_refit`,
`stage: refit`): it trains on every training frame for the number of optimizer steps the
fit kept (the fit's best epoch count times train frames / all training frames). Without
validation it keeps the last weights, which for neural forecasters can diverge in long
rollouts. Every test also writes `summary/*` values (mean and worst test NRMSE, SSIM,
heat-release error, gain and phase error per forcing frequency, validation MSE).

Full benchmark on a GPU server (run inside `screen`): every compressor-forecaster pair
(`hpo_<compressor>_<forecaster>.json`, the same budget each: 20 trials per stage, neural
forecasters at most 60 epochs with patience 10, batch size 256) plus the constant baseline
(`identity_constant.json`), seeds 0-9 (`SEEDS` to change), one log per pair in `logs/`:

```bash
PYTHON=.venv/bin/python Experiments/scripts/hpo.sh              # everything, all at once
GPUS="0 1" Experiments/scripts/hpo.sh pod_arx pod_gru constant  # a subset on GPUs 0 and 1
JOBS_PER_GPU=2 Experiments/scripts/hpo.sh                       # at most 2 pairs per GPU
```

The small models leave a GPU mostly idle (their time goes into launching many tiny GPU
operations), so throughput comes from running many processes per GPU: every pair starts at
once, interleaved over the GPUs, and each pair runs `parallel` (config, 3) HPO trials or seed
fits at a time in worker processes. The configs set `cpu_threads: 1` and `in_memory: false`,
so the processes share the operating system's file cache instead of each holding the data.
A trial that runs out of GPU memory counts as failed. Test metrics are computed in batches of
frames, SSIM on the GPU.

You can also pass a saved `seed_N/config.json` to `--config`; it already contains
the exact run name and seed. `--seed` overrides a single seed; `--seeds` runs multiple
seeds sequentially. Different processes may run different seeds under the same
explicit run name. A fresh fit refuses to overwrite an existing seed directory.
`test` and `--resume` require an explicit run name or a saved config.

Optuna stores its database and `best.json` inside the seed folder, with trial
artifacts under `seed_N/trial_0000/`, `seed_N/trial_0001/`, etc. Seeds do not share
optimizer state, learned compressors, checkpoints, or W&B IDs. No automatic
cross-seed metric averaging is applied.

W&B uses the experiment name as its **group**, and a run name such as
`pod_gru_<timestamp>/seed_0`. Training and evaluation reuse the seed's stored W&B ID
in online mode and write TensorBoard/JSONL metrics into that same seed folder.
Offline W&B sessions remain separate local session files until synced.

W&B defaults to **offline** so both log formats work without an account or network.
For FireMark online logging, copy `.env.example` to `.env` and fill in your API key:

```dotenv
WANDB_API_KEY=your_key_here
WANDB_ENTITY=FireMark
WANDB_PROJECT=rom-flamebench
WANDB_MODE=online
```

The logger loads `.env` from the working directory. Existing environment variables
win over `.env`; W&B environment settings win over JSON logging settings. Explicit
`mode: "disabled"` in a config always disables W&B. The key is never copied into
experiment configs/checkpoints, and `.env` is excluded from Git and Docker builds.
For Docker, add `--env-file .env` to `docker run`; do not bake credentials into an
image. Use `WANDB_MODE=offline` for disconnected runs. Offline logs can be uploaded
later with `wandb sync`.

Neural forecaster checkpoints include optimizer and RNG state. `--resume` requires
an identical config/metadata and continues from `last.pt`. Completed preprocessing
is reused; interrupted AE/VAE compressor fitting restarts that stage. Saved pickle
artifacts are for trusted local runs and currently require a compatible Python/
dependency environment; they are not a portable public model format. Model artifacts
are saved on CPU; `test --device cuda` or `test --device cpu` selects the evaluation device.

Start with one independent experiment per GPU:

```bash
PYTHON=.venv/bin/python bash Experiments/scripts/local.sh
sbatch Experiments/scripts/slurm.sh
```

The local example uses the three L40S GPUs, one model per GPU, running seeds 0, 1,
and 2 sequentially for each model. The Slurm example does the same with three array
jobs. Edit the explicit job/seed lists for other allocations. These scripts do not implement multi-GPU training of a single model.
Slurm partition/account settings belong to the target cluster. Use distinct
experiment names for different configurations and avoid concurrent writes to the
same seed. The old flat output layout is not auto-discovered or migrated.

For the Docker server:

```bash
docker build -t rom-flamebench .
docker run --rm --gpus '"device=0"' --ipc=host \
  -v "$PWD/Data:/workspace/Data:ro" \
  -v "$PWD/Experiments/Results:/workspace/Experiments/Results" \
  rom-flamebench python -m Experiments.run fit \
  --config Experiments/Configs/pod_gru.json
```

Prepare NPY data before mounting Data read-only. Use one container per experiment.

## Pre-commit

The hooks are defined in [.pre-commit-config.yaml](.pre-commit-config.yaml). For a
fresh clone, install the development tools and Git hook:

```bash
pip install -e '.[dev]'
pre-commit install --install-hooks
pre-commit run --all-files
```

The hook runs automatically on staged files at commit time. It checks JSON/YAML/TOML,
merge conflict markers, large added files, final newlines, trailing whitespace, and
Python syntax/name errors through Ruff. It does not run expensive model training or
reformat the whole codebase. Archived `legacy/` code and generated data/results are
excluded. Hook versions are pinned; see [pre-commit's setup guide](https://pre-commit.com/#quick-start).

## Tests

```bash
python -m pytest -q
```

Synthetic tests check disk conversion, non-overlapping partitions, train-only scaling,
POD/ARX recursive forecasts, volume-weighted Q, analytic gain/phase, neural checkpoint
resume, AE/VAE reconstruction, the fit/test CLI functions and an Optuna trial. These
are correctness checks, not evidence of forecasting accuracy on the flame data.

### Image preparation

`DataProcessing/process4convolution.py` adapts the mapping from `main` (commit
`6804043`). It bins cell centers in X/Z to eight decimal places and reverses Z.
The channel-first layout is `(time, field, reversed_z, x)`, equivalent to the
original script after transposing its output. There is no interpolation.

The supplied `Data/grid.vtu` maps 21,334 cells onto 206 × 104 pixels.
`Data/grid_indices.npz` stores axes, cell-to-pixel indices and the valid-cell mask.
The metadata now selects image files. Standard preparation rebuilds these images
from raw archives when needed:

```bash
python -m DataProcessing.prepare --metadata Data/metadata.json --source Data/Raw
```

For an older metadata file that still points to prepared cell arrays, convert once:

```bash
python -m DataProcessing.prepare --grid 'Data/grid.vtu'
```

Conversion streams batches to disk and keeps the original arrays. Images precede
normalization: fit per-field statistics on valid training cells only, then keep
empty pixels zero. AE/VAE reconstruction losses exclude empty pixels. Evaluation
restores original cell order before field metrics and volume-weighted Q integration.
Cell volumes remain a TODO. POD and dense AE/VAE flatten images internally; CAE and
ViTAE explicitly model spatial structure. Retrain models on the image layout;
old cell-layout checkpoints are not compatible.

### Image models

All experiment configs use `Data/metadata.json`, which selects float32 image arrays
and forcing in `Data/Images/Training` and `Data/Images/Test`. Raw archives and the
original cell arrays remain available. Arrays stay disk-backed.

- `cae`: three stride-2 convolution layers, a fixed-size latent state, and three
  transposed-convolution layers. Crop the decoder output to the original image size.
- `vit_ae`: non-overlapping patch embeddings, learned position embeddings, and
  spatial self-attention. A class token produces the latent state. The decoder
  expands the latent state into position-specific tokens, applies attention, and
  reconstructs image patches. Padding is cropped; fully empty encoder patches are masked.

Both share AE training, validation selection, masked reconstruction loss, checkpoints,
and TensorBoard/W&B logging. They are compact baselines, not pretrained foundation
models. `vit_ae` attention is spatial; the existing `transformer` forecaster applies
attention over time. Any registered forecaster can use either compressor. Forcing
enters the temporal forecaster together with compressed image snapshots.

```bash
python -m Experiments.run run --config Experiments/Configs/cae_gru.json --seeds 0 1 2
python -m Experiments.run run --config Experiments/Configs/vit_ae_gru.json --seeds 0 1 2
```

The supplied configs use CUDA. Set `device` to `cpu` in the config for a CPU run. `Walkthrough.ipynb`
exposes `compressor_name` with `pod`, `cae`, and `vit_ae` choices.
