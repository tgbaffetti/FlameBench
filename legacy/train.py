from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import gc
import json
import math
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

from scale_with_openmeasure import scale_train_tensor
from seed_utils import seed_everything, init_weights
from models import build_model, MODEL_REGISTRY

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Sliding window dataset — finestra fissa n_past+1 (n_past storia + 1 target).
# Niente più recursive/future_len variabile.
# =============================================================================
class SlidingWindowDataset(IterableDataset):
    def __init__(self, datasets, phis, n_past):
        self.datasets = datasets
        self.phis = phis
        self.window_len = n_past + 1
        self.sample_counts = []

        for data, phi in zip(self.datasets, self.phis):
            n_samples = data.shape[0] - self.window_len + 1
            if n_samples <= 0:
                raise ValueError(
                    f"Dataset troppo corto (Nt={data.shape[0]}) per n_past={n_past}"
                )
            if phi.shape[0] != data.shape[0]:
                raise ValueError(f"Forcing length {phi.shape[0]} != Nt={data.shape[0]}")
            self.sample_counts.append(n_samples)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start, step = 0, 1
        if worker_info is not None:
            start, step = worker_info.id, worker_info.num_workers

        global_index = 0
        for data, phi, n_samples in zip(self.datasets, self.phis, self.sample_counts):
            for i in range(n_samples):
                if global_index % step == start:
                    x_window = torch.as_tensor(data[i:i + self.window_len], dtype=torch.float32)
                    phi_window = torch.as_tensor(phi[i:i + self.window_len], dtype=torch.float32)
                    yield x_window, phi_window
                global_index += 1

    def __len__(self):
        return int(sum(self.sample_counts))


def build_scheduler(optimizer, scheduler_cfg):
    if not scheduler_cfg:
        return None
    warmup = int(scheduler_cfg.get("warmup_epochs", 0))
    div = float(scheduler_cfg.get("initial_lr_divisor", 100.0))
    every = int(scheduler_cfg.get("decay_every_epochs", 20))
    gamma = float(scheduler_cfg.get("decay_gamma", 0.8))
    start = 1.0 / max(div, 1e-9)

    def lr_lambda(e):
        if warmup > 0 and e < warmup:
            return start + (1 - start) * (e + 1) / warmup
        decay_steps = 0 if e < warmup else (e - warmup) // every
        return gamma ** decay_steps

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_training(model_name: str, settings: dict, config_label: str = "train.json"):
    seed_everything(42)

    print(f'Modello: "{model_name}"  —  config: "{config_label}"')

    if model_name not in settings.get("models", {}):
        raise KeyError(
            f"Nessuna configurazione per il modello '{model_name}' in train.json. "
            f"Modelli configurati: {list(settings.get('models', {}).keys())}"
        )

    dataset_cfg = settings["dataset"]
    model_cfg = settings["models"][model_name]
    training_cfg = model_cfg["training"]
    architecture_cfg = model_cfg["architecture"]
    output_cfg = model_cfg["output"]

    # ── dataset ──────────────────────────────────────────────────────────
    nf = dataset_cfg["Nf"]
    n_cells = dataset_cfg["Ncells"]
    data_paths = dataset_cfg["data_paths"]
    phi_paths = dataset_cfg["phi_paths"]
    grid_path = dataset_cfg.get("grid_path")

    if len(data_paths) != len(phi_paths):
        raise ValueError("data_paths e phi_paths devono avere la stessa lunghezza")

    # ── training ─────────────────────────────────────────────────────────
    epochs = training_cfg["epochs"]
    batch_size = training_cfg["batch_size"]
    lr = training_cfg["lr"]
    n_past = training_cfg["n_past"]
    scheduler_cfg = training_cfg.get("scheduler")
    save_checkpoint_each_epoch = training_cfg.get("save_checkpoint_each_epoch", True)

    models_dir = Path(output_cfg["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── caricamento dati: shape file = (n_cells, n_features, n_timesteps) ──
    # trasposizione interna a (n_timesteps, n_features, n_cells)
    train_snapshots_list = []
    train_phi_list = []
    nts = []
    for data_path, phi_path in zip(data_paths, phi_paths):
        raw = np.load(data_path)
        if raw.shape[0] != n_cells or raw.shape[1] != nf:
            raise ValueError(
                f"Shape inattesa per {data_path}: {raw.shape}, attesa (Ncells={n_cells}, Nf={nf}, Nt)"
            )
        dataset = np.transpose(raw, (2, 1, 0))  # -> (Nt, Nf, Ncells)

        phi = np.load(phi_path).astype(np.float32)
        nt = dataset.shape[0]
        if phi.shape != (nt,):
            raise ValueError(f"Shape phi inattesa {phi.shape}, attesa ({nt},) per {phi_path}")

        print(f'Dataset interpretato per {Path(data_path).name}: [{nt}, {nf}, {n_cells}]')
        train_snapshots_list.append(dataset)
        train_phi_list.append(phi)
        nts.append(nt)

    if grid_path is not None:
        grid = np.load(grid_path)
        if grid.shape != (n_cells, 3):
            raise ValueError(f"Shape grid inattesa {grid.shape}, attesa ({n_cells}, 3)")
    else:
        grid = None

    # ── normalizzazione phi ─────────────────────────────────────────────
    train_phi_concat = np.concatenate(train_phi_list, axis=0)
    phi_mean = float(train_phi_concat.mean())
    phi_std = float(train_phi_concat.std())
    if phi_std <= 0.0:
        raise ValueError("phi_std deve essere positivo")
    train_phi_scaled_list = [(phi - phi_mean) / phi_std for phi in train_phi_list]
    print(f"\nTraining snapshots: {sum(nts)} su {len(nts)} dataset")

    # ── scaling dati (scale_with_openmeasure) ───────────────────────────
    print("\nScaling training data ...")
    train_snapshots_concat = np.concatenate(train_snapshots_list, axis=0)
    train_snapshots_scaled_concat, rom = scale_train_tensor(train_snapshots_concat, grid)

    rom_path = models_dir / "rom.pkl"
    with open(rom_path, "wb") as f:
        pickle.dump(rom, f, protocol=4)
    print(f"ROM salvato in: {rom_path}")

    split_points = np.cumsum(nts)[:-1]
    train_snapshots_scaled_list = np.split(train_snapshots_scaled_concat, split_points, axis=0)
    print(f"max valore dopo scaling: {train_snapshots_scaled_concat.max():.6f}")
    print(f"min valore dopo scaling: {train_snapshots_scaled_concat.min():.6f}")

    print(f"\nn_past = {n_past}  (finestra = {n_past + 1} snapshot)")

    train_dataset = SlidingWindowDataset(train_snapshots_scaled_list, train_phi_scaled_list, n_past=n_past)

    # ── costruzione modello ──────────────────────────────────────────────
    model = build_model(model_name, architecture_cfg, nf=nf, n_cells=n_cells, n_past=n_past, device=Device)
    model.apply(init_weights)

    print(f"\nPreprocessing ({model_name}) ...")
    extra_state = model.preprocess(train_snapshots_scaled_concat) or {}

    if not model.requires_training_loop():
        # ── Modelli non a gradiente (POD-ARX, POD-NARX) ──────────────────
        print(f"\n[{model_name}] Fit closed-form (no gradient loop) ...")
        fit_extra = model.fit_closed_form(train_dataset) or {}
        ckpt = {
            "model_name": model_name,
            "model_state": model.state_dict(),   # vuoto, ok
            "epoch": 1,
            "loss": None,
            "settings": settings,
            "phi_mean": phi_mean,
            "phi_std": phi_std,
            "n_past": n_past,
        }
        ckpt.update(model.extra_checkpoint_data())
        ckpt.update(extra_state)
        ckpt.update(fit_extra)
        models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, models_dir / f"{model_name}_fit.pt")
        print(f"[{model_name}] Fit completato. Salvato in: {models_dir / f'{model_name}_fit.pt'}")

        # pulizia memoria e return anticipato
        del model, train_dataset
        del train_snapshots_concat, train_snapshots_scaled_concat
        del train_snapshots_list, train_snapshots_scaled_list
        del train_phi_list, train_phi_scaled_list, train_phi_concat
        gc.collect()
        print("Memoria liberata.\n")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = build_scheduler(optimizer, scheduler_cfg)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    t0 = time.perf_counter()
    history = []

    for epoch in range(epochs):
        model.train()
        losses = []
        e0 = time.perf_counter()

        for x_window, phi_window in loader:
            x_window = x_window.to(Device)
            phi_window = phi_window.to(Device)

            optimizer.zero_grad()
            loss = model.compute_loss(x_window, phi_window)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss non finita. Interrompo il training.")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        epoch_loss = float(np.mean(losses))
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        print(f"[{model_name}] Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f} - "
              f"time: {time.perf_counter() - e0:.2f}s")

        if not math.isfinite(epoch_loss):
            raise RuntimeError(f"Epoch {epoch + 1} ha prodotto una loss non finita. Stop.")

        if save_checkpoint_each_epoch:
            ckpt = {
                "model_name": model_name,
                "model_state": model.state_dict(),
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "settings": settings,
                "phi_mean": phi_mean,
                "phi_std": phi_std,
                "n_past": n_past,
            }
            ckpt.update(model.extra_checkpoint_data())
            ckpt.update(extra_state)
            torch.save(ckpt, models_dir / f"{model_name}_{epoch + 1:03d}_{epoch_loss:.6f}.pt")

        if scheduler is not None:
            scheduler.step()

    total_time = time.perf_counter() - t0
    print(f"\nTraining completato in {total_time:.2f}s. Modelli salvati in: {models_dir}")

    # ── pulizia memoria ──────────────────────────────────────────────────
    del model, optimizer, train_dataset
    del train_snapshots_concat, train_snapshots_scaled_concat
    del train_snapshots_list, train_snapshots_scaled_list
    del train_phi_list, train_phi_scaled_list, train_phi_concat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("Memoria liberata.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train.json")
    parser.add_argument("--model", type=str, required=True,
                         choices=list(MODEL_REGISTRY.keys()),
                         help="Nome del modello da allenare (deve avere una sezione in train.json)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        settings = json.load(f)

    run_training(args.model, settings, config_label=args.config)


if __name__ == "__main__":
    main()