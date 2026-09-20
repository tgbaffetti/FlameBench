"""Shared neural training, validation, checkpointing, and inference."""
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ..Model import Model


class DLModel(Model):
    hyperparams = {"lr": {"value": 0.001, "type": "float", "low": 1e-5, "high": 3e-3, "log": True},
                   "hidden": {"value": 64, "type": "categorical", "choices": [32, 64, 128]}}

    def __init__(self, network, device="cpu", lr=1e-3, epochs=100, patience=20, **kwargs):
        self.device = torch.device(device)
        self.network = network.to(self.device)
        self.lr, self.epochs, self.patience = lr, epochs, patience

    def fit(self, training, validation=None, logger=None, directory=None, resume=False, **kwargs):
        if validation is None:
            raise ValueError("Neural training requires held-out validation")
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.lr)
        start, best, stale, best_state = 0, float("inf"), 0, None
        checkpoint = Path(directory) / "last.pt" if directory else None
        if resume:
            if checkpoint is None or not checkpoint.exists():
                raise FileNotFoundError("Resume requires last.pt")
            state = torch.load(checkpoint, map_location=self.device, weights_only=False)
            self.network.load_state_dict(state["network"])
            optimizer.load_state_dict(state["optimizer"])
            start, best, stale, best_state = state["epoch"] + 1, state["best"], state["stale"], state["best_state"]
            torch.set_rng_state(state["torch_rng"].cpu())
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            if torch.cuda.is_available() and state.get("cuda_rng"):
                torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        for epoch in range(start, self.epochs):
            self.network.train()
            total = count = 0
            for history, forcing, target in training:
                history, forcing, target = [x.to(self.device) for x in (history, forcing, target)]
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(self.network(history, forcing), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(history)
                count += len(history)
            self.network.eval()
            val_total = val_count = 0
            with torch.no_grad():
                for history, forcing, target in validation:
                    history, forcing, target = [x.to(self.device) for x in (history, forcing, target)]
                    loss = nn.functional.mse_loss(self.network(history, forcing), target)
                    val_total += loss.item() * len(history)
                    val_count += len(history)
            value = val_total / val_count
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite validation loss")
            if value < best:
                best, stale = value, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
            else:
                stale += 1
            if logger:
                logger.log({"train/latent_mse": total/count, "validation/latent_mse": value}, epoch)
            if checkpoint:
                state = {"network": self.network.state_dict(), "optimizer": optimizer.state_dict(),
                         "epoch": epoch, "best": best, "stale": stale, "best_state": best_state,
                         "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                         "python_rng": random.getstate(),
                         "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
                temporary = checkpoint.with_suffix(".tmp")
                torch.save(state, temporary)
                temporary.replace(checkpoint)
            if stale >= self.patience:
                break
        if best_state is None:
            raise ValueError("No training epochs completed")
        self.network.load_state_dict(best_state)
        self.network.eval()
        return self

    def predict(self, history, forcing):
        self.network.eval()
        with torch.no_grad():
            x = torch.as_tensor(history, dtype=torch.float32, device=self.device)
            u = torch.as_tensor(forcing, dtype=torch.float32, device=self.device)
            return self.network(x, u).cpu().numpy()
