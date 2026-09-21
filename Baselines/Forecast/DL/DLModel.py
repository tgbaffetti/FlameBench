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
    # Class-level defaults so models pickled before these options existed still unpickle.
    residual = False
    noise_std = 0.0

    def __init__(self, network, device="cpu", lr=1e-3, epochs=100, patience=20,
                 unroll_grad="none", residual=False, noise_std=0.0, **kwargs):
        if unroll_grad not in {"none", "full"}:
            raise ValueError("unroll_grad must be 'none' (pushforward) or 'full'")
        if noise_std < 0:
            raise ValueError("noise_std must be nonnegative")
        self.device = torch.device(device)
        self.network = network.to(self.device)
        self.lr, self.epochs, self.patience = lr, epochs, patience
        self.unroll_grad = unroll_grad
        self.residual = residual
        self.noise_std = noise_std

    def step(self, history, forcing):
        """One transition; with residual=True the network learns the state delta."""
        out = self.network(history, forcing)
        return history[:, -1] + out if self.residual else out

    def compute_loss(self, history, forcing, target, train=False):
        """Single-step MSE, or the mean over an unrolled rollout when the batch
        carries (batch, horizon, ...) forcing/targets (pushforward detaches).
        Training-time noise injection (MeshGraphNets) perturbs the input states."""
        if train and self.noise_std > 0:
            history = history + self.noise_std * torch.randn_like(history)
        if target.dim() == 2:
            return nn.functional.mse_loss(self.step(history, forcing), target)
        loss = 0
        for k in range(target.shape[1]):
            predicted = self.step(history, forcing[:, k])
            loss = loss + nn.functional.mse_loss(predicted, target[:, k])
            step = predicted.detach() if self.unroll_grad == "none" else predicted
            history = torch.cat((history[:, 1:], step[:, None]), dim=1)
        return loss / target.shape[1]

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
                loss = self.compute_loss(history, forcing, target, train=True)
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
                    loss = self.compute_loss(history, forcing, target)
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
            return self.step(x, u).cpu().numpy()
