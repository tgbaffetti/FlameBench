"""Shared neural training, validation, checkpointing, and inference.

Schedule: fit trains the forecaster for `epochs` on the latents of a frozen compressor. fit_joint
then unfreezes a torch compressor and trains encoder, forecaster and decoder together for
`joint_epochs` with learning rate `joint_lr`. Both phases use loss = (1 - rollout_weight) * step-1
error + rollout_weight * mean error over the K recursive (fed-back) predictions of each window,
as in TransformerROM's TransformerLoss; K is the horizon of the dataset windows. The error
function is MSE, MAE, Huber or SmoothL1. Validation windows may have a longer horizon (K_eval);
validation also logs the error of each step.
"""
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from Baselines.losses import error_function
from DataProcessing.Dataset import keep_recent
from ..Model import Model


class DLModel(Model):
    """Keys starting with joint_ are tuned in the joint stage of the HPO, the others before it."""
    hyperparameters_ranges = {"lr": {"type": "float", "low": 1e-5, "high": 3e-3, "log": True},
                              "hidden": {"type": "categorical", "choices": [32, 64, 128]},
                              "layers": {"type": "int", "low": 1, "high": 4},
                              "dropout": {"type": "float", "low": 0.0, "high": 0.3},
                              "rollout_weight": {"type": "float", "low": 0.0, "high": 1.0},
                              "joint_lr": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
                              "joint_epochs": {"type": "categorical", "choices": [5, 10, 20]}}

    def __init__(self, network, Nx=9, Ni=0, device="cpu", lr=1e-3, epochs=100, patience=20,
                 rollout_weight=0.5, loss="mse", detach_rollout=False, joint_lr=1e-4, joint_epochs=0):
        if epochs < 1 or joint_epochs < 0 or not 0 <= rollout_weight <= 1:
            raise ValueError("Require epochs >= 1, joint_epochs >= 0 and 0 <= rollout_weight <= 1")
        error_function(loss)
        self.device = torch.device(device)
        self.network = network.to(self.device)
        self.Nx, self.Ni = Nx, Ni
        self.lr, self.epochs, self.patience = lr, epochs, patience
        self.joint_lr, self.joint_epochs = joint_lr, joint_epochs
        self.rollout_weight, self.loss, self.detach_rollout = rollout_weight, loss, detach_rollout

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
            if stale >= self.patience:
                start = self.epochs  # Early stopping already triggered; do not train further.
        for epoch in range(start, self.epochs):
            self.network.train()
            total = count = 0
            for batch in training:
                optimizer.zero_grad(set_to_none=True)
                loss, _ = self.rollout_loss(*self.tensors(batch))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(batch["target"])
                count += len(batch["target"])
            self.network.eval()
            val_total = val_count = 0
            val_steps = 0
            with torch.no_grad():
                for batch in validation:
                    loss, steps = self.rollout_loss(*self.tensors(batch))
                    val_total += loss.item() * len(batch["target"])
                    val_steps += steps.cpu() * len(batch["target"])
                    val_count += len(batch["target"])
            value = val_total / val_count
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite validation loss")
            if value < best:
                best, stale = value, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
            else:
                stale += 1
            if logger:
                logger.log({"train/latent_loss": total/count, "validation/latent_loss": value,
                            **self.step_errors("validation/latent", val_steps / val_count)}, epoch)
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

    def fit_joint(self, compressor, training, validation, logger=None):
        """Train encoder, forecaster and decoder on the rollout loss of the scaled image forecast.

        training/validation yield scaled image windows (ForecasterDataset without compressor);
        runs joint_epochs with learning rate joint_lr.
        A variational encoder contributes its posterior mean; the KL term is not used here.
        """
        modules = (compressor.encoder, self.network, compressor.decoder)
        parameters = [p for module in modules for p in module.parameters()]
        optimizer = torch.optim.AdamW(parameters, lr=self.joint_lr)
        mask = torch.as_tensor(training.dataset.mask, device=self.device)
        best, stale, best_state = float("inf"), 0, None
        for epoch in range(self.joint_epochs):
            for module in modules:
                module.train()
            total = count = 0
            for batch in training:
                loss, _ = self.joint_loss(compressor, batch, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite joint training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                total += loss.item() * len(batch["target"])
                count += len(batch["target"])
            for module in modules:
                module.eval()
            val_total = val_count = 0
            val_steps = 0
            with torch.no_grad():
                for batch in validation:
                    loss, steps = self.joint_loss(compressor, batch, mask)
                    val_total += loss.item() * len(batch["target"])
                    val_steps += steps.cpu() * len(batch["target"])
                    val_count += len(batch["target"])
            value = val_total / val_count
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite joint validation loss")
            if value < best:
                best, stale = value, 0
                best_state = [{k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
                              for module in modules]
            else:
                stale += 1
            if logger:
                logger.log({"joint/train_field_loss": total/count, "joint/validation_field_loss": value,
                            **self.step_errors("joint/validation_field", val_steps / val_count)}, epoch)
            if stale >= self.patience:
                break
        if best_state is not None:
            for module, state in zip(modules, best_state):
                module.load_state_dict(state)
        for module in modules:
            module.eval()
        return self

    def rollout_loss(self, states, forcing, target, decoder=None, mask=None):
        """Return (combined loss, error per step detached) over one recursive step per target.

        states (batch, length, rank); forcing (batch, length + steps - 1); target (batch, steps, ...),
        as in a ForecasterDataset sample. Without a decoder, targets are latents; with one,
        predictions are decoded and compared on valid pixels. Each prediction becomes the newest
        state row for the next step. By default gradients flow back through those fed-back
        predictions (backpropagation through time); detach_rollout cuts them, so each step trains
        only the call that produced it.
        """
        error = error_function(self.loss)
        length = states.shape[1]
        losses = []
        for step in range(target.shape[1]):
            predicted = self.network(states, keep_recent(forcing[:, step:step + length], self.Ni + 1))
            output, reference = (predicted if decoder is None else decoder(predicted)), target[:, step]
            if mask is not None:
                output, reference = output[..., mask], reference[..., mask]
            losses.append(error(output, reference))
            fed_back = predicted.detach() if self.detach_rollout else predicted
            states = keep_recent(torch.cat((states[:, 1:], fed_back[:, None]), dim=1), self.Nx + 1)
        errors = torch.stack(losses)
        return (1 - self.rollout_weight) * errors[0] + self.rollout_weight * errors.mean(), errors.detach()

    @staticmethod
    def step_errors(prefix, steps):
        return {f"{prefix}_step_{k + 1}": float(value) for k, value in enumerate(steps)}

    def tensors(self, batch):
        return [batch[key].to(self.device) for key in ("states", "forcing", "target")]

    def joint_loss(self, compressor, batch, mask):
        frames, forcing, target = self.tensors(batch)
        z = compressor.encoder(frames.flatten(0, 1))[:, :compressor.rank].reshape(*frames.shape[:2], -1)
        return self.rollout_loss(keep_recent(z, self.Nx + 1), forcing, target, compressor.decoder, mask)

    def to(self, device):
        self.device = torch.device(device)
        self.network.to(self.device)
        return self

    def predict(self, states, forcing):
        self.network.eval()
        with torch.no_grad():
            x = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            u = torch.as_tensor(forcing, dtype=torch.float32, device=self.device)
            return self.network(x, u).cpu().numpy()
