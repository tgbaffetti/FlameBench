"""Shared streaming training for image autoencoders; beta > 0 makes the encoder variational.

Every autoencoder works on deviations from the per-pixel training statistics, as POD does: the
encoder reads x - mean and the decoder returns mean + std * output (mean and std per field and
pixel, over the scaled training frames). A pixel that never varies in training (std 0, about a
third of them) is then reconstructed exactly. Without this the decoder adds small noise
everywhere; the largest cells, at the image border, are hundreds of times the median volume,
so that noise dominated the volume-integrated heat release (errors of 50-180% instead of 2-10%).
"""
from abc import abstractmethod
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from DataProcessing.loading import make_loader
from Baselines.losses import error_function
from ..Compressor import Compressor


class Shift(nn.Module):
    """Adds a fixed tensor (a buffer, so it follows .to(device) and is saved with the module)."""
    def __init__(self, offset):
        super().__init__()
        self.register_buffer("offset", offset)

    def forward(self, x):
        return x + self.offset


class Scale(nn.Module):
    """Multiplies by a fixed tensor (a buffer)."""
    def __init__(self, factor):
        super().__init__()
        self.register_buffer("factor", factor)

    def forward(self, x):
        return x * self.factor


def pixel_statistics(dataset, batch_size, loader_options=None):
    """Mean and standard deviation of each field and pixel over the frames of a dataset (float32)."""
    total = square = None
    for x in make_loader(dataset, batch_size, **(loader_options or {})):
        x = x.double()
        total = x.sum(0) if total is None else total + x.sum(0)
        square = x.square().sum(0) if square is None else square + x.square().sum(0)
    mean = total / len(dataset)
    return mean.float(), (square / len(dataset) - mean.square()).clamp(min=0).sqrt().float()


class AE(Compressor):
    """Base class: subclasses build self.encoder and self.decoder for (field, height, width) images.

    The encoder outputs rank values, or 2 * rank (mean, log-variance) when beta > 0.
    Evaluation uses the posterior mean for reproducible forecasts. loss is the training
    reconstruction error on valid pixels; checkpoint selection always uses validation MSE,
    matching the reported NRMSE. The KL term is averaged over latent dimensions, so beta equals
    the usual summed-KL beta divided by rank.
    """
    hyperparameters_ranges = {"lr": {"type": "float", "low": 1e-4, "high": 3e-3, "log": True},
                              "batch_size": {"type": "categorical", "choices": [8, 16, 32]}}
    rank_range = {"type": "categorical", "choices": [8, 16, 32, 64]}

    def __init__(self, rank=16, epochs=20, batch_size=16, lr=1e-3, device="cpu", beta=0.0, loss="mse"):
        if beta < 0:
            raise ValueError("beta must be nonnegative")
        error_function(loss)
        self.rank, self.epochs = rank, epochs
        self.batch_size, self.lr, self.device, self.beta, self.loss = batch_size, lr, device, beta, loss

    @property
    def encoder_outputs(self):
        return self.rank * (2 if self.beta else 1)

    @abstractmethod
    def build_networks(self):
        """Create self.encoder and self.decoder for self.shape."""

    def fit(self, dataset, validation=None, logger=None, loader_options=None, **kwargs):
        """Keep the weights of the epoch with the best validation MSE; best_epochs is its epoch count.

        Without validation (refit on all training data) it keeps the weights of the last epoch.
        """
        self.best_epochs = 0
        self.shape = tuple(dataset.field_shape)
        self.build_networks()
        mean, std = pixel_statistics(dataset, self.batch_size, loader_options)
        self.encoder = nn.Sequential(Shift(-mean), self.encoder)
        self.decoder = nn.Sequential(self.decoder, Scale(std), Shift(mean))
        self.to(self.device)
        mask = torch.as_tensor(dataset.mask, device=self.device)
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=self.lr)
        error = error_function(self.loss)
        training_loader = make_loader(dataset, self.batch_size, shuffle=True, **(loader_options or {}))
        validation_loader = None if validation is None else make_loader(validation, self.batch_size,
                                                                          **(loader_options or {}))
        best, best_state = float("inf"), None
        epochs = tqdm(range(self.epochs), desc=f"{type(self).__name__} fit")
        for epoch in epochs:
            self.encoder.train()
            self.decoder.train()
            total = count = 0
            for x in training_loader:
                x = x.to(self.device, non_blocking=True)
                encoded = self.encoder(x)
                if self.beta:
                    mu, logvar = encoded.chunk(2, dim=-1)
                    logvar = logvar.clamp(-20, 10)  # Keeps exp(logvar) finite; not a tuned value.
                    z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
                    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
                else:
                    z, kl = encoded, 0.0
                predicted = self.decoder(z)
                loss = error(predicted[..., mask], x[..., mask]) + self.beta * kl
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite autoencoder loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                total += loss.item() * len(x)
                count += len(x)
            self.encoder.eval()
            self.decoder.eval()
            if validation_loader is None:
                best_state = [{k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
                              for module in (self.encoder, self.decoder)]
                self.best_epochs = epoch + 1
                epochs.set_postfix(train=total / count)
                if logger:
                    logger.log({"Train/compressor_loss": total/count}, epoch + 1, axis="Train/compressor_epoch")
                continue
            sse = n = 0
            with torch.no_grad():
                for x in validation_loader:
                    x = x.to(self.device, non_blocking=True)
                    z = self.encoder(x)[:, :self.rank]
                    difference = (self.decoder(z) - x)[..., mask]
                    sse += difference.square().sum().item()
                    n += difference.numel()
            value = sse/n
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite reconstruction validation")
            if value < best:
                best, self.best_epochs = value, epoch + 1
                best_state = [{k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
                              for module in (self.encoder, self.decoder)]
            epochs.set_postfix(train=total / count, validation=value)
            if logger:
                logger.log({"Train/compressor_loss": total/count, "Validation/compressor_mse": value}, epoch + 1,
                           axis="Train/compressor_epoch")
        if best_state is None:
            raise ValueError("No compressor training epochs completed")
        for module, state in zip((self.encoder, self.decoder), best_state):
            module.load_state_dict(state)
            module.eval()
        return self

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        self.decoder.to(device)
        return self

    def encode(self, frames):
        self.encoder.eval()
        with torch.no_grad():
            x = self.encoder(torch.as_tensor(frames, dtype=torch.float32, device=self.device))
            return x[:, :self.rank].cpu().numpy()

    def decode(self, latent):
        self.decoder.eval()
        with torch.no_grad():
            z = torch.as_tensor(latent, dtype=torch.float32, device=self.device)
            return self.decoder(z).cpu().numpy()
