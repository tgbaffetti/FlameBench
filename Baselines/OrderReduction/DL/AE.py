"""Shared streaming training for image autoencoders; beta > 0 makes the encoder variational."""
from abc import abstractmethod
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from Baselines.losses import error_function
from ..Compressor import Compressor


class AE(Compressor):
    """Base class: subclasses build self.encoder and self.decoder for (field, height, width) images.

    The encoder outputs rank values, or 2 * rank (mean, log-variance) when beta > 0.
    Evaluation uses the posterior mean for reproducible forecasts. loss is the training
    reconstruction error on valid pixels; checkpoint selection always uses validation MSE,
    matching the reported NRMSE. The KL term is averaged over latent dimensions, so beta equals
    the usual summed-KL beta divided by rank.
    """
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
    def build(self):
        """Create self.encoder and self.decoder for self.shape."""

    def fit(self, dataset, validation=None, logger=None, **kwargs):
        if validation is None:
            raise ValueError(f"{type(self).__name__} requires a validation dataset")
        self.shape = tuple(dataset.field_shape)
        if len(self.shape) != 3 or dataset.mask is None:
            raise ValueError(f"{type(self).__name__} requires (field, height, width) images with a mask")
        self.build()
        self.to(self.device)
        mask = torch.as_tensor(dataset.mask, device=self.device)
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=self.lr)
        error = error_function(self.loss)
        best, best_state = float("inf"), None
        for epoch in range(self.epochs):
            self.encoder.train()
            self.decoder.train()
            total = count = 0
            for x in DataLoader(dataset, batch_size=self.batch_size, shuffle=True):
                x = x.to(self.device)
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
            sse = n = 0
            for x in DataLoader(validation, batch_size=self.batch_size):
                x = x.numpy()
                difference = (self.decode(self.encode(x)) - x)[..., validation.mask]
                sse += float(np.square(difference).sum())
                n += difference.size
            value = sse/n
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite reconstruction validation")
            if value < best:
                best = value
                best_state = [{k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
                              for module in (self.encoder, self.decoder)]
            if logger:
                logger.log({"compressor/train_loss": total/count, "compressor/validation_mse": value}, epoch)
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
