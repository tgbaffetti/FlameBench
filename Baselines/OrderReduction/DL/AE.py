"""Mesh-independent dense AE/VAE frame compressors with streaming training."""
import numpy as np
import torch
from torch import nn
from ..Compressor import Compressor


class AE(Compressor):
    name = "ae"
    def __init__(self, rank=16, hidden=128, epochs=20, batch_size=16, lr=1e-3, device="cpu", beta=0.0):
        self.rank, self.hidden, self.epochs = rank, hidden, epochs
        self.batch_size, self.lr, self.device, self.beta = batch_size, lr, device, beta

    def fit(self, dataset, scaler, validation=None, logger=None, **kwargs):
        if validation is None:
            raise ValueError("AE requires a validation dataset")
        self.shape = tuple(dataset.field_shape)
        size = int(np.prod(self.shape))
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(size, self.hidden), nn.SiLU(),
                                     nn.Linear(self.hidden, self.rank * (2 if self.beta else 1))).to(self.device)
        self.decoder = nn.Sequential(nn.Linear(self.rank, self.hidden), nn.SiLU(),
                                     nn.Linear(self.hidden, size)).to(self.device)
        optimizer = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
        best, best_state = float("inf"), None
        for epoch in range(self.epochs):
            self.encoder.train()
            self.decoder.train()
            total = count = 0
            for raw in dataset.snapshot_batches(self.batch_size):
                x = torch.as_tensor(scaler.transform(raw), device=self.device)
                encoded = self.encoder(x)
                if self.beta:
                    mu, logvar = encoded.chunk(2, dim=-1)
                    logvar = logvar.clamp(-20, 10)
                    z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
                    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
                else:
                    z, kl = encoded, 0.0
                predicted = self.decoder(z).reshape_as(x)
                loss = nn.functional.mse_loss(predicted, x) + self.beta * kl
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite autoencoder loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), 1.0)
                optimizer.step()
                total += loss.item() * len(x)
                count += len(x)
            self.encoder.eval()
            self.decoder.eval()
            error = n = 0
            for raw in validation.snapshot_batches(self.batch_size):
                x = scaler.transform(raw)
                reconstruction = self.decode(self.encode(x))
                error += float(np.square(reconstruction-x).sum())
                n += x.size
            value = error/n
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

    def encode(self, frames):
        self.encoder.eval()
        with torch.no_grad():
            x = self.encoder(torch.as_tensor(frames, dtype=torch.float32, device=self.device))
            return x[:, :self.rank].cpu().numpy()

    def decode(self, latent):
        self.decoder.eval()
        with torch.no_grad():
            z = torch.as_tensor(latent, dtype=torch.float32, device=self.device)
            return self.decoder(z).reshape(len(z), *self.shape).cpu().numpy()
