from abc import ABC, abstractmethod


class Compressor(ABC):
    """Frame compressor operating on feature-normalized (batch, field, cell) arrays."""
    @abstractmethod
    def fit(self, dataset, scaler, **kwargs):
        pass

    @abstractmethod
    def encode(self, frames):
        pass

    @abstractmethod
    def decode(self, latent):
        pass
