from abc import ABC, abstractmethod


class Compressor(ABC):
    """Frame compressor operating on feature-normalized (batch, field, height, width) arrays.

    fit receives a scaled CompressorDataset of training frames.
    """
    @abstractmethod
    def fit(self, dataset, **kwargs):
        pass

    @abstractmethod
    def encode(self, frames):
        pass

    @abstractmethod
    def decode(self, latent):
        pass
