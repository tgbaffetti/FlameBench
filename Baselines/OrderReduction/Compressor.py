from abc import ABC, abstractmethod


class Compressor(ABC):
    """Frame compressor operating on feature-normalized (batch, field, height, width) images.

    fit receives a scaled CompressorDataset of training frames. hyperparameters_ranges is the
    Optuna search space of the constructor keywords (see Experiments/HPO.py).
    """
    name = "abstract"
    hyperparameters_ranges = {}

    @classmethod
    def build(cls, hyperparameters, **context):
        """Create a compressor; context holds the values that are never tuned (rank, device)."""
        return cls(**context, **hyperparameters)

    @abstractmethod
    def fit(self, dataset, **kwargs):
        pass

    @abstractmethod
    def encode(self, frames):
        pass

    @abstractmethod
    def decode(self, latent):
        pass
