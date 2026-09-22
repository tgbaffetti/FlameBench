from abc import ABC, abstractmethod


class Compressor(ABC):
    """Frame compressor operating on feature-normalized (batch, field, height, width) images.

    fit receives a scaled CompressorDataset of training frames. hyperparameters_ranges is the
    Optuna search space of the constructor keywords (see Experiments/HPO.py).

    rank_range is the Optuna search space of the rank (the number of latent values per frame),
    or None when the rank cannot be tuned cheaply. When set, the HPO fits the compressor once at
    the largest rank of the range; each trial then calls truncated(rank) to get a smaller
    compressor without refitting, and scores it on the forecast objective. This only works when
    the smaller compressor is a subset of the larger one, as for POD (see POD.truncated). An
    autoencoder's latent values have no such order, so autoencoders leave rank_range as None.
    """
    name = "abstract"
    hyperparameters_ranges = {}
    rank_range = None

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
