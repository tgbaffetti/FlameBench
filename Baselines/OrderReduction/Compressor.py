from abc import ABC, abstractmethod


class Compressor(ABC):
    """Frame compressor operating on feature-normalized (batch, field, height, width) images.

    fit receives a scaled CompressorDataset of training frames. hyperparameters_ranges is the
    Optuna search space of the constructor keywords (see Experiments/HPO.py).

    rank_range is the Optuna search space of the rank (the number of latent values per frame),
    tuned on the forecast objective (reconstruction alone always prefers the largest rank), or
    None when the rank is not tuned. nested_ranks tells how the HPO gets a compressor of each
    sampled rank. True: a smaller compressor is a subset of a larger one, as for POD, so it is
    fitted once at the largest rank and each trial calls truncated(rank), which costs nothing
    (see POD.truncated). False: an autoencoder's latent values have no such order, and its
    best hyperparameters depend on the rank, so stage 1 tunes it separately at each rank; the
    categorical rank_range lists those ranks, and each costs `trials` autoencoder trainings.
    """
    name = "abstract"
    hyperparameters_ranges = {}
    rank_range = None
    nested_ranks = False
    deterministic = False  # True: every seed fits the same compressor, so the HPO fits it once.

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
