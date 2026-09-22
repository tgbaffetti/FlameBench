from abc import ABC, abstractmethod


class Model(ABC):
    """Forecaster of latent states, composed with a separate field compressor.

    fit avoids collision with torch.nn.Module.train(mode), which changes module mode.
    Constructors take explicit keywords only, so a misspelled config key raises TypeError.
    hyperparameters_ranges is the Optuna search space of the constructor keywords;
    dataset_ranges overrides entries of ForecasterDataset.hyperparameters_ranges, where a plain
    value fixes that entry (see Experiments/HPO.py).
    """
    name = "abstract"
    hyperparameters_ranges = {}
    dataset_ranges = {}

    @classmethod
    def build(cls, hyperparameters, **context):
        """Create a forecaster; context holds the values that are never tuned here (rank, Nx, Ni, device)."""
        return cls(**context, **hyperparameters)

    @abstractmethod
    def fit(self, training, validation=None, **kwargs):
        pass

    @abstractmethod
    def predict(self, states, forcing):
        """Next latent x(t+1) from rows s = t-length+1 .. t, length = max(Nx, Ni) + 1.

        states: (batch, length, latent), x(s); forcing: (batch, length), phi(s+1) - 1.
        Rows older than the last Nx + 1 (states) or Ni + 1 (forcing) are zero.
        """

    def test(self, dataset, compressor, scaler, directory, **kwargs):
        from Experiments.evaluation import evaluate
        return evaluate(self, dataset, compressor, scaler, directory, **kwargs)
