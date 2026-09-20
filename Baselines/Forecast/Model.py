from abc import ABC, abstractmethod


class Model(ABC):
    """Forecaster of latent states, composed with a separate field compressor.

    fit avoids collision with torch.nn.Module.train(mode), which changes module mode.
    """
    name = "abstract"
    hyperparams = {}

    @abstractmethod
    def fit(self, training, validation=None, **kwargs):
        pass

    @abstractmethod
    def predict(self, history, forcing):
        """history: (batch, Nx+1, latent); forcing: (batch, Ni+2), phi(t-Ni:t+1)."""

    def test(self, dataset, **kwargs):
        from Experiments.evaluation import evaluate
        return evaluate(self, dataset, **kwargs)

    @classmethod
    def HPO(cls, objective, trials=20, seed=42):
        import optuna
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=trials)
        return study
