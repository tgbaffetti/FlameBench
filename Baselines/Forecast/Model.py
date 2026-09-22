from abc import ABC, abstractmethod


class Model(ABC):
    """Forecaster of latent states, composed with a separate field compressor.

    fit avoids collision with torch.nn.Module.train(mode), which changes module mode.
    Constructors take explicit keywords only, so a misspelled config key raises TypeError.
    """
    name = "abstract"
    hyperparams = {}

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

    @classmethod
    def suggest(cls, trial):
        """Sample one value per entry of cls.hyperparams."""
        params = {}
        for key, spec in cls.hyperparams.items():
            if spec["type"] == "categorical":
                params[key] = trial.suggest_categorical(key, spec["choices"])
            elif spec["type"] == "int":
                params[key] = trial.suggest_int(key, spec["low"], spec["high"])
            else:
                params[key] = trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))
        return params

    @classmethod
    def HPO(cls, objective, trials=20, seed=42, storage=None):
        """Minimize objective(trial, params) over cls.hyperparams; diverged trials are skipped."""
        import optuna
        study = optuna.create_study(study_name=cls.name, storage=storage, load_if_exists=storage is not None,
                                    direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(lambda trial: objective(trial, cls.suggest(trial)), n_trials=trials,
                       catch=(FloatingPointError,))
        return study
