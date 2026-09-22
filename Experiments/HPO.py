"""Hyperparameter optimization of a compressor, a forecaster, or both, on validation data.

The search spaces are the hyperparameters_ranges of the given classes. Every value written in
the config sections compressor, dataset and forecaster is fixed; the other range entries are
tuned. A forecaster's dataset_ranges overrides the ForecasterDataset ranges (ARX fixes horizon 1).
The search runs in up to three stages, each keeping the best result of the stage before:
  1. compressor: validation reconstruction MSE (skipped when a trained compressor is given);
  2. forecaster and dataset (Nx, Ni, horizon), compressor frozen: validation field MSE of
     K_eval-step recursive forecasts;
  3. the forecaster's joint_* keys, training autoencoder and neural forecaster together: same
     objective. It runs for an autoencoder trained in stage 1, or a given one when
     fine_tune_compressor is true.
Errors are on scaled fields, valid pixels only. Each stage runs config["trials"] trials (default
2), or one fit when it has nothing to tune. Diverged trials are skipped.
"""
import copy
import warnings
import numpy as np
import optuna
from Baselines.OrderReduction.DL.AE import AE
from Baselines.Forecast.DL.DLModel import DLModel
from DataProcessing.Dataset import ForecasterDataset
from utils import seed_everything
from .pipeline import Pipeline


# Layer lists (e.g. GRU hiddens) are categorical choices; Optuna warns because a saved study
# could not store them, but studies here stay in memory.
warnings.filterwarnings("ignore", message="Choices for a categorical distribution")


def sample(trial, ranges, fixed, prefix):
    """The fixed values plus one value per remaining range entry; a plain entry is itself the value."""
    values = dict(fixed)
    for key, spec in ranges.items():
        if key in fixed:
            continue
        name = prefix + key
        if not isinstance(spec, dict):
            values[key] = spec
        elif spec["type"] == "categorical":
            values[key] = trial.suggest_categorical(name, spec["choices"])
        elif spec["type"] == "int":
            values[key] = trial.suggest_int(name, spec["low"], spec["high"])
        else:
            values[key] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    return values


def tunable(ranges, fixed):
    return any(isinstance(spec, dict) and key not in fixed for key, spec in ranges.items())


def minimize(train, trials, seed):
    """Run train(trial) -> (values, fitted objects, error) trials times; return the best (values, fitted)."""
    best = {"error": float("inf")}

    def objective(trial):
        seed_everything(seed)
        values, fitted, error = train(trial)
        if not np.isfinite(error):
            raise FloatingPointError("Diverged trial")
        if error < best["error"]:
            best.update(error=error, values=values, fitted=fitted)
        return error

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, catch=(FloatingPointError,), show_progress_bar=True)
    if "values" not in best:
        raise RuntimeError("Every trial diverged")
    return best["values"], best["fitted"]


def optimize(config, compressor_class=None, forecaster_class=None, dataset_class=ForecasterDataset,
             compressor=None, fine_tune_compressor=True):
    """Return the best config sections {"compressor", "dataset", "forecaster"} with the chosen values.

    Give compressor_class to tune a compressor, forecaster_class to tune a forecaster, or both.
    A trained compressor can replace compressor_class; it is frozen in stage 2, and fine-tuned in
    stage 3 only when fine_tune_compressor is true. The "compressor" section is then left out.
    """
    if (compressor_class is None) == (compressor is None):
        raise ValueError("Give either compressor_class or a trained compressor")
    pipeline = Pipeline(config)
    pipeline.fit_scaler()
    trials, seed = config.get("trials", 2), config.get("seed", 42)
    fixed = {key: {k: v for k, v in config.get(key, {}).items() if k != "name"}
             for key in ("compressor", "dataset", "forecaster")}
    result = {}
    if compressor is None:
        ranges = compressor_class.hyperparameters_ranges

        def train(trial):
            values = sample(trial, ranges, fixed["compressor"], "compressor.")
            return (values, *pipeline.train_compressor(compressor_class, values))

        values, compressor = minimize(train, trials if tunable(ranges, fixed["compressor"]) else 1, seed)
        result["compressor"] = {"name": compressor_class.name, **values}
        fine_tune_compressor = True
    if forecaster_class is None:
        return result

    dataset_ranges = {**dataset_class.hyperparameters_ranges, **forecaster_class.dataset_ranges}
    ranges = {k: v for k, v in forecaster_class.hyperparameters_ranges.items() if not k.startswith("joint_")}

    def train(trial):
        dataset_values = sample(trial, dataset_ranges, fixed["dataset"], "dataset.")
        values = sample(trial, ranges, fixed["forecaster"], "forecaster.")
        forecaster, error = pipeline.train_forecaster(forecaster_class, values, dataset_class, dataset_values, compressor)
        return (dataset_values, values), forecaster, error

    tuned = tunable(dataset_ranges, fixed["dataset"]) or tunable(ranges, fixed["forecaster"])
    (dataset_values, values), forecaster = minimize(train, trials if tuned else 1, seed)
    result["dataset"] = dataset_values
    result["forecaster"] = {"name": forecaster_class.name, **values}

    joint_ranges = {k: v for k, v in forecaster_class.hyperparameters_ranges.items() if k.startswith("joint_")}
    if issubclass(forecaster_class, DLModel):
        if not (joint_ranges and isinstance(compressor, AE) and fine_tune_compressor):
            result["forecaster"]["joint_epochs"] = 0
            return result
        fixed_joint = {k: v for k, v in fixed["forecaster"].items() if k.startswith("joint_")}

        def train(trial):
            values = sample(trial, joint_ranges, fixed_joint, "forecaster.")
            model, codec = copy.deepcopy(forecaster), copy.deepcopy(compressor)
            for key, value in values.items():
                setattr(model, key, value)
            return values, None, pipeline.fine_tune(model, codec, dataset_class, dataset_values)

        values, _ = minimize(train, trials if tunable(joint_ranges, fixed_joint) else 1, seed)
        result["forecaster"].update(values)
    return result
