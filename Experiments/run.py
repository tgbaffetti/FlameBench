"""CLI: prepare data separately, then fit/HPO and explicitly evaluate held-out tests."""
import argparse
import copy
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from DataProcessing.metadata import load_metadata
from DataProcessing.Dataset import TrainingDataset, TestDataset
from DataProcessing.scaling import FeatureScaler
from DataProcessing.latent import LatentDataset
from Baselines.OrderReduction.Linear.POD import POD
from Baselines.OrderReduction.DL.AE import AE
from Baselines.OrderReduction.DL.VAE import VAE
from Baselines.OrderReduction.Identity import Identity
from Baselines.Forecast.Classical.ARX import ARX, NARX, Constant
from Baselines.Forecast.DL.networks import GRU, LSTM, Transformer
from Baselines.Forecast.DL.deeponet import DeepONet
from Baselines.Forecast.DL.transolver import Transolver
from Baselines.Forecast.DL.meshgraphnet import MeshGraphNet
from .logging import ExperimentLogger
from .paths import new_run_name, run_directory
from .evaluation import evaluate
from utils import seed_everything, write_json, provenance

COMPRESSORS = {"pod": POD, "ae": AE, "vae": VAE, "identity": Identity}
MODELS = {"arx": ARX, "narx": NARX, "persistence": Constant, "gru": GRU, "lstm": LSTM,
          "transformer": Transformer, "deeponet": DeepONet, "transolver": Transolver,
          "meshgraphnet": MeshGraphNet}


def dump(path, value):
    temporary = Path(path).with_suffix(".tmp")
    with temporary.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def loader(dataset, cfg, shuffle=False):
    return DataLoader(dataset, batch_size=cfg.get("batch_size", 64), shuffle=shuffle,
                      num_workers=cfg.get("workers", 0), persistent_workers=cfg.get("workers", 0)>0)


def validation_rollout(model, latent_dataset, window=None):
    """Selection objective: recursive latent MSE on validation, no test data access.
    window=w restarts from the observed latents every w steps (windowed rollout)."""
    if window is not None and window < 1:
        raise ValueError("window must be a positive integer or None")
    sse = count = 0
    for data_path, phi_path in latent_dataset.paths:
        x, phi = np.load(data_path, mmap_mode="r"), np.load(phi_path, mmap_mode="r")
        first = latent_dataset.context
        history = np.array(x[first-latent_dataset.history:first])[None]
        for k in range(first, len(x)):
            if window and k > first and (k - first) % window == 0:
                history = np.array(x[k-latent_dataset.history:k])[None]
            forcing = np.array(phi[k-latent_dataset.Ni-1:k+1], dtype=np.float32)[None]
            predicted = model.predict(history, forcing)
            if not np.isfinite(predicted).all():
                return float("inf")
            sse += float(np.square(predicted[0].astype(np.float64)-x[k]).sum())
            count += x.shape[-1]
            history = np.concatenate((history[:, 1:], predicted[:, None]), axis=1)
    return sse/count


def fit(config, resume=False):
    cfg = copy.deepcopy(config)
    cfg.setdefault("seed", 42)
    if not cfg.get("run_name") and not resume:
        cfg["run_name"] = new_run_name(cfg)
    directory = run_directory(cfg)
    print(f"Run directory: {directory}", flush=True)
    seed_everything(cfg.get("seed", 42))
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    directory.mkdir(parents=True, exist_ok=resume)
    config_path = directory / "config.json"
    if config_path.exists() and not resume:
        raise FileExistsError(f"Run exists: {directory}; choose a new output or --resume")
    if resume and config_path.exists():
        old = json.loads(config_path.read_text())
        if old != cfg:
            raise ValueError("Resume requires the identical configuration")
    write_json(config_path, cfg)
    write_json(directory / "environment.json", provenance())
    metadata = load_metadata(cfg["metadata"])
    metadata_path = directory / "metadata.resolved.json"
    if resume and metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("Dataset metadata changed since original run")
    write_json(metadata_path, metadata)
    training = TrainingDataset(metadata, Nx=cfg["Nx"], Ni=cfg["Ni"], validation_fraction=cfg["validation_fraction"])
    validation = TrainingDataset(metadata, Nx=cfg["Nx"], Ni=cfg["Ni"], validation_fraction=cfg["validation_fraction"], partition="validation")
    logger = ExperimentLogger(directory, cfg, resume)
    try:
        preprocessing = directory / "preprocessing.pkl"
        if resume and preprocessing.exists():
            with preprocessing.open("rb") as file:
                scaler, compressor = pickle.load(file)
        else:
            scaler = FeatureScaler().fit(training.snapshot_batches(cfg.get("preprocessing_batch_size", 64)))
            cc = cfg["compressor"].copy()
            name = cc.pop("name")
            if name in {"ae", "vae"}:
                cc["device"] = cfg.get("device", "cpu")
            compressor = COMPRESSORS[name](**cc)
            compressor.fit(training, scaler, validation=validation, logger=logger)
            if isinstance(compressor, AE):
                compressor.device = "cpu"
                compressor.encoder.cpu()
                compressor.decoder.cpu()
            dump(preprocessing, (scaler, compressor))
        if isinstance(compressor, AE):
            compressor.device = cfg.get("device", "cpu")
            compressor.encoder.to(compressor.device)
            compressor.decoder.to(compressor.device)
        horizon = cfg["model"].get("unroll_steps", 1)
        if horizon > 1 and cfg["model"]["name"] in {"arx", "persistence"}:
            raise ValueError("unroll_steps applies to neural models only")
        train_z = LatentDataset(training, compressor, scaler, directory/"latent"/"training", horizon=horizon)
        val_z = LatentDataset(validation, compressor, scaler, directory/"latent"/"validation", horizon=horizon)
        mc = cfg["model"].copy()
        name = mc.pop("name")
        model = MODELS[name](rank=compressor.rank, Nx=cfg["Nx"], Ni=cfg["Ni"], device=cfg.get("device", "cpu"), **mc)
        model.fit(loader(train_z, cfg, shuffle=name not in {"arx", "narx", "persistence"}), loader(val_z, cfg),
                  logger=logger, directory=directory, resume=resume)
        score = validation_rollout(model, val_z, window=cfg.get("validation_window"))
        if not np.isfinite(score):
            raise FloatingPointError("Validation rollout diverged; model not eligible for selection")
        logger.log({"validation/rollout_latent_mse": score}, cfg["model"].get("epochs", 0))
        if hasattr(model, "network"):
            model.network.cpu()
            model.device = torch.device("cpu")
        dump(directory/"model.pkl", model)
        write_json(directory/"summary.json", {"validation_rollout_latent_mse": score,
                                              "validation_window": cfg.get("validation_window")})
        return score
    finally:
        logger.close()


def test(config):
    seed_everything(config.get("seed", 42))
    torch.set_num_threads(config.get("cpu_threads", 4))
    directory = run_directory(config)
    trained_config = json.loads((directory/"config.json").read_text())
    if config.get("seed", 42) != trained_config["seed"]:
        raise ValueError("Evaluation seed differs from the trained configuration")
    for key in ("Nx", "Ni", "compressor", "model"):
        if config[key] != trained_config[key]:
            raise ValueError(f"Evaluation {key} differs from the trained configuration")
    with (directory/"preprocessing.pkl").open("rb") as file:
        scaler, compressor = pickle.load(file)
    with (directory/"model.pkl").open("rb") as file:
        model = pickle.load(file)
    device = config.get("device", "cpu")
    if hasattr(model, "network"):
        model.device = torch.device(device)
        model.network.to(model.device)
    if isinstance(compressor, AE):
        compressor.device = device
        compressor.encoder.to(device)
        compressor.decoder.to(device)
    dataset = TestDataset(load_metadata(config["metadata"]), Nx=config["Nx"], Ni=config["Ni"])
    logger = ExperimentLogger(directory, config, resume=True)
    try:
        results = evaluate(model, dataset, compressor, scaler, directory, **config.get("evaluation", {}))
        for i, (name, result) in enumerate(results.items()):
            scalars = {f"test/{name}/mean_nrmse": result["mean_nrmse"],
                       f"test/{name}/heat_release_relative_l2": result.get("heat_release_relative_l2"),
                       f"test/{name}/seconds_per_step": result["seconds_per_step"]}
            for field, value in result["field_nrmse"].items():
                scalars[f"test/{name}/nrmse/{field}"] = value
            for metric in ("reference_gain", "predicted_gain", "relative_gain_error", "phase_error_deg"):
                if metric in result.get("gain_phase", {}):
                    scalars[f"test/{name}/{metric}"] = result["gain_phase"][metric]
            logger.log(scalars, i)
        return results
    finally:
        logger.close()


def hpo(config, trials):
    import optuna
    config = copy.deepcopy(config)
    if not config.get("run_name"):
        config["run_name"] = new_run_name(config)
    root = run_directory(config)
    root.mkdir(parents=True, exist_ok=True)
    print(f"HPO directory: {root}", flush=True)
    def objective(trial):
        cfg = copy.deepcopy(config)
        cfg["trial"] = trial.number
        for key, spec in MODELS[cfg["model"]["name"]].hyperparams.items():
            if spec["type"] == "categorical":
                value = trial.suggest_categorical(key, spec["choices"])
            elif spec["type"] == "int":
                value = trial.suggest_int(key, spec["low"], spec["high"])
            else:
                value = trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))
            cfg["model"][key] = value
        return fit(cfg)
    study = optuna.create_study(study_name="forecast", storage=f"sqlite:///{root.resolve() / 'optuna.db'}",
                load_if_exists=True, direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.get("seed",42)))
    study.optimize(objective, n_trials=trials, catch=(FloatingPointError,))
    write_json(root/"best.json", {"trial": study.best_trial.number, "value": study.best_value, "params": study.best_params})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit", "test", "run", "hpo"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--device")
    parser.add_argument("--output", help="Results root directory")
    parser.add_argument("--run-name", help="Shared model/timestamp folder; required for test or resume")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int)
    seeds.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    for key in ("device", "output", "run_name"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if not config.get("run_name"):
        if args.command == "test" or args.resume:
            parser.error("test/resume requires --run-name or run_name in the saved config")
        config["run_name"] = new_run_name(config)
    selected_seeds = args.seeds if args.seeds is not None else [
        args.seed if args.seed is not None else config.get("seed", 42)
    ]
    if len(set(selected_seeds)) != len(selected_seeds):
        parser.error("Seeds must be distinct")
    for seed in selected_seeds:
        config["seed"] = seed
        if args.command in {"fit", "run"}:
            fit(config, args.resume)
        if args.command in {"test", "run"}:
            test(config)
        if args.command == "hpo":
            hpo(config, args.trials)


if __name__ == "__main__":
    main()
