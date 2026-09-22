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
from DataProcessing.Dataset import CompressorDataset, ForecasterDataset, keep_recent
from DataProcessing.scaling import FeatureScaler
from Baselines.OrderReduction.Linear.POD import POD
from Baselines.OrderReduction.DL.AE import AE
from Baselines.OrderReduction.DL.CAE import CAE
from Baselines.OrderReduction.DL.ViTAE import ViTAE
from Baselines.Forecast.Classical.ARX import ARX, Constant
from Baselines.Forecast.DL.DLModel import DLModel
from Baselines.Forecast.DL.networks import GRU, LSTM, Transformer
from .evaluation import forcing_window
from .logging import ExperimentLogger
from .paths import new_run_name, run_directory
from utils import seed_everything, write_json, provenance

COMPRESSORS = {"pod": POD, "cae": CAE, "vit_ae": ViTAE}
MODELS = {"arx": ARX, "constant": Constant, "gru": GRU, "lstm": LSTM, "transformer": Transformer}


CONFIG_KEYS = {"metadata", "output", "run_name", "seed", "trial", "device", "cpu_threads", "validation_fraction", "blocks",
               "batch_size", "joint_batch_size", "workers", "preprocessing_batch_size", "compressor", "model", "logging",
               "evaluation", "Nx", "Ni", "validation_horizon"}


def check_config(config):
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")


def to_device(obj, device):
    """Move torch-backed compressors and models; numpy ones have no device."""
    if isinstance(obj, (AE, DLModel)):
        obj.to(device)
    return obj


def dump(path, value):
    temporary = Path(path).with_suffix(".tmp")
    with temporary.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def loader(dataset, cfg, shuffle=False, batch_size=None):
    return DataLoader(dataset, batch_size=batch_size or cfg.get("batch_size", 64), shuffle=shuffle,
                      num_workers=cfg.get("workers", 0), persistent_workers=cfg.get("workers", 0)>0)


def validation_rollout(model, compressor, dataset):
    """Selection objective: recursive MSE of scaled fields on validation, valid pixels only.

    dataset: scaled validation ForecasterDataset. Each segment is rolled out from its first
    `length` frames to its end. Field space keeps scores comparable when the compressor differs
    between trials (joint training). No test data access.
    """
    sse = count = 0
    for case, lo, hi in dataset.segments:
        _, phi = dataset.arrays(case)
        first = lo + dataset.length
        states = keep_recent(compressor.encode(dataset.frames(case, lo, first))[None], dataset.Nx + 1)
        for k in range(first, hi):
            predicted = model.predict(states, forcing_window(phi, k, dataset.length, dataset.Ni))
            if not np.isfinite(predicted).all():
                return float("inf")
            error = compressor.decode(predicted)[0] - dataset.frames(case, k, k + 1)[0]
            if dataset.mask is not None:
                error = error[..., dataset.mask]
            sse += float(np.square(error, dtype=np.float64).sum())
            count += error.size
            states = keep_recent(np.concatenate((states[:, 1:], predicted[:, None]), axis=1), dataset.Nx + 1)
    return sse/count


def fit(config, resume=False, preprocessing=None):
    """preprocessing: shared stage-A (scaler, compressor) pickle, created if missing (used by HPO trials)."""
    check_config(config)
    cfg = copy.deepcopy(config)
    cfg.setdefault("seed", 42)
    if not cfg.get("run_name") and not resume:
        cfg["run_name"] = new_run_name(cfg)
    directory = run_directory(cfg)
    print(f"Run directory: {directory}", flush=True)
    seed_everything(cfg["seed"])
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    if directory.exists() and not resume:
        raise FileExistsError(f"Run exists: {directory}; choose a new run name or --resume")
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.json"
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
    # Compressor and forecaster datasets share the same block split, so they see the same frames.
    split = dict(validation_fraction=cfg["validation_fraction"], blocks=cfg.get("blocks", 20))
    logger = ExperimentLogger(directory, cfg, resume)
    try:
        shared = preprocessing is not None
        preprocessing = Path(preprocessing) if shared else directory / "preprocessing.pkl"
        if (resume or shared) and preprocessing.exists():
            with preprocessing.open("rb") as file:
                scaler, compressor = pickle.load(file)
        else:
            frames = CompressorDataset(metadata, "train", **split)
            scaler = FeatureScaler(frames.mask).fit(DataLoader(frames, cfg.get("preprocessing_batch_size", 64)))
            cc = cfg["compressor"].copy()
            name = cc.pop("name")
            if issubclass(COMPRESSORS[name], AE):
                cc["device"] = cfg.get("device", "cpu")
            compressor = COMPRESSORS[name](**cc)
            compressor.fit(CompressorDataset(metadata, "train", scaler=scaler, **split),
                           validation=CompressorDataset(metadata, "validation", scaler=scaler, **split), logger=logger)
            dump(preprocessing, (scaler, to_device(compressor, "cpu")))
        to_device(compressor, cfg.get("device", "cpu"))
        # Training windows carry rollout_steps targets for the multi-step loss; validation windows
        # carry validation_horizon targets, so a model trained on K steps is checked on longer rollouts.
        # validation_horizon is raised to rollout_steps when shorter (HPO may sample any rollout_steps).
        rollout_steps = cfg["model"].get("rollout_steps", 1)
        validation_horizon = max(cfg.get("validation_horizon", 1), rollout_steps)
        windows = dict(Nx=cfg["Nx"], Ni=cfg["Ni"], scaler=scaler, **split)
        train_z = ForecasterDataset(metadata, "train", horizon=rollout_steps, compressor=compressor, **windows)
        val_z = ForecasterDataset(metadata, "validation", horizon=validation_horizon, compressor=compressor, **windows)
        validation = ForecasterDataset(metadata, "validation", horizon=validation_horizon, **windows)
        mc = cfg["model"].copy()
        name = mc.pop("name")
        model = MODELS[name](rank=compressor.rank, Nx=cfg["Nx"], Ni=cfg["Ni"], device=cfg.get("device", "cpu"), **mc)
        model.fit(loader(train_z, cfg, shuffle=isinstance(model, DLModel)), loader(val_z, cfg),
                  logger=logger, directory=directory, resume=resume)
        if isinstance(model, DLModel) and isinstance(compressor, AE) and model.frozen_epochs < model.epochs:
            batch_size = cfg.get("joint_batch_size", 8)
            training = ForecasterDataset(metadata, "train", horizon=rollout_steps, **windows)
            model.fit_joint(compressor, loader(training, cfg, shuffle=True, batch_size=batch_size),
                            loader(validation, cfg, batch_size=batch_size), logger=logger)
        score = validation_rollout(model, compressor, validation)
        if not np.isfinite(score):
            raise FloatingPointError("Validation rollout diverged; model not eligible for selection")
        logger.log({"validation/rollout_field_mse": score}, cfg["model"].get("epochs", 0))
        # Everything test() needs; the compressor is saved here because joint training changes it.
        dump(directory/"model.pkl", (scaler, to_device(compressor, "cpu"), to_device(model, "cpu")))
        write_json(directory/"summary.json", {"validation_rollout_field_mse": score})
        return score
    finally:
        logger.close()


def test(config):
    check_config(config)
    seed_everything(config.get("seed", 42))
    torch.set_num_threads(config.get("cpu_threads", 4))
    directory = run_directory(config)
    trained_config = json.loads((directory/"config.json").read_text())
    if config.get("seed", 42) != trained_config["seed"]:
        raise ValueError("Evaluation seed differs from the trained configuration")
    for key in ("Nx", "Ni", "compressor", "model"):
        if config[key] != trained_config[key]:
            raise ValueError(f"Evaluation {key} differs from the trained configuration")
    with (directory/"model.pkl").open("rb") as file:
        scaler, compressor, model = pickle.load(file)
    device = config.get("device", "cpu")
    to_device(model, device)
    to_device(compressor, device)
    dataset = ForecasterDataset(load_metadata(config["metadata"]), "test", Nx=config["Nx"], Ni=config["Ni"])
    logger = ExperimentLogger(directory, config, resume=True)
    try:
        results = model.test(dataset, compressor, scaler, directory, **config.get("evaluation", {}))
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
    """Tune cfg["model"] on the validation rollout, then test the best trial.

    Every trial shares one scaler and compressor (preprocessing.pkl in the HPO folder) and saves
    its fitted model in its own trial folder, so the best trial is tested without refitting.
    """
    config = copy.deepcopy(config)
    if not config.get("run_name"):
        config["run_name"] = new_run_name(config)
    root = run_directory(config)
    root.mkdir(parents=True, exist_ok=True)
    print(f"HPO directory: {root}", flush=True)
    def objective(trial, params):
        cfg = copy.deepcopy(config)
        cfg["trial"] = trial.number
        cfg["model"].update(params)
        return fit(cfg, preprocessing=root / "preprocessing.pkl")
    study = MODELS[config["model"]["name"]].HPO(objective, trials, config.get("seed", 42),
                                                storage=f"sqlite:///{root.resolve() / 'optuna.db'}")
    write_json(root/"best.json", {"trial": study.best_trial.number, "value": study.best_value, "params": study.best_params})
    best = copy.deepcopy(config)
    best["trial"] = study.best_trial.number
    best["model"].update(study.best_params)
    return study, test(best)


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
