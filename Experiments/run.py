"""CLI: prepare data separately, then fit (or tune, then fit) and explicitly evaluate held-out tests.

A config has three sections, each a class name plus constructor keywords: compressor, dataset
(Nx, Ni, horizon of the training windows) and forecaster. For hpo, write only the values to keep
fixed; the others are tuned (see Experiments/HPO.py) and the best config is then fitted and tested.
"""
import argparse
import copy
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from DataProcessing.metadata import load_metadata
from DataProcessing.Dataset import Dataset, ForecasterDataset
from Baselines.OrderReduction.Linear.POD import POD
from Baselines.OrderReduction.DL.AE import AE
from Baselines.OrderReduction.DL.CAE import CAE
from Baselines.OrderReduction.DL.ViTAE import ViTAE
from Baselines.Forecast.Classical.ARX import ARX, Constant
from Baselines.Forecast.DL.DLModel import DLModel
from Baselines.Forecast.DL.networks import GRU, LSTM, CNN, Transformer
from .HPO import optimize
from .logging import ExperimentLogger
from .paths import new_run_name, run_directory
from .pipeline import Pipeline
from utils import seed_everything, write_json, provenance

COMPRESSORS = {"pod": POD, "cae": CAE, "vit_ae": ViTAE}
FORECASTERS = {"arx": ARX, "constant": Constant, "gru": GRU, "lstm": LSTM, "cnn": CNN, "transformer": Transformer}


CONFIG_KEYS = {"metadata", "output", "run_name", "seed", "device", "cpu_threads", "validation_fraction", "blocks",
               "batch_size", "joint_batch_size", "workers", "dataloader", "in_memory", "preprocessing_batch_size", "K_eval", "trials",
               "compressor", "dataset", "forecaster", "logging", "evaluation"}


def check_config(config):
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")


def section(config, key):
    """(class name, constructor keywords) of a config section."""
    values = dict(config.get(key, {}))
    return values.pop("name", None), values


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


def fit(config, resume=False):
    """Fit scaler, compressor and forecaster (then joint training when joint_epochs > 0)."""
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
    pipeline = Pipeline(cfg, metadata)
    logger = ExperimentLogger(directory, cfg, resume)
    try:
        preprocessing = directory / "preprocessing.pkl"
        if resume and preprocessing.exists():
            with preprocessing.open("rb") as file:
                pipeline.scaler, compressor = pickle.load(file)
        else:
            pipeline.fit_scaler()
            name, values = section(cfg, "compressor")
            compressor, error = pipeline.train_compressor(COMPRESSORS[name], values, logger)
            logger.log({"validation/reconstruction_mse": error}, 0)
            dump(preprocessing, (pipeline.scaler, to_device(compressor, "cpu")))
        to_device(compressor, pipeline.device)
        name, values = section(cfg, "forecaster")
        _, dataset = section(cfg, "dataset")
        forecaster, score = pipeline.train_forecaster(FORECASTERS[name], values, ForecasterDataset, dataset, compressor,
                                                      logger, directory, resume)
        if isinstance(forecaster, DLModel) and isinstance(compressor, AE) and forecaster.joint_epochs:
            score = pipeline.fine_tune(forecaster, compressor, ForecasterDataset, dataset, logger)
        if not np.isfinite(score):
            raise FloatingPointError("Validation forecast diverged")
        logger.log({"validation/field_mse": score}, 0)
        # Everything test() needs; the compressor is saved here because joint training changes it.
        dump(directory / "model.pkl", (pipeline.scaler, to_device(compressor, "cpu"), to_device(forecaster, "cpu")))
        write_json(directory / "summary.json", {"validation_field_mse": score})
        return score
    finally:
        logger.close()


def test(config):
    check_config(config)
    seed_everything(config.get("seed", 42))
    torch.set_num_threads(config.get("cpu_threads", 4))
    directory = run_directory(config)
    trained_config = json.loads((directory / "config.json").read_text())
    if config.get("seed", 42) != trained_config["seed"]:
        raise ValueError("Evaluation seed differs from the trained configuration")
    for key in ("compressor", "dataset", "forecaster"):
        if config.get(key) != trained_config.get(key):
            raise ValueError(f"Evaluation {key} differs from the trained configuration")
    with (directory / "model.pkl").open("rb") as file:
        scaler, compressor, forecaster = pickle.load(file)
    device = config.get("device", "cpu")
    to_device(forecaster, device)
    to_device(compressor, device)
    dataset = ForecasterDataset(load_metadata(config["metadata"]), "test", Nx=forecaster.Nx, Ni=forecaster.Ni)
    logger = ExperimentLogger(directory, config, resume=True)
    try:
        results = forecaster.test(dataset, compressor, scaler, directory, **config.get("evaluation", {}))
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


def hpo(config):
    """Tune the config on validation data, then fit and test the best config; return (best config, test results)."""
    check_config(config)
    seed_everything(config.get("seed", 42))
    torch.set_num_threads(config.get("cpu_threads", 4))
    best = copy.deepcopy(config)
    best.update(optimize(config, COMPRESSORS[section(config, "compressor")[0]],
                         FORECASTERS[section(config, "forecaster")[0]]))
    fit(best)
    return best, test(best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit", "test", "run", "hpo"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--metadata", help="Dataset metadata.json, replacing the config value")
    parser.add_argument("--device")
    parser.add_argument("--output", help="Results root directory")
    parser.add_argument("--run-name", help="Shared model/timestamp folder; required for test or resume")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int)
    seeds.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    # Read each trajectory into RAM once (about 18 GB for all cases) instead of from the shared disk.
    Dataset.in_memory = config.get("in_memory", False)
    for key in ("metadata", "device", "output", "run_name"):
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
            hpo(config)


if __name__ == "__main__":
    main()
