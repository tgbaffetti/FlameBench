"""CLI: prepare data separately, then fit (or tune, then fit) and explicitly evaluate held-out tests.

A config has three sections, each a class name plus constructor keywords: compressor, dataset
(Nx, Ni, horizon of the training windows) and forecaster. For hpo, write only the values to keep
fixed; the others are tuned (see Experiments/HPO.py).

hpo runs the benchmark protocol of one compressor-forecaster pair, the same for every model:
  1. tune on the training/validation split (config seed);
  2. fit the best config on the same split once per seed, with early stopping on validation, and
     test each (run_name/seed_N, stage "fit"); config["parallel"] seeds run at a time. A deterministic part gives the same fit for every
     seed, so it is not fitted again: a deterministic compressor (POD) is fitted with the first
     seed and reused by the others, and when the forecaster (ARX) is deterministic too the whole
     model is, so only the first seed is run. A deterministic forecaster on a stochastic
     compressor (CAE + ARX) is still fitted per seed, because its input latents differ.
refit is an ablation, not part of the protocol: it refits a fitted config on all training data
(run_name + "_refit", stage "refit"). Without validation nothing can stop training early, so each
neural stage trains for the number of optimizer steps at which the fit kept its weights; the last
weights of a neural forecaster can then be unstable in long rollouts (see refit_config).
"""
import argparse
import copy
import json
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
import numpy as np
import torch
from DataProcessing.metadata import load_metadata
from DataProcessing.Dataset import CompressorDataset, Dataset, ForecasterDataset
from Baselines.OrderReduction.Linear.POD import POD
from Baselines.OrderReduction.Identity import Identity
from Baselines.OrderReduction.DL.AE import AE
from Baselines.OrderReduction.DL.CAE import CAE
from Baselines.OrderReduction.DL.ViTAE import ViTAE
from Baselines.Forecast.Classical.ARX import ARX, Constant
from Baselines.Forecast.DL.DLModel import DLModel
from Baselines.Forecast.DL.networks import GRU, LSTM, CNN, Transformer
from .evaluation import summarize
from .HPO import optimize
from .logging import ExperimentLogger
from .paths import new_run_name, run_directory
from .pipeline import Pipeline
from utils import seed_everything, write_json, provenance

COMPRESSORS = {"pod": POD, "cae": CAE, "vit_ae": ViTAE, "identity": Identity}
FORECASTERS = {"arx": ARX, "constant": Constant, "gru": GRU, "lstm": LSTM, "cnn": CNN, "transformer": Transformer}


CONFIG_KEYS = {"metadata", "output", "run_name", "seed", "device", "cpu_threads", "validation_fraction", "blocks",
               "batch_size", "joint_batch_size", "workers", "dataloader", "in_memory", "preprocessing_batch_size", "K_eval", "trials",
               "compressor", "dataset", "forecaster", "logging", "evaluation", "stage", "parallel"}


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


def fit(config, resume=False, preprocessing=None):
    """Fit scaler, compressor and forecaster (then joint training when joint_epochs > 0).

    preprocessing: an already fitted (scaler, compressor) to reuse instead of fitting them.
    """
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
    shared = preprocessing
    try:
        preprocessing = directory / "preprocessing.pkl"
        if resume and preprocessing.exists():
            with preprocessing.open("rb") as file:
                pipeline.scaler, compressor = pickle.load(file)
        elif shared is not None:
            pipeline.scaler, compressor = copy.deepcopy(shared)
            dump(preprocessing, (pipeline.scaler, to_device(compressor, "cpu")))
        else:
            pipeline.fit_scaler()
            name, values = section(cfg, "compressor")
            compressor, error = pipeline.train_compressor(COMPRESSORS[name], values, logger)
            if error is not None:  # None in a refit.
                logger.log({"Validation/reconstruction_mse": error})
            dump(preprocessing, (pipeline.scaler, to_device(compressor, "cpu")))
        to_device(compressor, pipeline.device)
        name, values = section(cfg, "forecaster")
        _, dataset = section(cfg, "dataset")
        forecaster, score = pipeline.train_forecaster(FORECASTERS[name], values, ForecasterDataset, dataset, compressor,
                                                      logger, directory, resume)
        if isinstance(forecaster, DLModel) and isinstance(compressor, AE) and forecaster.joint_epochs:
            score = pipeline.fine_tune(forecaster, compressor, ForecasterDataset, dataset, logger)
        # best_epochs: epochs up to the kept weights of each neural stage; refit() scales them.
        best_epochs = {"compressor": getattr(compressor, "best_epochs", None),
                       "forecaster": getattr(forecaster, "best_epochs", None),
                       "joint": getattr(forecaster, "best_joint_epochs", None)}
        logger.log({f"Validation/best_epoch_{stage}": epochs for stage, epochs in best_epochs.items() if epochs})
        if score is not None:  # None in a refit, which has no validation data.
            if not np.isfinite(score):
                raise FloatingPointError("Validation forecast diverged")
            logger.log({"Validation/field_mse": score})
            steps = pipeline.validation_steps
            logger.lines("Validation/error_vs_step", range(1, len(steps) + 1), {"field_mse": steps},
                         "Validation field MSE against rollout step", "rollout step")
        # Everything test() needs; the compressor is saved here because joint training changes it.
        dump(directory / "model.pkl", (pipeline.scaler, to_device(compressor, "cpu"), to_device(forecaster, "cpu")))
        write_json(directory / "summary.json", {"validation_field_mse": score, "train_frames": len(pipeline.frames("train")),
                                                "best_epochs": {k: v for k, v in best_epochs.items() if v}})
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
        log_test(logger, results, dataset.metadata, directory)
        return results
    finally:
        logger.close()


def log_test(logger, results, metadata, directory):
    """Test/<case>/... scalars, the Test/per_field_nrmse table, the Test/<case>/heat_release plot
    (predicted and reference integrated Q, when evaluated) and the Test_Summary/... numbers."""
    scalars = {}
    for name, result in results.items():
        gain_phase = result.get("gain_phase", {})
        scalars.update({f"Test/{name}/nrmse": result["mean_nrmse"], f"Test/{name}/ssim": result.get("mean_ssim"),
                        f"Test/{name}/heat_release_l2": result.get("heat_release_relative_l2"),
                        f"Test/{name}/gain_error": gain_phase.get("relative_gain_error"),
                        f"Test/{name}/phase_error_deg": gain_phase.get("phase_error_deg"),
                        f"Test/{name}/seconds_per_step": result["seconds_per_step"]})
    logger.log(scalars)
    fields = metadata["fields"]
    logger.table("Test/per_field_nrmse", ["case", *fields],
                 [[name, *(result["field_nrmse"][field] for field in fields)] for name, result in results.items()])
    for name in results:
        path = Path(directory) / f"{name}_Q.npz"
        if path.exists():
            with np.load(path) as q:
                logger.lines(f"Test/{name}/heat_release", q["time"], {"reference": q["reference"], "predicted": q["predicted"]},
                             f"{name}: integrated heat release", "time [s]")
    logger.summary(summarize(results, metadata["cases"]))


def refit_config(config):
    """The refit config of a fitted config: all training data, fixed epochs, run_name + "_refit".

    Each neural stage (compressor, forecaster, joint) keeps the number of optimizer steps at which
    the fit kept its weights (Goodfellow et al., Deep Learning, algorithm 7.2). An epoch over all
    training data has all_frames / train_frames times more steps, so
    epochs = max(1, round(best_epochs * train_frames / all_frames)); the ratio comes from the split.
    """
    fitted = json.loads((run_directory(config) / "summary.json").read_text())
    all_frames = len(CompressorDataset(load_metadata(config["metadata"]), "train", validation_fraction=0))
    ratio = fitted["train_frames"] / all_frames
    refit = copy.deepcopy(config)
    refit.update(validation_fraction=0, run_name=config["run_name"] + "_refit", stage="refit")
    for stage, (key, name) in {"compressor": ("compressor", "epochs"), "forecaster": ("forecaster", "epochs"),
                               "joint": ("forecaster", "joint_epochs")}.items():
        if stage in fitted.get("best_epochs", {}):
            refit[key][name] = max(1, round(fitted["best_epochs"][stage] * ratio))
    return refit


def refit(config, seeds):
    """Refit a fitted config on all training data once per seed and test each; return the refit config."""
    refitted = refit_config(config)
    for seed in seeds:
        refitted["seed"] = seed
        fit(refitted)
        test(refitted)
    return refitted


def hpo(config, seeds=None):
    """Tune with config["seed"], then fit and test the best config once per seed.

    Return (best config, {seed: test results}); seeds default to [config["seed"]].
    """
    check_config(config)
    seed_everything(config.get("seed", 42))
    torch.set_num_threads(config.get("cpu_threads", 4))
    config = {**config, "run_name": config.get("run_name") or new_run_name(config)}
    # The search is its own W&B run, "<run_name>/hpo", in the model's group (HPO/ keys).
    searched = {**config, "stage": "hpo"}
    logger = ExperimentLogger(Path(config["output"]) / config["run_name"] / "hpo", searched,
                              name=f"{config['run_name']}/hpo")
    try:
        best = copy.deepcopy(config)
        best.update(optimize(searched, COMPRESSORS[section(config, "compressor")[0]],
                             FORECASTERS[section(config, "forecaster")[0]], logger=logger))
        logger.table("HPO/best_config", ["section", "values"],
                     [[key, json.dumps(best[key])] for key in ("compressor", "dataset", "forecaster")])
    finally:
        logger.close()
    best["stage"] = "fit"
    compressor_class = COMPRESSORS[section(best, "compressor")[0]]
    seeds = list(seeds or [config.get("seed", 42)])
    if compressor_class.deterministic and FORECASTERS[section(best, "forecaster")[0]].deterministic:
        seeds = seeds[:1]  # Every seed would fit the same model.
    results, shared = {}, None
    if compressor_class.deterministic:  # Fitted with the first seed, reused by the others.
        results[seeds[0]] = fit_and_test(best, seeds[0])
        shared = str(run_directory({**best, "seed": seeds[0]}) / "preprocessing.pkl")
        seeds = seeds[1:]
    parallel = min(best.get("parallel", 1), len(seeds))
    if parallel > 1:  # Seeds are independent: config["parallel"] processes at a time.
        with ProcessPoolExecutor(parallel, mp_context=multiprocessing.get_context("spawn")) as pool:
            results.update(zip(seeds, pool.map(fit_and_test, repeat(best), seeds, repeat(shared))))
    else:
        results.update((seed, fit_and_test(best, seed, shared)) for seed in seeds)
    return best, results


def fit_and_test(config, seed, preprocessing=None):
    """Fit and test one seed of a config; preprocessing is the path of a (scaler, compressor) to reuse.

    A top-level function, so that hpo can run seeds in worker processes.
    """
    Dataset.in_memory = config.get("in_memory", False)
    config = {**config, "seed": seed}
    shared = None
    if preprocessing:
        with open(preprocessing, "rb") as file:
            shared = pickle.load(file)
    fit(config, preprocessing=shared)
    return test(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit", "test", "run", "hpo", "refit"],
                        help="hpo: tune once (config seed), then fit and test with each of --seeds; "
                             "refit (ablation): refit a fitted config (its config.json) on all "
                             "training data with each of --seeds")
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
    if args.command == "hpo":
        hpo(config, selected_seeds)
        return
    if args.command == "refit":
        refit(config, selected_seeds)
        return
    for seed in selected_seeds:
        config["seed"] = seed
        if args.command in {"fit", "run"}:
            fit(config, args.resume)
        if args.command in {"test", "run"}:
            test(config)


if __name__ == "__main__":
    main()
