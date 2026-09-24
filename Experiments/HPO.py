"""Hyperparameter optimization of a compressor, a forecaster, or both, on validation data.

The search spaces are the hyperparameters_ranges of the given classes. Every value written in
the config sections compressor, dataset and forecaster is fixed; the other range entries are
tuned. A forecaster's dataset_ranges overrides the ForecasterDataset ranges (ARX fixes horizon 1).
The search runs in up to three stages, each keeping the best result of the stage before:
  1. compressor: validation reconstruction MSE (skipped when a trained compressor is given).
     Reconstruction always improves with rank, so the rank is not chosen here. When the config
     gives no rank, POD (no hyperparameters) is fitted once at the largest rank of its
     rank_range, and an autoencoder is tuned separately at each rank of its rank_range
     (`trials` trials per rank), since its best hyperparameters depend on the rank; a rank
     where every trial fails is dropped from the stage-2 rank choices;
  2. forecaster, dataset (Nx, Ni, horizon) and the rank (when the config gives none), compressor
     frozen: validation field MSE of K_eval-step recursive forecasts. For POD each trial keeps
     the first `rank` modes of the stage-1 fit (see POD.truncated); an autoencoder trial uses
     the stage-1 autoencoder tuned at the sampled rank. Trials whose
     values cannot work together (a window longer than a data segment, a CNN needing more rows
     than Nx or Ni give) are skipped before training and do not count toward the trials;
  3. the forecaster's joint_* keys, training autoencoder and neural forecaster together: same
     objective. It runs for an autoencoder trained in stage 1, or a given one when
     fine_tune_compressor is true.
Errors are on scaled fields, valid pixels only. Each stage runs config["trials"] trials (default
2), or one fit when it has nothing to tune. Diverged trials are skipped.

With config["parallel"] = k > 1, each stage runs k trials at a time in k worker processes (one
pool per stage, on the same GPU); TPE then uses the constant-liar strategy, so trials running
together do not all sample the same point. A trial that runs out of GPU memory counts as failed,
like a diverged one.

Stage 1 does not depend on the forecaster, so its result at each rank is cached under
<output>/hpo_cache/ and reused by every pair with the same compressor, search space, fixed
values, constructor defaults, rank, trials, seed, data and split (see stage_one_cache). A file lock per rank stops
pairs that run at the same time from tuning the same rank twice: a pair tunes the ranks nobody
else is tuning, then waits for the others. The key
holds the metadata, not the data files: delete hpo_cache after changing prepared data.
"""
import copy
import fcntl
import hashlib
import inspect
import json
import multiprocessing
import pickle
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import optuna
import torch
from Baselines.OrderReduction.DL.AE import AE
from Baselines.Forecast.DL.DLModel import DLModel
from DataProcessing.Dataset import Dataset, ForecasterDataset
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
            values[key] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
        else:
            values[key] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    return values


def tunable(ranges, fixed):
    return any(isinstance(spec, dict) and key not in fixed for key, spec in ranges.items())


class SearchFailed(RuntimeError):
    """Every trial of a search diverged, ran out of GPU memory or was infeasible."""


# A trial that fails without stopping the search: diverged (nonfinite) or out of GPU memory.
TRIAL_FAILURES = (FloatingPointError, torch.cuda.OutOfMemoryError)
WORKER = {}  # In each process: the stage's "pipeline" (with the fitted scaler) and "codecs" by rank.


def move(obj, device):
    """Move a torch-backed compressor or forecaster; numpy ones (POD, ARX) have no device."""
    if hasattr(obj, "to"):
        obj.to(device)
    return obj


def start_worker(config, scaler, codecs):
    """Initializer of a worker process: its own pipeline with the fitted scaler, and the compressors."""
    Dataset.in_memory = config.get("in_memory", False)
    torch.set_num_threads(config.get("cpu_threads", 4))
    pipeline = Pipeline(config)
    pipeline.scaler = scaler
    WORKER.update(pipeline=pipeline, codecs={rank: move(codec, pipeline.device) for rank, codec in codecs.items()})


@contextmanager
def workers(config, pipeline, codecs=None):
    """The process pool of one stage (config["parallel"] workers), or None to run trials here."""
    codecs = codecs or {}
    if config.get("parallel", 1) <= 1:
        WORKER.update(pipeline=pipeline, codecs=codecs)
        yield None
        return
    portable = {rank: move(copy.deepcopy(codec), "cpu") for rank, codec in codecs.items()}
    with ProcessPoolExecutor(config["parallel"], mp_context=multiprocessing.get_context("spawn"),
                             initializer=start_worker, initargs=(config, pipeline.scaler, portable)) as pool:
        yield pool


def run_job(job, seed):
    """Run one trial's job = (function, arguments) in this process; returns (fitted object, error)."""
    seed_everything(seed)
    function, arguments = job
    return function(*arguments)


def train_compressor_job(compressor_class, values):
    compressor, error = WORKER["pipeline"].train_compressor(compressor_class, values)
    return move(compressor, "cpu"), error


def train_forecaster_job(forecaster_class, values, dataset_class, dataset_values, rank):
    """Forecaster on the compressor of this rank: stored, or a truncation of the largest (POD)."""
    codecs = WORKER["codecs"]
    codec = codecs[rank] if rank in codecs else codecs[max(codecs)].truncated(rank)
    forecaster, error = WORKER["pipeline"].train_forecaster(forecaster_class, values, dataset_class,
                                                            dataset_values, codec)
    return move(forecaster, "cpu"), error


def fine_tune_job(forecaster, compressor, values, dataset_class, dataset_values):
    pipeline = WORKER["pipeline"]
    model, codec = move(copy.deepcopy(forecaster), pipeline.device), move(copy.deepcopy(compressor), pipeline.device)
    for key, value in values.items():
        setattr(model, key, value)
    return None, pipeline.fine_tune(model, codec, dataset_class, dataset_values)


def minimize(propose, trials, seed, logger=None, name="search", pool=None):
    """Run trials until `trials` complete; return the best (values, fitted object).

    propose(trial) samples a trial's values in this process and returns (values, job); the job
    runs here (pool None) or in the pool, up to one per worker at a time. A proposal whose values
    cannot work together raises optuna.TrialPruned (see check_feasible): it costs no training and
    does not count toward `trials`; at most 10 * trials are sampled, so a search space that is
    almost all infeasible still ends. A job that diverges or runs out of GPU memory is a failed
    trial; when no trial completes it raises SearchFailed. With a logger, each completed trial logs HPO/<name>/objective and best_so_far against
    HPO/<name>/trial, and the end of the search logs the table of every trial and, when Optuna
    can estimate it, the importance of each parameter.
    """
    best = {"error": float("inf")}
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed, constant_liar=pool is not None))
    capacity = pool._max_workers if pool else 1
    running, asked = {}, 0

    def finish(trial, values, outcome):
        try:
            fitted, error = outcome()
        except TRIAL_FAILURES as failure:
            print(f"Trial {trial.number} of {name} failed: {type(failure).__name__}: {failure}", flush=True)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            return
        if not np.isfinite(error):
            print(f"Trial {trial.number} of {name} diverged", flush=True)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            return
        study.tell(trial, error)
        if error < best["error"]:
            best.update(error=error, values=values, fitted=fitted)
        if logger:
            logger.log({f"HPO/{name}/objective": error, f"HPO/{name}/best_so_far": best["error"]},
                       trial.number, axis=f"HPO/{name}/trial")

    def completed():
        return len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))

    while True:
        while len(running) < capacity and asked < 10 * trials and completed() + len(running) < trials:
            trial = study.ask()
            asked += 1
            try:
                values, job = propose(trial)
            except optuna.TrialPruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                continue
            if pool is None:
                finish(trial, values, lambda: run_job(job, seed))
            else:
                running[pool.submit(run_job, job, seed)] = (trial, values)
        if not running:
            break
        done, _ = wait(running, return_when=FIRST_COMPLETED)
        for future in done:
            trial, values = running.pop(future)
            finish(trial, values, future.result)
    if logger:
        log_study(logger, study, name)
    if "values" not in best:
        raise SearchFailed(f"Every trial of {name} diverged or was infeasible")
    return best["values"], best["fitted"]


def log_study(logger, study, name):
    """HPO/<name>/trials (number, state, objective and parameters of every trial; pruned trials
    were infeasible, failed ones diverged) and HPO/<name>/param_importance."""
    parameters = sorted({key for trial in study.trials for key in trial.params})
    rows = [[trial.number, trial.state.name.lower(), trial.value,
             *(json.dumps(trial.params.get(key)) for key in parameters)] for trial in study.trials]
    logger.table(f"HPO/{name}/trials", ["trial", "state", "objective", *parameters], rows)
    if len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))) < 10:
        return  # Too few trials for Optuna to tell parameters apart.
    try:
        importance = optuna.importance.get_param_importances(study)
    except (RuntimeError, ValueError, TypeError):
        # RuntimeError/ValueError: e.g. every trial had the same parameters. TypeError: Optuna's
        # PedAnova evaluator cannot hash list-valued categorical choices (hiddens, channels).
        return
    if importance:
        logger.table(f"HPO/{name}/param_importance", ["parameter", "importance"],
                     [[key, float(value)] for key, value in importance.items()])


def check_feasible(pipeline, forecaster_class, dataset_class, dataset_values, values, rank):
    """Raise optuna.TrialPruned when the sampled values cannot be trained together.

    Builds, without encoding any frames, the training and validation windows (they fail when a
    window, max(Nx, Ni) + 1 rows plus the horizon, is longer than a data segment) and the
    forecaster (it fails, for example, when a CNN's convolutions need more rows than
    max(Nx, Ni) + 1). Both raise ValueError, which is turned into a pruned trial.
    """
    try:
        windows = pipeline.windows(dataset_class, dataset_values, "train")
        pipeline.windows(dataset_class, dataset_values, "validation", validation=True)
        forecaster_class.build(pipeline.operator_defaults(forecaster_class, values),
                               input_size=rank + 1, output_size=rank, Nx=windows.Nx, Ni=windows.Ni,
                               device="cpu")
    except ValueError as error:
        raise optuna.TrialPruned(f"Infeasible values: {error}") from error


def stage_one_cache(config, pipeline, compressor_class, fixed, rank, trials, seed, tune, wait=True):
    """(values, compressor) of stage 1 at one rank: from <output>/hpo_cache/, or tune() and store it.

    Without config["output"] (direct calls, tests) nothing is cached. With wait=False it returns
    None at once when another process holds this rank's lock (it is tuning that rank).
    """
    if not config.get("output"):
        return tune()
    defaults = {name: parameter.default for name, parameter in
                inspect.signature(compressor_class.__init__).parameters.items()
                if parameter.default is not inspect.Parameter.empty and name not in ("rank", "device")}
    key = {"compressor": compressor_class.name, "ranges": compressor_class.hyperparameters_ranges,
           "defaults": defaults, "fixed": fixed, "rank": rank, "trials": trials, "seed": seed, "split": pipeline.split,
           "metadata": pipeline.metadata}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]
    directory = Path(config["output"]) / "hpo_cache" / f"{compressor_class.name}_rank{rank}_{digest}"
    directory.mkdir(parents=True, exist_ok=True)
    stored = directory / "stage1.pkl"
    with (directory / "lock").open("w") as lock:
        try:  # The lock is released when the file closes, also if this process dies.
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            return None
        if stored.exists():
            print(f"Stage 1 cache: {directory}", flush=True)
            with stored.open("rb") as file:
                values, compressor = pickle.load(file)
        else:
            values, compressor = tune()
            if hasattr(compressor, "to"):
                compressor.to("cpu")
            temporary = stored.with_suffix(".tmp")
            with temporary.open("wb") as file:
                pickle.dump((values, compressor), file, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(stored)
            (directory / "values.json").write_text(json.dumps({**key, "best": values}, indent=2, default=str))
    if hasattr(compressor, "to"):
        compressor.to(pipeline.device)
    return values, compressor


def optimize(config, compressor_class=None, forecaster_class=None, dataset_class=ForecasterDataset,
             compressor=None, fine_tune_compressor=True, logger=None):
    """Return the best config sections {"compressor", "dataset", "forecaster"} with the chosen values.

    Give compressor_class to tune a compressor, forecaster_class to tune a forecaster, or both.
    A trained compressor can replace compressor_class; it is frozen in stage 2, and fine-tuned in
    stage 3 only when fine_tune_compressor is true. The "compressor" section is then left out.
    With a logger, every stage logs its trials under HPO/stage1 (HPO/stage1_rank<r> per rank),
    HPO/stage2 and HPO/stage3; a stage 1 read from the cache logs nothing.
    """
    if (compressor_class is None) == (compressor is None):
        raise ValueError("Give either compressor_class or a trained compressor")
    if forecaster_class is None and compressor is not None:
        raise ValueError("Nothing to tune: a trained compressor needs a forecaster_class")
    pipeline = Pipeline(config)
    pipeline.fit_scaler()
    trials, seed = config.get("trials", 2), config.get("seed", 42)
    fixed = {key: {k: v for k, v in config.get(key, {}).items() if k != "name"}
             for key in ("compressor", "dataset", "forecaster")}
    result = {}
    # The rank is chosen in stage 2, on the forecast objective (see Compressor.rank_range).
    rank_range = compressor_class.rank_range if compressor is None else None
    free_rank = rank_range is not None and "rank" not in fixed["compressor"]
    tune_rank = free_rank and forecaster_class is not None
    # Without a forecaster POD is still fitted at its largest rank: any smaller rank is a truncation.
    nested = free_rank and compressor_class.nested_ranks
    if compressor is None:
        ranges = compressor_class.hyperparameters_ranges

        def tune_compressor(pool, rank=None, wait=True):
            """Stage 1 at one rank (None: the config's rank or the class default): (values, compressor)."""
            def propose(trial):
                values = sample(trial, ranges, fixed["compressor"], "compressor.")
                if rank is not None:
                    values["rank"] = rank
                return values, (train_compressor_job, (compressor_class, values))

            count = trials if tunable(ranges, fixed["compressor"]) else 1
            name = "stage1" if rank is None or nested else f"stage1_rank{rank}"
            return stage_one_cache(config, pipeline, compressor_class, fixed["compressor"], rank, count, seed,
                                   lambda: minimize(propose, count, seed, logger, name, pool), wait)

        with workers(config, pipeline) as pool:
            if tune_rank and not nested:
                if rank_range["type"] != "categorical":
                    raise ValueError("A compressor without nested ranks needs a categorical rank_range")
                # An autoencoder's best hyperparameters depend on its rank: stage 1 runs once per
                # rank. Ranks that another pair is tuning right now are skipped first and awaited
                # last, so pairs running at the same time share the stage-1 work. A rank where
                # every trial fails is left out of the stage-2 choices.
                tuned, failed = {}, set()
                for wait_for_others in (False, True):
                    for rank in rank_range["choices"]:
                        if rank in tuned or rank in failed:
                            continue
                        try:
                            found = tune_compressor(pool, rank, wait_for_others)
                        except SearchFailed as error:
                            print(f"Stage 1 skips rank {rank}: {error}", flush=True)
                            failed.add(rank)
                            continue
                        if found is not None:
                            tuned[rank] = found
                if not tuned:
                    raise SearchFailed(f"Stage 1 failed at every rank {rank_range['choices']}")
                rank_range = {**rank_range, "choices": [r for r in rank_range["choices"] if r in tuned]}
            else:
                # POD is fitted once at the largest rank; each stage-2 trial truncates it.
                tuned = {None: tune_compressor(pool, rank_range["high"] if nested else None)}
                compressor = tuned[None][1]
        fine_tune_compressor = True

    def compressor_of_rank(rank):
        """The stage-1 (values, compressor) at a rank: POD truncated, an autoencoder tuned at that rank."""
        if nested:
            return {**tuned[None][0], "rank": rank}, compressor.truncated(rank)
        return tuned[rank]

    if forecaster_class is None:
        result["compressor"] = {"name": compressor_class.name, **tuned[None][0]}
        return result

    dataset_ranges = {**dataset_class.hyperparameters_ranges, **forecaster_class.dataset_ranges}
    ranges = {k: v for k, v in forecaster_class.hyperparameters_ranges.items() if not k.startswith("joint_")}
    if nested:  # POD caps the rank at the data size, so the fit may hold fewer modes than asked.
        high = min(rank_range["high"], compressor.rank)
        rank_range = {**rank_range, "low": min(rank_range["low"], high), "high": high}

    def propose(trial):
        rank = sample(trial, {"rank": rank_range}, {}, "compressor.")["rank"] if tune_rank else compressor.rank
        dataset_values = sample(trial, dataset_ranges, fixed["dataset"], "dataset.")
        values = sample(trial, ranges, fixed["forecaster"], "forecaster.")
        check_feasible(pipeline, forecaster_class, dataset_class, dataset_values, values, rank)
        return (rank, dataset_values, values), (train_forecaster_job,
                                                (forecaster_class, values, dataset_class, dataset_values, rank))

    tuned_any = tune_rank or tunable(dataset_ranges, fixed["dataset"]) or tunable(ranges, fixed["forecaster"])
    codecs = {codec.rank: codec for _, codec in tuned.values()} if tune_rank else {compressor.rank: compressor}
    with workers(config, pipeline, codecs) as pool:
        (rank, dataset_values, values), forecaster = minimize(propose, trials if tuned_any else 1, seed, logger,
                                                              "stage2", pool)
    codec_values, compressor = compressor_of_rank(rank) if tune_rank else (None, compressor)
    forecaster = move(forecaster, pipeline.device)
    if compressor_class is not None:
        result["compressor"] = {"name": compressor_class.name, **(codec_values if tune_rank else tuned[None][0])}
    result["dataset"] = dataset_values
    result["forecaster"] = {"name": forecaster_class.name, **values}

    joint_ranges = {k: v for k, v in forecaster_class.hyperparameters_ranges.items() if k.startswith("joint_")}
    if issubclass(forecaster_class, DLModel):
        if not (joint_ranges and isinstance(compressor, AE) and fine_tune_compressor):
            result["forecaster"]["joint_epochs"] = 0
            return result
        fixed_joint = {k: v for k, v in fixed["forecaster"].items() if k.startswith("joint_")}

        def propose(trial):
            values = sample(trial, joint_ranges, fixed_joint, "forecaster.")
            return values, (fine_tune_job, (forecaster, compressor, values, dataset_class, dataset_values))

        with workers(config, pipeline) as pool:
            values, _ = minimize(propose, trials if tunable(joint_ranges, fixed_joint) else 1, seed, logger,
                                 "stage3", pool)
        result["forecaster"].update(values)
    return result
