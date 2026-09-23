"""Training steps shared by run.py and HPO.py.

The config gives the data (metadata, validation_fraction, blocks), the device, the batch sizes
and K_eval, the number of recursive steps of the validation windows. The scaler is fitted once on
the training frames and never tuned. validation_fraction 0 is the refit after the HPO: all training
frames are used, nothing is validated, and the validation errors are None.
"""
import numpy as np
from DataProcessing.loading import make_loader
from DataProcessing.metadata import load_metadata
from DataProcessing.Dataset import CompressorDataset
from DataProcessing.scaling import FeatureScaler
from Baselines.Forecast.Classical.ARX import Constant
from Baselines.Forecast.DL.DLModel import DLModel
from .evaluation import reconstruction_error, validation_error


class Pipeline:
    def __init__(self, config, metadata=None):
        self.config = config
        self.metadata = metadata or load_metadata(config["metadata"])
        self.split = dict(validation_fraction=config.get("validation_fraction", 0.2), blocks=config.get("blocks", 20))
        self.device = config.get("device", "cpu")
        self.K_eval = config.get("K_eval", 1)
        self.joint_batch_size = config.get("joint_batch_size", 8)
        self.loader_options = {"num_workers": config.get("workers", 0), **config.get("dataloader", {})}
        self.scaler = None
        self.refit = self.split["validation_fraction"] == 0
        self.validation_steps = None  # Field MSE at each rollout step of the last validation_error.
        self._frozen = {}  # (compressor id, Nx, Ni) -> (compressor, frozen-state reference error)

    def loader(self, dataset, shuffle=False, batch_size=None):
        return make_loader(dataset, batch_size=batch_size or self.config.get("batch_size", 64), shuffle=shuffle,
                           **self.loader_options)

    def fit_scaler(self):
        frames = CompressorDataset(self.metadata, "train", **self.split)
        self.scaler = FeatureScaler(frames.mask).fit(self.loader(frames, batch_size=self.config.get("preprocessing_batch_size", 64)))
        return self.scaler

    def frames(self, partition):
        return CompressorDataset(self.metadata, partition, scaler=self.scaler, **self.split)

    def windows(self, dataset_class, hyperparameters, partition, compressor=None, validation=False):
        """Scaled forecaster windows; validation windows have K_eval targets and, as images, stride K_eval."""
        hyperparameters, stride = dict(hyperparameters), 1
        if validation:
            hyperparameters["horizon"] = self.K_eval
            stride = 1 if compressor else self.K_eval
        return dataset_class.build(hyperparameters, metadata=self.metadata, partition=partition, scaler=self.scaler,
                                   compressor=compressor, stride=stride, **self.split)

    def train_compressor(self, compressor_class, hyperparameters, logger=None):
        """Build and fit a compressor; return it with its validation reconstruction MSE."""
        compressor = compressor_class.build(hyperparameters, device=self.device)
        validation = None if self.refit else self.frames("validation")
        compressor.fit(self.frames("train"), validation=validation, logger=logger, loader_options=self.loader_options)
        if self.refit:
            return compressor, None
        return compressor, reconstruction_error(compressor, validation, loader_options=self.loader_options)

    def train_forecaster(self, forecaster_class, hyperparameters, dataset_class, dataset_hyperparameters, compressor,
                         logger=None, directory=None, resume=False):
        """Build and fit a forecaster on the latents of a frozen compressor; return it with its validation error.

        A forecaster that learns nothing (trainable False) gets windows of frames that are never
        encoded, so a large identity compressor costs no memory.
        """
        encoder = compressor if forecaster_class.trainable else None
        training = self.windows(dataset_class, dataset_hyperparameters, "train", encoder)
        validation = None if self.refit else self.loader(
            self.windows(dataset_class, dataset_hyperparameters, "validation", encoder, validation=True))
        # A row holds a latent state and the forcing; the forecaster returns the next latent state.
        forecaster = forecaster_class.build(hyperparameters, input_size=compressor.rank + 1, output_size=compressor.rank,
                                            Nx=training.Nx, Ni=training.Ni, device=self.device)
        forecaster.fit(self.loader(training, shuffle=isinstance(forecaster, DLModel)), validation,
                       logger=logger, directory=directory, resume=resume)
        return forecaster, self.validation_error(forecaster, compressor, dataset_class, dataset_hyperparameters)

    def fine_tune(self, forecaster, compressor, dataset_class, dataset_hyperparameters, logger=None):
        """Train a neural forecaster and an autoencoder together on images; return the validation error."""
        training = self.windows(dataset_class, dataset_hyperparameters, "train")
        validation = None if self.refit else self.loader(
            self.windows(dataset_class, dataset_hyperparameters, "validation", validation=True),
            batch_size=self.joint_batch_size)
        forecaster.fit_joint(compressor, self.loader(training, shuffle=True, batch_size=self.joint_batch_size),
                             validation, logger=logger)
        return self.validation_error(forecaster, compressor, dataset_class, dataset_hyperparameters)

    def frozen_reference(self, compressor, dataset_class, dataset_hyperparameters):
        """Validation error of repeating the last state forever: the scale of the divergence gate.

        Cached per compressor and window shape; the entry pins the compressor so its id stays
        valid. After joint fine-tuning the reference is slightly stale (same object, new
        decoder) — irrelevant at the default gate factor of 100.
        """
        dataset = self.windows(dataset_class, dataset_hyperparameters, "validation", validation=True)
        key = (id(compressor), dataset.Nx, dataset.Ni)
        if key not in self._frozen:
            frozen = Constant(Nx=dataset.Nx, Ni=dataset.Ni)
            self._frozen[key] = (compressor, validation_error(frozen, compressor, dataset,
                                                              self.joint_batch_size, self.loader_options))
        return self._frozen[key][1]

    def validation_error(self, forecaster, compressor, dataset_class, dataset_hyperparameters):
        """Field MSE of K_eval-step recursive forecasts: the selection objective (None in a refit).

        A finite score worse than divergence_factor (default 100) times the frozen-state
        reference returns infinity: finite-but-astronomical rollouts (observed up to 1e42) must
        not survive selection, and both fit and the HPO already treat infinity as diverged.
        """
        if self.refit:
            return None
        dataset = self.windows(dataset_class, dataset_hyperparameters, "validation", validation=True)
        error, self.validation_steps = validation_error(forecaster, compressor, dataset, self.joint_batch_size,
                                                        self.loader_options, per_step=True)
        if error is not None and np.isfinite(error):
            reference = self.frozen_reference(compressor, dataset_class, dataset_hyperparameters)
            bound = self.config.get("divergence_factor", 100) * max(reference, np.finfo(np.float64).tiny)
            if error > bound:
                self.validation_steps = None
                return float("inf")
        return error
