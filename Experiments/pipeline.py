"""Training steps shared by run.py and HPO.py.

The config gives the data (metadata, validation_fraction, blocks), the device, the batch sizes
and K_eval, the number of recursive steps of the validation windows. The scaler is fitted once on
the training frames and never tuned.
"""
from torch.utils.data import DataLoader
from DataProcessing.metadata import load_metadata
from DataProcessing.Dataset import CompressorDataset
from DataProcessing.scaling import FeatureScaler
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
        self.scaler = None

    def loader(self, dataset, shuffle=False, batch_size=None):
        workers = self.config.get("workers", 0)
        return DataLoader(dataset, batch_size=batch_size or self.config.get("batch_size", 64), shuffle=shuffle,
                          num_workers=workers, persistent_workers=workers > 0)

    def fit_scaler(self):
        frames = CompressorDataset(self.metadata, "train", **self.split)
        self.scaler = FeatureScaler(frames.mask).fit(DataLoader(frames, self.config.get("preprocessing_batch_size", 64)))
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
        validation = self.frames("validation")
        compressor.fit(self.frames("train"), validation=validation, logger=logger)
        return compressor, reconstruction_error(compressor, validation)

    def train_forecaster(self, forecaster_class, hyperparameters, dataset_class, dataset_hyperparameters, compressor,
                         logger=None, directory=None, resume=False):
        """Build and fit a forecaster on the latents of a frozen compressor; return it with its validation error."""
        training = self.windows(dataset_class, dataset_hyperparameters, "train", compressor)
        validation = self.windows(dataset_class, dataset_hyperparameters, "validation", compressor, validation=True)
        forecaster = forecaster_class.build(hyperparameters, rank=compressor.rank, Nx=training.Nx, Ni=training.Ni,
                                            device=self.device)
        forecaster.fit(self.loader(training, shuffle=isinstance(forecaster, DLModel)), self.loader(validation),
                       logger=logger, directory=directory, resume=resume)
        return forecaster, self.validation_error(forecaster, compressor, dataset_class, dataset_hyperparameters)

    def fine_tune(self, forecaster, compressor, dataset_class, dataset_hyperparameters, logger=None):
        """Train a neural forecaster and an autoencoder together on images; return the validation error."""
        training = self.windows(dataset_class, dataset_hyperparameters, "train")
        validation = self.windows(dataset_class, dataset_hyperparameters, "validation", validation=True)
        forecaster.fit_joint(compressor, self.loader(training, shuffle=True, batch_size=self.joint_batch_size),
                             self.loader(validation, batch_size=self.joint_batch_size), logger=logger)
        return self.validation_error(forecaster, compressor, dataset_class, dataset_hyperparameters)

    def validation_error(self, forecaster, compressor, dataset_class, dataset_hyperparameters):
        """Field MSE of K_eval-step recursive forecasts: the selection objective."""
        dataset = self.windows(dataset_class, dataset_hyperparameters, "validation", validation=True)
        return validation_error(forecaster, compressor, dataset, self.joint_batch_size)
