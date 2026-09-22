"""Datasets for the compressor (single frames) and the forecaster (time windows).

Partitions: "train" and "validation" come from the training cases, "test" from the test cases.
Each training trajectory is cut in time into `blocks` equal blocks. round(blocks *
validation_fraction) evenly spaced blocks are validation and the others are training, so both
partitions cover the whole forcing history (for example every frequency of a sweep). Each run of
consecutive blocks is one segment. Samples never cross a segment edge, so no frame is used by both
partitions, and the compressor and the forecaster see exactly the same frames.

Frames are images (field, height, width) with a mask of valid pixels. With a scaler, frames are
feature-normalized; without one (to fit the scaler) they are raw.
"""
from bisect import bisect_right
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset as TorchDataset


def split_segments(length, partition, validation_fraction, blocks):
    """(start, stop) frame ranges of one training trajectory that belong to the partition."""
    count = round(blocks * validation_fraction)
    if not 0 < count < blocks:
        raise ValueError("validation_fraction must select at least one block and leave one for training")
    edges = [block * length // blocks for block in range(blocks + 1)]
    validation = {int((k + 0.5) * blocks / count) for k in range(count)}
    ranges = []
    for block in range(blocks):
        if (block in validation) != (partition == "validation"):
            continue
        if ranges and ranges[-1][1] == edges[block]:
            ranges[-1] = (ranges[-1][0], edges[block + 1])
        else:
            ranges.append((edges[block], edges[block + 1]))
    return ranges


def keep_recent(rows, count):
    """Copy of rows (numpy array or torch tensor) with all but the last `count` rows of axis 1 zeroed."""
    rows = rows * 1
    rows[:, :rows.shape[1] - count] = 0
    return rows


LOADED = {}  # path -> array read fully into memory, shared by every dataset of the process


def load(path, in_memory):
    """Memory map of a .npy file, or with in_memory the whole array, read once per process."""
    if not in_memory:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    if path not in LOADED:
        LOADED[path] = np.load(path, allow_pickle=False)
    return LOADED[path]


class Dataset(TorchDataset):
    """Frames of one partition, read lazily from memory-mapped trajectories.

    Set Dataset.in_memory = True to read each trajectory into memory once instead; every later
    dataset reuses it. This needs RAM for all the images (about 18 GB) but avoids slow disks.

    hyperparameters_ranges is the Optuna search space of the constructor keywords; build creates
    a dataset from them, with context holding the values that are never tuned.
    """
    hyperparameters_ranges = {}
    in_memory = False

    @classmethod
    def build(cls, hyperparameters, **context):
        return cls(**context, **hyperparameters)

    def __init__(self, metadata, partition, validation_fraction=0.2, blocks=20, scaler=None):
        if partition not in {"train", "validation", "test"}:
            raise ValueError(partition)
        self.metadata, self.partition, self.scaler = metadata, partition, scaler
        self.segments, self._maps, shape = [], {}, None
        for case in metadata["cases"]:
            if case["split"] != ("test" if partition == "test" else "training"):
                continue
            x, phi = self.arrays(case)
            if x.ndim != 4 or x.shape[1] != len(metadata["fields"]) or phi.shape != (len(x),):
                raise ValueError(f"Invalid shape/alignment: {case['name']}")
            if shape is not None and x.shape[1:] != shape:
                raise ValueError("All trajectories must share field/cell shape")
            shape = x.shape[1:]
            ranges = [(0, len(x))] if partition == "test" else split_segments(len(x), partition,
                                                                               validation_fraction, blocks)
            self.segments += [(case, start, stop) for start, stop in ranges]
        if not self.segments:
            raise ValueError(f"No cases for {partition}")
        self.field_shape = shape
        with np.load(metadata["grid_indices"]) as grid:
            self.mask = grid["mask"]
            self.grid_indices = (grid["rows"], grid["columns"])
            self.volumes = grid["volumes"] if "volumes" in grid else None
        if self.mask.shape != shape[1:]:
            raise ValueError("Image mask does not match data")
        self._maps = {}  # Workers open their own read-only memory maps.

    def arrays(self, case):
        name = case["name"]
        if name not in self._maps:
            self._maps[name] = (load(case["data"], self.in_memory), load(case["phi"], self.in_memory))
        return self._maps[name]

    def frames(self, case, start, stop):
        """Frames start:stop of a case as float32, scaled when the dataset has a scaler."""
        frames = np.array(self.arrays(case)[0][start:stop], dtype=np.float32)
        return frames if self.scaler is None else self.scaler.transform(frames)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_maps"] = {}
        state["in_memory"] = False  # Spawned workers mmap files instead of loading whole trajectories.
        return state


class CompressorDataset(Dataset):
    """Sample: one frame (fields, ...) of the partition."""
    def __init__(self, metadata, partition, validation_fraction=0.2, blocks=20, scaler=None):
        super().__init__(metadata, partition, validation_fraction, blocks, scaler)
        self.ends = list(np.cumsum([stop - start for _, start, stop in self.segments]))

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        segment = bisect_right(self.ends, index)
        case, start, _ = self.segments[segment]
        time = start + index - (self.ends[segment - 1] if segment else 0)
        return torch.from_numpy(self.frames(case, time, time + 1)[0])


class ForecasterDataset(Dataset):
    """Sample: the rows the forecaster reads at time t, plus the rollout forcing and targets.

    With length = max(Nx, Ni) + 1 rows s = t-length+1 .. t and K = horizon:
      states  (length, ...): x(s); rows older than the last Nx + 1 are zero.
      forcing (length + K - 1,): phi(s + 1) - 1 for s = t-length+1 .. t+K-1, the deviation from
              the unforced flow; rows older than the last Ni + 1 are zero. Rollout step k reads
              forcing[k : k+length] and pads it again with keep_recent.
      target  (K, ...): x(t+1 .. t+K).
    With a compressor the states are latents, encoded once and kept in memory; without one they
    are frames (joint training). stride > 1 keeps every stride-th window only, e.g. stride = K
    gives back-to-back rollouts that compare each frame once (validation).
    """
    hyperparameters_ranges = {"Nx": {"type": "int", "low": 0, "high": 5},
                              "Ni": {"type": "int", "low": 0, "high": 5},
                              "horizon": {"type": "categorical", "choices": [1, 5, 10, 20]}}

    def __init__(self, metadata, partition, Nx=9, Ni=0, horizon=1, validation_fraction=0.2, blocks=20,
                 scaler=None, compressor=None, stride=1, batch_size=64):
        for value in (Nx, Ni, horizon):
            if type(value) is not int or value < 0:
                raise ValueError("Nx, Ni and horizon must be nonnegative integers")
        if horizon < 1 or stride < 1:
            raise ValueError("horizon and stride must be positive integers")
        super().__init__(metadata, partition, validation_fraction, blocks, scaler)
        self.Nx, self.Ni, self.horizon, self.stride = Nx, Ni, horizon, stride
        self.length = max(Nx, Ni) + 1
        self.ends, total = [], 0
        for case, start, stop in self.segments:
            if stop - start < self.length + horizon:
                raise ValueError(f"Segment {case['name']} [{start}, {stop}) too short; use fewer blocks")
            total += (stop - start - self.length - horizon) // stride + 1
            self.ends.append(total)
        self.latents = None
        if compressor is not None:
            progress = tqdm(total=sum(stop - start for _, start, stop in self.segments), desc=f"Encoding {partition}")
            self.latents = []
            for case, start, stop in self.segments:
                chunks = []
                for begin in range(start, stop, batch_size):
                    chunks.append(compressor.encode(self.frames(case, begin, min(begin + batch_size, stop))))
                    progress.update(len(chunks[-1]))
                self.latents.append(np.concatenate(chunks))
            progress.close()

    def states(self, segment, start, stop):
        case, first, _ = self.segments[segment]
        if self.latents is None:
            return self.frames(case, start, stop)
        return self.latents[segment][start - first:stop - first]

    def window(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        segment = bisect_right(self.ends, index)
        case, start, _ = self.segments[segment]
        offset = index - (self.ends[segment - 1] if segment else 0)
        t = start + self.length - 1 + offset * self.stride
        return segment, case, t

    def target_frames(self, index):
        """Scaled image targets for scoring, even when __getitem__ returns latent targets."""
        _, case, t = self.window(index)
        return self.frames(case, t + 1, t + self.horizon + 1)

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        segment, case, t = self.window(index)
        states = self.states(segment, t - self.length + 1, t + 1)
        forcing = np.array(self.arrays(case)[1][t - self.length + 2:t + self.horizon + 1], dtype=np.float32) - 1
        forcing[:self.length - 1 - self.Ni] = 0
        return {"states": torch.from_numpy(keep_recent(states[None], self.Nx + 1)[0]),
                "forcing": torch.from_numpy(forcing),
                "target": torch.from_numpy(np.array(self.states(segment, t + 1, t + self.horizon + 1)))}
