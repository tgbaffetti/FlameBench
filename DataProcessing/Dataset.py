"""Map-style datasets: a sample never crosses a trajectory or split boundary."""
from abc import ABC, abstractmethod
from bisect import bisect_right
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset, ABC):
    @property
    @abstractmethod
    def input_signals(self):
        """Serializable metadata describing the forcing in each trajectory."""

    @abstractmethod
    def segments(self):
        """Return (case, inclusive_start, exclusive_stop) tuples."""


class WindowDataset(Dataset):
    def __init__(self, metadata, split, Nx=9, Ni=0, validation_fraction=0.2, partition="train"):
        if type(Nx) is not int or type(Ni) is not int or Nx < 0 or Ni < 0:
            raise ValueError("Nx and Ni must be nonnegative integers")
        if not 0 < validation_fraction < 1:
            raise ValueError("0 < validation_fraction < 1 required")
        if partition not in {"train", "validation", "test"}:
            raise ValueError(partition)
        self.metadata = metadata
        self.Nx, self.Ni = Nx, Ni
        self.history = Nx + 1
        self.context = max(Nx, Ni) + 1
        self._segments, self.ends, self._maps = [], [], {}
        total = 0
        shape = None
        for case in metadata["cases"]:
            if case["split"] != split:
                continue
            x, phi = self.arrays(case)
            if x.ndim != 3 or x.shape[1] != len(metadata["fields"]) or phi.shape != (len(x),):
                raise ValueError(f"Invalid shape/alignment: {case['name']}")
            if shape is not None and x.shape[1:] != shape:
                raise ValueError("All trajectories must share field/cell shape")
            shape = x.shape[1:]
            lo, hi = 0, len(x)
            if split == "training":
                cut = int(len(x) * (1 - validation_fraction))
                lo, hi = (0, cut) if partition == "train" else (cut, len(x))
            if hi - lo <= self.context:
                raise ValueError(f"Segment too short: {case['name']}")
            self._segments.append((case, lo, hi))
            total += hi - lo - self.context
            self.ends.append(total)
        if not self._segments:
            raise ValueError(f"No cases for {split}")
        self.field_shape = shape
        self._maps = {}  # Workers open their own read-only memory maps.

    @property
    def input_signals(self):
        return [{k: v for k, v in case.items() if k not in {"data", "phi"}}
                for case, _, _ in self._segments]

    def segments(self):
        return list(self._segments)

    def arrays(self, case):
        name = case["name"]
        if name not in self._maps:
            self._maps[name] = (np.load(case["data"], mmap_mode="r", allow_pickle=False),
                                np.load(case["phi"], mmap_mode="r", allow_pickle=False))
        return self._maps[name]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_maps"] = {}
        return state

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        segment = bisect_right(self.ends, index)
        case, lo, _ = self._segments[segment]
        offset = index - (self.ends[segment - 1] if segment else 0)
        stop = lo + self.context + offset
        start = stop - self.history
        x, phi = self.arrays(case)
        return {"history": torch.from_numpy(np.array(x[start:stop], dtype=np.float32)),
                "forcing": torch.from_numpy(np.array(phi[stop - self.Ni - 1:stop + 1], dtype=np.float32)),
                "target": torch.from_numpy(np.array(x[stop], dtype=np.float32))}

    def snapshot_batches(self, batch_size=64):
        for case, lo, hi in self._segments:
            x, _ = self.arrays(case)
            for start in range(lo, hi, batch_size):
                yield np.array(x[start:min(start + batch_size, hi)], dtype=np.float32)


class TrainingDataset(WindowDataset):
    def __init__(self, metadata, Nx=9, Ni=0, validation_fraction=0.2, partition="train"):
        if partition not in {"train", "validation"}:
            raise ValueError(partition)
        super().__init__(metadata, "training", Nx, Ni, validation_fraction, partition)


class TestDataset(WindowDataset):
    __test__ = False

    def __init__(self, metadata, Nx=9, Ni=0):
        super().__init__(metadata, "test", Nx, Ni, partition="test")
