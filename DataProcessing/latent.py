"""Precompute small latent trajectories to disk; preserve original split bounds."""
from bisect import bisect_right
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    def __init__(self, source, compressor, scaler, directory, batch_size=64, horizon=1):
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.Nx, self.Ni = source.Nx, source.Ni
        self.horizon = horizon
        self.history, self.context = source.history, source.context
        self.paths, self.ends, self._maps = [], [], {}
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        total = 0
        for case, lo, hi in source.segments():
            data, phi = source.arrays(case)
            path = directory / f"{case['name']}_{lo}_{hi}.npy"
            out = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(hi-lo, compressor.rank))
            for start in range(lo, hi, batch_size):
                end = min(start + batch_size, hi)
                out[start-lo:end-lo] = compressor.encode(scaler.transform(np.array(data[start:end])))
            out.flush()
            del out
            phi_path = directory / f"{case['name']}_{lo}_{hi}_phi.npy"
            np.save(phi_path, np.asarray(phi[lo:hi], dtype=np.float32))
            self.paths.append((str(path), str(phi_path)))
            if hi-lo <= self.context + self.horizon - 1:
                raise ValueError(f"Segment too short for horizon {self.horizon}: {case['name']}")
            total += hi-lo-self.context-(self.horizon-1)
            self.ends.append(total)

    def __len__(self):
        return self.ends[-1]

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_maps'] = {}
        return state

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        k = bisect_right(self.ends, index)
        end = self.context + index - (self.ends[k-1] if k else 0)
        start = end - self.history
        if k not in self._maps:
            self._maps[k] = tuple(np.load(p, mmap_mode="r") for p in self.paths[k])
        x, phi = self._maps[k]
        history = torch.from_numpy(np.array(x[start:end]))
        if self.horizon == 1:
            return (history,
                    torch.from_numpy(np.array(phi[end-self.Ni-1:end+1])),
                    torch.from_numpy(np.array(x[end])))
        # Unrolled sample: one forcing window and one target per rollout step.
        forcing = np.stack([phi[end+j-self.Ni-1:end+j+1] for j in range(self.horizon)])
        return (history,
                torch.from_numpy(forcing),
                torch.from_numpy(np.array(x[end:end+self.horizon])))
