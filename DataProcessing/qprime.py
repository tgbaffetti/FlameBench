"""Precompute integrated heat release q(t) = sum_c Q_c(t) * V_c for every case.

Writes Data/Qseries/<case>.npy, float64 (nt,), from the prepared arrays and
metadata["cell_volumes"]. These are the targets for the 0-D flame-response
baselines and match the q' computed in Experiments/evaluation.py.

Run after prepare.py and grid.py: python -m DataProcessing.qprime
"""
import argparse
from pathlib import Path
import numpy as np
from .metadata import load_metadata


def qseries(data_path, q_index, volumes, block=256):
    data = np.load(data_path, mmap_mode="r")
    out = np.empty(len(data), dtype=np.float64)
    for start in range(0, len(data), block):
        stop = min(start + block, len(data))
        out[start:stop] = data[start:stop, q_index].astype(np.float64) @ volumes
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="Data/metadata.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata = load_metadata(args.metadata)
    if not metadata.get("cell_volumes"):
        raise ValueError("Set cell_volumes in the metadata first (python -m DataProcessing.grid)")
    volumes = np.load(metadata["cell_volumes"], allow_pickle=False)
    q_index = metadata["fields"].index("mix:Q")
    out_dir = Path(args.metadata).resolve().parent / "Qseries"
    out_dir.mkdir(exist_ok=True)
    for case in metadata["cases"]:
        target = out_dir / f"{case['name']}.npy"
        if target.exists() and not args.overwrite:
            print(f"Keeping {target}", flush=True)
            continue
        q = qseries(case["data"], q_index, volumes)
        if not np.isfinite(q).all():
            raise ValueError(f"Nonfinite q in {case['name']}")
        np.save(target, q)
        print(f"{case['name']}: q mean {q.mean():.4e} W, min {q.min():.4e}, max {q.max():.4e}", flush=True)


if __name__ == "__main__":
    main()
