"""Stream NPZ extraction and transpose into disk-backed (time, field, cell) NPY."""
import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
import numpy as np
from .metadata import load_metadata


def convert(source, target, fields, block_cells=128):
    source, target = Path(source), Path(target)
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Temporary extraction lives on disk, never in an in-memory BytesIO.
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        raw_path = source
        if source.suffix == ".npz":
            raw_path = Path(tmp) / "raw.npy"
            with zipfile.ZipFile(source) as archive:
                if archive.namelist() != ["data.npy"]:
                    raise ValueError(f"Expected exactly data.npy in {source}")
                with archive.open("data.npy") as src, raw_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        staged = Path(tmp) / "prepared.npy"
        if fields:
            if raw.ndim != 3 or raw.shape[1] != fields:
                raise ValueError(f"Expected (cells, {fields}, time); got {raw.shape}")
            nc, nf, nt = raw.shape
            out = np.lib.format.open_memmap(staged, mode="w+", dtype="float32", shape=(nt, nf, nc))
            for start in range(0, nc, block_cells):
                block = raw[start:start + block_cells]
                if not np.isfinite(block).all():
                    raise ValueError(f"Nonfinite data in {source}")
                out[:, :, start:start + block_cells] = block.transpose(2, 1, 0)
        else:
            if raw.ndim != 1 or not np.isfinite(raw).all():
                raise ValueError(f"Expected finite 1D forcing: {source}")
            out = np.lib.format.open_memmap(staged, mode="w+", dtype="float32", shape=raw.shape)
            out[:] = raw
        out.flush()
        del out, raw
        staged.replace(target)


def prepare(metadata_path, source_root, overwrite=False):
    metadata = load_metadata(metadata_path)
    for case in metadata["cases"]:
        for key in ("data", "phi"):
            target = Path(case[key])
            source = Path(source_root) / case["source_" + key]
            if target.exists() and not overwrite:
                print(f"Keeping {target}", flush=True)
                continue
            print(f"Converting {source} -> {target}", flush=True)
            convert(source, target, len(metadata["fields"]) if key == "data" else 0)
        data = np.load(case["data"], mmap_mode="r")
        phi = np.load(case["phi"], mmap_mode="r")
        expected = round(case["duration"] / metadata["dt"]) + 1
        if data.shape[0] != len(phi) or len(phi) != expected:
            raise ValueError(f"Timestamp/length mismatch in {case['name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="DataProcessing/metadata.json")
    parser.add_argument("--source", default="data")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(args.metadata, args.source, args.overwrite)
