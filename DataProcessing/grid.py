"""Derive per-cell quantities from the CFD grid (data/grid.vtu is not in git; ask on Teams).

Writes, into the metadata's data_root:
- cell_volumes.npy: (n_cells,) float64, |cell volume| in m^3 (206 cells have inverted
  orientation in the source grid, hence the absolute value).
- coordinates.npy: (n_cells, 3) float64 cell centers in m.
- grid_indices.npz: col_idx/row_idx (n_cells,) int64 mapping cell -> (x, z) pixel, plus
  nx, nz, x_vals, z_vals. Same rounding-based binning as process4convolution.py; pixels
  not covered by any cell (the angled side) stay unassigned.
- edges.npy: (2, E) int64 directed cell-adjacency (face-sharing neighbors, both directions),
  for graph models (MeshGraphNets).

Run: python -m DataProcessing.grid [--grid data/grid.vtu] [--metadata DataProcessing/metadata.json]
Then reference the files from the metadata ("cell_volumes", "coordinates", "grid_indices").
"""
import argparse
from pathlib import Path
import numpy as np
from .metadata import load_metadata

DECIMALS = 8


def derive(grid_path):
    import pyvista as pv
    grid = pv.read(grid_path)
    volumes = np.abs(grid.compute_cell_sizes(length=False, area=False, volume=True)["Volume"])
    if not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError("Cell volumes must be finite and nonzero")
    centers = grid.cell_centers().points.astype(np.float64)
    if centers[:, 1].std() > 1e-6:
        raise ValueError("Mesh is expected to be planar in Y")
    x = np.round(centers[:, 0], DECIMALS)
    z = np.round(centers[:, 2], DECIMALS)
    x_vals, col_idx = np.unique(x, return_inverse=True)
    z_vals, row_idx = np.unique(z, return_inverse=True)
    counts = np.zeros((len(x_vals), len(z_vals)), dtype=np.int64)
    np.add.at(counts, (col_idx, row_idx), 1)
    if counts.max() > 1:
        raise ValueError("Two cells map to one pixel; increase DECIMALS resolution")
    edges = [(i, j) for i in range(grid.n_cells)
             for j in grid.cell_neighbors(i, connections="faces")]
    edges = np.array(edges, dtype=np.int64).T
    if edges.shape[0] != 2 or not len(set(map(tuple, edges.T))) == edges.shape[1]:
        raise ValueError("Malformed cell adjacency")
    return {"cell_volumes": volumes, "coordinates": centers, "edges": edges,
            "grid_indices": {"col_idx": col_idx, "row_idx": row_idx,
                             "nx": len(x_vals), "nz": len(z_vals),
                             "x_vals": x_vals, "z_vals": z_vals}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="data/grid.vtu")
    parser.add_argument("--metadata", default="DataProcessing/metadata.json")
    args = parser.parse_args()
    out = Path(load_metadata(args.metadata)["data_root"])
    derived = derive(args.grid)
    np.save(out / "cell_volumes.npy", derived["cell_volumes"])
    np.save(out / "coordinates.npy", derived["coordinates"])
    np.save(out / "edges.npy", derived["edges"])
    np.savez(out / "grid_indices.npz", **derived["grid_indices"])
    g = derived["grid_indices"]
    print(f"{len(derived['cell_volumes'])} cells -> {out}; image {g['nx']}x{g['nz']}, "
          f"coverage {len(derived['cell_volumes'])/(g['nx']*g['nz']):.1%}, "
          f"{derived['edges'].shape[1]} directed edges")


if __name__ == "__main__":
    main()
