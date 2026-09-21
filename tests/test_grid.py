import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from DataProcessing.grid import derive


def synthetic_grid(tmp_path, spacing=(0.5, 1.0, 0.25), dimensions=(4, 2, 3)):
    grid = pv.ImageData(dimensions=dimensions, spacing=spacing).cast_to_unstructured_grid()
    path = tmp_path / "grid.vtu"
    grid.save(path)
    return path, grid


def test_derive_volumes_coordinates_indices(tmp_path):
    path, grid = synthetic_grid(tmp_path)
    out = derive(path)
    assert out["cell_volumes"].shape == (grid.n_cells,)
    assert np.allclose(out["cell_volumes"], 0.5 * 1.0 * 0.25)
    assert out["coordinates"].shape == (grid.n_cells, 3)
    g = out["grid_indices"]
    assert (g["nx"], g["nz"]) == (3, 2)  # (dimensions - 1) cells per axis in x and z
    # cell -> pixel is injective and reconstructs the cell centers
    pixels = set(zip(g["col_idx"].tolist(), g["row_idx"].tolist()))
    assert len(pixels) == grid.n_cells
    assert np.allclose(g["x_vals"][g["col_idx"]], np.round(out["coordinates"][:, 0], 8))
    assert np.allclose(g["z_vals"][g["row_idx"]], np.round(out["coordinates"][:, 2], 8))
    edges = out["edges"]
    # 3x2 planar cell grid: 2*(2*horizontal + 3*vertical... ) -> count via pixel adjacency
    assert edges.shape[0] == 2 and edges.shape[1] == 2 * 7
    assert (edges[0] != edges[1]).all()
    # symmetric: every directed edge has its reverse
    pairs = set(map(tuple, edges.T))
    assert all((b, a) in pairs for a, b in pairs)


def test_derive_rejects_nonplanar(tmp_path):
    grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    path = tmp_path / "grid.vtu"
    grid.save(path)
    with pytest.raises(ValueError, match="planar"):
        derive(path)
