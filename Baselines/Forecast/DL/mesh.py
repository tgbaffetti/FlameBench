"""Cell-center coordinates for mesh-native operators, from the saved image-grid mapping."""
import numpy as np


def cell_coordinates(path):
    """(cells, 2) float32 cell centers, each axis normalized to [0, 1], in original cell order.

    Reads the ImageGrid .npz (rows/columns index into the reversed-z and x axes), so operators
    need no coordinate file beyond the grid mapping every dataset already loads.
    """
    if path is None:
        raise ValueError("Mesh operators require the grid mapping (.npz of ImageGrid.save)")
    with np.load(path) as grid:
        xz = np.stack([grid["x"][grid["columns"]], grid["z"][grid["rows"]]], axis=1).astype(np.float32)
    xz -= xz.min(axis=0)
    span = xz.max(axis=0)
    span[span == 0] = 1
    return xz / span
