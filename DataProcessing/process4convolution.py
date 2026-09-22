"""Cell-to-image mapping adapted from main's process4convolution.py (6804043).

Same rounded X/Z bins and reversed Z axis; channel-first output for PyTorch.
No interpolation: each cell occupies exactly one pixel. The physical cell volumes are kept
for volume integrals such as the total heat release.
"""
import numpy as np


class ImageGrid:
    def __init__(self, centers, volumes=None, decimals=8):
        centers = np.asarray(centers)
        if volumes is not None:
            volumes = np.asarray(volumes, dtype=np.float32)
            if volumes.shape != (len(centers),) or not np.isfinite(volumes).all() or np.any(volumes <= 0):
                raise ValueError("Cell volumes must be finite, positive, and aligned with cells")
        self.volumes = volumes
        if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
            raise ValueError("Expected finite (cells, 3) centers")
        if centers[:, 1].std() > 1e-6:
            raise ValueError("Cell centers must lie in the X/Z plane")
        self.x, columns = np.unique(np.round(centers[:, 0], decimals), return_inverse=True)
        z, rows = np.unique(np.round(centers[:, 2], decimals), return_inverse=True)
        self.z = z[::-1]
        self.rows, self.columns = len(z) - 1 - rows, columns
        self.shape = (len(z), len(self.x))
        if len(np.unique(self.rows * len(self.x) + columns)) != len(centers):
            raise ValueError("Multiple cells map to one pixel; adjust rounding tolerance")
        self.mask = np.zeros(self.shape, dtype=bool)
        self.mask[self.rows, self.columns] = True

    @classmethod
    def read(cls, path):
        import pyvista as pv
        mesh = pv.read(path)
        # Legacy volumes, but with the absolute value: the wedge cells on the axis come out
        # negative only because of their node ordering.
        volumes = np.abs(mesh.compute_cell_sizes(length=False, area=True, volume=True).cell_data["Volume"])
        volumes = volumes.astype(np.float32)
        return cls(mesh.cell_centers().points, volumes)

    def images(self, cells):
        if cells.shape[-1] != len(self.rows):
            raise ValueError("Mesh cell count does not match data")
        result = np.zeros((*cells.shape[:-1], *self.shape), dtype=cells.dtype)
        result[..., self.rows, self.columns] = cells
        return result

    def cells(self, images):
        return images[..., self.rows, self.columns]

    def save(self, path):
        arrays = dict(rows=self.rows, columns=self.columns, mask=self.mask, x=self.x, z=self.z)
        if self.volumes is not None:
            arrays["volumes"] = self.volumes
        np.savez(path, **arrays)
