import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
GRID_PATH = 'data/grid.vtu'
DATA_PATH = 'data/step_A03.npy'
features = ['p', 'U1', 'U3', 'rho', 'T', 'mix:Q', 'CH4', 'O2', 'H2O', 'CO2', 'OH']
n_features = len(features)
DECIMALS = 8   # rounding tolerance for X/Z binning (tune if needed)
# -----------------------------------------

grid = pv.read(GRID_PATH)
DataMatrix = np.load(DATA_PATH)

xyz = grid.cell_centers().points
n_cells = xyz.shape[0]
nt = DataMatrix.shape[1]

print(f'--> n_cells: {n_cells}, nt: {nt}')
print(f'--> check n_cells*n_features == DataMatrix.shape[0]: '
      f'{n_cells*n_features} == {DataMatrix.shape[0]}')

# --- planarity check (Y should be ~constant) ---
y = xyz[:,1]
print(f'Y: min={y.min():.6e}, max={y.max():.6e}, std={y.std():.6e}')
if y.std() > 1e-6:
    print('WARNING: Y not constant, mesh not perfectly planar!')
else:
    print('OK: cell centers essentially on same plane (Y ~ constant).')

x = xyz[:,0]
z = xyz[:,2]

# --- build X, Z axes via binning ---
x_round = np.round(x, DECIMALS)
z_round = np.round(z, DECIMALS)

x_vals = np.unique(x_round)
z_vals = np.unique(z_round)

nx = len(x_vals)
nz = len(z_vals)
print(f'--> nx (cols): {nx}, nz (rows): {nz}, nx*nz={nx*nz} '
      f'(>= n_cells={n_cells} expected, diff = missing cells due to angled side)')

# value -> index map
x_idx_map = {v: i for i, v in enumerate(x_vals)}
z_idx_map = {v: i for i, v in enumerate(z_vals)}

col_idx = np.array([x_idx_map[v] for v in x_round])
row_idx = np.array([z_idx_map[v] for v in z_round])

# --- build tensor (nt, n_features, nz, nx), 0 outside grid ---
Data_tensor = np.zeros((nt, n_features, nz, nx))

for f in range(n_features):
    field_block = DataMatrix[f*n_cells:(f+1)*n_cells, :]   # (n_cells, nt)
    for t in range(nt):
        Data_tensor[t, f, row_idx, col_idx] = field_block[:, t]
    print(f'({f+1}/{n_features}) feature {features[f]} filled into matrix.')

Data_tensor = Data_tensor[:, :, ::-1, :]  # same Z flip as original script

print('--> Data_tensor shape: [nt, nf, nz, nx] =', Data_tensor.shape)

"""
# --- visual check ---
i_T = features.index('T')
plt.imshow(Data_tensor[1000, i_T, :, :], cmap='hot_r')
plt.savefig("delete_T_check_plot.png", dpi=300, bbox_inches="tight")
plt.close()

i_OH = features.index('OH')
plt.imshow(Data_tensor[1000, i_OH, :, :], cmap='hot_r')
plt.savefig("delete_OH_check_plot.png", dpi=300, bbox_inches="tight")
plt.close()
"""

np.save('data4convolution/step_A03.npy', Data_tensor)
print('--> SAVED FILE!')