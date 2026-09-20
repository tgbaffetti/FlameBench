import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import os

# ---------------- CONFIG ----------------
GRID_PATH = 'data/grid.vtu'
DATA_DIR = 'data'
FILE_NAMES = [
    'sineSweep_f1_f80_A02.npy',
    'sineSweep_f1_f80_A04.npy',
    'step_A03.npy',
    'step_A05.npy',
    'sine_f10_A03.npy',
    'sine_f10_A05.npy',
    'sine_f40_A03.npy',
    'sine_f40_A05.npy',
]
OUTPUT_DIR = 'data4convolution'
features = ['p', 'U1', 'U3', 'rho', 'T', 'mix:Q', 'CH4', 'O2', 'H2O', 'CO2', 'OH']
n_features = len(features)
DECIMALS = 8   # rounding tolerance for X/Z binning (tune if needed)
# -----------------------------------------

grid = pv.read(GRID_PATH)
xyz = grid.cell_centers().points
n_cells = xyz.shape[0]

# --- planarity check (Y should be ~constant), grid-only, done once ---
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
print(f'--> nx: {nx}, nz: {nz}, nx*nz={nx*nz} '
      f'(>= n_cells={n_cells} expected, diff = missing cells due to angled side)')

x_idx_map = {v: i for i, v in enumerate(x_vals)}
z_idx_map = {v: i for i, v in enumerate(z_vals)}

col_idx = np.array([x_idx_map[v] for v in x_round])
row_idx = np.array([z_idx_map[v] for v in z_round])

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOOP OVER FILES ----------------
for FILE_NAME in FILE_NAMES:
    print(f'\n=== Processing {FILE_NAME} ===')
    DATA_PATH = os.path.join(DATA_DIR, FILE_NAME)
    DataMatrix = np.load(DATA_PATH)
    nt = DataMatrix.shape[2]

    print(f'--> n_cells: {n_cells}, nt: {nt}')
    print(f'--> check n_cells == DataMatrix.shape[0]: {n_cells} == {DataMatrix.shape[0]}')
    print(f'--> check n_features == DataMatrix.shape[1]: {n_features} == {DataMatrix.shape[1]}')

    Data_tensor = np.zeros((nx, nz, n_features, nt))

    for f in range(n_features):
        field_block = DataMatrix[:, f, :]
        Data_tensor[col_idx, row_idx, f, :] = field_block
        print(f'({f+1}/{n_features}) feature {features[f]} filled into matrix.')

    Data_tensor = Data_tensor[:, ::-1, :, :]

    print('--> Data_tensor shape:', Data_tensor.shape)

    """
    # --- visual check ---
    i_T = features.index('T')
    plt.imshow(Data_tensor[:, :, i_T, 1000].T, cmap='hot_r')
    plt.savefig(f"delete_T_check_plot_{FILE_NAME}.png", dpi=300, bbox_inches="tight")
    plt.close()

    i_OH = features.index('OH')
    plt.imshow(Data_tensor[:, :, i_OH, 1000].T, cmap='hot_r')
    plt.savefig(f"delete_OH_check_plot_{FILE_NAME}.png", dpi=300, bbox_inches="tight")
    plt.close()
    """

    np.save(os.path.join(OUTPUT_DIR, FILE_NAME), Data_tensor)
    print(f'--> SAVED FILE: {os.path.join(OUTPUT_DIR, FILE_NAME)}')

print('\n--> ALL FILES DONE!')