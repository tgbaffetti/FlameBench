import numpy as np
from openmeasure.sparse_sensing import ROM

def scale_train_tensor(data_train_tensor, grid):
    """
    data_train_tensor: [Nt, Nf, Nz, Nx]
    grid: [Nz*Nx, 3] (stesso ordine del flatten usato nel dataset)
    """
    Nt, Nf, Nz, Nx = data_train_tensor.shape

    data_train_mat = data_train_tensor.reshape(Nt, Nf * Nz * Nx).T
    rom = ROM(data_train_mat, Nf, grid)
    # data_train_mat_scaled = rom.scale_data()
    data_train_mat_scaled = rom.scale_data(scale_type='range')
    data_train_scaled = data_train_mat_scaled.T.reshape(Nt, Nf, Nz, Nx)

    return data_train_scaled, rom

def scale_test_tensor(data_test_tensor, rom):
    """
    data_train_tensor: [Nt, Nf, Nz, Nx]
    """
    Nt, Nf, Nz, Nx = data_test_tensor.shape

    data_test_mat = data_test_tensor.reshape(Nt, Nf * Nz * Nx).T
    data_test_mat_scaled = (data_test_mat - rom.X_cnt) / rom.X_scl
    data_test_scaled = data_test_mat_scaled.T.reshape(Nt, Nf, Nz, Nx)

    return data_test_scaled

def rescale_back_output(output_tensor, rom):
    """
    output_tensor: [Nt, Nf, Nz, Nx]
    """
    Nt, Nf, Nz, Nx = output_tensor.shape

    output_mat = output_tensor.reshape(Nt, Nf * Nz * Nx).T
    output_mat_rescaled = rom.X_scl*output_mat + rom.X_cnt
    output_rescaled = output_mat_rescaled.T.reshape(Nt, Nf, Nz, Nx)

    return output_rescaled