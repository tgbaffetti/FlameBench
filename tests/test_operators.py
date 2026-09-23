"""Mesh-native operators (DeepONet, Transolver) on the identity-compressor path."""
import json
import numpy as np
import pytest
import torch

from Baselines.Forecast.DL.mesh import cell_coordinates
from Baselines.Forecast.DL.deeponet import DeepONet, DeepONetNetwork
from Baselines.Forecast.DL.transolver import Transolver, TransolverNetwork
from tests.test_benchmark import GRID, metadata, SETTINGS  # noqa: F401  (fixture)

FIELDS, CELLS = 2, 4
RANK = FIELDS * CELLS


@pytest.fixture
def grid_path(tmp_path):
    GRID.save(tmp_path / 'grid.npz')
    return str(tmp_path / 'grid.npz')


def test_cell_coordinates_normalized_original_order(grid_path):
    coords = cell_coordinates(grid_path)
    assert coords.shape == (CELLS, 2) and coords.min() == 0 and coords.max() == 1
    # Original cell order: GRID centers x = 0,1,0,1 and z = 0,0,1,2.
    np.testing.assert_allclose(coords[:, 0], [0, 1, 0, 1])
    np.testing.assert_allclose(coords[:, 1], [0, 0, .5, 1])
    with pytest.raises(ValueError, match='grid'):
        cell_coordinates(None)


def network(cls, grid_path, **kwargs):
    small = dict(deeponet=dict(sensors=3, p=4, hidden=8, branch_layers=2, trunk_layers=2),
                 transolver=dict(dim=8, depth=2, heads=2, slices=4, mlp_ratio=2))[cls.name]
    args = dict(input_size=RANK + 1, output_size=RANK, Nx=1, Ni=1, grid=grid_path,
                fields=FIELDS, device='cpu', **small)
    args.update(kwargs)
    return cls(**args)


@pytest.mark.parametrize('cls', [DeepONet, Transolver])
def test_operator_contract(cls, grid_path):
    model = network(cls, grid_path)
    states, forcing = torch.zeros(3, 2, RANK), torch.zeros(3, 2)
    out = model.predict(states.numpy(), forcing.numpy())
    assert out.shape == (3, RANK) and np.isfinite(out).all()
    with pytest.raises(ValueError, match='fields'):
        cls(input_size=RANK + 1, output_size=RANK, grid=grid_path)
    with pytest.raises(ValueError, match='incompatible'):
        network(cls, grid_path, fields=3)


@pytest.mark.parametrize('cls', [DeepONetNetwork, TransolverNetwork])
def test_operator_learns_teacher(cls, grid_path):
    torch.manual_seed(0)
    coords = cell_coordinates(grid_path)
    build = (lambda: cls(RANK, 1, 1, coords, FIELDS, 3, 4, 8, 2, 2)) if cls is DeepONetNetwork \
        else (lambda: cls(RANK, 1, 1, coords, FIELDS, 8, 2, 2, 4, 2))
    net, teacher = build(), build()
    states, forcing = torch.randn(16, 2, RANK), torch.rand(16, 2)
    with torch.no_grad():
        target = teacher(states, forcing)  # Realizable: produced by a same-shape teacher.
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    initial = torch.nn.functional.mse_loss(net(states, forcing), target).item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(net(states, forcing), target)
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.5 * initial


def test_deeponet_end_to_end(metadata, tmp_path):
    from Experiments.run import fit, test as run_test
    path = tmp_path / 'metadata.json'
    path.write_text(json.dumps(metadata))
    cfg = {'metadata': str(path), 'output': str(tmp_path / 'run'), 'run_name': 'deeponet', 'seed': 42,
           **SETTINGS, 'batch_size': 8,
           'compressor': {'name': 'identity'}, 'dataset': {'Nx': 1, 'Ni': 1, 'horizon': 2},
           'forecaster': {'name': 'deeponet', 'grid': metadata['grid_indices'], 'fields': 2,
                          'sensors': 4, 'p': 4, 'hidden': 8, 'branch_layers': 2, 'trunk_layers': 2,
                          'epochs': 2, 'patience': 2},
           'evaluation': {'heat_release': True}}
    assert np.isfinite(fit(cfg))
    result = run_test(cfg)['sine']
    assert result['status'] in {'completed', 'diverged'}  # Two epochs prove the path, not accuracy.
    if result['status'] == 'completed':
        assert np.isfinite(result['mean_nrmse'])
