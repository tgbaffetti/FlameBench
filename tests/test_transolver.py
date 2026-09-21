import numpy as np
import pytest
import torch

from Baselines.Forecast.DL.transolver import PhysicsAttention, Transolver, TransolverNetwork

FIELDS, CELLS = 2, 12
RANK = FIELDS * CELLS


def coords():
    rng = np.random.default_rng(0)
    xyz = rng.uniform(size=(CELLS, 3))
    xyz[:, 1] = 0.5
    return xyz


def network(**overrides):
    kwargs = dict(Nx=2, Ni=1, coordinates=coords(), fields=FIELDS,
                  dim=16, depth=2, heads=2, slices=4)
    kwargs.update(overrides)
    return TransolverNetwork(RANK, **kwargs)


def test_physics_attention_shape():
    attention = PhysicsAttention(dim=16, heads=2, slices=4)
    assert attention(torch.randn(3, CELLS, 16)).shape == (3, CELLS, 16)


def test_network_shapes_and_rejections():
    out = network()(torch.randn(3, 3, RANK), torch.rand(3, 3))
    assert out.shape == (3, RANK)
    with pytest.raises(ValueError, match="incompatible"):
        network(fields=5)
    with pytest.raises(ValueError, match="coordinates"):
        TransolverNetwork(RANK, coordinates=None)


def test_network_learns():
    torch.manual_seed(0)
    net = network()
    history, forcing = torch.randn(16, 3, RANK), torch.rand(16, 3)
    with torch.no_grad():
        target = network()(history, forcing)  # realizable: produced by a same-shape teacher
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    initial = torch.nn.functional.mse_loss(net(history, forcing), target).item()
    for _ in range(50):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(net(history, forcing), target)
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.5 * initial


def test_model_predict_shape():
    model = Transolver(RANK, Nx=2, Ni=1, coordinates=coords(), fields=FIELDS,
                       dim=16, depth=2, heads=2, slices=4, device="cpu")
    predicted = model.predict(np.zeros((4, 3, RANK), dtype=np.float32),
                              np.ones((4, 3), dtype=np.float32))
    assert predicted.shape == (4, RANK) and np.isfinite(predicted).all()
