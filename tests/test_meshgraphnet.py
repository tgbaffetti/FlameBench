import numpy as np
import pytest
import torch

from Baselines.Forecast.DL.meshgraphnet import MeshGraphNet, MGNNetwork

FIELDS, CELLS = 2, 12
RANK = FIELDS * CELLS


def coords():
    rng = np.random.default_rng(0)
    xyz = rng.uniform(size=(CELLS, 3))
    xyz[:, 1] = 0.5
    return xyz


def edges():
    # ring graph, both directions
    forward = np.stack((np.arange(CELLS), (np.arange(CELLS) + 1) % CELLS))
    return np.concatenate((forward, forward[::-1]), axis=1)


def network(**overrides):
    kwargs = dict(Nx=2, Ni=1, coordinates=coords(), edges=edges(), fields=FIELDS,
                  dim=16, hidden=16, passes=2)
    kwargs.update(overrides)
    return MGNNetwork(RANK, **kwargs)


def test_network_shapes_and_rejections():
    out = network()(torch.randn(3, 3, RANK), torch.rand(3, 3))
    assert out.shape == (3, RANK)
    with pytest.raises(ValueError, match="incompatible"):
        network(fields=5)
    with pytest.raises(ValueError, match="requires"):
        MGNNetwork(RANK, coordinates=coords(), edges=None)
    with pytest.raises(ValueError, match="indices"):
        network(edges=np.array([[0, 1], [1, 99]]).T)


def test_message_passing_propagates():
    # With 2 passes on a ring, perturbing one cell must affect its neighbors' output.
    torch.manual_seed(0)
    net = network()
    base = torch.zeros(1, 3, RANK)
    poked = base.clone()
    poked[0, :, 0] = 5.0  # cell 0, field 0
    difference = (net(poked, torch.ones(1, 3)) - net(base, torch.ones(1, 3))).abs()
    difference = difference.reshape(FIELDS, CELLS).sum(dim=0)
    assert difference[1] > 1e-6 and difference[CELLS - 1] > 1e-6


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
    model = MeshGraphNet(RANK, Nx=2, Ni=1, coordinates=coords(), edges=edges(),
                         fields=FIELDS, dim=16, hidden=16, passes=2, device="cpu")
    predicted = model.predict(np.zeros((4, 3, RANK), dtype=np.float32),
                              np.ones((4, 3), dtype=np.float32))
    assert predicted.shape == (4, RANK) and np.isfinite(predicted).all()
