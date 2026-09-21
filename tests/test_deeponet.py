import numpy as np
import pytest
import torch

from Baselines.OrderReduction.Identity import Identity
from Baselines.Forecast.DL.deeponet import DeepONet, DeepONetNetwork


class _Frames:
    field_shape = (2, 12)


FIELDS, CELLS = 2, 12
RANK = FIELDS * CELLS


def coords():
    rng = np.random.default_rng(0)
    xyz = rng.uniform(size=(CELLS, 3))
    xyz[:, 1] = 0.5  # planar in Y
    return xyz


def test_identity_roundtrip():
    compressor = Identity().fit(_Frames(), scaler=None)
    frames = np.arange(2 * FIELDS * CELLS, dtype=np.float32).reshape(2, FIELDS, CELLS)
    latent = compressor.encode(frames)
    assert latent.shape == (2, RANK) and compressor.rank == RANK
    assert np.array_equal(compressor.decode(latent), frames)


def network(**overrides):
    kwargs = dict(Nx=2, Ni=1, coordinates=coords(), fields=FIELDS, sensors=5,
                  p=8, hidden=16, branch_layers=2, trunk_layers=2)
    kwargs.update(overrides)
    return DeepONetNetwork(RANK, **kwargs)


def test_network_shapes_and_rejections():
    net = network()
    out = net(torch.randn(3, 3, RANK), torch.rand(3, 3))
    assert out.shape == (3, RANK)
    with pytest.raises(ValueError, match="incompatible"):
        network(fields=5)
    with pytest.raises(ValueError, match="coordinates"):
        DeepONetNetwork(RANK, coordinates=None)


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
    model = DeepONet(RANK, Nx=2, Ni=1, coordinates=coords(), fields=FIELDS,
                     sensors=5, p=8, hidden=16, device="cpu")
    predicted = model.predict(np.zeros((4, 3, RANK), dtype=np.float32),
                              np.ones((4, 3), dtype=np.float32))
    assert predicted.shape == (4, RANK) and np.isfinite(predicted).all()
