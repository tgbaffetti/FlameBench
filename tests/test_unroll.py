import numpy as np
import pytest
import torch

from DataProcessing.latent import LatentDataset
from Baselines.Forecast.DL.networks import GRU

RANK = 3


class _Source:
    """Minimal WindowDataset stand-in: one trajectory of 20 steps."""
    Nx, Ni = 2, 1
    history, context = 3, 3

    def segments(self):
        return [({"name": "case"}, 0, 20)]

    def arrays(self, case):
        rng = np.random.default_rng(0)
        return rng.normal(size=(20, RANK)).astype(np.float32), \
            np.linspace(1.0, 1.2, 20, dtype=np.float32)


class _Identity:
    rank = RANK

    def encode(self, frames):
        return np.asarray(frames, dtype=np.float32)


class _Scaler:
    def transform(self, frames):
        return frames


def dataset(tmp_path, horizon):
    return LatentDataset(_Source(), _Identity(), _Scaler(), tmp_path / f"h{horizon}", horizon=horizon)


def test_horizon_shapes_and_length(tmp_path):
    single, unrolled = dataset(tmp_path, 1), dataset(tmp_path, 4)
    assert len(unrolled) == len(single) - 3
    history, forcing, target = unrolled[0]
    assert history.shape == (3, RANK)
    assert forcing.shape == (4, _Source.Ni + 2)
    assert target.shape == (4, RANK)
    # Step k's forcing window ends at phi(t+k+1), matching evaluation.forcing_window.
    x, phi = _Source().arrays(None)
    assert np.allclose(forcing[2], phi[_Source.context + 2 - _Source.Ni - 1:_Source.context + 3])
    assert np.allclose(target[2], x[_Source.context + 2])
    with pytest.raises(ValueError, match="horizon"):
        dataset(tmp_path, 0)
    with pytest.raises(ValueError, match="too short|Segment"):
        dataset(tmp_path, 18)


@pytest.mark.parametrize("unroll_grad", ["none", "full"])
def test_compute_loss_unrolled(unroll_grad):
    torch.manual_seed(0)
    model = GRU(rank=RANK, Nx=2, Ni=1, hidden=8, layers=1, device="cpu", unroll_grad=unroll_grad)
    history = torch.randn(5, 3, RANK)
    forcing = torch.rand(5, 4, 3)
    target = torch.randn(5, 4, RANK)
    loss = model.compute_loss(history, forcing, target)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.network.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_compute_loss_rejects_bad_mode():
    with pytest.raises(ValueError, match="unroll_grad"):
        GRU(rank=RANK, Nx=2, Ni=1, hidden=8, layers=1, device="cpu", unroll_grad="typo")


def test_residual_and_noise():
    torch.manual_seed(0)
    model = GRU(rank=RANK, Nx=2, Ni=1, hidden=8, layers=1, device="cpu",
                residual=True, noise_std=0.1)
    history = torch.randn(5, 3, RANK)
    forcing = torch.rand(5, 3)
    # residual: prediction = last state + network delta
    delta = model.network(history, forcing)
    assert torch.allclose(model.step(history, forcing), history[:, -1] + delta)
    predicted = model.predict(history.numpy(), forcing.numpy())
    assert np.allclose(predicted, (history[:, -1] + delta).detach().numpy(), atol=1e-6)
    # noise only at training time: eval losses are deterministic, train losses are not
    target = torch.randn(5, RANK)
    eval_losses = {model.compute_loss(history, forcing, target).item() for _ in range(3)}
    train_losses = {model.compute_loss(history, forcing, target, train=True).item() for _ in range(3)}
    assert len(eval_losses) == 1 and len(train_losses) == 3
    with pytest.raises(ValueError, match="noise_std"):
        GRU(rank=RANK, Nx=2, Ni=1, hidden=8, layers=1, device="cpu", noise_std=-1)
