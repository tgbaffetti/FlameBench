import json
import numpy as np
import pytest
import torch

from Baselines.ZeroD import ZEROD_MODELS
from Experiments.zerod import case_signals, fit
from Experiments.zerod import test as evaluate_zerod


def synthetic_dataset(tmp_path, nt=400, gain=0.5):
    """q responds linearly and instantaneously to phi: learnable by every model."""
    rng = np.random.default_rng(0)
    (tmp_path / "Qseries").mkdir()
    cases = []
    for name, split, waveform, freq in [("trainA", "training", "sweep", None),
                                        ("trainB", "training", "sweep", None),
                                        ("sine1", "test", "sine", 10.0)]:
        t = np.arange(nt) * 5e-4
        if waveform == "sine":
            phi = 1 + 0.3 * np.sin(2 * np.pi * freq * t)
        else:
            phi = 1 + 0.3 * np.sin(2 * np.pi * (1 + 20 * t) * t) + 0.01 * rng.standard_normal(nt)
        q = 2.0 * (1 + gain * (phi - 1))
        np.save(tmp_path / f"phi_{name}.npy", phi.astype(np.float32))
        np.save(tmp_path / "Qseries" / f"{name}.npy", q)
        np.save(tmp_path / f"{name}.npy", np.zeros((nt, 1, 1), dtype=np.float32))
        cases.append({"name": name, "split": split, "waveform": waveform, "frequency_hz": freq,
                      "duration": float(nt - 1) * 5e-4, "data": f"{name}.npy", "phi": f"phi_{name}.npy"})
    metadata = {"dt": 5e-4, "fields": ["mix:Q"], "units": ["W/m^3"],
                "initial_snapshot_is_steady": True, "cell_volumes": None, "coordinates": None,
                "forcing_definition": "test", "cases": cases}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    return path


def config(tmp_path, name):
    return {"metadata": str(synthetic_dataset(tmp_path)), "output": str(tmp_path / "Results"),
            "run_name": f"zerod_{name}_test", "seed": 0, "device": "cpu",
            "validation_fraction": 0.2, "batch_size": 64, "window": 8,
            "model": {"name": name, "hidden": 32, "layers": 2, "lr": 0.01, "epochs": 30, "patience": 10},
            "evaluation": {"gain_phase_start": 0.05}}


@pytest.mark.parametrize("name", ["mlp", "gru"])
def test_forward_shapes(name):
    model = ZEROD_MODELS[name](window=8, hidden=16, layers=1)
    assert model(torch.zeros(5, 8)).shape == (5,)


def test_case_signals_padding_and_targets(tmp_path):
    metadata_path = synthetic_dataset(tmp_path)
    case = json.loads(metadata_path.read_text())["cases"][0]
    case["phi"] = str(tmp_path / case["phi"])
    inputs, targets, q, phi, q0 = case_signals(metadata_path, case, window=8)
    assert inputs.shape == (len(phi), 8) and targets.shape == (len(phi),)
    assert q0 == q[0]
    # Window k ends at phi'(k); history before t=0 is padded with the steady phi'=0.
    assert np.allclose(inputs[0, :-1], 0) and np.isclose(inputs[0, -1], phi[0] - 1)
    assert np.isclose(targets[0], q[0] / q0 - 1)


def test_fit_and_test_learn_linear_response(tmp_path):
    cfg = config(tmp_path, "mlp")
    fit(cfg)
    results = evaluate_zerod(cfg)
    entry = results["sine1"]
    assert entry["heat_release_relative_l2"] < 0.05
    assert entry["gain_phase"]["relative_gain_error"] < 0.1
    assert (tmp_path / "Results" / cfg["run_name"] / "seed_0" / "metrics.json").exists()
