"""0-D flame-response runner: phi window -> q'(t)/q0, trained on Data/Qseries targets.

Comparable with the field models through the same q' metrics (relative L2, FTF
gain/phase); it fills only the heat-release columns of the benchmark table.

  python -m Experiments.zerod run --config Experiments/Configs/zerod_mlp.json
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from Baselines.ZeroD import ZEROD_MODELS
from DataProcessing.metadata import load_metadata
from utils import write_json, seed_everything
from .metrics import relative_l2, gain_phase
from .paths import run_directory, new_run_name


def qseries_path(metadata_path, case):
    return Path(metadata_path).resolve().parent / "Qseries" / f"{case['name']}.npy"


def case_signals(metadata_path, case, window):
    """Return (inputs, targets, q, phi, q0): windows of phi'-taps and q'/q0 targets."""
    phi = np.load(case["phi"], allow_pickle=False).astype(np.float64)
    q = np.load(qseries_path(metadata_path, case), allow_pickle=False)
    if len(phi) != len(q):
        raise ValueError(f"phi/q length mismatch in {case['name']}")
    q0 = float(q[0])
    if q0 == 0:
        raise ValueError(f"Zero steady heat release in {case['name']}")
    # Pre-forcing history equals the steady value phi=1 (phi' = 0), as in evaluation.py.
    padded = np.concatenate((np.ones(window - 1), phi)) - 1.0
    inputs = np.lib.stride_tricks.sliding_window_view(padded, window).astype(np.float32)
    targets = (q / q0 - 1.0).astype(np.float32)
    return inputs, targets, q, phi, q0


def fit(config):
    cfg = config
    seed_everything(cfg["seed"])
    directory = run_directory(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(cfg["metadata"])
    window, device = cfg["window"], cfg.get("device", "cpu")
    inputs, targets = [], []
    for case in metadata["cases"]:
        if case["split"] != "training":
            continue
        x, y, *_ = case_signals(cfg["metadata"], case, window)
        cut = int(len(y) * (1 - cfg["validation_fraction"]))
        inputs.append((x[:cut], x[cut:]))
        targets.append((y[:cut], y[cut:]))
    train = TensorDataset(torch.from_numpy(np.concatenate([i[0] for i in inputs])),
                          torch.from_numpy(np.concatenate([t[0] for t in targets])))
    val = TensorDataset(torch.from_numpy(np.concatenate([i[1] for i in inputs])),
                        torch.from_numpy(np.concatenate([t[1] for t in targets])))
    mc = dict(cfg["model"])
    model = ZEROD_MODELS[mc.pop("name")](window=window, **mc).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["model"].get("lr", 1e-3))
    loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val, batch_size=cfg["batch_size"])
    best, best_state, patience_left = float("inf"), None, cfg["model"].get("patience", 20)
    for epoch in range(cfg["model"]["epochs"]):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = float(np.mean([torch.nn.functional.mse_loss(model(x.to(device)), y.to(device)).item()
                                   for x, y in val_loader]))
        print(f"epoch {epoch}: validation_mse {score:.6e}", flush=True)
        if score < best:
            best, patience_left = score, cfg["model"].get("patience", 20)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best_state)
    torch.save(best_state, directory / "model.pt")
    write_json(directory / "config.json", cfg)
    write_json(directory / "summary.json", {"validation_mse": best})
    return model


def test(config):
    cfg = config
    directory = run_directory(cfg)
    metadata = load_metadata(cfg["metadata"])
    window, device = cfg["window"], cfg.get("device", "cpu")
    mc = dict(cfg["model"])
    model = ZEROD_MODELS[mc.pop("name")](window=window, **mc).to(device)
    model.load_state_dict(torch.load(directory / "model.pt", map_location=device, weights_only=True))
    model.eval()
    results = {}
    for case in metadata["cases"]:
        if case["split"] != "test":
            continue
        x, _, q_ref, phi, q0 = case_signals(cfg["metadata"], case, window)
        with torch.no_grad():
            qhat = model(torch.from_numpy(x).to(device)).cpu().numpy().astype(np.float64)
        q_pred = q0 * (1.0 + qhat)
        times = np.arange(len(q_ref)) * metadata["dt"]
        result = {"heat_release_relative_l2": relative_l2(q_pred, q_ref),
                  "forecast_steps": len(q_ref), "window": window}
        np.savez(directory / f"{case['name']}_Q.npz", time=times, reference=q_ref, predicted=q_pred)
        if case["waveform"] == "sine":
            result["gain_phase"] = gain_phase(q_pred, q_ref, phi, times, case["frequency_hz"],
                                              q0, cfg["evaluation"]["gain_phase_start"])
        results[case["name"]] = result
        write_json(directory / "metrics.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit", "test", "run"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    for key in ("device", "run_name", "seed"):
        if getattr(args, key.replace("-", "_"), None) is not None:
            config[key] = getattr(args, key.replace("-", "_"))
    if not config.get("run_name"):
        if args.command == "test":
            parser.error("test requires --run-name")
        config["run_name"] = new_run_name({"compressor": {"name": "zerod"}, "model": config["model"]})
    if args.command in {"fit", "run"}:
        fit(config)
    if args.command in {"test", "run"}:
        test(config)


if __name__ == "__main__":
    main()
