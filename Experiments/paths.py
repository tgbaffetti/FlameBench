"""One experiment group, one folder per seed."""
from datetime import datetime, timezone
from pathlib import Path


def new_run_name(config):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"{config['compressor']['name']}_{config['forecaster']['name']}_{timestamp}"


def run_directory(config):
    name = config.get("run_name")
    if not isinstance(name, str) or not name:
        raise ValueError("Specify run_name (or --run-name) to identify the experiment")
    if name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("run_name must be a single folder name")
    seed = config.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32-1")
    return Path(config["output"]) / name / f"seed_{seed}"
