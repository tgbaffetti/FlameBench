"""Run directory layout: output / run_name / seed_<s> [/ trial_<nnnn>]."""
from datetime import datetime, timezone
from pathlib import Path


def new_run_name(config):
    """Group name shared by every seed of one configuration."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{config['compressor']['name']}_{config['model']['name']}_{stamp}"


def run_directory(config):
    name = config.get("run_name")
    if not name:
        raise ValueError("run_name is required to locate a run directory")
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("run_name must be a single folder name")
    directory = Path(config.get("output", "Experiments/Results")) / name / f"seed_{config.get('seed', 42)}"
    if "trial" in config:
        directory = directory / f"trial_{config['trial']:04d}"
    return directory
