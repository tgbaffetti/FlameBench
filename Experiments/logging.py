"""TensorBoard, JSONL and W&B share exactly the same scalar metrics."""
import json
import os
import uuid
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv


class ExperimentLogger:
    def __init__(self, directory, config, resume=False):
        load_dotenv(Path.cwd() / ".env", override=False)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(directory / "tensorboard"))
        self.file = (directory / "metrics.jsonl").open("a")
        self.run = None
        wb = config.get("logging", {}).get("wandb", {})
        mode = os.getenv("WANDB_MODE", wb.get("mode", "offline"))
        if wb.get("mode") == "disabled":
            mode = "disabled"  # Explicitly disabled test runs never contact W&B.
        if mode != "disabled":
            import wandb
            id_path = directory / "wandb_id.txt"
            run_id = id_path.read_text().strip() if resume and id_path.exists() else uuid.uuid4().hex[:8]
            group = config.get("run_name")
            name = f"{group}/seed_{config.get('seed', 42)}" if group else directory.name
            if "trial" in config:
                name += f"/trial_{config['trial']:04d}"
            self.run = wandb.init(project=os.getenv("WANDB_PROJECT", wb.get("project", "rom-flamebench")),
                                  entity=os.getenv("WANDB_ENTITY", wb.get("entity", "FireMark")), mode=mode, config=config,
                                  dir=str(directory.resolve()), name=name, group=group, id=run_id,
                                  resume="allow" if resume and mode == "online" else None)
            id_path.write_text(run_id)

    def log(self, values, step):
        for name, value in values.items():
            if value is not None:
                self.writer.add_scalar(name, value, step)
        self.file.write(json.dumps({"step": step, **values}, allow_nan=False) + "\n")
        self.file.flush()
        if self.run:
            self.run.log({"epoch": step, **values})

    def close(self):
        self.writer.close()
        self.file.close()
        if self.run:
            self.run.finish()
