"""TensorBoard, JSONL and W&B share exactly the same scalar metrics.

W&B shows one workspace section per key prefix:
  Train/         training losses per epoch, one epoch axis per stage (Train/<stage>_epoch)
  Validation/    validation losses per epoch, best epochs, final errors, error against rollout step
  Test/          per test case: scalars, a per-field NRMSE table and the integrated heat release
  Test_Summary/  the numbers that rank models (also in the run summary, for bar charts)
  HPO/           objective and best-so-far per trial, trial tables, parameter importance (HPO run)

log() records scalars, optionally against their own step axis; summary() the final numbers;
table() and lines() a table and a line plot (W&B and JSONL only). Runs are grouped by
model_name = "<compressor>_<forecaster>" (one group per model: its HPO run and every seed) and
tagged with the config's logging.wandb.tags plus its stage ("hpo", "fit" or "refit").
"""
import json
import os
import uuid
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv


def model_name(config):
    """ "<compressor>_<forecaster>", or None when a section has no name."""
    names = [config.get(key, {}).get("name") for key in ("compressor", "forecaster")]
    return "_".join(names) if all(names) else None


class ExperimentLogger:
    def __init__(self, directory, config, resume=False, name=None):
        load_dotenv(Path.cwd() / ".env", override=False)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(directory / "tensorboard"))
        self.file = (directory / "metrics.jsonl").open("a")
        self.run, self.axes = None, {}
        wb = config.get("logging", {}).get("wandb", {})
        mode = os.getenv("WANDB_MODE", wb.get("mode", "offline"))
        if wb.get("mode") == "disabled":
            mode = "disabled"  # Explicitly disabled test runs never contact W&B.
        if mode != "disabled":
            import wandb
            id_path = directory / "wandb_id.txt"
            run_id = id_path.read_text().strip() if resume and id_path.exists() else uuid.uuid4().hex[:8]
            run_name, stage = config.get("run_name"), config.get("stage")
            name = name or (f"{run_name}/seed_{config.get('seed', 42)}" if run_name else directory.name)
            model = model_name(config)
            self.run = wandb.init(project=os.getenv("WANDB_PROJECT", wb.get("project", "rom-flamebench")),
                                  entity=os.getenv("WANDB_ENTITY", wb.get("entity", "FireMark")), mode=mode,
                                  config={**config, "model_name": model},
                                  tags=list(wb.get("tags", [])) + ([stage] if stage else []),
                                  group=model or run_name, job_type=stage or "fit", name=name,
                                  dir=str(directory.resolve()), id=run_id,
                                  resume="allow" if resume and mode == "online" else None)
            id_path.write_text(run_id)

    def log(self, values, step=0, axis=None):
        """Scalars at a step. axis names the step in W&B (e.g. "Train/forecaster_epoch"), so each
        training stage and HPO stage gets its own x-axis; without it W&B uses its own counter."""
        for name, value in values.items():
            if value is not None:
                self.writer.add_scalar(name, value, step)
        self.file.write(json.dumps({"step": step, **values}, allow_nan=False) + "\n")
        self.file.flush()
        values = {name: value for name, value in values.items() if value is not None}
        if self.run and values:  # None (e.g. no gain for a step input) would make empty W&B panels.
            if axis:
                for name in values:
                    if self.axes.get(name) != axis:
                        self.run.define_metric(name, step_metric=axis)
                        self.axes[name] = axis
                self.run.log({axis: step, **values})
            else:
                self.run.log(values)

    def summary(self, values):
        """One final value per metric: W&B run summary (a column of the runs table, so bar charts
        and parallel coordinates compare runs) and one history row (W&B builds its automatic
        panels from the history only), plus a {"summary": ...} JSONL line and TensorBoard
        scalars at step 0."""
        for name, value in values.items():
            if value is not None:
                self.writer.add_scalar(name, value, 0)
        self.file.write(json.dumps({"summary": values}, allow_nan=False) + "\n")
        self.file.flush()
        if self.run:
            logged = {k: v for k, v in values.items() if v is not None}
            self.run.log(logged)
            self.run.summary.update(logged)

    def table(self, key, columns, rows):
        self.file.write(json.dumps({"table": key, "columns": columns, "rows": rows}, allow_nan=False) + "\n")
        self.file.flush()
        if self.run:
            import wandb
            self.run.log({key: wandb.Table(columns=columns, data=rows)})

    def lines(self, key, x, series, title, x_name):
        """A line plot of one or more series (name -> values) against x."""
        x = [float(v) for v in x]
        series = {name: [float(v) for v in values] for name, values in series.items()}
        self.file.write(json.dumps({"lines": key, "x": x, **series}, allow_nan=False) + "\n")
        self.file.flush()
        if self.run:
            import wandb
            self.run.log({key: wandb.plot.line_series(xs=x, ys=list(series.values()), keys=list(series),
                                                      title=title, xname=x_name)})

    def close(self):
        self.writer.close()
        self.file.close()
        if self.run:
            self.run.finish()
