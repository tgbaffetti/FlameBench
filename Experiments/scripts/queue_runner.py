"""Run a queue of configs sequentially: fit, full-AR test, one-step test each.

Usage: python -m Experiments.scripts.queue_runner CONFIG [CONFIG ...]
Failures are reported and the queue continues; markers go to stdout for monitoring.
"""
import json
import sys
import traceback
from pathlib import Path
from Experiments.run import fit, test
from Experiments.paths import new_run_name


def main():
    for config_path in sys.argv[1:]:
        name = Path(config_path).stem
        cfg = json.loads(Path(config_path).read_text())
        cfg["run_name"] = new_run_name(cfg)
        print(f"QUEUE_START {name} -> {cfg['run_name']}", flush=True)
        try:
            fit(cfg)
            for restart in (None, 1):
                cfg["evaluation"]["restart_every"] = restart
                test(cfg)
            print(f"QUEUE_OK {name}", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"QUEUE_FAILED {name}", flush=True)
    print("QUEUE_DONE", flush=True)


if __name__ == "__main__":
    main()
