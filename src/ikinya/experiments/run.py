import argparse
import json
import os
import time

from ikinya.core.seed import set_seed
from ikinya.core.logging import RunLogger
from ikinya.experiments.registry import get_experiment

# IMPORTANT: import experiments so they register
import ikinya.experiments.exp_smoke  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    run_name = cfg["run"]["name"]
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + run_name
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    resolved_path = os.path.join(run_dir, "config.resolved.json")
    with open(resolved_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    set_seed(int(cfg["run"]["seed"]))
    logger = RunLogger(run_dir)

    experiment_fn = get_experiment(args.exp)
    experiment_fn(cfg, run_dir, logger)

    logger.close()
    print("RUN_DIR:", run_dir)


if __name__ == "__main__":
    main()