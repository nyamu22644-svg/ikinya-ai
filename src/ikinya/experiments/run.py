import argparse
import json
import os
import time

from ikinya.core.seed import set_seed
from ikinya.core.logging import RunLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to JSON config")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    run_name = cfg["run"]["name"]
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + run_name
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # save resolved config
    resolved_path = os.path.join(run_dir, "config.resolved.json")
    with open(resolved_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # seed + logger
    set_seed(int(cfg["run"]["seed"]))
    logger = RunLogger(run_dir)
    logger.log_event(0, "run_start", {"config": args.config})
    logger.close()

    print("RUN_DIR:", run_dir)


if __name__ == "__main__":
    main()