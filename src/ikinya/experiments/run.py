import argparse
import json
import os
import time
from datetime import datetime

from ikinya.core.seed import set_seed
from ikinya.core.logging import RunLogger
from ikinya.experiments.registry import get_experiment

# ensure experiments register themselves
import ikinya.experiments.exp_smoke  # noqa
import ikinya.experiments.exp_lm_char  # noqa


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_run_dir(cfg: dict) -> str:
    name = cfg.get("run", {}).get("name", "run")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join("runs", f"{ts}_{name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to JSON config")
    ap.add_argument("--exp", required=True, help="Experiment name registered in registry")
    ap.add_argument(
        "--run_dir",
        default=None,
        help="If set, use this run directory (resume/continue). Example: runs\\2026-02-24_12-51-22_lm_char_small",
    )
    args = ap.parse_args()

    cfg = _load_json(args.config)

    run_seed = int(cfg.get("run", {}).get("seed", 1337))
    set_seed(run_seed)

    run_dir = args.run_dir if args.run_dir else _make_run_dir(cfg)
    _ensure_dir(run_dir)

    logger = RunLogger(run_dir)

    resolved = {
        "config_path": args.config,
        "exp": args.exp,
        "run_dir": run_dir,
        "created_at": time.time(),
        "cfg": cfg,
    }
    with open(os.path.join(run_dir, "config.resolved.json"), "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2)

    print("RUN_DIR:", run_dir)

    experiment_fn = get_experiment(args.exp)
    experiment_fn(cfg, run_dir, logger)

    logger.close()


if __name__ == "__main__":
    main()