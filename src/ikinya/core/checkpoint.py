import os
import random
import numpy as np
import torch
from typing import Any, Dict


def save_checkpoint(path: str, *, step: int, model, optimizer, config: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "config": config,
    }

    torch.save(checkpoint, path)


def load_checkpoint(path: str, *, model, optimizer=None, map_location="cpu") -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # restore RNG state (best effort)
    rng = checkpoint.get("rng", {})
    try:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
    except Exception:
        pass

    return checkpoint