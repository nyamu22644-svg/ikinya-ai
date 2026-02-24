from typing import Callable, Dict


_EXPERIMENTS: Dict[str, Callable] = {}


def register(name: str):
    def decorator(fn: Callable):
        _EXPERIMENTS[name] = fn
        return fn
    return decorator


def get_experiment(name: str) -> Callable:
    if name not in _EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{name}'. Available: {list(_EXPERIMENTS.keys())}")
    return _EXPERIMENTS[name]