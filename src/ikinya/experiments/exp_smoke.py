from ikinya.core.logging import RunLogger
from ikinya.experiments.registry import register


@register("smoke")
def run_smoke(cfg: dict, run_dir: str, logger: RunLogger):
    logger.log_event(0, "smoke_test", {"message": "experiment system working"})
    logger.log_metrics(1, {"loss": 0.123})