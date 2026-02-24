import csv
import json
import os
import time
from typing import Dict, Any, Optional


class RunLogger:
    """
    Minimal research-grade logger:
    - CSV metrics (easy plotting)
    - JSONL event stream (structured lab notebook)
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

        self.csv_path = os.path.join(run_dir, "metrics.csv")
        self.jsonl_path = os.path.join(run_dir, "events.jsonl")

        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer: Optional[csv.DictWriter] = None

        self._jsonl_file = open(self.jsonl_path, "a", encoding="utf-8")

    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        row = {"step": step, **metrics}

        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=list(row.keys())
            )
            if self._csv_file.tell() == 0:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(row)
        self._csv_file.flush()

        self._write_jsonl({"type": "metrics", "step": step, **metrics})

    def log_event(self, step: int, event_type: str, payload: Dict[str, Any]) -> None:
        self._write_jsonl({"type": event_type, "step": step, **payload})

    def _write_jsonl(self, obj: Dict[str, Any]) -> None:
        obj = {"timestamp": time.time(), **obj}
        self._jsonl_file.write(json.dumps(obj) + "\n")
        self._jsonl_file.flush()

    def close(self) -> None:
        self._csv_file.close()
        self._jsonl_file.close()