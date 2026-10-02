
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

from .base import IntervalScheduler


class TraceScheduler(IntervalScheduler):

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.trace_path = config.get("trace_path")
        self.repeat = config.get("repeat", True)
        self.shuffle = config.get("shuffle", False)

        self._intervals: List[float] = []
        self._load_trace()

    def _load_trace(self) -> None:
        if not self.trace_path:
            raise ValueError("TraceScheduler requires 'trace_path' in config")

        path = Path(self.trace_path)
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")

        intervals: List[float] = []

        if path.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    intervals = [float(x) for x in data]
                elif isinstance(data, dict) and "intervals" in data:
                    intervals = [float(x) for x in data["intervals"]]
        else:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        try:
                            intervals.append(float(line))
                        except ValueError:
                            continue

        if not intervals:
            raise ValueError(f"No intervals found in trace file: {self.trace_path}")

        if self.shuffle:
            import random
            random.shuffle(intervals)

        self._intervals = intervals
        self._index = 0

    def next_interval(self, user_id: int) -> float:
        if not self._intervals:
            return 1.0  # fallback

        interval = self._intervals[self._index]

        self._index += 1
        if self._index >= len(self._intervals):
            if self.repeat:
                self._index = 0
                if self.shuffle:
                    import random
                    random.shuffle(self._intervals)
            else:
                self._index = len(self._intervals) - 1
                return float("inf")

        return interval
