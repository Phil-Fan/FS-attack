
from __future__ import annotations

from .base import IntervalScheduler


class BurstScheduler(IntervalScheduler):

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.base_interval = config.get("base_interval", 1.0)
        self.burst_interval = config.get("burst_interval", 0.1)
        self.cycle_time = config.get("cycle_time", 20.0)
        self.burst_time = config.get("burst_time", 5.0)

    def next_interval(self, user_id: int) -> float:
        elapsed = self._elapsed()
        in_burst = (elapsed % self.cycle_time) < self.burst_time
        return self.burst_interval if in_burst else self.base_interval
