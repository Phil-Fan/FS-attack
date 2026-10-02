
from __future__ import annotations

import random

from .base import IntervalScheduler


class FixedScheduler(IntervalScheduler):

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.base_interval = config.get("base_interval", 1.0)
        self.jitter_range = config.get("jitter_range", 0.5)

    def next_interval(self, user_id: int) -> float:
        jitter = random.uniform(-self.jitter_range, self.jitter_range) * self.base_interval
        return max(0, self.base_interval + jitter)
