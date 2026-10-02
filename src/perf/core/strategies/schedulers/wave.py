
from __future__ import annotations

import math

from .base import IntervalScheduler


class WaveScheduler(IntervalScheduler):

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.base_interval = config.get("base_interval", 1.0)
        self.period = config.get("period", 10.0)
        self.amplitude_ratio = config.get("amplitude_ratio", 0.5)
        self.amplitude = self.amplitude_ratio * self.base_interval

    def next_interval(self, user_id: int) -> float:
        elapsed = self._elapsed()
        sine_value = math.sin(2 * math.pi * elapsed / self.period)
        interval = self.base_interval + self.amplitude * sine_value
        return max(0, interval)
