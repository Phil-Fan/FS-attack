
from __future__ import annotations

import random

from .base import IntervalScheduler


class PoissonScheduler(IntervalScheduler):

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.base_interval = config.get("base_interval", 1.0)
        self.lambda_value = config.get("lambda_value", 1.0)

    def next_interval(self, user_id: int) -> float:
        # Mean is 1/lambda, so we scale by base_interval
        return random.expovariate(self.lambda_value) * self.base_interval
