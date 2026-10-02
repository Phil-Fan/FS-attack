
from __future__ import annotations

import time
from abc import ABC, abstractmethod


class IntervalScheduler(ABC):

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.start_time = time.time()

    @abstractmethod
    def next_interval(self, user_id: int) -> float:
        raise NotImplementedError

    def _elapsed(self) -> float:
        return time.time() - self.start_time
