
from __future__ import annotations

from .strategies.base import Strategy, get_strategy, available_strategies
from .strategies.selectors import (
    PromptSelector,
    get_prompt_selector,
    available_selectors,
)
from .strategies.schedulers import (
    IntervalScheduler,
    get_interval_scheduler,
    available_schedulers,
)


@Strategy.register("simple")
class SimpleStrategy(Strategy):

    def get_name(self) -> str:
        return "simple"


__all__ = [
    "Strategy",
    "SimpleStrategy",
    "get_strategy",
    "available_strategies",
    "PromptSelector",
    "get_prompt_selector",
    "available_selectors",
    "IntervalScheduler",
    "get_interval_scheduler",
    "available_schedulers",
]
