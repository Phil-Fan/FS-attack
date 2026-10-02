
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .selectors import PromptSelector, get_prompt_selector
from .schedulers import IntervalScheduler, get_interval_scheduler


class Strategy(ABC):

    _strategies: Dict[str, type[Strategy]] = {}

    def __init__(self, config: Dict):
        self.config = config
        self._prompt_selectors: Dict[str, PromptSelector] = {}
        self._interval_scheduler: Optional[IntervalScheduler] = None

        selector_cfg = config.get("prompt_selector") or config.get("prompt_selectors")
        if selector_cfg:
            if isinstance(selector_cfg, dict):
                if "type" in selector_cfg:
                    self._prompt_selectors["default"] = get_prompt_selector(
                        selector_cfg
                    )
                else:
                    for role, role_cfg in selector_cfg.items():
                        if isinstance(role_cfg, dict):
                            self._prompt_selectors[role] = get_prompt_selector(role_cfg)

        scheduler_cfg = (
            config.get("interval_scheduler")
            or config.get("interval_strategy")
            or config.get("interval")
        )
        if isinstance(scheduler_cfg, dict):
            self._interval_scheduler = get_interval_scheduler(scheduler_cfg)

    @classmethod
    def register(cls, name: str) -> callable:

        def decorator(strategy_class: type[Strategy]) -> type[Strategy]:
            cls._strategies[name] = strategy_class
            return strategy_class

        return decorator

    @abstractmethod
    def get_name(self) -> str:
        pass

    def get_interval(self, user_id: int) -> float:
        if self._interval_scheduler is not None:
            return self._interval_scheduler.next_interval(user_id)
        # Default fallback
        return 1.0

    def get_backoff_wait(self, user_id: int) -> float:
        """State-driven Back-off, if the active selector implements it.

        Returns t_wait (> 0) when the system is at/over saturation
        (Delta_mem <= 0) and injection should pause; 0.0 otherwise. Selectors
        without an adaptive controller (e.g. FileListSelector) return 0.0.
        """
        selector = self._prompt_selectors.get("default")
        if selector is None and self._prompt_selectors:
            selector = next(iter(self._prompt_selectors.values()))
        if selector is not None and hasattr(selector, "get_backoff_wait"):
            return float(selector.get_backoff_wait())
        return 0.0

    def pick_prompt(
        self,
        user_id: int,
        prompt_tiers: Dict[str, List[str]],
        tier: Optional[str] = None,
    ) -> str:
        # This is optional and depends on the strategy implementation
        selector = self._prompt_selectors.get("default")
        if selector is None and self._prompt_selectors:
            # Fallback to first available selector
            selector = next(iter(self._prompt_selectors.values()))

        if selector is not None:
            return selector.select_one(prompt_tiers, tier=tier)

        # Fallback: return first available prompt from specified tier
        if tier is not None:
            tier_prompts = prompt_tiers.get(tier, [])
            if tier_prompts:
                import random

                return random.choice(tier_prompts)
            raise ValueError(f"No prompts available in tier: {tier}")

        # Fallback to first available tier
        for tier_prompts in prompt_tiers.values():
            if tier_prompts:
                import random

                return random.choice(tier_prompts)

        raise ValueError("No prompts available")


def get_strategy(
    config: Dict, default_type: str = "simple"
) -> Optional[Strategy]:
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dict, got {type(config)}")

    strategy_type = config.get("type", default_type)

    if strategy_type not in Strategy._strategies:
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            f"Available: {list(Strategy._strategies.keys())}"
        )

    strategy_class = Strategy._strategies[strategy_type]
    return strategy_class(config)


def available_strategies() -> List[str]:
    return list(Strategy._strategies.keys())
