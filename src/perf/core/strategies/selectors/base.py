
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class PromptSelector(ABC):

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def get_pool(
        self, prompt_tiers: Dict[str, List[str]], tier: Optional[str] = None
    ) -> List[str]:
        raise NotImplementedError

    def select_one(
        self, prompt_tiers: Dict[str, List[str]], tier: Optional[str] = None
    ) -> str:
        import random

        pool = self.get_pool(prompt_tiers, tier=tier)
        if not pool:
            raise ValueError("No prompts available in the selected pool")
        return random.choice(pool)
