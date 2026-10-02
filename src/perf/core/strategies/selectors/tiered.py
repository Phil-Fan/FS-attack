
from __future__ import annotations

import random
from typing import Dict, List, Optional

from .base import PromptSelector


class TieredSelector(PromptSelector):

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.order = config.get("order", ["high", "mid", "low"])
        self.shuffle = config.get("shuffle", True)
        self.mode = config.get("mode", "priority")
        self._rr_index = 0

    def get_pool(
        self, prompt_tiers: Dict[str, List[str]], tier: Optional[str] = None
    ) -> List[str]:
        if tier is not None:
            prompts = prompt_tiers.get(tier, [])
            pool = list(prompts)
            if self.shuffle:
                random.shuffle(pool)
            return pool

        if self.mode == "round_robin" and self.order:
            for _ in range(len(self.order)):
                tier_name = self.order[self._rr_index % len(self.order)]
                self._rr_index += 1
                prompts = prompt_tiers.get(tier_name, [])
                if prompts:
                    pool = list(prompts)
                    if self.shuffle:
                        random.shuffle(pool)
                    return pool
            return []

        # Otherwise, use priority order
        for tier_name in self.order:
            prompts = prompt_tiers.get(tier_name, [])
            if prompts:
                pool = list(prompts)
                if self.shuffle:
                    random.shuffle(pool)
                return pool
        return []
