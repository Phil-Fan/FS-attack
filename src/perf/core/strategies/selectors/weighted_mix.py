
from __future__ import annotations

import random
from typing import Dict, List, Optional

from .base import PromptSelector


class WeightedMixSelector(PromptSelector):

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.weights = config.get("weights", {})
        self.default_weight = config.get("default_weight", 0.0)

        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        else:
            self.weights = {}

    def get_pool(
        self, prompt_tiers: Dict[str, List[str]], tier: Optional[str] = None
    ) -> List[str]:
        if tier is not None:
            return list(prompt_tiers.get(tier, []))

        tier_weights: list[tuple[str, float]] = []
        available_tiers = [t for t in prompt_tiers if prompt_tiers.get(t)]

        for tier_name in available_tiers:
            weight = self.weights.get(tier_name, self.default_weight)
            if weight > 0:
                tier_weights.append((tier_name, weight))

        if not tier_weights:
            # Fallback to first available tier
            if available_tiers:
                return list(prompt_tiers[available_tiers[0]])
            return []

        tiers, weights = zip(*tier_weights)
        selected_tier = random.choices(tiers, weights=weights, k=1)[0]

        return list(prompt_tiers.get(selected_tier, []))
