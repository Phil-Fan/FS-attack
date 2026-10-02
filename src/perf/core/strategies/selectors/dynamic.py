
from __future__ import annotations

import random
import logging
from typing import Dict, List, Optional

from .base import PromptSelector

logger = logging.getLogger("attack")


class DynamicPromptSelector(PromptSelector):

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)

        # Paper-aligned (Sec. 4 / Appendix Algorithm 1) controller parameters.
        # The phase is decided from the memory gap Delta_mem = C_sat - U_sys.
        # Legacy absolute thresholds (fill_threshold / squeeze_threshold) are
        # still accepted and converted into the equivalent margins so existing
        # configs keep working.
        self.c_sat = float(config.get("c_sat", config.get("C_sat", 97.5)))

        if "delta_large" in config:
            self.delta_large = float(config["delta_large"])
        elif "fill_threshold" in config:
            self.delta_large = self.c_sat - float(config["fill_threshold"])
        else:
            self.delta_large = 47.5  # C_sat - delta_large = 50% Fill/Transition boundary

        if "delta_small" in config:
            self.delta_small = float(config["delta_small"])
        elif "squeeze_threshold" in config:
            self.delta_small = self.c_sat - float(config["squeeze_threshold"])
        else:
            self.delta_small = 17.5  # C_sat - delta_small = 80% Transition/Squeeze boundary

        # Back-off: pause injection for t_wait when Delta_mem <= 0.
        # `backoff_enabled=False` reproduces the paper's "no adaptive back-off"
        # ablation (the tier logic stays, but the controller never sleeps).
        self.t_wait = float(config.get("t_wait", 1.0))
        self.backoff_enabled = bool(config.get("backoff_enabled", True))
        self.max_state_age = config.get("max_state_age", 5.0)

        default_mapping = {
            "fill": ["high", "mid"],
            "transition": ["mid"],
            "squeeze": ["low", "mid"],
            "backoff": ["low"],
        }
        self.tier_mapping: dict[str, list[str]] = config.get("tier_mapping", default_mapping)

        self._phase_counts = {
            "fill": 0,
            "transition": 0,
            "squeeze": 0,
            "backoff": 0,
            "unknown": 0,
        }

        self._last_selected_tier: Optional[str] = None

        # Lazily import state_manager to avoid circular dependency
        self._state_manager = None

    def _get_state_manager(self):
        if self._state_manager is None:
            from ...state_manager import get_state_manager
            self._state_manager = get_state_manager()
        return self._state_manager

    def _get_current_phase(self) -> str:
        state_manager = self._get_state_manager()

        if state_manager.is_stale(self.max_state_age):
            logger.debug("ITL state is stale, defaulting to fill phase")
            return "unknown"

        phase = state_manager.get_attack_phase(
            c_sat=self.c_sat,
            delta_large=self.delta_large,
            delta_small=self.delta_small,
        )
        return phase

    def get_backoff_wait(self) -> float:
        """Return the Back-off sleep duration t_wait.

        Per Algorithm 1, when Delta_mem = C_sat - U_sys <= 0 the system is at or
        beyond saturation and the attacker pauses injection for t_wait instead
        of dispatching another request, avoiding self-preemption. Returns 0.0
        when the controller is not in the Back-off regime (or when back-off is
        disabled for ablation).
        """
        if not self.backoff_enabled:
            return 0.0
        if self._get_current_phase() == "backoff":
            return self.t_wait
        return 0.0

    def get_pool(
        self, prompt_tiers: Dict[str, List[str]], tier: Optional[str] = None
    ) -> List[str]:
        if tier is not None:
            self._last_selected_tier = tier
            return list(prompt_tiers.get(tier, []))

        phase = self._get_current_phase()
        self._phase_counts[phase] += 1

        tier_order = self.tier_mapping.get(phase, ["high", "mid", "low"])

        logger.debug(
            f"[DynamicSelector] Phase: {phase}, Tier order: {tier_order}"
        )

        for tier_name in tier_order:
            prompts = prompt_tiers.get(tier_name, [])
            if prompts:
                self._last_selected_tier = tier_name
                # Return a copy to avoid modifying the original data
                return list(prompts)

        for tier_name in prompt_tiers:
            if prompt_tiers[tier_name]:
                logger.warning(
                    f"[DynamicSelector] Phase {phase} tiers empty, fallback to '{tier_name}'"
                )
                self._last_selected_tier = tier_name
                return list(prompt_tiers[tier_name])

        logger.warning("[DynamicSelector] No prompts available")
        self._last_selected_tier = None
        return []

    def get_last_tier(self) -> Optional[str]:
        return self._last_selected_tier

    def get_phase_stats(self) -> Dict[str, int]:
        return self._phase_counts.copy()

    def reset_stats(self) -> None:
        for phase in self._phase_counts:
            self._phase_counts[phase] = 0
