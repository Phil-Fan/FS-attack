
from __future__ import annotations

import threading
import time
from typing import Optional
from dataclasses import dataclass


@dataclass
class ITLState:
    kv_usage_class: int
    kv_usage_percent: float
    timestamp: float
    itl_window: Optional[list[float]]


class ITLStateManager:

    _instance: Optional[ITLStateManager] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._state: Optional[ITLState] = None
        self._state_lock = threading.RLock()
        self._initialized = True

    def update(self, kv_usage_class: int, kv_usage_percent: float, itl_window: Optional[list[float]] = None) -> None:
        with self._state_lock:
            self._state = ITLState(
                kv_usage_class=kv_usage_class,
                kv_usage_percent=kv_usage_percent,
                timestamp=time.time(),
                itl_window=itl_window,
            )

    def get_state(self) -> Optional[ITLState]:
        with self._state_lock:
            return self._state

    def is_stale(self, max_age_seconds: float = 5.0) -> bool:
        with self._state_lock:
            if self._state is None:
                return True
            age = time.time() - self._state.timestamp
            return age > max_age_seconds

    def get_kv_usage_percent(self) -> Optional[float]:
        with self._state_lock:
            if self._state is None:
                return None
            return self._state.kv_usage_percent

    def get_memory_gap(self, c_sat: float = 97.5) -> Optional[float]:
        """Return the memory gap Delta_mem = C_sat - U_sys (in percent)."""
        kv_usage = self.get_kv_usage_percent()
        if kv_usage is None:
            return None
        return c_sat - kv_usage

    def get_attack_phase(
        self,
        c_sat: float = 97.5,
        delta_large: float = 47.5,
        delta_small: float = 17.5,
    ) -> str:
        """Map the estimated KV usage to an F&S phase.

        Follows the paper's formulation (Sec.~4 / Appendix Algorithm~1). Let
        Delta_mem = C_sat - U_sys; the controller selects:

          Delta_mem > delta_large                 -> "fill"       (dispatch P_High)
          delta_small < Delta_mem <= delta_large  -> "transition" (dispatch P_Mid)
          0 < Delta_mem <= delta_small            -> "squeeze"    (dispatch P_Low)
          Delta_mem <= 0                          -> "backoff"    (sleep t_wait)

        Defaults use the paper's saturation threshold C_sat = 97.5% (the
        "97.5-100% High band"), giving the Fill/Transition boundary at
        C_sat - delta_large = 50% and the Transition/Squeeze boundary at
        C_sat - delta_small = 80%.
        """
        kv_usage = self.get_kv_usage_percent()
        if kv_usage is None:
            return "unknown"

        delta_mem = c_sat - kv_usage
        if delta_mem <= 0.0:
            return "backoff"
        if delta_mem <= delta_small:
            return "squeeze"
        if delta_mem <= delta_large:
            return "transition"
        return "fill"

    def reset(self) -> None:
        with self._state_lock:
            self._state = None

    def get_debug_info(self) -> dict:
        with self._state_lock:
            if self._state is None:
                return {"status": "no_state"}
            return {
                "status": "active",
                "kv_usage_class": self._state.kv_usage_class,
                "kv_usage_percent": self._state.kv_usage_percent,
                "timestamp": self._state.timestamp,
                "age_seconds": time.time() - self._state.timestamp,
                "attack_phase": self.get_attack_phase(),
                "itl_window_size": len(self._state.itl_window) if self._state.itl_window else 0,
            }


def get_state_manager() -> ITLStateManager:
    return ITLStateManager()


__all__ = [
    "ITLStateManager",
    "ITLState",
    "get_state_manager",
]
