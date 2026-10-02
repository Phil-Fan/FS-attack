
from .base import IntervalScheduler
from .fixed import FixedScheduler
from .poisson import PoissonScheduler
from .burst import BurstScheduler
from .wave import WaveScheduler
from .trace import TraceScheduler

_AVAILABLE_SCHEDULERS: dict[str, type[IntervalScheduler]] = {
    "fixed": FixedScheduler,
    "poisson": PoissonScheduler,
    "burst": BurstScheduler,
    "wave": WaveScheduler,
    "trace": TraceScheduler,
}


def get_interval_scheduler(config: dict) -> IntervalScheduler:
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dict, got {type(config)}")

    scheduler_type = config.get("type", "fixed")

    if scheduler_type not in _AVAILABLE_SCHEDULERS:
        raise ValueError(
            f"Unknown interval scheduler type: {scheduler_type}. "
            f"Available: {list(_AVAILABLE_SCHEDULERS.keys())}"
        )

    scheduler_class = _AVAILABLE_SCHEDULERS[scheduler_type]
    return scheduler_class(config)


def available_schedulers() -> list[str]:
    return list(_AVAILABLE_SCHEDULERS.keys())


__all__ = [
    "IntervalScheduler",
    "FixedScheduler",
    "PoissonScheduler",
    "BurstScheduler",
    "WaveScheduler",
    "TraceScheduler",
    "get_interval_scheduler",
    "available_schedulers",
]
