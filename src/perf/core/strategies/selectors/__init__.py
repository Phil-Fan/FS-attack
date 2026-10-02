
from .base import PromptSelector
from .tiered import TieredSelector
from .file_list import FileListSelector
from .weighted_mix import WeightedMixSelector
from .dynamic import DynamicPromptSelector

_AVAILABLE_SELECTORS: dict[str, type[PromptSelector]] = {
    "tiered": TieredSelector,
    "file_list": FileListSelector,
    "weighted_mix": WeightedMixSelector,
    "dynamic": DynamicPromptSelector,
}


def get_prompt_selector(config: dict) -> PromptSelector:
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dict, got {type(config)}")

    selector_type = config.get("type", "tiered")

    if selector_type not in _AVAILABLE_SELECTORS:
        raise ValueError(
            f"Unknown prompt selector type: {selector_type}. "
            f"Available: {list(_AVAILABLE_SELECTORS.keys())}"
        )

    selector_class = _AVAILABLE_SELECTORS[selector_type]
    return selector_class(config)


def available_selectors() -> list[str]:
    return list(_AVAILABLE_SELECTORS.keys())


__all__ = [
    "PromptSelector",
    "TieredSelector",
    "FileListSelector",
    "WeightedMixSelector",
    "get_prompt_selector",
    "available_selectors",
]
