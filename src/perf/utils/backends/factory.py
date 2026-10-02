
from typing import Dict, Optional, Type
import logging

from .base import BaseClassifier
from .lightgbm_backend import LightGBMBackend
from .tsai_backend import TsaiBackend

logger = logging.getLogger(__name__)


_BACKEND_CLASSES: Dict[str, Type[BaseClassifier]] = {
    "lightgbm": LightGBMBackend,
    "tsai": TsaiBackend,
}


def get_available_backends() -> list:
    available = []
    for name, cls in _BACKEND_CLASSES.items():
        try:
            if name == "lightgbm":
                import lightgbm
                available.append(name)
            elif name == "tsai":
                try:
                    import tsai
                    available.append(name)
                except ImportError:
                    pass
        except ImportError:
            pass
    return available


def create_classifier(
    backend: str = "lightgbm",
    config: Optional[Dict] = None,
    num_classes: int = 10,
    **kwargs
) -> BaseClassifier:
    backend = backend.lower()

    if backend not in _BACKEND_CLASSES:
        available = ", ".join(_BACKEND_CLASSES.keys())
        raise ValueError(
            f"Unknown backend: {backend}. Available backends: {available}"
        )

    logger.info(f"Creating {backend} classifier with {num_classes} classes")

    classifier_cls = _BACKEND_CLASSES[backend]
    return classifier_cls(config=config, num_classes=num_classes, **kwargs)
