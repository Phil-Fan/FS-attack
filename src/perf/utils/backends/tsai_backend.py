
import logging
from typing import Dict, Optional
import numpy as np

try:
    from tsai.all import *
    TSAI_AVAILABLE = True
except ImportError:
    TSAI_AVAILABLE = False
    logging.warning("Tsai not installed. Install with: pip install tsai")

from .base import BaseClassifier

logger = logging.getLogger(__name__)


def get_tsai_config(num_classes: int = 10) -> Dict:
    return {
        "arch": "LSTM",
        "hidden_size": 128,
        "n_layers": 3,
        "dropout": 0.2,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_classes": num_classes,
    }


class TsaiBackend(BaseClassifier):

    @property
    def backend_name(self) -> str:
        return "tsai"

    def __init__(self, config: Optional[Dict] = None, num_classes: int = 10):
        if not TSAI_AVAILABLE:
            raise ImportError(
                "Tsai is not installed. Install with: pip install tsai[pytorch]\n"
                "See: https://github.com/timeseriesAI/tsai"
            )

        if config is None:
            config = get_tsai_config(num_classes=num_classes)

        super().__init__(config, num_classes)
        self.model = None
        self.learner = None
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.backend_name} backend is not yet implemented.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.backend_name} backend is not yet implemented.")

    def load(self, model_dir: str) -> None:
        raise NotImplementedError(f"{self.backend_name} backend is not yet implemented.")
