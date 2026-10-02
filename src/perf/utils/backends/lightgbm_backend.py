
import joblib
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import logging

try:
    import lightgbm as lgb
except (ImportError, OSError):
    # OSError covers native-library load failures (e.g. a missing libomp).
    lgb = None
    logging.warning("LightGBM not installed. Install with: pip install lightgbm")

from .base import BaseClassifier

logger = logging.getLogger(__name__)


class LightGBMBackend(BaseClassifier):

    @property
    def backend_name(self) -> str:
        return "lightgbm"

    def __init__(self, config: Optional[Dict] = None, num_classes: int = 10):
        if lgb is None:
            raise ImportError("LightGBM is not installed. Run: pip install lightgbm")

        super().__init__(config or {}, num_classes)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        return self.model.predict_proba(X)

    def load(self, model_dir: str) -> None:
        model_dir = Path(model_dir)

        model_path = model_dir / 'itl_kv_model.pkl'
        self.model = joblib.load(model_path)

        config_path = model_dir / 'itl_kv_config.json'
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            self.config = config_data['config']
            self.num_classes = config_data['num_classes']

        logger.info(f"Model loaded from {model_dir}")

    def save(self, model_dir: str) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / 'itl_kv_model.pkl'
        joblib.dump(self.model, model_path)

        config_path = model_dir / 'itl_kv_config.json'
        config_data = {
            'backend': self.backend_name,
            'config': self.config,
            'num_classes': self.num_classes,
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Model saved to {model_dir}")
