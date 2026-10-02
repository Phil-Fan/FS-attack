
import numpy as np
import pandas as pd
from typing import List, Union
from pathlib import Path
import logging
import json

try:
    from .backends import create_classifier
except ImportError:
    from backends import create_classifier

logger = logging.getLogger(__name__)


class ITLPredictor:

    _instance_cache = {}

    @classmethod
    def get_cached(cls, model_dir: str) -> 'ITLPredictor':
        model_dir = str(Path(model_dir).resolve())
        if model_dir not in cls._instance_cache:
            cls._instance_cache[model_dir] = cls(model_dir)
        return cls._instance_cache[model_dir]

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self._classifier = create_classifier(backend="lightgbm", config=None, num_classes=10)
        self._load_model()
        self.kv_bins = self._load_kv_bins()
        self.num_classes = 10

    def _load_model(self):
        self._classifier.load(str(self.model_dir))
        logger.info(f"Model loaded from {self.model_dir}")

    def _load_kv_bins(self) -> np.ndarray:
        return np.linspace(0, 100, 11)

    def _extract_features(self, itl_array: np.ndarray) -> dict:
        return {
            'mean': np.mean(itl_array),
            'std': np.std(itl_array),
            'min': np.min(itl_array),
            'max': np.max(itl_array),
            'median': np.median(itl_array),
            'q25': np.percentile(itl_array, 25),
            'q75': np.percentile(itl_array, 75),
            'q90': np.percentile(itl_array, 90),
            'length': len(itl_array),
            'cv': np.std(itl_array) / np.mean(itl_array) if np.mean(itl_array) > 0 else 0,
        }

    def predict_class(self, itl_array: Union[List[float], np.ndarray]) -> int:
        if isinstance(itl_array, np.ndarray):
            itl_array = itl_array.tolist()

        features = self._extract_features(np.array(itl_array))
        feature_names = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'q90', 'length', 'cv']
        X = pd.DataFrame([features], columns=feature_names)
        return int(self._classifier.predict(X)[0])

    def predict(self, itl_array: Union[List[float], np.ndarray, str]) -> str:
        if isinstance(itl_array, str):
            itl_array = [float(x.strip()) for x in itl_array.split(',')]
        elif isinstance(itl_array, np.ndarray):
            itl_array = itl_array.tolist()

        pred_class = self.predict_class(itl_array)
        print(f"ITL array: {itl_array}")
        print(f"Predicted class: {pred_class}")
        return self._class_to_range(pred_class)

    def _class_to_range(self, pred_class: int) -> str:
        if pred_class < len(self.kv_bins) - 1:
            kv_min = self.kv_bins[pred_class]
            kv_max = self.kv_bins[pred_class + 1]
            return f"{kv_min:.1f}-{kv_max:.1f}"
        return f"{self.kv_bins[-2]:.1f}+"


def predict_kv_range(itl_array: Union[List[float], np.ndarray, str]) -> str:
    predictor = ITLPredictor.get_cached(".")
    return predictor.predict(itl_array)


def predict_kv_class(itl_array: Union[List[float], np.ndarray]) -> int:
    predictor = ITLPredictor.get_cached(".")
    return predictor.predict_class(itl_array)
