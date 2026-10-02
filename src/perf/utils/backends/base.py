
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd


class BaseClassifier(ABC):

    def __init__(self, config: Dict, num_classes: int):
        self.config = config
        self.num_classes = num_classes
        self.model = None

    @property
    @abstractmethod
    def backend_name(self) -> str:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, model_dir: str) -> None:
        pass

    @abstractmethod
    def load(self, model_dir: str) -> None:
        pass

    def get_feature_importance(self) -> pd.DataFrame:
        raise NotImplementedError(f"{self.backend_name} does not support feature importance")
