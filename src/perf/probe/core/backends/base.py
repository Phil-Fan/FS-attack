
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
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, model_dir: str) -> None:
        pass

    @abstractmethod
    def load(self, model_dir: str) -> None:
        pass

    def get_feature_importance(self) -> pd.DataFrame:
        raise NotImplementedError(f"{self.backend_name} does not support feature importance")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

        y_pred = self.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        }

        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm

        return metrics
