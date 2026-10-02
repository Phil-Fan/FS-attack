
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


def get_lightgbm_config(num_classes: int = 10) -> Dict:
    return {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": num_classes,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
        "device_type": "cpu",  # Can be changed to "gpu"
        "learning_rate": 0.05,
        "n_estimators": 300,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "num_classes": num_classes,
    }


class LightGBMBackend(BaseClassifier):

    @property
    def backend_name(self) -> str:
        return "lightgbm"

    def __init__(self, config: Optional[Dict] = None, num_classes: int = 10):
        if lgb is None:
            raise ImportError("LightGBM is not installed. Run: pip install lightgbm")

        if config is None:
            config = get_lightgbm_config(num_classes=num_classes)
        else:
            # Ensure num_class is set correctly
            config['num_class'] = config.get('num_classes', num_classes)
            config['num_classes'] = config.get('num_classes', num_classes)

        super().__init__(config, num_classes)

        self.train_acc: Optional[float] = None

    def train(
        self,
        X_train,
        y_train: np.ndarray,
        X_val=None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        logger.info(f"Training {self.backend_name} classifier (num_classes={self.num_classes})...")

        self.model = lgb.LGBMClassifier(**self.config)

        # Ensure X_train is 2D (LightGBM requires 2D input)
        feature_names = None
        if isinstance(X_train, np.ndarray):
            if X_train.ndim == 3:
                X_train = X_train.squeeze(1)
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            X_train = pd.DataFrame(X_train, columns=feature_names)

        eval_set = None
        if X_val is not None and y_val is not None:
            # Ensure X_val is 2D
            if isinstance(X_val, np.ndarray):
                if X_val.ndim == 3:
                    X_val = X_val.squeeze(1)
                if feature_names is None:
                    feature_names = [f'feature_{i}' for i in range(X_val.shape[1])]
                X_val = pd.DataFrame(X_val, columns=feature_names)
            eval_set = [(X_val, y_val)]
            logger.info("Using validation set")

        self.model.fit(X_train, y_train, eval_set=eval_set)

        self.train_acc = (self.model.predict(X_train) == y_train).mean()

        logger.info(f"Training completed. Best iteration: {self.model.best_iteration_}")
        logger.info(f"Train accuracy: {self.train_acc:.4f}")

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        # Ensure X is 2D
        if isinstance(X, np.ndarray) and X.ndim == 3:
            X = X.squeeze(1)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        # Ensure X is 2D
        if isinstance(X, np.ndarray) and X.ndim == 3:
            X = X.squeeze(1)
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        importance = self.model.feature_importances_

        return pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(importance))],
            'importance': importance
        }).sort_values('importance', ascending=False)

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
