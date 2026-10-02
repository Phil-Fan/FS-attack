
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, f1_score

try:
    from .backends import create_classifier
except ImportError:
    from backends import create_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ITLSequenceClassifier:

    def __init__(
        self,
        num_classes: int = 10,
        backend: str = "lightgbm",
        use_sequence_features: bool = True,
        tsai_config: Dict = None
    ):
        self.num_classes = num_classes
        self.backend = backend
        self.use_sequence_features = use_sequence_features

        if backend == "tsai":
            self._classifier = create_classifier(
                backend=backend,
                config=tsai_config,
                num_classes=num_classes
            )
            self.use_sequence_features = False
        else:
            self._classifier = create_classifier(
                backend=backend,
                config=None,
                num_classes=num_classes
            )

        self.feature_cols = None
        self.train_acc = None

    @property
    def model(self):
        return self._classifier.model

    def _extract_sequence_features(self, itl_sequences: List[np.ndarray]) -> np.ndarray:
        features = []

        for seq in itl_sequences:
            seq = np.array(seq)
            feat = {
                'mean': np.mean(seq),
                'std': np.std(seq),
                'min': np.min(seq),
                'max': np.max(seq),
                'median': np.median(seq),
                'q25': np.percentile(seq, 25),
                'q75': np.percentile(seq, 75),
                'q90': np.percentile(seq, 90),
                'length': len(seq),
                'cv': np.std(seq) / np.mean(seq) if np.mean(seq) > 0 else 0,
            }
            features.append(feat)

        return pd.DataFrame(features)

    def prepare_data(
        self,
        data_df: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        itl_sequences = [
            np.array([float(x) for x in seq_str.split(',')])
            for seq_str in data_df['itl'].values
        ]

        if self.use_sequence_features:
            features_df = self._extract_sequence_features(itl_sequences)
            self.feature_cols = features_df.columns.tolist()
            X = features_df.values
        else:
            from .backends.tsai_backend import prepare_sequence_data
            X = np.array(itl_sequences)
            self.feature_cols = None

        y = data_df['kv_label'].values if 'kv_label' in data_df.columns else None

        if self.feature_cols is not None:
            logger.info(f"Number of features: {X.shape[1]}")
        logger.info(f"Number of samples: {X.shape[0]}")

        return X, y

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame
    ):
        if self.backend == "tsai":
            from .backends.tsai_backend import prepare_sequence_data
            X_train, y_train, X_eval, y_eval = prepare_sequence_data(train_df, eval_df)
            logger.info(f"Train set (tsai): {X_train.shape}")
            logger.info(f"Eval set (tsai): {X_eval.shape}")
        else:
            X_train, y_train = self.prepare_data(train_df)
            X_eval, y_eval = self.prepare_data(eval_df)
            logger.info(f"Train set: {X_train.shape[0]} samples")
            logger.info(f"Eval set: {X_eval.shape[0]} samples")

        self._classifier.train(X_train, y_train, X_eval, y_eval)
        self.train_acc = self._classifier.train_acc

        if self.feature_cols is not None:
            X_eval_for_pred = pd.DataFrame(X_eval, columns=self.feature_cols)
        else:
            X_eval_for_pred = X_eval
        y_pred = self._classifier.predict(X_eval_for_pred)
        eval_acc = accuracy_score(y_eval, y_pred)
        f1_macro = f1_score(y_eval, y_pred, average='macro')

        logger.info(f"Train set Accuracy: {self.train_acc:.4f}")
        logger.info(f"Eval set Accuracy: {eval_acc:.4f}, F1-macro: {f1_macro:.4f}")

        return {
            'train_acc': self.train_acc,
            'eval_acc': eval_acc,
            'f1_macro': f1_macro
        }

    def predict(self, data_df: pd.DataFrame) -> np.ndarray:
        if self.backend == "tsai":
            from .backends.tsai_backend import parse_itl_sequence
            import numpy as np

            itl_sequences = [
                parse_itl_sequence(s)
                for s in data_df['itl'].values
            ]

            max_len = max(len(s) for s in itl_sequences) if itl_sequences else 100
            X = np.array([
                np.pad(seq, (0, max_len - len(seq)), mode='constant', constant_values=0)
                if len(seq) < max_len else seq[:max_len]
                for seq in itl_sequences
            ])
            X = X.reshape(X.shape[0], 1, X.shape[1])
        else:
            X, _ = self.prepare_data(data_df)
            # Convert the numpy array back to a DataFrame to preserve feature names and avoid sklearn warnings
            if self.feature_cols is not None:
                X = pd.DataFrame(X, columns=self.feature_cols)

        return self._classifier.predict(X)

    def save(self, model_dir: str):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        self._classifier.save(str(model_dir))

        import json
        meta = {
            'num_classes': self.num_classes,
            'backend': self.backend,
            'feature_cols': self.feature_cols,
            'use_sequence_features': self.use_sequence_features
        }
        with open(model_dir / 'sequence_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Model saved to: {model_dir}")

    def load(self, model_dir: str):
        model_dir = Path(model_dir)

        self._classifier.load(str(model_dir))

        import json
        with open(model_dir / 'sequence_meta.json', 'r') as f:
            meta = json.load(f)

        self.num_classes = meta['num_classes']
        self.backend = meta['backend']
        self.feature_cols = meta['feature_cols']
        self.use_sequence_features = meta['use_sequence_features']

        logger.info(f"Model loaded from {model_dir}")
