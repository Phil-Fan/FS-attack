
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from .base_task import BaseTask, TaskConfig, TaskResult
from core.backends.factory import create_classifier
from core.data.loader import load_sequence_data
from core.metrics.timer import Timer
from core.metrics.classifier import compute_metrics

logger = logging.getLogger(__name__)


class TrainTask(BaseTask):

    def __init__(
        self,
        config: TaskConfig,
        model_type: str = "all",
        epochs: int = 50,
        batch_size: int = 64
    ):
        super().__init__(config)
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.results: Dict[str, Dict] = {}

    def _execute(self):
        logger.info(f"Loading data: {self.config.data_dir}")
        data = load_sequence_data(self.config.data_dir)
        X_train, y_train = data["train"]
        X_val, y_val = data["val"]

        if self.model_type == "all":
            model_types = ["minirocket", "inceptiontime", "lightgbm"]
        else:
            model_types = [self.model_type]

        logger.info(f"Training models: {model_types}")

        for model_name in model_types:
            result = self._train_model(model_name, X_train, y_train, X_val, y_val)
            self.results[model_name] = result

            self.result.metrics[f"{model_name}/val_accuracy"] = result["val_accuracy"]
            self.result.metrics[f"{model_name}/train_time"] = result["train_time"]

        self._summarize_results()

    def _train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict:
        logger.info(f"Training model: {model_name}")

        num_classes = len(np.unique(np.concatenate([y_train, y_val])))

        if model_name in ["minirocket", "inceptiontime"]:
            backend = "tsai"
            model_subtype = model_name
        else:
            backend = "lightgbm"
            model_subtype = "lightgbm"

        classifier_kwargs = {}
        if backend == "tsai":
            classifier_kwargs["model_type"] = model_subtype

        classifier = create_classifier(
            backend=backend,
            num_classes=num_classes,
            **classifier_kwargs
        )

        with Timer() as train_timer:
            classifier.train(X_train, y_train, X_val, y_val)

        y_pred = classifier.predict(X_val)
        metrics = compute_metrics(y_val, y_pred)

        logger.info(f"{model_name} validation accuracy: {metrics['accuracy']:.4f}")

        if self.config.output_dir:
            self._save_model(model_name, classifier, metrics)

        return {
            "classifier": classifier,
            "val_accuracy": metrics["accuracy"],
            "train_time": train_timer.duration,
            "metrics": metrics,
            "predictions": y_pred
        }

    def _save_model(self, model_name: str, classifier, metrics: Dict):
        model_dir = self.config.output_dir / f"model_{model_name}"
        model_dir.mkdir(parents=True, exist_ok=True)

        classifier.save(str(model_dir))

        config = {
            "model_type": model_name,
            "epochs": self.epochs,
            "val_accuracy": metrics["accuracy"],
            "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        }
        with open(model_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Model saved: {model_dir}")

    def _summarize_results(self):
        if not self.results:
            return

        best_model = max(self.results.keys(), key=lambda k: self.results[k]["val_accuracy"])
        best_acc = self.results[best_model]["val_accuracy"]

        self.result.metrics["best_model"] = best_model
        self.result.metrics["best_accuracy"] = best_acc

        logger.info(f"Best model: {best_model} (accuracy: {best_acc:.4f})")

        summary = {
            "model_type": self.model_type,
            "epochs": self.epochs,
            "results": {
                name: {
                    "val_accuracy": r["val_accuracy"],
                    "train_time": r["train_time"]
                }
                for name, r in self.results.items()
            },
            "best_model": best_model,
            "best_accuracy": best_acc
        }

        self._save_metrics("training_summary.json")
