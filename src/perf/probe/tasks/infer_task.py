
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from .base_task import BaseTask, TaskConfig
from core.backends.factory import create_classifier
from core.data.loader import load_sequence_data
from core.metrics.timer import Timer
from core.metrics.classifier import compute_metrics

logger = logging.getLogger(__name__)


class InferTask(BaseTask):

    def __init__(
        self,
        config: TaskConfig,
        model_dir: Path,
        output_predictions: bool = True
    ):
        super().__init__(config)
        self.model_dir = Path(model_dir)
        self.output_predictions = output_predictions
        self.classifier = None

    def _execute(self):
        logger.info(f"Loading model: {self.model_dir}")
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Model configuration does not exist: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            model_config = json.load(f)

        model_type = model_config.get("model_type", "unknown")
        logger.info(f"Model type: {model_type}")

        logger.info(f"Loading data: {self.config.data_dir}")
        data = load_sequence_data(self.config.data_dir)
        X_test, y_test = data["test"]

        logger.info(f"Test set size: {len(y_test)}")

        self._load_model(model_type, model_config)

        self._run_inference(X_test, y_test)

    def _load_model(self, model_type: str, model_config: Dict):
        if model_type in ["minirocket", "inceptiontime"]:
            backend = "tsai"
        else:
            backend = "lightgbm"

        self.classifier = create_classifier(
            backend=backend,
            num_classes=model_config.get("num_classes", 2)
        )

        self.classifier.load(str(self.model_dir))
        logger.info("Model loading complete")

    def _run_inference(self, X_test: np.ndarray, y_test: np.ndarray):
        with Timer() as infer_timer:
            y_pred = self.classifier.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)

        num_samples = len(y_test)
        metrics["inference_time_seconds"] = infer_timer.duration
        metrics["samples_per_second"] = num_samples / infer_timer.duration
        metrics["ms_per_sample"] = infer_timer.duration * 1000 / num_samples

        self.result.metrics = metrics

        logger.info(f"Inference complete:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"  Inference time: {infer_timer.duration:.4f} seconds")
        logger.info(f"  Throughput: {metrics['samples_per_second']:.2f} samples/sec")

        if self.config.output_dir:
            self._save_results(y_pred, metrics)

    def _save_results(self, predictions: np.ndarray, metrics: Dict):
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.output_predictions:
            np.save(output_dir / "predictions.npy", predictions)
            self.result.artifacts["predictions"] = output_dir / "predictions.npy"
            logger.info(f"Predictions saved: {output_dir / 'predictions.npy'}")

        metrics_to_save = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        with open(output_dir / "metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)

        logger.info(f"Metrics saved: {output_dir / 'metrics.json'}")
