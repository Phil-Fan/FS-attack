
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .base_task import BaseTask, TaskConfig
from core.backends.factory import create_classifier
from core.data.loader import load_sequence_data
from core.metrics.timer import Timer
from core.metrics.classifier import compute_metrics

logger = logging.getLogger(__name__)


class BenchmarkTask(BaseTask):

    def __init__(
        self,
        config: TaskConfig,
        model_types: Optional[List[str]] = None,
        epochs: int = 30,
        batch_size: int = 64
    ):
        super().__init__(config)
        self.model_types = model_types or ["minirocket", "inceptiontime", "lightgbm"]
        self.epochs = epochs
        self.batch_size = batch_size
        self.all_results: List[Dict] = []

    def _execute(self):
        logger.info(f"Loading data: {self.config.data_dir}")
        data = load_sequence_data(self.config.data_dir)
        X_train, y_train = data["train"]
        X_val, y_val = data["val"]

        num_classes = len(np.unique(np.concatenate([y_train, y_val])))
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Benchmarking models: {self.model_types}")

        for model_name in self.model_types:
            result = self._benchmark_model(
                model_name, X_train, y_train, X_val, y_val, num_classes
            )
            self.all_results.append(result)

        self._summarize_results()

    def _benchmark_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int
    ) -> Dict:
        logger.info(f"Benchmarking model: {model_name}")

        result = {"model": model_name}

        if model_name in ["minirocket", "inceptiontime"]:
            backend = "tsai"
        else:
            backend = "lightgbm"

        classifier = create_classifier(
            backend=backend,
            num_classes=num_classes
        )

        logger.info(f"  Training...")
        with Timer() as train_timer:
            classifier.train(X_train, y_train, X_val, y_val)
        result["train_time"] = train_timer.duration
        logger.info(f"  Training time: {train_timer.duration:.2f} seconds")

        logger.info(f"  Running inference...")
        with Timer() as infer_timer:
            y_pred = classifier.predict(X_val)
        result["inference_time"] = infer_timer.duration
        result["samples_per_second"] = len(y_val) / infer_timer.duration
        logger.info(f"  Inference time: {infer_timer.duration:.4f} seconds")
        logger.info(f"  Throughput: {result['samples_per_second']:.2f} samples/sec")

        metrics = compute_metrics(y_val, y_pred)
        result.update({
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
        })
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Macro: {metrics['f1_macro']:.4f}")

        return result

    def _summarize_results(self):
        if not self.all_results:
            return

        summary = {
            "timestamp": np.datetime64('now').astype(str),
            "model_types": self.model_types,
            "epochs": self.epochs,
            "results": self.all_results
        }

        metric_keys = ["train_time", "inference_time", "samples_per_second",
                       "accuracy", "f1_macro"]

        stats = {}
        for key in metric_keys:
            values = [r[key] for r in self.all_results if key in r]
            if values:
                stats[f"{key}/mean"] = float(np.mean(values))
                stats[f"{key}/std"] = float(np.std(values))
                stats[f"{key}/min"] = float(np.min(values))
                stats[f"{key}/max"] = float(np.max(values))

        summary["statistics"] = stats

        best_by_accuracy = max(self.all_results, key=lambda r: r.get("accuracy", 0))
        best_by_throughput = max(self.all_results, key=lambda r: r.get("samples_per_second", 0))

        summary["best_model/accuracy"] = best_by_accuracy["model"]
        summary["best_model/throughput"] = best_by_throughput["model"]

        self.result.metrics = summary

        logger.info("\n" + "="*50)
        logger.info("Benchmark Summary")
        logger.info("="*50)
        logger.info(f"Best accuracy model: {best_by_accuracy['model']} ({best_by_accuracy.get('accuracy', 0):.4f})")
        logger.info(f"Highest throughput model: {best_by_throughput['model']} ({best_by_throughput.get('samples_per_second', 0):.2f} samples/sec)")

        if self.config.output_dir:
            self._save_results()

    def _save_results(self):
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "benchmark_results.json", 'w', encoding='utf-8') as f:
            json.dump(self.result.metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved: {output_dir / 'benchmark_results.json'}")

        df_results = pd.DataFrame(self.all_results)
        csv_path = output_dir / "benchmark_comparison.csv"
        df_results.to_csv(csv_path, index=False)
        logger.info(f"Comparison table saved: {csv_path}")
