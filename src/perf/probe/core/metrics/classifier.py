
from typing import Dict, Optional
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dict: bool = True
) -> Dict[str, float]:
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
        }

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["num_classes"] = len(np.unique(np.concatenate([y_true, y_pred])))

    return metrics


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None
) -> str:
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=False
    )
    return report


def compute_metrics_detailed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None
) -> Dict:
    metrics = compute_metrics(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    metrics["classification_report"] = report

    return metrics


def compute_inference_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    inference_time: float
) -> Dict[str, float]:
    metrics = compute_metrics(y_true, y_pred)
    num_samples = len(y_true)

    metrics["inference_time_seconds"] = inference_time
    metrics["samples_per_second"] = num_samples / inference_time if inference_time > 0 else 0
    metrics["ms_per_sample"] = inference_time * 1000 / num_samples if num_samples > 0 else 0

    return metrics
