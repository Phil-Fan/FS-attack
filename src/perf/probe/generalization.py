#!/usr/bin/env python3

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from core.backends.factory import create_classifier
from core.data.loader import load_sequence_data

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-dataset generalization test: rows are training datasets, columns are test datasets."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory containing the per-dataset subfolders",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Names of the dataset subfolders to compare; the order determines the matrix order",
    )
    parser.add_argument(
        "--model",
        choices=["lightgbm", "minirocket", "inceptiontime"],
        default="lightgbm",
        help="Model type, defaults to lightgbm",
    )
    parser.add_argument(
        "--model-dir-name",
        default=None,
        help="Custom model directory name, defaults to training/model_<model>",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory, defaults to <root>/generalization_<model>",
    )
    parser.add_argument(
        "--no-confusion",
        action="store_true",
        help="Do not generate the confusion matrix grid plot",
    )
    parser.add_argument(
        "--min-class-support",
        type=int,
        default=30,
        help="Minimum number of train/test samples for a reliable class, defaults to 30",
    )
    return parser.parse_args()


def resolve_dataset(root: Path, value: str) -> Tuple[str, Path]:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    if not (path / "data" / "train.csv").is_file():
        raise FileNotFoundError(f"Missing train.csv: {path / 'data'}")
    if not (path / "data" / "eval.csv").is_file():
        raise FileNotFoundError(f"Missing eval.csv: {path / 'data'}")
    return path.name, path


def get_model_dir(dataset_dir: Path, model: str, custom_name: str | None) -> Path:
    model_name = custom_name or f"model_{model}"
    model_dir = dataset_dir / "training" / model_name
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Model configuration does not exist: {model_dir / 'config.json'}")
    return model_dir


def load_model(model_dir: Path, model: str):
    with open(model_dir / "config.json", "r", encoding="utf-8") as file:
        model_config = json.load(file)

    backend = "tsai" if model in {"minirocket", "inceptiontime"} else "lightgbm"
    classifier_kwargs = {"model_type": model} if backend == "tsai" else {}
    classifier = create_classifier(
        backend=backend,
        num_classes=int(model_config.get("num_classes", 2)),
        **classifier_kwargs,
    )
    classifier.load(str(model_dir))
    return classifier


def evaluate(
    datasets: List[Tuple[str, Path]],
    model: str,
    model_dir_name: str | None,
    min_class_support: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[Tuple[str, str], np.ndarray], int, pd.DataFrame]:
    test_data = {}
    train_labels = {}
    train_counts = {}
    test_counts = {}
    support_rows = []
    max_class = 0
    for name, dataset_dir in datasets:
        data = load_sequence_data(dataset_dir / "data")
        X_test, y_test = data["test"]
        test_data[name] = (X_test, y_test)
        train_df = pd.read_csv(dataset_dir / "data" / "train.csv")
        eval_df = pd.read_csv(dataset_dir / "data" / "eval.csv")
        min_label = min(
            int(train_df["kv_label"].min()),
            int(eval_df["kv_label"].min()),
        )
        normalized_train_labels = train_df["kv_label"].astype(int) - min_label
        train_support = sorted(normalized_train_labels.unique().tolist())
        test_support = sorted(np.unique(y_test).astype(int).tolist())
        train_labels[name] = set(train_support)
        train_counts[name] = normalized_train_labels.value_counts().to_dict()
        test_counts[name] = pd.Series(y_test).value_counts().to_dict()
        support_rows.append({
            "dataset": name,
            "train_classes": ",".join(map(str, train_support)),
            "test_classes": ",".join(map(str, test_support)),
            "num_train_classes": len(train_support),
            "num_test_classes": len(test_support),
            "min_class_support": min_class_support,
            "train_class_counts": json.dumps(
                normalized_train_labels.value_counts().sort_index().to_dict()
            ),
            "test_class_counts": json.dumps(
                pd.Series(y_test).value_counts().sort_index().to_dict()
            ),
        })
        if len(y_test):
            max_class = max(max_class, int(np.max(y_test)) + 1)

    names = [name for name, _ in datasets]
    metrics = {
        metric: pd.DataFrame(index=names, columns=names, dtype=float)
        for metric in (
            "accuracy",
            "f1_macro",
            "coverage",
            "supported_accuracy",
            "supported_f1_macro",
            "reliable_coverage",
            "reliable_accuracy",
            "reliable_f1_macro",
        )
    }
    confusion_matrices: Dict[Tuple[str, str], np.ndarray] = {}

    for train_name, train_dir in datasets:
        model_dir = get_model_dir(train_dir, model, model_dir_name)
        logger.info("Loading model: %s", model_dir)
        classifier = load_model(model_dir, model)
        max_class = max(max_class, int(classifier.num_classes))

        for test_name, (X_test, y_test) in test_data.items():
            y_pred = np.asarray(classifier.predict(X_test)).astype(int)
            test_classes = sorted(np.unique(y_test).astype(int).tolist())
            seen_mask = np.isin(y_test, list(train_labels[train_name]))
            common_classes = sorted(
                set(test_classes).intersection(train_labels[train_name])
            )
            reliable_classes = sorted(
                cls for cls in common_classes
                if train_counts[train_name].get(cls, 0) >= min_class_support
                and test_counts[test_name].get(cls, 0) >= min_class_support
            )

            metrics["accuracy"].loc[train_name, test_name] = float(
                np.mean(y_pred == y_test)
            )
            metrics["f1_macro"].loc[train_name, test_name] = float(
                f1_score(
                    y_test,
                    y_pred,
                    labels=test_classes,
                    average="macro",
                    zero_division=0,
                )
            )
            metrics["coverage"].loc[train_name, test_name] = float(
                np.mean(seen_mask)
            )
            if seen_mask.any():
                metrics["supported_accuracy"].loc[train_name, test_name] = float(
                    np.mean(y_pred[seen_mask] == y_test[seen_mask])
                )
                metrics["supported_f1_macro"].loc[train_name, test_name] = float(
                    f1_score(
                        y_test[seen_mask],
                        y_pred[seen_mask],
                        labels=common_classes,
                        average="macro",
                        zero_division=0,
                    )
                )
            else:
                metrics["supported_accuracy"].loc[train_name, test_name] = np.nan
                metrics["supported_f1_macro"].loc[train_name, test_name] = np.nan
            reliable_mask = np.isin(y_test, reliable_classes)
            metrics["reliable_coverage"].loc[train_name, test_name] = float(
                np.mean(reliable_mask)
            )
            if reliable_mask.any():
                metrics["reliable_accuracy"].loc[train_name, test_name] = float(
                    np.mean(y_pred[reliable_mask] == y_test[reliable_mask])
                )
                metrics["reliable_f1_macro"].loc[train_name, test_name] = float(
                    f1_score(
                        y_test[reliable_mask],
                        y_pred[reliable_mask],
                        labels=reliable_classes,
                        average="macro",
                        zero_division=0,
                    )
                )
            else:
                metrics["reliable_accuracy"].loc[train_name, test_name] = np.nan
                metrics["reliable_f1_macro"].loc[train_name, test_name] = np.nan
            if len(y_test):
                max_class = max(max_class, int(np.max(y_pred)) + 1)
            confusion_matrices[(train_name, test_name)] = (
                y_test,
                y_pred,
            )

    n_classes = max_class
    padded_matrices = {}
    for key, (y_true, y_pred) in confusion_matrices.items():
        padded_matrices[key] = confusion_matrix(
            y_true, y_pred, labels=list(range(n_classes))
        )
    return metrics, padded_matrices, n_classes, pd.DataFrame(support_rows)


def save_long_results(
    metrics: Dict[str, pd.DataFrame], output_path: Path
) -> None:
    rows = []
    accuracy = metrics["accuracy"]
    for train_name in accuracy.index:
        for test_name in accuracy.columns:
            rows.append({
                "train_dataset": train_name,
                "test_dataset": test_name,
                **{
                    name: matrix.loc[train_name, test_name]
                    for name, matrix in metrics.items()
                },
            })
    pd.DataFrame(rows).to_csv(output_path, index=False, float_format="%.6f")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    root = args.root.resolve()
    output = (args.output or root / f"generalization_{args.model}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))

    datasets = [resolve_dataset(root, value) for value in args.datasets]
    names = [name for name, _ in datasets]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate dataset names; please pass different subfolders")

    if args.min_class_support < 1:
        raise ValueError("--min-class-support must be greater than 0")
    metrics, confusion_matrices, n_classes, support_df = evaluate(
        datasets, args.model, args.model_dir_name, args.min_class_support
    )

    for name, matrix in metrics.items():
        matrix.to_csv(output / f"{name}_matrix.csv", float_format="%.6f")
    support_df.to_csv(output / "label_support.csv", index=False)
    save_long_results(metrics, output / "generalization_results.csv")

    from core.visualizer import (
        plot_generalization_heatmap,
        plot_confusion_matrices_grid,
        save_generalization_report,
    )

    plot_generalization_heatmap(
        metrics["accuracy"],
        output / "accuracy_matrix.png",
        title=f"{args.model} Generalization",
    )
    plot_generalization_heatmap(
        metrics["f1_macro"],
        output / "f1_macro_matrix.png",
        title=f"{args.model} Generalization",
    )
    plot_generalization_heatmap(
        metrics["coverage"],
        output / "coverage_matrix.png",
        title=f"{args.model} Label Coverage",
    )
    plot_generalization_heatmap(
        metrics["supported_accuracy"],
        output / "supported_accuracy_matrix.png",
        title=f"{args.model} Supported Accuracy",
    )
    plot_generalization_heatmap(
        metrics["supported_f1_macro"],
        output / "supported_f1_macro_matrix.png",
        title=f"{args.model} Supported F1",
    )
    plot_generalization_heatmap(
        metrics["reliable_coverage"],
        output / "reliable_coverage_matrix.png",
        title=f"{args.model} Reliable Label Coverage",
    )
    plot_generalization_heatmap(
        metrics["reliable_accuracy"],
        output / "reliable_accuracy_matrix.png",
        title=f"{args.model} Reliable Accuracy",
    )
    plot_generalization_heatmap(
        metrics["reliable_f1_macro"],
        output / "reliable_f1_macro_matrix.png",
        title=f"{args.model} Reliable F1",
    )
    save_generalization_report(
        metrics["supported_accuracy"], output / "generalization_report.txt"
    )
    if not args.no_confusion:
        plot_confusion_matrices_grid(
            confusion_matrices,
            names,
            output / "confusion_matrices.png",
            n_classes=n_classes,
        )

    logger.info("Generalization test complete, results directory: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
