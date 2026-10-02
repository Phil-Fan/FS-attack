
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib/seaborn not installed")


def load_metrics(model_dir: Path) -> Dict:
    config_path = model_dir / "config.json"
    metrics_path = model_dir / "metrics.json"

    metrics = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            metrics.update(json.load(f))
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics.update(json.load(f))

    return metrics


def plot_confusion_matrix_comparison(
    results_dir: Path,
    output_path: Path
):
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available")
        return

    model_dirs = list(results_dir.glob("*/training/model_*"))
    if not model_dirs:
        logger.warning("No model results found")
        return

    datasets = {}
    for model_dir in model_dirs:
        dataset_name = model_dir.parent.parent.name
        model_name = model_dir.name.replace("model_", "")
        key = f"{dataset_name}/{model_name}"
        datasets[key] = model_dir

    n_models = len(datasets)
    if n_models == 0:
        return

    fig, axes = plt.subplots(
        max(1, (n_models + 2) // 3), 3,
        figsize=(15, 5 * max(1, (n_models + 2) // 3))
    )
    if n_models == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    for idx, (key, model_dir) in enumerate(datasets.items()):
        if idx >= len(axes):
            break

        metrics = load_metrics(model_dir)
        cm = metrics.get("confusion_matrix", [])

        if cm:
            ax = axes[idx]
            cm_array = np.array(cm)
            sns.heatmap(
                cm_array,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax,
                cbar=False
            )
            ax.set_title(f"{key}\nAcc: {metrics.get('val_accuracy', 0):.2%}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Confusion matrix saved: {output_path}")
    plt.close()


def plot_accuracy_comparison(
    results_dir: Path,
    summary_csv: Path,
    output_path: Path
):
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available")
        return

    if not summary_csv.exists():
        logger.warning(f"Summary file not found: {summary_csv}")
        return

    data = []
    with open(summary_csv, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                data.append({
                    'dataset': parts[0],
                    'model': parts[1],
                    'accuracy': float(parts[2]) if parts[2] else 0,
                    'f1': float(parts[3]) if parts[3] else 0
                })

    if not data:
        return

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    pivot_acc = df.pivot(index='dataset', columns='model', values='accuracy')
    pivot_acc.plot(kind='bar', ax=ax1, colormap='Set2')
    ax1.set_title('Classification Accuracy by Dataset')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Dataset')
    ax1.legend(title='Model', bbox_to_anchor=(1.02, 1))
    ax1.set_ylim(0, 1.1)
    ax1.tick_params(axis='x', rotation=45)

    ax2 = axes[1]
    pivot_f1 = df.pivot(index='dataset', columns='model', values='f1')
    pivot_f1.plot(kind='bar', ax=ax2, colormap='Set2')
    ax2.set_title('F1 Macro Score by Dataset')
    ax2.set_ylabel('F1 Macro')
    ax2.set_xlabel('Dataset')
    ax2.legend(title='Model', bbox_to_anchor=(1.02, 1))
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Accuracy comparison saved: {output_path}")
    plt.close()


def plot_heatmap(
    results_dir: Path,
    summary_csv: Path,
    output_path: Path
):
    if not MATPLOTLIB_AVAILABLE:
        return

    if not summary_csv.exists():
        return

    data = []
    with open(summary_csv, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                data.append({
                    'dataset': parts[0],
                    'model': parts[1],
                    'accuracy': float(parts[2]) if parts[2] else 0
                })

    if not data:
        return

    df = pd.DataFrame(data)
    pivot = df.pivot(index='dataset', columns='model', values='accuracy')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2%',
        cmap='RdYlGn',
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5
    )
    ax.set_title('Model Accuracy Heatmap')
    ax.set_xlabel('Model')
    ax.set_ylabel('Dataset')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Heatmap saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("results_dir", type=Path, help="Results root directory")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = args.output or results_dir / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = results_dir / "training_summary.csv"

    plot_confusion_matrix_comparison(
        results_dir,
        output_dir / "confusion_matrix_comparison.png"
    )

    plot_accuracy_comparison(
        results_dir,
        summary_csv,
        output_dir / "accuracy_comparison.png"
    )

    plot_heatmap(
        results_dir,
        summary_csv,
        output_dir / "accuracy_heatmap.png"
    )

    print(f"\nVisualization results saved to: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
