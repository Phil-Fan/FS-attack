
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LinearSegmentedColormap
import logging

logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix",
    class_names: Optional[List[str]] = None,
    figsize: tuple = (10, 8),
    n_classes: int = 10
) -> None:
    if class_names is not None:
        n_classes = len(class_names)
    else:
        unique_labels = sorted(set(np.concatenate([y_true, y_pred])))
        n_classes = len(unique_labels)
        class_names = [f'{i}' for i in unique_labels]

    # Use the labels argument to ensure a fixed-size confusion matrix
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=figsize)

    color_matrix = np.zeros((n_classes, n_classes, 4))

    green_cmap = plt.cm.Greens
    red_cmap = plt.cm.Reds

    max_val = cm.max() if cm.max() > 0 else 1

    for i in range(n_classes):
        for j in range(n_classes):
            value = cm[i, j]

            if value == 0:
                color_matrix[i, j] = [1, 1, 1, 1]
            elif i == j:
                intensity = value / max_val
                intensity = max(intensity, 0.3)
                color_matrix[i, j] = green_cmap(intensity)
            else:
                intensity = value / max_val
                intensity = max(intensity, 0.3)
                color_matrix[i, j] = red_cmap(intensity)

    im = ax.imshow(color_matrix, interpolation='nearest')

    for i in range(n_classes):
        for j in range(n_classes):
            text_color = 'white' if cm[i, j] > max_val / 2 else 'black'
            text = ax.text(j, i, format(cm[i, j], 'd'),
                          ha="center", va="center", color=text_color, fontsize=10)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_ylabel('True', fontsize=11, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=green_cmap(0.6), label='Correct (diagonal)'),
        Patch(facecolor=red_cmap(0.6), label='Incorrect (off-diagonal)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    accuracy = cm.trace() / cm.sum() if cm.sum() > 0 else 0
    ax.set_title(f'{title}\nAccuracy: {accuracy:.4f}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices_grid(
    confusion_matrices: Dict[tuple, np.ndarray],
    dataset_names: List[str],
    output_path: Path,
    figsize_per_cell: int = 4,
    n_classes: int = 10
) -> None:
    n = len(dataset_names)
    figsize = (figsize_per_cell * n + 3, figsize_per_cell * n)

    fig, axes = plt.subplots(n, n, figsize=figsize)

    if n == 1:
        axes = np.array([[axes]])

    index_labels = [chr(65 + i) for i in range(n)]

    class_names = [f'{i}' for i in range(n_classes)]

    green_cmap = plt.cm.Greens
    red_cmap = plt.cm.Reds

    for i, train_name in enumerate(dataset_names):
        for j, test_name in enumerate(dataset_names):
            ax = axes[i, j]
            key = (train_name, test_name)

            if key not in confusion_matrices:
                ax.set_facecolor('#f0f0f0')
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='#666666')
                ax.set_xticks([])
                ax.set_yticks([])
                _set_grid_axis_labels(ax, index_labels, i, j, n)
                continue

            cm = confusion_matrices[key]

            # Ensure the confusion matrix is n_classes x n_classes
            if cm.shape != (n_classes, n_classes):
                new_cm = np.zeros((n_classes, n_classes), dtype=cm.dtype)
                min_rows = min(cm.shape[0], n_classes)
                min_cols = min(cm.shape[1], n_classes)
                new_cm[:min_rows, :min_cols] = cm[:min_rows, :min_cols]
                cm = new_cm

            color_matrix = np.zeros((n_classes, n_classes, 4))
            max_val = cm.max() if cm.max() > 0 else 1

            for row in range(n_classes):
                for col in range(n_classes):
                    value = cm[row, col]

                    if value == 0:
                        color_matrix[row, col] = [1, 1, 1, 1]
                    elif row == col:
                        intensity = value / max_val
                        intensity = max(intensity, 0.3)
                        color_matrix[row, col] = green_cmap(intensity)
                    else:
                        intensity = value / max_val
                        intensity = max(intensity, 0.3)
                        color_matrix[row, col] = red_cmap(intensity)

            ax.imshow(color_matrix, interpolation='nearest')

            for row in range(n_classes):
                for col in range(n_classes):
                    text_color = 'white' if cm[row, col] > max_val / 2 else 'black'
                    ax.text(col, row, format(cm[row, col], 'd'),
                           ha="center", va="center", color=text_color, fontsize=8)

            ax.set_xticks(np.arange(n_classes))
            ax.set_yticks(np.arange(n_classes))
            ax.set_xticklabels(class_names, fontsize=8)
            ax.set_yticklabels(class_names, fontsize=8)

            accuracy = cm.trace() / cm.sum() if cm.sum() > 0 else 0
            ax.set_title(f'{accuracy:.2f}', fontsize=11, fontweight='bold', pad=5)

            _set_grid_axis_labels(ax, index_labels, i, j, n)

    _add_dataset_legend(fig, dataset_names, index_labels)

    plt.suptitle('Confusion Matrices (Diagonal = Accuracy)',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _set_grid_axis_labels(ax, index_labels: List[str], i: int, j: int, n: int) -> None:
    if i == n - 1:
        ax.set_xlabel(f'Test: {index_labels[j]}', fontsize=9)
    else:
        ax.set_xlabel('')

    if j == 0:
        ax.set_ylabel(f'Train: {index_labels[i]}', fontsize=9)
    else:
        ax.set_ylabel('')


def _add_dataset_legend(fig, dataset_names: List[str], index_labels: List[str]) -> None:
    from matplotlib.patches import Patch

    legend_elements = []
    for idx, name in zip(index_labels, dataset_names):
        legend_elements.append(
            Patch(facecolor='lightgray', edgecolor='black', label=f'{idx}: {name}')
        )

    fig.legend(
        handles=legend_elements,
        loc='center right',
        bbox_to_anchor=(0.98, 0.5),
        fontsize=9,
        framealpha=0.9,
        title='Datasets',
        title_fontsize=10,
        edgecolor='gray'
    )


def plot_generalization_heatmap(
    result_df: pd.DataFrame,
    output_path: Path,
    title: str = "Generalization Matrix",
    figsize: tuple = (10, 8),
    cmap: str = 'RdYlGn',
    use_legend: bool = True
) -> None:
    n_rows = len(result_df)
    n_cols = len(result_df.columns)
    fig_width = figsize[0] + 3 if use_legend else figsize[0]
    fig, ax = plt.subplots(figsize=(fig_width, figsize[1]))

    row_labels = [chr(65 + i) for i in range(n_rows)]
    col_labels = [chr(97 + i) for i in range(n_cols)]  # a, b, c, ... (avoid conflicts with row labels)

    result_df_short = result_df.copy()
    result_df_short.index = row_labels
    result_df_short.columns = col_labels
    result_df_short = result_df_short.astype(float)

    sns.heatmap(
        result_df_short,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Accuracy'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_xlabel('Test Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Train Dataset', fontsize=12, fontweight='bold')
    ax.set_title(f'{title} (Accuracy)', fontsize=14, fontweight='bold', pad=20)

    if use_legend:
        _add_heatmap_legend(ax, result_df.index.tolist(), result_df.columns.tolist(), row_labels, col_labels)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _add_heatmap_legend(
    ax,
    row_datasets: List[str],
    col_datasets: List[str],
    row_labels: List[str],
    col_labels: List[str]
) -> None:
    def simplify_name(name: str) -> str:
        parts = name.split('__')
        if len(parts) >= 3:
            payload = parts[1].replace('payload_', '')
            ratio = parts[3].split('_')[1] if len(parts) > 3 else ''
            return f'{payload}/{ratio}'
        return name[:30]

    legend_text = "Train (rows):\n"
    for label, name in zip(row_labels, row_datasets):
        legend_text += f"  {label}: {simplify_name(name)}\n"

    if col_datasets != row_datasets:
        legend_text += "\nTest (cols):\n"
        for label, name in zip(col_labels, col_datasets):
            legend_text += f"  {label}: {simplify_name(name)}\n"

    ax.text(
        1.02, 0.5,
        legend_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )


def plot_training_metrics(
    train_acc: float,
    test_acc: float,
    f1_macro: float,
    output_path: Path,
    dataset_name: str = "Model"
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    metrics = ['Train Acc', 'Test Acc', 'F1-macro']
    values = [train_acc, test_acc, f1_macro]
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.4f}', ha='center', va='bottom',
               fontsize=11, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset_name} - Training Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_training_report(
    dataset_name: str,
    train_acc: float,
    test_acc: float,
    f1_macro: float,
    num_train_samples: int,
    num_test_samples: int,
    num_classes: int,
    output_path: Path
) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Training Report: {dataset_name}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Train Accuracy:  {train_acc:.4f}\n")
        f.write(f"  Test Accuracy:   {test_acc:.4f}\n")
        f.write(f"  F1-macro:        {f1_macro:.4f}\n\n")

        f.write("Dataset Info:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Num Classes:     {num_classes}\n")
        f.write(f"  Train Samples:   {num_train_samples}\n")
        f.write(f"  Test Samples:    {num_test_samples}\n")


def save_generalization_report(
    result_df: pd.DataFrame,
    output_path: Path
) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Generalization Test Report\n")
        f.write("=" * 60 + "\n\n")

        f.write("Accuracy Matrix:\n")
        f.write("-" * 40 + "\n")
        result_df.to_csv(f, float_format='%.4f')
        f.write("\n")

        diagonal_acc = [result_df.iloc[i, i]
                       for i in range(len(result_df))
                       if not pd.isna(result_df.iloc[i, i])]
        if diagonal_acc:
            f.write(f"\nSame-dataset avg accuracy: {np.mean(diagonal_acc):.4f}\n")

        off_diag = []
        for i in range(len(result_df)):
            for j in range(len(result_df)):
                if i != j and not pd.isna(result_df.iloc[i, j]):
                    off_diag.append(result_df.iloc[i, j])
        if off_diag:
            f.write(f"Cross-dataset avg accuracy: {np.mean(off_diag):.4f}\n")
            f.write(f"Cross-dataset std dev:       {np.std(off_diag):.4f}\n")


def plot_dataset_statistics(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str = "Dataset"
) -> None:
    import matplotlib.pyplot as plt

    train_df = train_df.copy()
    eval_df = eval_df.copy()
    train_df['seq_length'] = train_df['itl'].apply(lambda x: len(str(x).split(',')))
    eval_df['seq_length'] = eval_df['itl'].apply(lambda x: len(str(x).split(',')))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Dataset Statistics: {dataset_name}', fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    ax.hist(train_df['kv_usage'], bins=50, alpha=0.6, label='Train', color='#3498db', edgecolor='black')
    ax.hist(eval_df['kv_usage'], bins=50, alpha=0.6, label='Eval', color='#e74c3c', edgecolor='black')
    ax.set_xlabel('KV Usage (%)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('KV Usage Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    train_labels = train_df['kv_label'].value_counts().sort_index()
    eval_labels = eval_df['kv_label'].value_counts().sort_index()
    x = np.arange(len(train_labels))
    width = 0.35
    ax.bar(x - width/2, train_labels.values, width, label='Train', color='#3498db', alpha=0.8, edgecolor='black')
    ax.bar(x + width/2, eval_labels.values, width, label='Eval', color='#e74c3c', alpha=0.8, edgecolor='black')
    ax.set_xlabel('KV Label (Class)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('KV Label Distribution', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(train_labels.index)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[0, 2]
    ax.hist(train_df['seq_length'], bins=50, alpha=0.6, label='Train', color='#3498db', edgecolor='black')
    ax.hist(eval_df['seq_length'], bins=50, alpha=0.6, label='Eval', color='#e74c3c', edgecolor='black')
    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Sequence Length Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    train_users = train_df['userid'].nunique()
    eval_users = eval_df['userid'].nunique()
    total_users = len(set(train_df['userid'].unique()) | set(eval_df['userid'].unique()))
    categories = ['Train\nUsers', 'Eval\nUsers', 'Total\nUsers']
    values = [train_users, eval_users, total_users]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('User Statistics', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(value)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax = axes[1, 1]
    categories = ['Train\nSamples', 'Eval\nSamples', 'Total\nSamples']
    values = [len(train_df), len(eval_df), len(train_df) + len(eval_df)]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Sample Statistics', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(value)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax = axes[1, 2]
    ax.axis('off')
    stats_data = {
        'Metric': [
            'Train Samples',
            'Eval Samples',
            'Train Users',
            'Eval Users',
            'Avg Seq Length (Train)',
            'Avg Seq Length (Eval)',
            'KV Usage Mean (Train)',
            'KV Usage Mean (Eval)',
            'KV Usage Std (Train)',
            'KV Usage Std (Eval)',
            'Num Classes'
        ],
        'Value': [
            len(train_df),
            len(eval_df),
            train_df['userid'].nunique(),
            eval_df['userid'].nunique(),
            f"{train_df['seq_length'].mean():.2f}",
            f"{eval_df['seq_length'].mean():.2f}",
            f"{train_df['kv_usage'].mean():.2f}",
            f"{eval_df['kv_usage'].mean():.2f}",
            f"{train_df['kv_usage'].std():.2f}",
            f"{eval_df['kv_usage'].std():.2f}",
            train_df['kv_label'].nunique()
        ]
    }
    table = ax.table(
        cellText=[[str(v)] for v in stats_data['Value']],
        rowLabels=stats_data['Metric'],
        cellLoc='left',
        loc='center',
        colWidths=[0.6]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title('Dataset Summary', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = output_dir / 'dataset_statistics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Dataset statistics charts saved: {output_path}")


def plot_itl_analysis(
    itl_aligned_path: Path,
    output_dir: Path,
    time_limit: int
) -> None:
    from scipy import stats
    from scipy.stats import gaussian_kde

    df = pd.read_csv(itl_aligned_path)

    required_cols = ['time', 'user_id', 'kv_usage', 'itl_ms']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return

    df['timestamp'] = pd.to_datetime(df['time'])
    df['time_seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

    df = df[df['time_seconds'] <= time_limit].copy()
    logger.info(f"Applied time limit: {time_limit}s, records after filtering: {len(df)}")

    logger.info(f"ITL analysis data loaded: {len(df)} rows, {df['user_id'].nunique()} users")

    original_len = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['kv_usage', 'itl_ms'])

    for col in ['kv_usage', 'itl_ms']:
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        removed = (~mask).sum()
        if removed > 0:
            logger.debug(f"  Removed outliers for {col}: {removed}")
        df = df[mask]

    filtered_len = len(df)
    if original_len != filtered_len:
        logger.info(f"Data cleaning: {original_len} -> {filtered_len} rows")

    output_dir.mkdir(parents=True, exist_ok=True)

    unique_users = sorted(df['user_id'].unique())
    n_users = len(unique_users)
    cmap = plt.colormaps.get_cmap('tab20')
    user_colors = {uid: cmap(i / n_users) for i, uid in enumerate(unique_users)}

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    for uid in unique_users:
        user_df = df[df['user_id'] == uid]
        axes[0].scatter(user_df['time_seconds'], user_df['itl_ms'],
                       c=[user_colors[uid]], alpha=0.3, s=5, label=f'User {uid}', rasterized=True)

    axes[0].set_xlabel('Time (seconds from start)', fontsize=12)
    axes[0].set_ylabel('ITL (ms)', fontsize=12)
    axes[0].set_title('ITL over Time by User', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)

    for uid in unique_users:
        user_df = df[df['user_id'] == uid]
        axes[1].scatter(user_df['time_seconds'], user_df['kv_usage'],
                       c=[user_colors[uid]], alpha=0.3, s=5, label=f'User {uid}', rasterized=True)

    axes[1].set_xlabel('Time (seconds from start)', fontsize=12)
    axes[1].set_ylabel('KV Cache Usage', fontsize=12)
    axes[1].set_title('KV Cache Usage over Time by User', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=8, ncol=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'itl_fig1_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: itl_fig1_timeseries.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    kv_series = df['kv_usage'].dropna()
    if len(kv_series) > 0:
        scatter = axes[0, 0].scatter(df['kv_usage'], df['itl_ms'], alpha=0.2, s=5,
                                    c=df['user_id'].map(user_colors), rasterized=True)

        slope, intercept, r_value, p_value, std_err = stats.linregress(df['kv_usage'], df['itl_ms'])
        x_line = np.linspace(kv_series.min(), kv_series.max(), 100)
        y_line = slope * x_line + intercept
        axes[0, 0].plot(x_line, y_line, 'r-', linewidth=3,
                  label=f'Linear: k={slope:.2f}, b={intercept:.2f}\nR²={r_value**2:.3f}')

    axes[0, 0].set_xlabel('KV Cache Usage', fontsize=12)
    axes[0, 0].set_ylabel('ITL (ms)', fontsize=12)
    axes[0, 0].set_title('[2.1] ITL vs KV - Linear Regression', fontsize=13, fontweight='bold')
    axes[0, 0].legend(loc='upper left', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    for uid in unique_users:
        user_df = df[df['user_id'] == uid]
        axes[0, 1].scatter(user_df['kv_usage'], user_df['itl_ms'],
                          c=[user_colors[uid]], alpha=0.3, s=10, rasterized=True)

        if len(user_df) > 2:
            kv_series_u = user_df['kv_usage'].dropna()
            if len(kv_series_u) > 0:
                slope_u, intercept_u, r_value_u, _, _ = stats.linregress(user_df['kv_usage'], user_df['itl_ms'])
                x_line_u = np.linspace(kv_series_u.min(), kv_series_u.max(), 100)
                y_line_u = slope_u * x_line_u + intercept_u
                axes[0, 1].plot(x_line_u, y_line_u, color=user_colors[uid],
                               linestyle='--', linewidth=2, alpha=0.8,
                               label=f'U{uid}: y={slope_u:.2f}x+{intercept_u:.2f}, R²={r_value_u**2:.2f}')

    axes[0, 1].set_xlabel('KV Cache Usage', fontsize=12)
    axes[0, 1].set_ylabel('ITL (ms)', fontsize=12)
    axes[0, 1].set_title('[2.2] ITL vs KV - Per-User Regression', fontsize=13, fontweight='bold')
    axes[0, 1].legend(loc='upper left', fontsize=8, ncol=2)
    axes[0, 1].grid(True, alpha=0.3)

    sample_size = min(10000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42).copy()
    sample_df = sample_df.replace([np.inf, -np.inf], np.nan)
    sample_df = sample_df.dropna(subset=['kv_usage', 'itl_ms'])

    if len(sample_df) >= 2:
        xy = np.vstack([sample_df['kv_usage'], sample_df['itl_ms']])
        try:
            z = gaussian_kde(xy)(xy)
            scatter = axes[1, 0].scatter(sample_df['kv_usage'], sample_df['itl_ms'],
                                         c=z, s=10, cmap='viridis', alpha=0.6, rasterized=True)
            plt.colorbar(scatter, ax=axes[1, 0], label='Density')
        except (ValueError, np.linalg.LinAlgError):
            axes[1, 0].scatter(sample_df['kv_usage'], sample_df['itl_ms'], s=10, alpha=0.6, rasterized=True)

    axes[1, 0].set_xlabel('KV Cache Usage', fontsize=12)
    axes[1, 0].set_ylabel('ITL (ms)', fontsize=12)
    axes[1, 0].set_title('[2.3] ITL vs KV - 2D Density', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    df['kv_bin'] = pd.cut(df['kv_usage'], bins=20)
    pivot_table = df.pivot_table(values='itl_ms', index='kv_bin', columns='user_id', aggfunc='mean')
    sns.heatmap(pivot_table, cmap='RdYlBu_r', cbar_kws={'label': 'Mean ITL (ms)'}, ax=axes[1, 1])
    axes[1, 1].set_xlabel('User ID', fontsize=12)
    axes[1, 1].set_ylabel('KV Cache Usage (binned)', fontsize=12)
    axes[1, 1].set_title('[2.4] Heatmap: Mean ITL by KV Usage and User', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'itl_fig2_scatter_regression.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: itl_fig2_scatter_regression.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    x_data = df['kv_usage'].dropna().values
    y_data = df['itl_ms'].dropna().values
    if len(x_data) == 0 or len(y_data) == 0:
        logger.warning("ITL visualization skipped: no valid KV/ITL data for nonlinear analysis")
        plt.close('all')
        return
    
    sample_idx = np.random.choice(len(x_data), size=min(15000, len(x_data)), replace=False)
    x_sample = x_data[sample_idx]
    y_sample = y_data[sample_idx]

    z1 = np.polyfit(x_sample, y_sample, 1)
    z2 = np.polyfit(x_sample, y_sample, 2)
    z3 = np.polyfit(x_sample, y_sample, 3)

    p1 = np.poly1d(z1)
    p2 = np.poly1d(z2)
    p3 = np.poly1d(z3)

    k, b = z1
    r2_linear = 1 - np.sum((y_sample - p1(x_sample)) ** 2) / np.sum((y_sample - y_sample.mean()) ** 2)
    r2_quad = 1 - np.sum((y_sample - p2(x_sample)) ** 2) / np.sum((y_sample - y_sample.mean()) ** 2)
    r2_cubic = 1 - np.sum((y_sample - p3(x_sample)) ** 2) / np.sum((y_sample - y_sample.mean()) ** 2)

    x_fit = np.linspace(x_data.min(), x_data.max(), 100)

    axes[0, 0].scatter(x_sample, y_sample, alpha=0.2, s=5, color='gray', rasterized=True)
    axes[0, 0].plot(x_fit, p1(x_fit), 'r-', linewidth=2.5, label=f'Linear: y = {k:.2f}x + {b:.2f}, R² = {r2_linear:.3f}')
    axes[0, 0].plot(x_fit, p2(x_fit), 'g-', linewidth=2.5, label=f'Quadratic: R² = {r2_quad:.3f}')
    axes[0, 0].plot(x_fit, p3(x_fit), 'b-', linewidth=2.5, label=f'Cubic: R² = {r2_cubic:.3f}')

    axes[0, 0].set_xlabel('KV Cache Usage', fontsize=12)
    axes[0, 0].set_ylabel('ITL (ms)', fontsize=12)
    axes[0, 0].set_title('[3.1] Polynomial Regression', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10, loc='upper left', framealpha=0.9)
    axes[0, 0].grid(True, alpha=0.3)

    df['kv_segment'] = pd.cut(df['kv_usage'], bins=12)
    segment_stats = df.groupby('kv_segment', observed=True)['itl_ms'].agg(['mean', 'median', 'std', 'count'])

    x_seg = np.arange(len(segment_stats))
    width = 0.25

    axes[0, 1].bar(x_seg - width, segment_stats['mean'], width, label='Mean', color='steelblue', alpha=0.8)
    axes[0, 1].bar(x_seg, segment_stats['median'], width, label='Median', color='forestgreen', alpha=0.8)
    axes[0, 1].bar(x_seg + width, segment_stats['std'], width, label='Std Dev', color='coral', alpha=0.8)

    axes[0, 1].set_xlabel('KV Cache Usage Segment', fontsize=12)
    axes[0, 1].set_ylabel('ITL (ms)', fontsize=12)
    axes[0, 1].set_title('[3.2] ITL Statistics by KV Segment', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(x_seg)
    axes[0, 1].set_xticklabels([f'{int(idx.left)}-{int(idx.right)}' for idx in segment_stats.index], rotation=45, ha='right', fontsize=8)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    colors_q = ['blue', 'cyan', 'green', 'orange', 'red']

    kv_groups = df.groupby('kv_usage')['itl_ms'].quantile(quantiles).unstack()

    for q, color in zip(quantiles, colors_q):
        axes[1, 0].plot(kv_groups.index, kv_groups[q], 'o-', color=color,
                       linewidth=2, markersize=4, label=f'Q{int(q*100)}', alpha=0.8)

    axes[1, 0].set_xlabel('KV Cache Usage', fontsize=12)
    axes[1, 0].set_ylabel('ITL (ms)', fontsize=12)
    axes[1, 0].set_title('[3.3] Quantile Regression', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    user_corr_data = []
    for uid in unique_users:
        user_df = df[df['user_id'] == uid]
        corr = user_df[['itl_ms', 'kv_usage']].corr().iloc[0, 1]
        user_corr_data.append({'user_id': uid, 'correlation': corr})

    corr_df = pd.DataFrame(user_corr_data)

    x_pos = np.arange(len(corr_df))
    bars = axes[1, 1].bar(x_pos, corr_df['correlation'],
                         color=[user_colors[uid] for uid in corr_df['user_id']], alpha=0.7, edgecolor='black')
    axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([f'User {uid}' for uid in corr_df['user_id']])
    axes[1, 1].set_ylabel('Correlation Coefficient', fontsize=12)
    axes[1, 1].set_title('[3.4] ITL-KV Correlation by User', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    for bar, corr in zip(bars, corr_df['correlation']):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{corr:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'itl_fig3_nonlinear_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: itl_fig3_nonlinear_analysis.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    df['predicted_itl'] = slope * df['kv_usage'] + intercept
    df['residual'] = df['itl_ms'] - df['predicted_itl']

    for uid in unique_users:
        user_df = df[df['user_id'] == uid]
        axes[0, 0].scatter(user_df['kv_usage'], user_df['residual'],
                          c=[user_colors[uid]], alpha=0.3, s=10, rasterized=True)

    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('KV Cache Usage', fontsize=12)
    axes[0, 0].set_ylabel('Residual (ms)', fontsize=12)
    axes[0, 0].set_title('[4.1] Residual Plot', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    df['itl_per_kv'] = df['itl_ms'] / df['kv_usage']

    for uid in unique_users:
        user_df = df[df['user_id'] == uid]
        axes[0, 1].scatter(user_df['kv_usage'], user_df['itl_per_kv'],
                          c=[user_colors[uid]], alpha=0.3, s=10, rasterized=True,
                          label=f'User {uid}')

    axes[0, 1].set_xlabel('KV Cache Usage', fontsize=12)
    axes[0, 1].set_ylabel('ITL / KV Ratio (ms/unit)', fontsize=12)
    axes[0, 1].set_title('[4.2] ITL per KV Unit', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[0, 1].grid(True, alpha=0.3)

    kv_bins = pd.cut(df['kv_usage'], bins=5)
    for i, kv_interval in enumerate(kv_bins.cat.categories):
        subset = df[kv_bins == kv_interval]
        if len(subset) > 50:
            axes[1, 0].hist(subset['itl_ms'], bins=50, alpha=0.5,
                           label=f'KV: {kv_interval}', density=True)

    axes[1, 0].set_xlabel('ITL (ms)', fontsize=12)
    axes[1, 0].set_ylabel('Density', fontsize=12)
    axes[1, 0].set_title('[4.3] ITL Distribution by KV Range', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    df['kv_quantile'] = pd.qcut(df['kv_usage'], q=8, duplicates='drop')
    kv_box_data = [group['itl_ms'].values for name, group in df.groupby('kv_quantile', observed=True)]
    kv_labels = [f'{int(interval.left)}-{int(interval.right)}' for interval in df['kv_quantile'].cat.categories]

    bp = axes[1, 1].boxplot(kv_box_data, tick_labels=kv_labels, patch_artist=True, showmeans=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    axes[1, 1].set_xlabel('KV Cache Usage (quantile bins)', fontsize=12)
    axes[1, 1].set_ylabel('ITL (ms)', fontsize=12)
    axes[1, 1].set_title('[4.4] ITL Distribution by KV Quantiles', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'itl_fig4_deep_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: itl_fig4_deep_analysis.png")

    logger.info(f"ITL analysis visualization complete: {output_dir}")
