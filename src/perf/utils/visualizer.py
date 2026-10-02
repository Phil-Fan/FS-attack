
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LinearSegmentedColormap
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

KV_THRESHOLDS = {
    'low': 50,
    'high': 97.5,
    'max': 100
}

CACHE_SUBDIR = "cache_visualizer"
CACHE_VERSION = 2


def get_file_signature(file_path: Path) -> tuple:
    if file_path is None or not file_path.exists():
        return (0, 0, None)
    stat = file_path.stat()
    return (stat.st_mtime, stat.st_size, str(file_path))


def cached(cache_dir: Path, source_files: list, cache_key: str = "default"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_file = cache_dir / f"{cache_key}.pkl"

            if not cache_file.exists():
                return None

            # Check whether all source files are unchanged
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)

                cached_sigs = cached.get('source_signatures', {})
                all_match = True
                for src_path in source_files:
                    if src_path is None:
                        continue
                    src_sig = get_file_signature(src_path)
                    cached_sig = cached_sigs.get(str(src_path))
                    if cached_sig is None or cached_sig != src_sig:
                        all_match = False
                        break

                if all_match:
                    logger.info(f"Using cache: {cache_file.relative_to(cache_dir.parent)}")
                    return cached.get('result')
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

            return None
        return wrapper
    return decorator


def get_cached_scheduler_data(log_path: Path) -> dict | None:
    cache_dir = log_path.parent / CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{log_path.stem}.pkl"

    if cache_file.exists():
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)

            cached_version = cached.get("cache_version", 0)
            if cached_version != CACHE_VERSION:
                logger.debug(f"Cache expired: version changed ({cached_version} -> {CACHE_VERSION})")
                return None

            # Primary check: file size (unchanged file content is what matters)
            cached_size = cached.get('log_size', 0)
            actual_size = log_path.stat().st_size

            if cached_size == actual_size:
                logger.info(f"Using cache: {cache_file.relative_to(log_path.parent)}")
                return cached['data']
            else:
                logger.debug(f"Cache expired: file size changed ({cached_size} -> {actual_size})")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    return None


def save_cached_scheduler_data(log_path: Path, data: dict) -> None:
    cache_dir = log_path.parent / CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{log_path.stem}.pkl"

    try:
        import pickle
        cache_data = {
            'cache_version': CACHE_VERSION,
            'log_size': log_path.stat().st_size,
            'data': data
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"Saved cache: {cache_file.relative_to(log_path.parent)}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def parse_scheduler_log(log_path: Path) -> Dict:
    cached_data = get_cached_scheduler_data(log_path)
    if cached_data is not None:
        return cached_data

    steps, times = [], []
    rc_data, wc_data = [], []
    kv_usage = []
    event_data = {
        "preempted": {}, "scheduled": {}, "queued": {}, "finished": {},
    }
    all_user_types = set()

    if not log_path.exists():
        logger.warning(f"Scheduler log not found: {log_path}")
        return {"steps": steps, "times": times, "rc_data": rc_data, "wc_data": wc_data,
                "kv_usage": kv_usage, "event_data": event_data, "all_user_types": all_user_types}

    with open(log_path, "r") as f:
        for line in f:
            step_match = STEP_PATTERN.match(line)
            event_match = EVENT_PATTERN.match(line)

            if step_match:
                ts = datetime.strptime(step_match.group("time"), "%Y-%m-%d %H:%M:%S,%f")
                step = int(step_match.group("step"))
                r_c = eval(step_match.group("r_c"))
                w_c = eval(step_match.group("w_c"))
                kv = step_match.group("kv")
                gpu_kv = step_match.group("gpu_kv")
                cpu_kv = step_match.group("cpu_kv")

                if kv is not None:
                    kv_val = float(kv)
                else:
                    gpu_kv = float(gpu_kv)
                    cpu_kv = float(cpu_kv)
                    kv_val = max(gpu_kv, cpu_kv)

                steps.append(step)
                times.append(ts)
                rc_data.append(r_c)
                wc_data.append(w_c)
                kv_usage.append(kv_val)

            elif event_match:
                ts = datetime.strptime(event_match.group("time"), "%Y-%m-%d %H:%M:%S,%f")
                step = int(event_match.group("step"))
                event_type = event_match.group("event")
                user_type = event_match.group("user_type")

                # In sglang, retracted is the effective preemption signal.
                if event_type == "retracted":
                    event_type = "preempted"

                if user_type == "warmup":
                    continue

                all_user_types.add(user_type)

                if event_type in event_data:
                    if user_type not in event_data[event_type]:
                        event_data[event_type][user_type] = []
                    event_data[event_type][user_type].append((step, ts))

    result = {"steps": steps, "times": times, "rc_data": rc_data, "wc_data": wc_data,
              "kv_usage": kv_usage, "event_data": event_data, "all_user_types": all_user_types}

    save_cached_scheduler_data(log_path, result)

    logger.info(f"Parsed {len(steps)} step records and {len(all_user_types)} user types from scheduler log")
    return result


# ============ USENIX-style font settings ============
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# USENIX-style font sizes
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
TITLE_SIZE = 16
TITLE_WEIGHT = 'bold'
LEGEND_SIZE = 11

FIG_SIZE_SMALL = (6, 6)
FIG_SIZE_LARGE = (12, 6)
FIG_SIZE_WIDE = (16, 10)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.0)

STEP_PATTERN = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - (?:scheduler - )?DEBUG - "
    r"\[step (?P<step>\d+)\] r_c: (?P<r_c>\{.*?\}) w_c: (?P<w_c>\{.*?\})(?: s_c: (?P<s_c>\{.*?\}))??; "
    r"(?:(?:kv:(?P<kv>[\d\.]+)%)|(?:gpu_kv:(?P<gpu_kv>[\d\.]+)%, cpu_kv:(?P<cpu_kv>[\d\.]+)%)); "
    r"computed: (?P<computed>\{.*?\}); cached: (?P<cached>\{.*?\})"
)

EVENT_PATTERN = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - (?:scheduler - )?DEBUG - "
    r"\[step (?P<step>\d+)\] \[(?P<event>preempted|retracted|scheduled|queued|finished|swap_in|swap_out|preempt_recompute|preempt_swap)\] \[type: (?P<user_type>\w+)\]"
)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix",
    class_names: Optional[List[str]] = None,
    figsize: tuple = FIG_SIZE_SMALL,
    n_classes: int = 10
) -> None:
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if class_names is None:
        class_names = [f'{i}' for i in range(n_classes)]

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
                intensity = max(value / max_val, 0.3)
                color_matrix[i, j] = green_cmap(intensity)
            else:
                intensity = max(value / max_val, 0.3)
                color_matrix[i, j] = red_cmap(intensity)

    ax.imshow(color_matrix, interpolation='nearest')

    for i in range(n_classes):
        for j in range(n_classes):
            text_color = 'white' if cm[i, j] > max_val / 2 else 'black'
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center", color=text_color, fontsize=10)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted', fontsize=AXIS_LABEL_SIZE, fontweight=TITLE_WEIGHT)
    ax.set_ylabel('True', fontsize=AXIS_LABEL_SIZE, fontweight=TITLE_WEIGHT)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=green_cmap(0.6), label='Correct (diagonal)'),
        Patch(facecolor=red_cmap(0.6), label='Incorrect (off-diagonal)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=LEGEND_SIZE)

    accuracy = cm.trace() / cm.sum() if cm.sum() > 0 else 0
    ax.set_title(f'{title}\nAccuracy: {accuracy:.4f}', fontsize=TITLE_SIZE, fontweight=TITLE_WEIGHT)

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
                        intensity = max(value / max_val, 0.3)
                        color_matrix[row, col] = green_cmap(intensity)
                    else:
                        intensity = max(value / max_val, 0.3)
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
                fontsize=TITLE_SIZE, fontweight=TITLE_WEIGHT, y=1.00)
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
    figsize: tuple = FIG_SIZE_SMALL,
    cmap: str = 'RdYlGn',
    use_legend: bool = True
) -> None:
    n_rows = len(result_df)
    n_cols = len(result_df.columns)
    fig_width = figsize[0] + 3 if use_legend else figsize[0]
    fig, ax = plt.subplots(figsize=(fig_width, figsize[1]))

    row_labels = [chr(65 + i) for i in range(n_rows)]
    col_labels = [chr(97 + i) for i in range(n_cols)]

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

    ax.set_xlabel('Test Dataset', fontsize=AXIS_LABEL_SIZE, fontweight=TITLE_WEIGHT)
    ax.set_ylabel('Train Dataset', fontsize=AXIS_LABEL_SIZE, fontweight=TITLE_WEIGHT)
    ax.set_title(f'{title} (Accuracy)', fontsize=TITLE_SIZE, fontweight=TITLE_WEIGHT, pad=20)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)

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
    fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)

    metrics = ['Train Acc', 'Test Acc', 'F1-macro']
    values = [train_acc, test_acc, f1_macro]
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.4f}', ha='center', va='bottom',
               fontsize=LEGEND_SIZE, fontweight=TITLE_WEIGHT)

    ax.set_ylabel('Score', fontsize=AXIS_LABEL_SIZE, fontweight=TITLE_WEIGHT)
    ax.set_title(f'{dataset_name} - Training Metrics', fontsize=TITLE_SIZE, fontweight=TITLE_WEIGHT, pad=20)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_itl_analysis(
    itl_aligned_path: Path,
    output_dir: Path,
    time_limit: int,
    results_df: Optional[pd.DataFrame] = None,
    scheduler_log_path: Optional[Path] = None
) -> None:
    from scipy import stats
    from scipy.stats import gaussian_kde

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(itl_aligned_path)

    required_cols = ['time', 'user_id', 'kv_usage', 'itl_ms']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return

    has_user_type = 'user_type' in df.columns
    if has_user_type:
        df['user_type'] = df['user_type'].astype(str).str.strip().str.lower()

    df['timestamp'] = pd.to_datetime(df['time'])

    if has_user_type:
        df = df[~df['user_type'].isin(['warmup', 'prober'])].copy()
        logger.info(f"After excluding warmup/prober users: {len(df)} rows")

    if not df.empty:
        base_time = df['timestamp'].min()
        logger.info(f"Time origin (first non-warmup user time): {base_time}")
    else:
        logger.error("No data")
        return

    df['time_seconds'] = (df['timestamp'] - base_time).dt.total_seconds()
    time_limit = float(df['time_seconds'].max()) if not df.empty else time_limit
    df = df[df['time_seconds'] <= time_limit].copy()
    logger.info(f"Time range: 0-{time_limit:.2f}s (from ITL token), records: {len(df)}")

    logger.info(f"ITL analysis data loaded: {len(df)} rows, {df['user_id'].nunique()} users")
    if has_user_type:
        user_type_counts = df['user_type'].value_counts()
        logger.info(f"User type distribution: {user_type_counts.to_dict()}")

    original_len = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['kv_usage', 'itl_ms'])

    for col in ['kv_usage', 'itl_ms']:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df = df[mask]

    filtered_len = len(df)
    if original_len != filtered_len:
        logger.info(f"Data cleaning: {original_len} -> {filtered_len} rows")
    if df.empty:
        logger.warning("ITL visualization skipped: no valid data after filtering")
        return

    unique_users = sorted(df['user_id'].unique())
    n_users = len(unique_users)
    cmap = plt.colormaps.get_cmap('tab20')
    user_colors = {uid: cmap(i / n_users) for i, uid in enumerate(unique_users)}

    # Helper function: apply the USENIX style (no numbered titles)
    def apply_usenix_style(ax, xlabel, ylabel, title=None):
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE, fontweight=TITLE_WEIGHT)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, fontweight=TITLE_WEIGHT, labelpad=10)
        if title:
            ax.set_title(title, fontsize=TITLE_SIZE, fontweight=TITLE_WEIGHT)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.3)

    from matplotlib.ticker import FuncFormatter

    def _format_thousands(val, _pos):
        abs_val = abs(val)
        sign = '-' if val < 0 else ''
        if abs_val >= 1000:
            label = f"{abs_val / 1000:.1f}".rstrip('0').rstrip('.')
            return f"{sign}{label}K"
        return f"{val:.0f}"

    def _select_violin_user_types(series: pd.Series) -> list[str]:
        if series is None or series.empty:
            return ['all']
        cleaned = series.dropna().astype(str).str.strip().str.lower()
        counts = cleaned.value_counts()
        if counts.empty:
            return ['all']
        ordered = list(counts.index)
        if 'normal' in ordered:
            ordered = ['normal'] + [t for t in ordered if t != 'normal']
        if len(ordered) >= 2:
            return ordered[:2]
        return [ordered[0]]

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=FIG_SIZE_WIDE)
    gs = GridSpec(
        5,
        1,
        figure=fig,
        height_ratios=[1, 1, 1, 1, 1],
        hspace=0.15,
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(5)]

    # USENIX-style attack scenario color scheme
    USER_TYPE_COLORS = {
        'normal': '#0072B2',
        'malicious': '#D62728',
        'baseline': '#E57373',
    }

    sched_data = None
    if scheduler_log_path is not None:
        sched_data = parse_scheduler_log(scheduler_log_path)

    itl_violin_payload = None
    ttft_violin_payload = None

    results_user_type_map = None
    if results_df is None:
        inferred_results_csv = itl_aligned_path.parent.parent / "logs" / "results.csv"
        if inferred_results_csv.exists():
            try:
                results_df = pd.read_csv(inferred_results_csv)
                logger.info(f"Loaded TTFT data: {len(results_df)} rows ({inferred_results_csv})")
            except Exception as exc:
                logger.warning(f"Failed to load results.csv: {exc}")

    if results_df is not None and isinstance(results_df, pd.DataFrame):
        results_df = results_df.copy()
        if 'user_type' in results_df.columns:
            results_df['user_type'] = results_df['user_type'].astype(str).str.strip().str.lower()
        if 'user_id' in results_df.columns and 'user_type' in results_df.columns:
            results_user_type_map = (
                results_df.groupby('user_id')['user_type']
                .apply(lambda s: s.value_counts().index[0])
                .to_dict()
            )

    if not has_user_type and results_user_type_map:
        df['user_type'] = df['user_id'].map(results_user_type_map)
        has_user_type = True

    if results_df is not None and 'time' in results_df.columns:
        if 'user_type' in results_df.columns:
            results_df = results_df[~results_df['user_type'].isin(['warmup', 'prober'])]

        if not results_df.empty and isinstance(results_df, pd.DataFrame):
            results_df['timestamp'] = pd.to_datetime(results_df['time'])
            results_df['time_seconds'] = (results_df['timestamp'] - base_time).dt.total_seconds()
            results_df = results_df[results_df['time_seconds'] <= time_limit]
            results_df = results_df.dropna(subset=['ttft'])

            if 'user_type' in results_df.columns:
                for ut in results_df['user_type'].unique():
                    ut_df = results_df[results_df['user_type'] == ut]
                    c = USER_TYPE_COLORS.get(ut, '#333333')
                    axes[0].scatter(ut_df['time_seconds'], ut_df['ttft'],
                                   alpha=0.3, s=5, c=c, rasterized=True, label=ut)
                handles, labels = axes[0].get_legend_handles_labels()
                new_handles = []
                for h in handles:
                    color = h.get_facecolor()[0] if hasattr(h, 'get_facecolor') and len(h.get_facecolor()) > 0 else '#333333'
                    new_h = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color,
                                      markersize=10)
                    new_handles.append(new_h)
                axes[0].legend(new_handles, labels, loc='upper right', fontsize=LEGEND_SIZE, frameon=True)
            else:
                axes[0].scatter(results_df['time_seconds'], results_df['ttft'],
                               alpha=0.3, s=5, c='blue', rasterized=True)

    if sched_data is not None and sched_data.get('steps') and sched_data.get('times'):
        times_list = sched_data['times']
        kv_usage = sched_data['kv_usage']

        high_load_times = []
        for t, kv in zip(times_list, kv_usage):
            rel_time = (t - base_time).total_seconds()
            if 0 <= rel_time <= time_limit and kv >= KV_THRESHOLDS['high']:
                high_load_times.append(rel_time)

        if high_load_times:
            high_load_times = sorted(high_load_times)
            start = high_load_times[0]
            prev = high_load_times[0]
            for t in high_load_times[1:]:
                if t - prev > 1:
                    axes[0].axvspan(start, prev, ymin=0, ymax=1, color='#EF9A9A', alpha=0.5)
                    start = t
                prev = t
            axes[0].axvspan(start, prev, ymin=0, ymax=1, color='#EF9A9A', alpha=0.5)

    axes[0].set_ylabel('TTFT (ms)', fontsize=AXIS_LABEL_SIZE, fontweight=TITLE_WEIGHT, labelpad=10)
    axes[0].tick_params(axis='both', labelsize=TICK_SIZE)

    if has_user_type:
        for ut in df['user_type'].unique():
            ut_df = df[df['user_type'] == ut]
            c = USER_TYPE_COLORS.get(ut, '#333333')
            axes[1].scatter(ut_df['time_seconds'], ut_df['itl_ms'],
                           alpha=0.5, s=8, c=c, rasterized=True, label=ut, edgecolors='white', linewidth=0.3)
        handles, labels = axes[1].get_legend_handles_labels()
        new_handles = []
        for h in handles:
            color = h.get_facecolor()[0] if hasattr(h, 'get_facecolor') and len(h.get_facecolor()) > 0 else '#333333'
            new_h = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color,
                              markersize=10)
            new_handles.append(new_h)
        axes[1].legend(new_handles, labels, loc='upper right', fontsize=LEGEND_SIZE, frameon=True)
    else:
        axes[1].scatter(df['time_seconds'], df['itl_ms'],
                       alpha=0.5, s=8, c='#0072B2', rasterized=True, edgecolors='white', linewidth=0.3)
    axes[1].set_ylabel('ITL (ms)', fontsize=AXIS_LABEL_SIZE, fontweight=TITLE_WEIGHT, labelpad=10)
    axes[1].tick_params(axis='both', labelsize=TICK_SIZE)

    # Generate ITL violin plot data (by KV region and user type)
    if len(df) > 0:
        df['kv_region'] = pd.cut(df['kv_usage'],
                                  bins=[0, KV_THRESHOLDS['low'], KV_THRESHOLDS['high'], 100],
                                  labels=['Low\n(0-50%)', 'Med\n(50-97.5%)', 'High\n(97.5-100%)'])
        regions = ['Low\n(0-50%)', 'Med\n(50-97.5%)', 'High\n(97.5-100%)']

        user_types = _select_violin_user_types(df['user_type']) if has_user_type else ['all']
        base_positions = np.arange(1, len(regions) + 1)
        offset = 0.18 if len(user_types) > 1 else 0.0
        violin_data = []
        positions = []
        violin_user_types = []
        for region_idx, region in enumerate(regions):
            for ut_idx, user_type in enumerate(user_types):
                user_df = df[df['user_type'] == user_type] if has_user_type else df
                values = user_df[user_df['kv_region'] == region]['itl_ms'].dropna()
                violin_data.append(values)
                if len(user_types) == 1:
                    pos = base_positions[region_idx]
                else:
                    pos = base_positions[region_idx] + (-offset if ut_idx == 0 else offset)
                positions.append(pos)
                violin_user_types.append(user_type)

        itl_violin_payload = {
            'regions': regions,
            'user_types': user_types,
            'base_positions': base_positions,
            'positions': positions,
            'violin_data': violin_data,
            'violin_user_types': violin_user_types,
        }

    # Generate TTFT violin plot data (by KV region and user type)
    if results_df is not None and len(results_df) > 0:
        if 'user_type' in results_df.columns:
            results_df['user_type'] = results_df['user_type'].astype(str).str.strip().str.lower()

        # Ensure results_df has a time_seconds column
        if 'time_seconds' not in results_df.columns:
            if 'time' in results_df.columns:
                results_df['timestamp'] = pd.to_datetime(results_df['time'])
                if not results_df['timestamp'].isna().all():
                    results_df['time_seconds'] = (results_df['timestamp'] - base_time).dt.total_seconds()
                    results_df = results_df[results_df['time_seconds'] <= time_limit]
                else:
                    logger.warning("ITL visualization skipped: results_df timestamp is invalid, cannot generate the TTFT violin plot")
                    results_df = None
            else:
                logger.warning("ITL visualization skipped: results_df is missing the time column, cannot generate the TTFT violin plot")
                results_df = None

        if results_df is not None and len(results_df) > 0 and 'time_seconds' in results_df.columns:
            if 'kv_usage' in results_df.columns or 'user_id' in results_df.columns:
                merged = results_df.merge(df[['time_seconds', 'kv_usage']].drop_duplicates(),
                                           on='time_seconds', how='left')
                merged['kv_region'] = pd.cut(merged['kv_usage'].fillna(50),
                                              bins=[0, KV_THRESHOLDS['low'], KV_THRESHOLDS['high'], 100],
                                              labels=['Low\n(0-50%)', 'Med\n(50-97.5%)', 'High\n(97.5-100%)'])
                ttft_regions = ['Low\n(0-50%)', 'Med\n(50-97.5%)', 'High\n(97.5-100%)']

                user_types = _select_violin_user_types(merged['user_type']) if 'user_type' in merged.columns else ['all']
                if 'user_type' in merged.columns and user_types != ['all']:
                    merged = merged[merged['user_type'].isin(user_types)]
                base_positions = np.arange(1, len(ttft_regions) + 1)
                offset = 0.18 if len(user_types) > 1 else 0.0
                ttft_violin_data = []
                positions = []
                violin_user_types = []

                for region_idx, region in enumerate(ttft_regions):
                    for ut_idx, user_type in enumerate(user_types):
                        user_merged = merged[merged['user_type'] == user_type] if 'user_type' in merged.columns else merged
                        values = user_merged[user_merged['kv_region'] == region]['ttft'].dropna()
                        ttft_violin_data.append(values)
                        if len(user_types) == 1:
                            pos = base_positions[region_idx]
                        else:
                            pos = base_positions[region_idx] + (-offset if ut_idx == 0 else offset)
                        positions.append(pos)
                        violin_user_types.append(user_type)
                ttft_violin_payload = {
                    'regions': ttft_regions,
                    'user_types': user_types,
                    'base_positions': base_positions,
                    'positions': positions,
                    'violin_data': ttft_violin_data,
                    'violin_user_types': violin_user_types,
                }

    df_kv = df.groupby('time_seconds')['kv_usage'].mean().reset_index()
    axes[3].plot(df_kv['time_seconds'], df_kv['kv_usage'],
                 linewidth=2, alpha=0.8, color='#0072B2', label='KV usage')

    if sched_data is not None and sched_data.get('steps') and sched_data.get('times'):
        times_list = sched_data['times']
        kv_usage = sched_data['kv_usage']

        high_load = []
        med_load = []
        low_load = []

        for t, kv in zip(times_list, kv_usage):
            rel_time = (t - base_time).total_seconds()
            if 0 <= rel_time <= time_limit:
                if kv >= KV_THRESHOLDS['high']:
                    high_load.append(rel_time)
                elif kv >= KV_THRESHOLDS['low']:
                    med_load.append(rel_time)
                else:
                    low_load.append(rel_time)

        def draw_bg_ranges_y_limited(time_list, y_min, y_max, color, alpha):
            if not time_list:
                return
            time_list = sorted(time_list)
            start = time_list[0]
            prev = time_list[0]
            for t in time_list[1:]:
                if t - prev > 1:
                    axes[3].axvspan(start, prev, ymin=(y_min / 100), ymax=(y_max / 100), color=color, alpha=alpha)
                    start = t
                prev = t
            axes[3].axvspan(start, prev, ymin=(y_min / 100), ymax=(y_max / 100), color=color, alpha=alpha)

        draw_bg_ranges_y_limited(low_load, 0, KV_THRESHOLDS['low'], '#A5D6A7', 0.5)
        draw_bg_ranges_y_limited(med_load, KV_THRESHOLDS['low'], KV_THRESHOLDS['high'], '#FFCC80', 0.5)
        draw_bg_ranges_y_limited(high_load, KV_THRESHOLDS['high'], KV_THRESHOLDS['max'], '#EF9A9A', 0.5)

    # Keep only the 97.5 boundary line (dashed)
    axes[3].axhline(y=KV_THRESHOLDS['high'], color='gray', linestyle='--', alpha=0.6, linewidth=1.2, label='Threshold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#A5D6A7', edgecolor='gray', alpha=0.7, label=f'Low (0-{KV_THRESHOLDS["low"]}%)'),
        Patch(facecolor='#FFCC80', edgecolor='gray', alpha=0.7, label=f'Med ({KV_THRESHOLDS["low"]}-{KV_THRESHOLDS["high"]}%)'),
        Patch(facecolor='#EF9A9A', edgecolor='gray', alpha=0.7, label=f'High ({KV_THRESHOLDS["high"]}-{KV_THRESHOLDS["max"]}%)'),
    ]
    axes[3].legend(handles=legend_elements, loc='upper right', fontsize=LEGEND_SIZE, frameon=True)
    axes[3].set_ylabel('KV Cache(%)', fontsize=AXIS_LABEL_SIZE, fontweight=TITLE_WEIGHT, labelpad=10)
    axes[3].tick_params(axis='both', labelsize=TICK_SIZE)

    # Subplot 2: Event timeline - changed to a horizontal violin plot
    all_user_types = sched_data.get('all_user_types', set()) if sched_data else set()
    if sched_data is not None and sched_data.get('event_data'):
        event_data = sched_data['event_data']
        event_types = ['preempted', 'scheduled', 'finished']

        event_times_by_type = {}
        for event_type in event_types:
            event_times_by_type[event_type] = []
            for user_type in sorted(all_user_types):
                if user_type == 'warmup':
                    continue
                if event_type in event_data and user_type in event_data[event_type]:
                    events = event_data[event_type][user_type]
                    if events:
                        for _, ts in events:
                            rel_time = (ts - base_time).total_seconds()
                            if 0 <= rel_time <= time_limit:
                                event_times_by_type[event_type].append(rel_time)

        violin_data = []
        violin_labels = []
        violin_positions = []
        colors_violin = ['#D62728', '#2CA02C', '#1F77B4']
        filtered_colors = []
        filtered_event_types = []
        
        for idx, event_type in enumerate(event_types):
            data = event_times_by_type.get(event_type, [])
            if len(data) > 0:
                violin_data.append(data)
                violin_labels.append(event_type.capitalize())
                violin_positions.append(len(violin_positions) + 1)
                filtered_event_types.append(event_type)
                if idx < len(colors_violin):
                    filtered_colors.append(colors_violin[idx])

        # Only draw the violin plot when there is data
        if violin_data and len(violin_data) > 0:
            # Draw the horizontal violin plot
            violin_parts = axes[2].violinplot(
                violin_data,
                positions=violin_positions,
                vert=False,
                showmeans=True,
                showmedians=True
            )

            # Set the violin plot colors
            for i, body in enumerate(violin_parts['bodies']):
                if i < len(filtered_colors):
                    body.set_facecolor(filtered_colors[i])
                    body.set_alpha(0.6)

            if violin_positions:
                axes[2].set_yticks(violin_positions)
                axes[2].set_yticklabels(violin_labels)
                axes[2].set_ylim(0.3, len(violin_positions) + 0.7)

            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor=color, edgecolor='gray', label=event_type.capitalize())
                for color, event_type in zip(filtered_colors, filtered_event_types)
            ]
            if legend_handles:
                axes[2].legend(handles=legend_handles, loc='upper right', fontsize=LEGEND_SIZE, frameon=True)
        else:
            axes[2].text(0.5, 0.5, 'No event data available', 
                        ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
            axes[2].set_ylim(0, 1)

    axes[2].set_ylabel('', fontsize=AXIS_LABEL_SIZE, fontweight=TITLE_WEIGHT, labelpad=10)
    axes[2].tick_params(axis='both', labelsize=TICK_SIZE)

    if sched_data is not None and sched_data.get('steps'):
        running_counts = [sum(d.values()) if d else 0 for d in sched_data['rc_data']]
        waiting_counts = [sum(d.values()) if d else 0 for d in sched_data['wc_data']]

        if sched_data.get('times'):
            times_list = sched_data['times']
            if len(times_list) > 1:
                time_seconds = [(t - base_time).total_seconds() for t in times_list]
                filtered_data = [(ts, rc, wc) for ts, rc, wc in zip(time_seconds, running_counts, waiting_counts) if ts >= 0]
                if filtered_data:
                    filtered_ts, filtered_rc, filtered_wc = zip(*filtered_data)
                    axes[4].plot(filtered_ts, filtered_rc, linewidth=1.5, alpha=0.8,
                               color='#2CA02C', label='Running')
                    axes[4].plot(filtered_ts, filtered_wc, linewidth=1.5, alpha=0.8,
                               color='#D62728', label='Waiting')
                handles, labels = axes[4].get_legend_handles_labels()
                new_handles = []
                for h in handles:
                    color = h.get_color() if hasattr(h, 'get_color') else '#333333'
                    new_h = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color,
                                      markersize=10)
                    new_handles.append(new_h)
                axes[4].legend(new_handles, labels, loc='upper right', fontsize=LEGEND_SIZE, frameon=True)

        max_time = min(df['time_seconds'].max(), time_limit)
        axes[4].set_xlim(0, max_time)

    apply_usenix_style(axes[4], 'Time (seconds)', 'Queue Size')

    max_time_all = max(df['time_seconds'].max(), 1)
    for ax in axes[:5]:
        ax.set_xlim(0, max_time_all)
        ax.yaxis.set_major_formatter(FuncFormatter(_format_thousands))

    for i, ax in enumerate(axes):
        if i < 4:
            ax.set_xticklabels([])
        ax.tick_params(axis='x', labelbottom=False if i < 4 else True)
        ax.grid(True, alpha=0.3)

    fig.align_ylabels(axes)
    plt.tight_layout()

    # Add independent exception handling for each image generation so that one failure does not affect the others
    try:
        plt.savefig(output_dir / 'itl_fig1_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: itl_fig1_timeseries.png")
    except Exception as exc:
        logger.warning(f"  Failed to generate itl_fig1_timeseries.png: {exc}")
        plt.close('all')

    # ============== Figure 1b: TTFT/ITL violin plots ==============
    if itl_violin_payload is not None or ttft_violin_payload is not None:
        fig, vaxes = plt.subplots(1, 2, figsize=(12, 4))

        violin_type_colors = {
            'normal': '#1F77B4',
            'malicious': '#FF7F0E',
            'baseline': '#2CA02C',
            'all': '#4C78A8',
        }
        fallback_palette = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']

        def draw_violin(ax, payload, ylabel, show_xlabel):
            if payload is None:
                ax.axis('off')
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=10)
                return

            filtered_data = []
            filtered_positions = []
            filtered_user_types = []
            for data, pos, ut in zip(
                payload['violin_data'],
                payload['positions'],
                payload['violin_user_types'],
            ):
                if isinstance(data, pd.Series):
                    data_array = data.values
                else:
                    data_array = np.asarray(data)
                
                data_array = data_array[np.isfinite(data_array)]
                
                # Keep only cases with valid data
                if len(data_array) == 0:
                    continue
                
                filtered_data.append(data_array)
                filtered_positions.append(pos)
                filtered_user_types.append(ut)

            if not filtered_data:
                ax.axis('off')
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=10)
                return

            # Ensure filtered_data and filtered_positions have the same length
            if len(filtered_data) != len(filtered_positions):
                logger.warning(f"Data length mismatch: data={len(filtered_data)}, positions={len(filtered_positions)}")
                ax.axis('off')
                ax.text(0.5, 0.5, 'Data length mismatch', ha='center', va='center', fontsize=10)
                return

            parts = ax.violinplot(
                filtered_data,
                positions=filtered_positions,
                widths=0.28,
                vert=True,
                showmeans=True,
                showmedians=True,
            )

            # Color mapping: prefer known types; otherwise assign in order of appearance
            color_map = {
                ut: violin_type_colors.get(ut, fallback_palette[idx % len(fallback_palette)])
                for idx, ut in enumerate(payload['user_types'])
            }
            for i, body in enumerate(parts['bodies']):
                ut = filtered_user_types[i]
                color = color_map.get(ut, '#666666')
                body.set_facecolor(color)
                body.set_edgecolor('#333333')
                body.set_linewidth(0.6)
                body.set_alpha(0.85)

            ax.set_xticks(payload['base_positions'])
            ax.set_xticklabels(payload['regions'], fontsize=9)
            ax.set_ylabel(ylabel, fontsize=10, fontweight=TITLE_WEIGHT, labelpad=8)
            ax.tick_params(axis='both', labelsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            if not show_xlabel:
                ax.set_xlabel('')

            if len(payload['user_types']) > 1 and len(set(filtered_user_types)) > 1:
                from matplotlib.patches import Patch

                legend_handles = [
                    Patch(facecolor=color_map.get(ut, '#666666'), edgecolor='#333333', label=ut.capitalize())
                    for ut in payload['user_types']
                ]
                ax.legend(handles=legend_handles, loc='upper right', fontsize=8, frameon=True)

        draw_violin(vaxes[0], ttft_violin_payload, 'TTFT (ms)', show_xlabel=False)
        draw_violin(vaxes[1], itl_violin_payload, 'ITL (ms)', show_xlabel=True)

        for ax in vaxes:
            ax.yaxis.set_major_formatter(FuncFormatter(_format_thousands))

        plt.tight_layout()

        try:
            plt.savefig(output_dir / 'itl_fig1_violins.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"  Saved: itl_fig1_violins.png")
        except Exception as exc:
            logger.warning(f"  Failed to generate itl_fig1_violins.png: {exc}")
            plt.close('all')

    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_LARGE)

    if has_user_type:
        baseline_df = df[df['user_type'] == 'baseline'].copy()
        malicious_df = df[df['user_type'] == 'malicious'].copy()

        if not baseline_df.empty:
            axes[0, 0].scatter(baseline_df['kv_usage'], baseline_df['itl_ms'],
                              alpha=0.3, s=10, c='#E57373', rasterized=True,
                              label='Baseline Users')
            if len(baseline_df) > 2:
                kv_series = baseline_df['kv_usage'].dropna()
                if len(kv_series) > 0:
                    slope_b, intercept_b, r_value_b, _, _ = stats.linregress(baseline_df['kv_usage'], baseline_df['itl_ms'])
                    x_line_b = np.linspace(kv_series.min(), kv_series.max(), 100)
                    y_line_b = slope_b * x_line_b + intercept_b
                    axes[0, 0].plot(x_line_b, y_line_b, color='#E57373',
                                   linestyle='--', linewidth=2, alpha=0.8,
                                   label=f'Baseline: R²={r_value_b**2:.3f}')

        if not malicious_df.empty:
            axes[0, 0].scatter(malicious_df['kv_usage'], malicious_df['itl_ms'],
                              alpha=0.3, s=10, c='#D62728', rasterized=True,
                              label='Malicious Users')
            if len(malicious_df) > 2:
                kv_series = malicious_df['kv_usage'].dropna()
                if len(kv_series) > 0:
                    slope_m, intercept_m, r_value_m, _, _ = stats.linregress(malicious_df['kv_usage'], malicious_df['itl_ms'])
                    x_line_m = np.linspace(kv_series.min(), kv_series.max(), 100)
                    y_line_m = slope_m * x_line_m + intercept_m
                    axes[0, 0].plot(x_line_m, y_line_m, color='#D62728',
                              linestyle='-', linewidth=3, alpha=0.8,
                              label=f'Malicious: R²={r_value_m**2:.3f}')

        apply_usenix_style(axes[0, 0], 'KV Cache Usage (%)', 'ITL (ms)', 'Baseline & Malicious Users')
        axes[0, 0].legend(loc='upper left', fontsize=LEGEND_SIZE, frameon=True)

        if baseline_df.empty and malicious_df.empty:
            axes[0, 0].text(0.5, 0.5, 'No baseline/malicious user data', ha='center', va='center',
                           transform=axes[0, 0].transAxes, fontsize=12)
    else:
        kv_series = df['kv_usage'].dropna()
        if len(kv_series) > 0:
            axes[0, 0].scatter(df['kv_usage'], df['itl_ms'], alpha=0.2, s=5,
                                        c=df['user_id'].map(user_colors), rasterized=True)
            slope, intercept, r_value, _, _ = stats.linregress(df['kv_usage'], df['itl_ms'])
            x_line = np.linspace(kv_series.min(), kv_series.max(), 100)
            y_line = slope * x_line + intercept
            axes[0, 0].plot(x_line, y_line, 'r-', linewidth=3,
                      label=f'Linear: k={slope:.2f}, b={intercept:.2f}\nR²={r_value**2:.3f}')
        apply_usenix_style(axes[0, 0], 'KV Cache Usage (%)', 'ITL (ms)', 'All Users')
        axes[0, 0].legend(loc='upper left', fontsize=LEGEND_SIZE, frameon=True)

    if has_user_type:
        normal_df = df[df['user_type'] == 'normal'].copy()
        prober_df = df[df['user_type'] == 'prober'].copy()

        if not normal_df.empty:
            axes[0, 1].scatter(normal_df['kv_usage'], normal_df['itl_ms'],
                              alpha=0.3, s=10, c='#0072B2', rasterized=True,
                              label='Normal Users')
            if len(normal_df) > 2:
                kv_series = normal_df['kv_usage'].dropna()
                if len(kv_series) > 0:
                    slope_n, intercept_n, r_value_n, _, _ = stats.linregress(normal_df['kv_usage'], normal_df['itl_ms'])
                    x_line_n = np.linspace(kv_series.min(), kv_series.max(), 100)
                    y_line_n = slope_n * x_line_n + intercept_n
                    axes[0, 1].plot(x_line_n, y_line_n, color='#0072B2',
                                   linestyle='--', linewidth=2, alpha=0.8,
                                   label=f'Normal: R²={r_value_n**2:.3f}')

        if not prober_df.empty:
            axes[0, 1].scatter(prober_df['kv_usage'], prober_df['itl_ms'],
                              alpha=0.5, s=10, c='#9467bd', marker='^',
                              rasterized=True, label='Prober Users')
            if len(prober_df) > 2:
                kv_series = prober_df['kv_usage'].dropna()
                if len(kv_series) > 0:
                    slope_p, intercept_p, r_value_p, _, _ = stats.linregress(prober_df['kv_usage'], prober_df['itl_ms'])
                    x_line_p = np.linspace(kv_series.min(), kv_series.max(), 100)
                    y_line_p = slope_p * x_line_p + intercept_p
                    axes[0, 1].plot(x_line_p, y_line_p, color='#9467bd',
                                   linestyle=':', linewidth=2, alpha=0.8,
                                   label=f'Prober: R²={r_value_p**2:.3f}')

        apply_usenix_style(axes[0, 1], 'KV Cache Usage (%)', 'ITL (ms)', 'Normal & Prober Users')
        axes[0, 1].legend(loc='upper left', fontsize=LEGEND_SIZE)

        if normal_df.empty and prober_df.empty:
            axes[0, 1].text(0.5, 0.5, 'No normal/prober user data', ha='center', va='center',
                           transform=axes[0, 1].transAxes, fontsize=12)
    else:
        for uid in unique_users:
            user_df = df[df['user_id'] == uid]
            axes[0, 1].scatter(user_df['kv_usage'], user_df['itl_ms'],
                              c=[user_colors[uid]], alpha=0.3, s=10, rasterized=True)

            if len(user_df) > 2:
                kv_series = user_df['kv_usage'].dropna()
                if len(kv_series) > 0:
                    slope_u, intercept_u, r_value_u, _, _ = stats.linregress(user_df['kv_usage'], user_df['itl_ms'])
                    x_line_u = np.linspace(kv_series.min(), kv_series.max(), 100)
                    y_line_u = slope_u * x_line_u + intercept_u
                    axes[0, 1].plot(x_line_u, y_line_u, color=user_colors[uid],
                                   linestyle='--', linewidth=2, alpha=0.8,
                                   label=f'U{uid}: k={slope_u:.2f}, R²={r_value_u**2:.2f}')

        apply_usenix_style(axes[0, 1], 'KV Cache Usage (%)', 'ITL (ms)', 'Per-User Regression')
        axes[0, 1].legend(loc='upper left', fontsize=LEGEND_SIZE, ncol=2)

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
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            cax = inset_axes(axes[1, 0], width="3%", height="45%", loc="upper right", borderpad=1)
            cbar = plt.colorbar(scatter, cax=cax)
            cbar.set_label('Density')
            cbar.ax.tick_params(labelsize=8)
            cax.yaxis.set_ticks_position('right')
        except (ValueError, np.linalg.LinAlgError):
            axes[1, 0].scatter(sample_df['kv_usage'], sample_df['itl_ms'], s=10, alpha=0.6, rasterized=True)

    apply_usenix_style(axes[1, 0], 'KV Cache Usage (%)', 'ITL (ms)', 'ITL vs KV - 2D Density')
    axes[1, 0].yaxis.set_major_formatter(FuncFormatter(_format_thousands))

    df['kv_bin'] = pd.cut(df['kv_usage'], bins=20)
    pivot_table = df.pivot_table(values='itl_ms', index='kv_bin', columns='user_id', aggfunc='mean', observed=True)
    if not pivot_table.empty:
        pivot_table = pivot_table.iloc[::-1]

    heatmap = sns.heatmap(pivot_table, cmap='RdYlBu_r', cbar=False, ax=axes[1, 1])

    num_users = len(pivot_table.columns)
    if num_users > 0:
        max_labels = 8
        step = max(1, int(np.ceil(num_users / max_labels)))
        tick_idx = np.arange(0, num_users, step)
        axes[1, 1].set_xticks(tick_idx + 0.5)
        axes[1, 1].set_xticklabels(
            [str(pivot_table.columns[i]) for i in tick_idx],
            rotation=45,
            ha='right',
            fontsize=8,
        )

    num_bins = len(pivot_table.index)
    if num_bins > 0:
        kv_series = df['kv_usage'].dropna()
        if len(kv_series) > 0:
            kv_min = kv_series.min()
            kv_max = kv_series.max()
        else:
            kv_min = kv_max = 0
        if num_bins == 1:
            axes[1, 1].set_yticks([0.5])
            axes[1, 1].set_yticklabels([f"{kv_max:.0f}%"], fontsize=8)
        else:
            axes[1, 1].set_yticks([0.5, num_bins - 0.5])
            axes[1, 1].set_yticklabels([f"{kv_max:.0f}%", f"{kv_min:.0f}%"], fontsize=8)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cax = inset_axes(axes[1, 1], width="3%", height="45%", loc="upper right", borderpad=1)
    cbar = plt.colorbar(heatmap.collections[0], cax=cax)
    cbar.set_label('Mean ITL (ms)')
    cbar.ax.tick_params(labelsize=8)
    cax.yaxis.set_ticks_position('right')

    apply_usenix_style(axes[1, 1], 'User ID', 'KV Cache (%)', 'Heatmap: Mean ITL')

    axes[0, 0].yaxis.set_major_formatter(FuncFormatter(_format_thousands))
    axes[0, 1].yaxis.set_major_formatter(FuncFormatter(_format_thousands))

    plt.tight_layout()

    try:
        plt.savefig(output_dir / 'itl_fig2_scatter_regression.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: itl_fig2_scatter_regression.png")
    except Exception as exc:
        logger.warning(f"  Failed to generate itl_fig2_scatter_regression.png: {exc}")
        plt.close('all')

    logger.info(f"ITL analysis visualization complete: {output_dir}")
