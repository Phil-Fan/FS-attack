#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
import re
from pathlib import Path
import sys
import argparse
import logging
from collections import defaultdict
from matplotlib.patches import Patch, FancyArrowPatch
try:
    from utils.logging_utils import setup_logging
except ImportError:
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from .paths import collect_results_csvs, parse_run_id, get_actual_duration
except ImportError:
    import os
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paths import collect_results_csvs, parse_run_id, get_actual_duration

try:
    from scipy import stats
except ImportError:
    stats = None


sns.set_theme(style="whitegrid", context="notebook")


def parse_filename(filename):
    file_path = Path(filename)
    stem = file_path.stem

    timestamp_match = re.match(r"(\d{8}_\d{6})", stem)
    timestamp_prefix = timestamp_match.group(1) if timestamp_match else None

    payload_type = None
    dataset_name = None
    ratio_value = None

    for segment in stem.split("__"):
        if segment.startswith("payload_"):
            payload_type = segment.replace("payload_", "", 1)
        elif segment.startswith("file_"):
            dataset_name = segment.replace("file_", "", 1)
        elif segment.startswith("ratio_"):
            ratio_value = segment.replace("ratio_", "", 1)

    if not payload_type or not dataset_name or not ratio_value:
        run_id = file_path.parent.parent.name if file_path.name == "results.csv" else None
        if run_id:
            parsed = parse_run_id(run_id)
            payload_type = payload_type or parsed.get("payload")
            dataset_name = dataset_name or parsed.get("dataset")
            ratio_value = ratio_value or parsed.get("ratio")

    return timestamp_prefix, payload_type, dataset_name, ratio_value


def extract_attack_meta(payload_type):
    if not payload_type:
        return "unknown", "base"
    match = re.search(r"(.*?)_?attack([0-9p\\.]+)", payload_type)
    if match:
        method = match.group(1) or payload_type
        strength = match.group(2).replace("p", ".")
        return method, strength
    return payload_type, "base"


def normalize_ratio(ratio_value):
    if not ratio_value:
        return 0.0, "0"
    cleaned = ratio_value.replace("p", ".")
    try:
        numeric = float(cleaned)
    except ValueError:
        return 0.0, ratio_value
    display = f"{numeric:.2f}".rstrip("0").rstrip(".")
    if display == "":
        display = "0"
    return numeric, display


def parse_timestamp_column(series):
    candidate_formats = [
        "%Y-%m-%d %H:%M:%S,%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in candidate_formats:
        try:
            return pd.to_datetime(series, format=fmt, errors="raise")
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(series, errors="coerce")


def linear_fit_with_ci(x_vals, y_vals, pred_x):
    if len(x_vals) < 2:
        y_pred = np.full_like(pred_x, y_vals[0] if len(y_vals) else np.nan, dtype=float)
        return y_pred, y_pred, y_pred

    if np.nanstd(x_vals) == 0:
        y_pred = np.full_like(pred_x, np.nanmean(y_vals), dtype=float)
        return y_pred, y_pred, y_pred

    X = np.vstack([np.ones_like(x_vals), x_vals]).T
    beta, residuals, rank, _ = np.linalg.lstsq(X, y_vals, rcond=None)
    pred_design = np.vstack([np.ones_like(pred_x), pred_x]).T
    y_pred = pred_design @ beta

    dof = len(x_vals) - 2
    if dof <= 0:
        return y_pred, y_pred, y_pred

    if residuals.size > 0:
        sse = residuals[0]
    else:
        sse = np.sum((y_vals - X @ beta) ** 2)
    mse = sse / dof if dof > 0 else 0.0
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        y_pred = np.full_like(pred_x, np.nanmean(y_vals), dtype=float)
        return y_pred, y_pred, y_pred
    se = np.array(
        [np.sqrt(row @ xtx_inv @ row.T * mse) for row in pred_design]
    )
    if stats:
        t_val = stats.t.ppf(0.975, dof)
    else:
        t_val = 1.96
    delta = t_val * se
    lower = y_pred - delta
    upper = y_pred + delta
    return y_pred, lower, upper

def plot_summary_by_payload(csv_files, user_type, output_dir=None):

    metrics = [
        ("latency", "Latency (s)"),
        ("ttft", "TTFT (ms)"),
        ("tpot", "TPOT (ms)"),
    ]

    grouped = defaultdict(lambda: defaultdict(list))
    for csv_file in csv_files:
        _, payload_type, dataset_name, ratio_value = parse_filename(csv_file)
        method, strength = extract_attack_meta(payload_type)
        dataset = dataset_name or "unknown"
        ratio = ratio_value or "0p0"
        grouped[(method, dataset, strength)][ratio].append(csv_file)

    if not grouped:
        logger.warning("Warning: could not extract payload/dataset/ratio information from the filename")
        return

    distinct_colors = [
        "#1f77b4",  # Blue   (corresponds to: Engorgio)
        "#ff7f0e",  # Orange (corresponds to: LoopLLM)
        "#9467bd",  # Purple (corresponds to: Extend Attack)
        "#d62728",
        "#2ca02c",  # Green  (corresponds to: Direct Generation)
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#AEC7E8",
        "#FFBB78",
        "#98DF8A",
        "#FF9896",
        "#C5B0D5",
        "#911EB4"   # User Fixed Purple (unchanged)
    ]

    items = sorted(grouped.items())
    if tqdm:
        items = tqdm(items, desc=f"slo-{user_type}", leave=False, dynamic_ncols=True)
    for (method, dataset, strength), ratio_files in items:
        label_method = method.replace("__", "-")
        label_dataset = dataset.replace("__", "-")
        logger.info(
            "Processing experiment: method=%s, dataset=%s, strength=%s, user=%s",
            label_method,
            label_dataset,
            strength,
            user_type,
        )

        ratio_series = []
        ratio_meta = []
        first_source = None
        for ratio_key, files in sorted(
            ratio_files.items(), key=lambda kv: normalize_ratio(kv[0])[0]
        ):
            combined_frames = []
            for csv_file in files:
                try:
                    df = pd.read_csv(csv_file)
                except Exception as exc:
                    logger.warning("Warning: failed to read %s: %s", csv_file, exc)
                    continue

                required = ["user_type", "latency", "ttft", "tpot"]
                if not all(col in df.columns for col in required):
                    logger.warning("Warning: file %s is missing required fields; skipping", csv_file)
                    continue

                if "timestamp" not in df.columns:
                    df = df.reset_index(drop=True)
                    df["timestamp"] = pd.to_datetime(df.index.astype(float), unit="s")

                df["timestamp"] = parse_timestamp_column(df["timestamp"])
                df["user_type"] = df["user_type"].astype(str).str.lower()
                df = df[df["user_type"] != "warmup"].copy()
                mask = df["user_type"].str.contains(user_type, case=False, na=False)
                selected = df.loc[mask].dropna(subset=["timestamp"]).copy()
                if selected.empty:
                    continue

                # Filter the data based on the actual duration
                csv_path = Path(csv_file)
                if csv_path.name == "results.csv":
                    run_dir = csv_path.parent.parent
                    actual_duration = get_actual_duration(run_dir)
                    if actual_duration is not None:
                        start_time = selected["timestamp"].min()
                        selected["elapsed_seconds"] = (
                            (selected["timestamp"] - start_time).dt.total_seconds()
                        )
                        # Keep only data within the experiment duration
                        selected = selected[selected["elapsed_seconds"] <= actual_duration].copy()
                        selected = selected.drop(columns=["elapsed_seconds"])
                        if selected.empty:
                            logger.warning("No data after filtering (duration limit: %.1fs)", actual_duration)
                            continue

                combined_frames.append(selected)
                if first_source is None:
                    first_source = csv_file

            if not combined_frames:
                continue

            merged = pd.concat(combined_frames, ignore_index=True)
            merged = merged.sort_values("timestamp").reset_index(drop=True)

            overall_mask = pd.Series(True, index=merged.index)
            for metric_name in ("latency", "ttft"):
                if metric_name in merged.columns:
                    metric_values = merged[metric_name]
                    valid_vals = metric_values.dropna()
                    if not valid_vals.empty:
                        mu = valid_vals.mean()
                        sigma = valid_vals.std()
                        if sigma > 0:
                            lower = mu - 3 * sigma
                            upper = mu + 3 * sigma
                            metric_mask = metric_values.between(lower, upper, inclusive="both")
                            metric_mask = metric_mask | metric_values.isna()
                            overall_mask &= metric_mask

            filtered = merged.loc[overall_mask].copy()
            if filtered.empty:
                continue

            filtered["sample_idx"] = np.arange(1, len(filtered) + 1)
            ratio_series.append((ratio_key, filtered))
            numeric_ratio, ratio_label = normalize_ratio(ratio_key)
            ratio_meta.append({"label": ratio_label, "numeric": numeric_ratio})

        if not ratio_series:
            logger.warning("Warning: no usable data for this combination")
            continue

        colors = distinct_colors[:len(ratio_series)]
        
        segment_width = 1.0
        segment_padding = 0.18
        segment_info = []
        for idx_seg, meta in enumerate(ratio_meta):
            start = idx_seg * (segment_width + segment_padding)
            end = start + segment_width
            center = (start + end) / 2
            segment_info.append(
                {
                    "start": start,
                    "end": end,
                    "center": center,
                    "label": meta["label"],
                    "numeric": meta["numeric"],
                }
            )

        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 8), sharex=False)
        fig.subplots_adjust(hspace=0.18)
        fig.patch.set_facecolor("white")
        legend_handles = [
            Patch(facecolor=color, edgecolor=color, label=meta["label"])
            for color, meta in zip(colors, ratio_meta)
        ]

        for ax_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[ax_idx]
            ax.set_facecolor("#f8f9fb")
            for spine in ax.spines.values():
                spine.set_alpha(0.3)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            box_sequences = []
            box_colors = []
            box_labels = []
            box_frames = []
            mean_summaries = []
            for color, seg, (ratio_key, df_ratio), meta in zip(
                colors, segment_info, ratio_series, ratio_meta
            ):
                if metric not in df_ratio.columns or df_ratio.empty:
                    continue
                metric_series = df_ratio[metric].dropna()
                if metric_series.empty:
                    continue

                filtered_df = df_ratio.loc[df_ratio[metric].notna()].copy()
                count = len(filtered_df)
                if count == 1:
                    x_values = np.array([seg["center"]])
                else:
                    x_values = np.linspace(seg["start"], seg["end"], count)
                filtered_df = filtered_df.assign(seg_x=x_values)
                
                sns.scatterplot(
                    x="seg_x",
                    y=metric,
                    data=filtered_df,
                    ax=ax,
                    color=color,
                    s=18,
                    alpha=0.75,
                    edgecolor="#f6f6f6",
                    linewidth=0.3,
                )

                values = filtered_df[metric].to_numpy()
                if len(values) > 0:
                    box_sequences.append(values)
                    box_colors.append(color)
                    box_labels.append(meta["label"])
                    box_frames.append(
                        pd.DataFrame(
                            {
                                "value": values,
                                "label": meta["label"],
                                "numeric": meta["numeric"],
                            }
                        )
                    )
                    mean_val = float(values.mean())
                    median_val = float(np.median(values))
                    ax.hlines(
                        median_val,
                        seg["start"],
                        seg["end"],
                        colors=color,
                        linestyles=(0, (4, 3)),
                        linewidth=1.1,
                    )
                    mean_summaries.append(
                        {
                            "numeric": meta["numeric"],
                            "label": meta["label"],
                            "mean": mean_val,
                            "median": median_val,
                            "center": seg["center"],
                            "box_pos": len(box_sequences),
                            "segment": seg,
                        }
                    )

            ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(axis="y", alpha=0.4, linestyle="--", linewidth=0.7)
            ax.grid(False, axis="x")
            centers = [seg["center"] for seg in segment_info]
            labels = [seg["label"] for seg in segment_info]
            ax.set_xticks(centers)
            ax.set_xticklabels(labels, fontsize=14)

            if metric == "latency" and len(mean_summaries) >= 2:
                numeric_vals = np.array([item["numeric"] for item in mean_summaries], dtype=float)
                mean_vals = np.array([item["mean"] for item in mean_summaries], dtype=float)
                x_min = segment_info[0]["start"]
                x_max = segment_info[-1]["end"]
                display_x = np.linspace(x_min, x_max, 400)
                numeric_min = numeric_vals.min()
                numeric_max = numeric_vals.max()
                if numeric_max == numeric_min:
                    mapped_numeric = np.full_like(display_x, numeric_min, dtype=float)
                else:
                    mapped_numeric = np.interp(
                        display_x,
                        [x_min, x_max],
                        [numeric_min, numeric_max],
                    )
                y_pred, lower_ci, upper_ci = linear_fit_with_ci(numeric_vals, mean_vals, mapped_numeric)
                ax.plot(display_x, y_pred, color="#2c3e50", linewidth=1.4, linestyle="-")
                ax.fill_between(display_x, lower_ci, upper_ci, color="#2c3e50", alpha=0.15)

            if ax_idx == len(metrics) - 1:
                ax.set_xlabel('Malicious User Ratio', fontsize=16, fontweight='bold')
            else:
                ax.set_xticklabels([])
                ax.set_xlabel("")

            ax_box = None
            if box_sequences:
                box_df = pd.concat(box_frames, ignore_index=True)
                divider = make_axes_locatable(ax)
                ax_box = divider.append_axes("right", size="12%", pad=0.12)
                palette = {label: color for label, color in zip(box_labels, box_colors)}
                ax_box.set_xticklabels([])
                sns.boxplot(
                    data=box_df,
                    x="label",
                    y="value",
                    order=box_labels,
                    palette=palette,
                    ax=ax_box,
                    width=0.55,
                    fliersize=0,
                    hue="label",
                    legend=False,
                )

                ax_box.set_ylim(ax.get_ylim())
                ax_box.set_xticks([])
                ax_box.set_yticklabels([])
                ax_box.set_xlabel("")
                ax_box.set_ylabel("")
                ax_box.set_facecolor("#f8f9fb")
                ax_box.yaxis.grid(True, alpha=0.4, linestyle="--", linewidth=0.6)
                for spine in ax_box.spines.values():
                    spine.set_alpha(0.2)
                ax_box.spines['top'].set_visible(False)
                ax_box.spines['right'].set_visible(False)

            if metric == "latency":
                def find_entry(target):
                    for entry in mean_summaries:
                        if abs(entry["numeric"] - target) < 1e-6:
                            return entry
                    return None

                base_entry = find_entry(0.0)
                target_entry = find_entry(0.75)
                if base_entry and target_entry:
                    seg = target_entry.get("segment")
                    if seg:
                        arrow_x = seg["end"] + segment_padding * 0.6
                        start_y = base_entry["median"]
                        end_y = target_entry["median"]
                        arrow = FancyArrowPatch(
                            (arrow_x, start_y),
                            (arrow_x, end_y),
                            arrowstyle="-|>",
                            mutation_scale=12,
                            linewidth=1.1,
                            color="#1f1f1f",
                        )
                        ax.add_patch(arrow)
                        if base_entry["median"] != 0:
                            delta_pct = (end_y - start_y) / base_entry["median"] * 100
                        else:
                            delta_pct = 0.0
                        text_y = (start_y + end_y) / 2
                        ax.text(
                            arrow_x - 0.03,
                            text_y,
                            f"{delta_pct:+.1f}%",
                            fontsize=15,
                            fontweight="bold",
                            ha="right",
                            va="center",
                            color="#1f1f1f",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                        )
                        current_ymax = ax.get_ylim()[1]
                        if text_y > current_ymax:
                            ax.set_ylim(ax.get_ylim()[0], text_y * 1.1)

        plt.tight_layout(rect=[0, 0.06, 1, 0.98])

        sample_path = first_source or csv_files[0]
        output_base = output_dir or Path(sample_path).parent
        safe_method = re.sub(r"[^a-zA-Z0-9_-]", "-", label_method)
        safe_dataset = re.sub(r"[^a-zA-Z0-9_-]", "-", label_dataset)
        safe_strength = re.sub(r"[^a-zA-Z0-9._-]", "-", strength)
        output_path = (
            Path(output_base)
            / f"2_slo_side_{safe_method}_{safe_dataset}_{safe_strength}_{user_type}.png"
        )

        try:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("Summary figure saved to: %s", output_path)
        except Exception as exc:
            logger.error("Error: could not save the summary figure to %s: %s", output_path, exc)

        plt.close(fig)

def plot_metrics(csv_path, output_path=None):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error("Error: could not read CSV file %s: %s", csv_path, e)
        return
    
    required_columns = ['timestamp', 'user_type', 'latency', 'ttft', 'tpot']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error("Error: CSV file is missing required columns: %s", missing_columns)
        return
    
    df['timestamp'] = parse_timestamp_column(df['timestamp'])
    if df['timestamp'].isna().all():
        logger.error("Error: the timestamp column could not be parsed; all results are empty")
        return
    
    # Ensure user_type is a string type
    df['user_type'] = df['user_type'].astype(str).str.lower()
    
    df = df[df['user_type'] != 'warmup'].copy()
    
    normal_df = df[df['user_type'].str.contains('normal', case=False, na=False)].copy()
    malicious_df = df[df['user_type'].str.contains('malicious', case=False, na=False)].copy()
    
    normal_df = normal_df.sort_values('timestamp')
    malicious_df = malicious_df.sort_values('timestamp')
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    metrics = [
        ('latency', 'Latency (s)', axes[0]),
        ('ttft', 'TTFT (ms)', axes[1]),
        ('tpot', 'TPOT (ms)', axes[2])
    ]
    
    for metric, ylabel, ax in metrics:
        if not normal_df.empty:
            sns.lineplot(
                data=normal_df,
                x='timestamp',
                y=metric,
                ax=ax,
                label='Normal',
                marker='o',
                linewidth=1.5,
                alpha=0.7,
            )
        
        if not malicious_df.empty:
            sns.lineplot(
                data=malicious_df,
                x='timestamp',
                y=metric,
                ax=ax,
                label='Malicious',
                marker='s',
                linewidth=1.5,
                alpha=0.7,
            )
        
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    axes[-1].set_xlabel('Timestamp', fontsize=11)
    
    plt.tight_layout()
    
    # Get the original filename and annotate it in the bottom right (after tight_layout to ensure correct position)
    csv_file = Path(csv_path)
    filename = csv_file.name
    
    fig.text(0.99, 0.01, filename, ha='right', va='bottom', 
             fontsize=8, alpha=0.6, family='monospace', transform=fig.transFigure)
    
    if output_path is None:
        filename_stem = csv_file.stem
        filename_without_ts = re.sub(r'^\d{8}_\d{6}__', '', filename_stem)
        if not filename_without_ts:
            filename_without_ts = filename_stem
        output_path = csv_file.parent / f"{filename_without_ts}.png"
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info("Image saved to: %s", output_path)
    except Exception as e:
        logger.error("Error: could not save the image to %s: %s", output_path, e)
        return
    
    plt.close()


def plot_summary(csv_files, user_type, output_path=None):
    metrics = [
        ('latency', 'Latency (s)'),
        ('ttft', 'TTFT (ms)'),
        ('tpot', 'TPOT (ms)')
    ]
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    file_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            required_columns = ['timestamp', 'user_type', 'latency', 'ttft', 'tpot']
            if not all(col in df.columns for col in required_columns):
                continue
            
            df['timestamp'] = parse_timestamp_column(df['timestamp'])
            df['user_type'] = df['user_type'].astype(str).str.lower()
            
            df = df[df['user_type'] != 'warmup'].copy()
            
            user_df = df[df['user_type'].str.contains(user_type, case=False, na=False)].copy()
            
            if not user_df.empty:
                user_df = user_df.sort_values('timestamp')
                
                stats = {}
                for metric, _ in metrics:
                    if metric in user_df.columns:
                        values = user_df[metric].dropna()
                        if len(values) > 0:
                            stats[metric] = {
                                'mean': values.mean(),
                                'p99': values.quantile(0.99)
                            }
                
                file_data.append({
                    'file': Path(csv_file).stem,
                    'data': user_df,
                    'stats': stats
                })
        except Exception as e:
            logger.warning("Warning: error reading file %s: %s", csv_file, e)
            continue
    
    if not file_data:
        logger.warning("Warning: no data found for %s users", user_type)
        plt.close()
        return
    
    colors = cm.tab10(range(len(file_data)))
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+']
    
    for idx, (metric, ylabel) in enumerate(metrics):
        ax_line = fig.add_subplot(gs[idx, 0])
        ax_box = fig.add_subplot(gs[idx, 1])
        
        box_data = []
        box_labels = []
        
        box_palette = {}
        box_stats = []
        for file_idx, file_info in enumerate(file_data):
            df = file_info['data']
            file_name = file_info['file']
            stats = file_info['stats']
            
            if not df.empty and metric in df.columns:
                marker = markers[file_idx % len(markers)]
                color = colors[file_idx]
                
                sns.lineplot(
                    data=df,
                    x='timestamp',
                    y=metric,
                    ax=ax_line,
                    label=file_name,
                    marker=marker,
                    linewidth=1.5,
                    alpha=0.7,
                    color=color,
                )
                
                if metric in stats and len(df) > 0:
                    mean_val = stats[metric]['mean']
                    p99_val = stats[metric]['p99']
                    
                    start_time = df['timestamp'].iloc[0]
                    end_time = df['timestamp'].iloc[-1]
                    
                    # Draw the mean as a dashed line over this segment (not added to the legend)
                    ax_line.hlines(mean_val, start_time, end_time, 
                                 colors=color, linestyles='--', linewidth=1.5, 
                                 alpha=0.8)
                    
                    # Draw p99 as a dotted line over this segment (not added to the legend)
                    ax_line.hlines(p99_val, start_time, end_time, 
                                 colors=color, linestyles=':', linewidth=1.0, 
                                 alpha=0.6)
                
                values = df[metric].dropna()
                if len(values) > 0:
                    box_data.append(values.values)
                    # Shorten the filename for the boxplot label (keep only the key part)
                    # Extract the ratio information, e.g. ratio_0p5 -> 0p5
                    short_name = file_name
                    if 'ratio' in file_name:
                        ratio_match = re.search(r'ratio_([\w]+)', file_name)
                        if ratio_match:
                            short_name = ratio_match.group(1)  # keep only the ratio value, e.g. "0p5"
                        else:
                            parts = file_name.split('__')
                            if len(parts) > 1:
                                ratio_parts = [p for p in parts if 'ratio' in p.lower()]
                                if ratio_parts:
                                    ratio_match = re.search(r'ratio_([\w]+)', ratio_parts[0])
                                    if ratio_match:
                                        short_name = ratio_match.group(1)
                                    else:
                                        short_name = ratio_parts[0].replace('ratio_', '')
                                else:
                                    short_name = parts[-1].replace('.csv', '')[:15]
                            else:
                                short_name = file_name[:15]
                    else:
                        short_name = file_name[:15] + '...' if len(file_name) > 15 else file_name
                    box_labels.append(short_name)
                    box_palette.setdefault(short_name, color)
                    box_stats.append(stats.get(metric) if metric in stats else None)
        
        ax_line.set_ylabel(ylabel, fontsize=11)
        ax_line.legend(loc='best', fontsize=7, ncol=1)
        ax_line.grid(True, alpha=0.3)
        
        ax_line.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax_line.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax_line.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        if box_data:
            box_frames = [
                pd.DataFrame({'value': data, 'label': label})
                for data, label in zip(box_data, box_labels)
            ]
            df_box = pd.concat(box_frames, ignore_index=True)
            order = list(dict.fromkeys(box_labels))
            sns.boxplot(
                data=df_box,
                x='label',
                y='value',
                order=order,
                palette=box_palette,
                ax=ax_box,
                width=0.5,
                fliersize=0,
            )
            
            label_to_pos = {label: idx for idx, label in enumerate(order)}
            for label, stats_info in zip(box_labels, box_stats):
                if stats_info is not None:
                    p99_val = stats_info['p99']
                    ax_box.text(label_to_pos[label], p99_val, f'p99={p99_val:.2f}',
                              fontsize=6, ha='center', va='center',
                              bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='yellow', alpha=0.5))
            
            ax_box.set_ylabel(ylabel, fontsize=11)
            ax_box.grid(True, alpha=0.3, axis='y')
            ax_box.tick_params(axis='x', rotation=45, labelsize=7)
        
        if idx == len(metrics) - 1:
            ax_line.set_xlabel('Timestamp', fontsize=11)
    
    plt.tight_layout()
    
    if output_path is None:
        first_file = Path(csv_files[0])
        output_path = first_file.parent / f"summary_{user_type}.png"
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info("Summary figure saved to: %s", output_path)
    except Exception as e:
        logger.error("Error: could not save the summary figure to %s: %s", output_path, e)
    
    plt.close()


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description='Visualize the performance metrics in CSV files')
    parser.add_argument('--path', type=str, help='Path to a CSV file or a directory containing CSV files')
    parser.add_argument('--output', type=str, default=None, 
                       help='Path to the output directory (optional, defaults to the directory containing the CSV file)')
    parser.add_argument('--sweep-ts', type=str, default=None,
                       help='Only process CSV files matching this timestamp prefix (format: YYYYMMDD_HHMMSS)')
    
    args = parser.parse_args()
    
    input_path = Path(args.path)
    if not input_path.exists():
        logger.error("Error: path does not exist: %s", input_path)
        sys.exit(1)
    
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    if input_path.is_dir():
        csv_files = collect_results_csvs(input_path, args.sweep_ts)
        if not csv_files:
            # Support the legacy layout: CSV files placed directly in the directory
            csv_files = list(input_path.glob('*.csv'))

        if not csv_files:
            logger.warning("Warning: no CSV files found in directory %s", input_path)
            sys.exit(0)

        logger.info("Found %d CSV files; generating summary figures...", len(csv_files))
        csv_file_paths = [str(f) for f in csv_files]

        if csv_file_paths:
            base_output = output_dir if output_dir else str(input_path)
            plot_summary_by_payload(csv_file_paths, 'normal', base_output)
            plot_summary_by_payload(csv_file_paths, 'malicious', base_output)
            logger.info("\nSummary figure generation complete!")
    
    elif input_path.is_file():
        if input_path.suffix.lower() != '.csv':
            logger.error("Error: file is not in CSV format: %s", input_path)
            sys.exit(1)
        
        if args.output:
            sub_dir = Path(args.output) / "sub"
            sub_dir.mkdir(parents=True, exist_ok=True)
            filename_stem = input_path.stem
            filename_without_ts = re.sub(r'^\d{8}_\d{6}__', '', filename_stem)
            if not filename_without_ts:
                filename_without_ts = filename_stem
            output_path = sub_dir / f"{filename_without_ts}.png"
        else:
            output_path = None
        
        plot_metrics(str(input_path), str(output_path) if output_path else None)
    
    else:
        logger.error("Error: path is neither a file nor a directory: %s", input_path)
        sys.exit(1)


if __name__ == '__main__':
    main()
