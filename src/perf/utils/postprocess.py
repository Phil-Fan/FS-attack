#!/usr/bin/env python3

from __future__ import annotations

import argparse
import bisect
import csv
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.sequence_processor import SequenceProcessor
from utils import visualizer
from utils.logging_utils import setup_logging

try:
    from visualize.summarize_sweep import calculate_waiting_ratio
except ImportError:
    try:
        visualize_path = Path(__file__).resolve().parent.parent / "visualize"
        if visualize_path.exists():
            if str(visualize_path) not in sys.path:
                sys.path.insert(0, str(visualize_path))
            from summarize_sweep import calculate_waiting_ratio
        else:
            calculate_waiting_ratio = None
    except ImportError:
        calculate_waiting_ratio = None

logger = logging.getLogger(__name__)


STEP_PATTERN = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - (?:scheduler - )?DEBUG - "
    r"\[step (?P<step>\d+)\] r_c: (?P<r_c>\{.*?\}) w_c: (?P<w_c>\{.*?\})(?: s_c: (?P<s_c>\{.*?\}))??; "
    r"(?:(?:kv:(?P<kv>[\d\.]+)%)|(?:gpu_kv:(?P<gpu_kv>[\d\.]+)%, cpu_kv:(?P<cpu_kv>[\d\.]+)%)); "
    r"computed: (?P<computed>\{.*?\}); cached: (?P<cached>\{.*?\})"
)


def print_info(msg: str) -> None:
    logger.info(msg)


def find_latest_scheduler_log(logs_dir: Path) -> Optional[Path]:
    if not logs_dir.exists():
        return None
    candidates = [p for p in logs_dir.glob("*.log") if not p.name.endswith("_server.log")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_kv_from_log(log_file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Log file does not exist: {log_file_path}")

    steps = []
    times = []
    kv_usage = []

    print_info(f"Parsing log file: {log_file_path}")

    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            step_match = STEP_PATTERN.match(line)
            if step_match:
                ts = datetime.strptime(step_match.group("time"), "%Y-%m-%d %H:%M:%S,%f")
                kv = step_match.group("kv")
                gpu_kv = step_match.group("gpu_kv")
                cpu_kv = step_match.group("cpu_kv")

                if kv is not None:
                    kv = float(kv)
                else:
                    gpu_kv = float(gpu_kv)
                    cpu_kv = float(cpu_kv)
                    kv = max(gpu_kv, cpu_kv)

                steps.append(int(step_match.group("step")))
                times.append(ts)
                kv_usage.append(kv)

    result = []
    for step, time_obj, kv in zip(steps, times, kv_usage):
        result.append({
            "step": step,
            "datetime": time_obj,
            "kv_usage": kv,
        })

    print_info(f"Extracted {len(result)} KV records")
    return result


class KVUsageAligner:
    def __init__(self, kv_data: Optional[List[Dict[str, Any]]] = None):
        self._kv_data: List[Dict[str, Any]] = []
        self._sorted_times: List[datetime] = []
        if kv_data:
            self._load_kv_data(kv_data)

    def _load_kv_data(self, kv_data: List[Dict[str, Any]]) -> None:
        valid_kv_data = []
        for entry in kv_data:
            time_obj = entry.get("datetime")
            if time_obj:
                valid_kv_data.append({
                    "datetime": time_obj,
                    "kv_usage": entry.get("kv_usage"),
                })

        valid_kv_data.sort(key=lambda x: x["datetime"])
        self._kv_data = valid_kv_data
        self._sorted_times = [entry["datetime"] for entry in valid_kv_data]

    @classmethod
    def from_log_file(cls, log_file_path: str) -> "KVUsageAligner":
        kv_data = extract_kv_from_log(log_file_path)
        return cls(kv_data)

    def find_closest_kv(self, target_time: datetime) -> Optional[float]:
        if not self._sorted_times:
            return None

        idx = bisect.bisect_left(self._sorted_times, target_time)
        if idx == 0:
            return self._kv_data[0]["kv_usage"]
        if idx >= len(self._sorted_times):
            return self._kv_data[-1]["kv_usage"]

        left = self._sorted_times[idx - 1]
        right = self._sorted_times[idx]
        if abs(target_time - left) <= abs(target_time - right):
            return self._kv_data[idx - 1]["kv_usage"]
        return self._kv_data[idx]["kv_usage"]


def align_itl_with_kv(
    itl_csv_path: Path,
    scheduler_log: Path,
    output_path: Path,
    aligner: KVUsageAligner | None = None,
) -> None:
    print_info("Aligning ITL data with KV cache")
    if aligner is None:
        aligner = KVUsageAligner.from_log_file(str(scheduler_log))
    aligned_rows = []

    with open(itl_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_str = row.get("time", "")
            if not time_str:
                continue
            try:
                if "," in time_str:
                    it = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f")
                else:
                    it = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")

                kv_usage = aligner.find_closest_kv(it)
                if kv_usage is None:
                    kv_usage = 0.0
                row["kv_usage"] = kv_usage
                aligned_rows.append(row)
            except ValueError:
                continue

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        if aligned_rows:
            writer = csv.DictWriter(f, fieldnames=aligned_rows[0].keys())
            writer.writeheader()
            writer.writerows(aligned_rows)

    print_info(f"Alignment complete: {len(aligned_rows)} records -> {output_path}")


def build_datasets(itl_aligned_path: Path, data_dir: Path, processor: SequenceProcessor) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.csv"
    eval_path = data_dir / "eval.csv"
    if train_path.exists() and eval_path.exists():
        print_info("  Skipped: train.csv / eval.csv already exist")
        try:
            return pd.read_csv(train_path), pd.read_csv(eval_path)
        except Exception:
            return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(itl_aligned_path)
    if df['kv_usage'].isna().any():
        df['kv_usage'] = df['kv_usage'].fillna(0)

    df['timestamp'] = pd.to_datetime(df['time'])
    df['time_seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    df = df[df['time_seconds'] <= processor.time_limit].copy()

    req_groups = df.groupby(['user_id', 'num_run'])

    sequences = []
    for (user_id, num_run), group in req_groups:
        group = group.sort_values('time_seconds')
        itl_values = group['itl_ms'].values
        kv_values = group['kv_usage'].values

        windows = processor._sliding_windows(itl_values, kv_values)
        for window_id, window in enumerate(windows):
            sequences.append({
                'user_id': user_id,
                'num_req': num_run,
                'window_id': window_id,
                'itl_sequence': window['itl'],
                'kv_usage': window['kv_usage'],
                'seq_length': len(window['itl']),
            })

    seq_df = pd.DataFrame(sequences)
    if seq_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    seq_df['kv_label'] = pd.cut(
        seq_df['kv_usage'],
        bins=processor.kv_bins,
        labels=range(processor.num_classes),
        include_lowest=True
    ).astype(int)

    np.random.seed(processor.random_seed)
    unique_users = seq_df['user_id'].unique()
    np.random.shuffle(unique_users)

    n_test = int(len(unique_users) * processor.test_ratio)
    test_users = unique_users[:n_test]
    train_users = unique_users[n_test:]

    train_df = seq_df[seq_df['user_id'].isin(train_users)]
    eval_df = seq_df[seq_df['user_id'].isin(test_users)]

    train_viz_df = processor._save_dataset_csv(train_df, train_path)
    eval_viz_df = processor._save_dataset_csv(eval_df, eval_path)

    return train_viz_df, eval_viz_df


def process_run(
    run_dir: Path,
    scheduler_log: Path,
    num_classes: int,
    time_limit: int,
    test_ratio: float,
    aligner: KVUsageAligner | None = None,
) -> None:
    itl_csv = run_dir / "logs" / "itl.csv"
    if not itl_csv.exists():
        print_info(f"Skipped (missing itl.csv): {run_dir}")
        return

    data_dir = run_dir / "data"
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    itl_aligned_path = data_dir / "itl_align.csv"
    if not itl_aligned_path.exists():
        align_itl_with_kv(itl_csv, scheduler_log, itl_aligned_path, aligner=aligner)
    else:
        print_info(f"  Skipped: {itl_aligned_path.name} already exists")

    processor = SequenceProcessor(
        time_limit=time_limit,
        test_ratio=test_ratio,
        num_classes=num_classes,
    )

    train_df, eval_df = build_datasets(itl_aligned_path, data_dir, processor)

    results_df = None
    results_csv = run_dir / "logs" / "results.csv"
    if results_csv.exists():
        try:
            results_df = pd.read_csv(results_csv)
            print_info(f"  Loaded TTFT data: {len(results_df)} rows")
        except Exception as exc:
            print_info(f"  Failed to load results.csv: {exc}")

    scheduler_log_path = scheduler_log if scheduler_log else None

    visualizer.plot_itl_analysis(itl_aligned_path, figures_dir, time_limit,
                                     results_df=results_df, scheduler_log_path=scheduler_log_path)

    if calculate_waiting_ratio is not None:
        try:
            waiting_ratios = calculate_waiting_ratio(scheduler_log, run_dir)
            if waiting_ratios:
                waiting_ratio_path = data_dir / "waiting_ratios.json"
                import json
                with open(waiting_ratio_path, "w", encoding="utf-8") as f:
                    json.dump(waiting_ratios, f, indent=2, ensure_ascii=False)
                print_info(f"  Queueing-time ratio saved: {waiting_ratio_path}")
                for user_type, ratio in waiting_ratios.items():
                    print_info(f"    {user_type}: {ratio:.4f}")
        except Exception as exc:
            print_info(f"  Queueing-time ratio calculation skipped: {exc}")

    print_info(f"Done: {run_dir}")


def collect_experiment_metadata(sweep_root: Path):
    import json

    experiments = []

    # Supports arbitrary prefix formats (e.g. 450s_*, 1800s_*, loopllm_qwen_*, etc.)
    for run_dir in sorted(sweep_root.iterdir()):
        if not run_dir.is_dir():
            continue
        # Skip non-experiment directories (e.g. figures, data, etc.)
        if not (run_dir / "logs").exists():
            continue

        meta_file = run_dir / "logs" / "experiment_meta.json"
        if not meta_file.exists():
            continue

        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
        except Exception as e:
            print_info(f"Warning: failed to read {meta_file}: {e}")
            continue

        # Examples: 1800s_mix_qwen_0_6b_75p, 450s_loopllm_qwen_0_6b_75p
        dir_name = run_dir.name
        match = re.match(r'^(\d+s_)?(\w+)_[^_]+_(\d+(?:p\d+)?)p$', dir_name)
        if match:
            # prefix = match.group(1)  # optional time prefix, e.g. "1800s_"
            strategy = match.group(2)
            ratio = match.group(3).replace('p', '.')
            label = f"{strategy}+{ratio}p"
        else:
            strategy = dir_name
            ratio = "unknown"
            label = dir_name

        start_time = datetime.fromisoformat(meta["start_time"])
        end_time = datetime.fromisoformat(meta["actual_end_time"])

        experiments.append({
            "start_time": start_time,
            "end_time": end_time,
            "strategy": strategy,
            "ratio": ratio,
            "label": label,
            "run_dir": run_dir
        })

    experiments.sort(key=lambda x: x["start_time"])
    return experiments


def generate_per_experiment_itl_plots(
    sweep_root: Path,
    scheduler_log: Path,
    num_classes: int = 10,
    time_limit: int = 1200,
    test_ratio: float = 0.2,
) -> None:

    experiments = collect_experiment_metadata(sweep_root)
    if not experiments:
        print_info(f"No experiments found")
        return

    print_info(f"Found {len(experiments)} experiments, generating an independent ITL visualization for each...")

    need_aligner = any(
        not (exp["run_dir"] / "data" / "itl_align.csv").exists()
        for exp in experiments
    )
    aligner = KVUsageAligner.from_log_file(str(scheduler_log)) if need_aligner else None

    for exp in experiments:
        exp_dir = exp["run_dir"]
        exp_label = exp["label"]
        exp_start = exp["start_time"]
        exp_end = exp["end_time"]

        print_info(f"\nProcessing experiment: {exp_label} ({exp_dir.name})")

        itl_csv = exp_dir / "logs" / "itl.csv"
        if not itl_csv.exists():
            print_info(f"  Skipped (missing itl.csv): {exp_dir}")
            continue

        data_dir = exp_dir / "data"
        exp_figures_dir = exp_dir / "figures"
        exp_figures_dir.mkdir(parents=True, exist_ok=True)

        itl_aligned_path = data_dir / "itl_align.csv"
        try:
            if itl_aligned_path.exists():
                print_info(f"  Skipped: {itl_aligned_path.name} already exists")
            else:
                align_itl_with_kv(itl_csv, scheduler_log, itl_aligned_path, aligner=aligner)
        except Exception as exc:
            print_info(f"  ITL alignment skipped: {exc}")
            continue

        # Use the configured nominal duration directly (do not use the actual duration)
        exp_time_limit = time_limit

        processor = SequenceProcessor(
            time_limit=exp_time_limit,
            test_ratio=test_ratio,
            num_classes=num_classes,
        )

        train_df, eval_df = build_datasets(itl_aligned_path, data_dir, processor)
        results_df = None
        results_csv = exp_dir / "logs" / "results.csv"
        if results_csv.exists():
            try:
                results_df = pd.read_csv(results_csv)
                print_info(f"  Loaded TTFT data: {len(results_df)} rows")
            except Exception as exc:
                print_info(f"  Failed to load results.csv: {exc}")
        visualizer.plot_itl_analysis(itl_aligned_path, exp_figures_dir, exp_time_limit,
                                        results_df=results_df, scheduler_log_path=scheduler_log)
        print_info(f"  ITL visualization: {exp_figures_dir / 'itl_fig1_timeseries.png'}")

    print_info(f"\nITL visualization generation complete for all experiments!")


def find_run_dirs(root: Path) -> List[Path]:
    run_dirs = set()
    for itl_csv in root.rglob("logs/itl.csv"):
        run_dirs.add(itl_csv.parent.parent)
    return sorted(run_dirs)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Perf postprocess for ITL alignment and visualization")
    parser.add_argument("--root", required=True, help="Run root directory")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--time-limit", type=int, default=1200)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--scheduler-log", default=None, help="Override scheduler log path")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    if args.scheduler_log:
        scheduler_log = Path(args.scheduler_log).resolve()
    else:
        scheduler_log = find_latest_scheduler_log(root / "logs")

    if not scheduler_log or not scheduler_log.exists():
        raise FileNotFoundError("Scheduler log file not found")

    run_dirs = find_run_dirs(root)
    if not run_dirs:
        raise FileNotFoundError(f"Run directory not found: {root}")

    print_info(f"Scheduler log: {scheduler_log}")
    print_info(f"Found {len(run_dirs)} run directories")

    need_aligner = any(
        not (run_dir / "data" / "itl_align.csv").exists()
        for run_dir in run_dirs
    )
    aligner = KVUsageAligner.from_log_file(str(scheduler_log)) if need_aligner else None

    for run_dir in run_dirs:
        process_run(
            run_dir,
            scheduler_log,
            args.num_classes,
            args.time_limit,
            args.test_ratio,
            aligner=aligner,
        )


if __name__ == "__main__":
    main()
logger = logging.getLogger(__name__)
