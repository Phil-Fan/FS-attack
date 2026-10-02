#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

try:
    from .paths import load_config
    from .sys_side import (
        parse_log_file,
        initialize_data_structures,
        step_pattern,
    )
except ImportError:
    import os
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paths import load_config
    from sys_side import (
        parse_log_file,
        initialize_data_structures,
        step_pattern,
    )


@dataclass
class SweepSummaryRow:

    run_id: str
    combo_id: str
    model_name: str
    duration: int
    total_users: int
    normal_user_count: int
    baseline_user_count: int
    malicious_user_count: int
    normal_dataset: str
    attack_method: str
    attack_strength: str
    malicious_ratio: float

    total_requests: int = 0
    successful_requests: int = 0
    overall_avg_latency: float = 0.0
    overall_avg_prompt_tokens: float = 0.0
    overall_avg_total_tokens: float = 0.0
    overall_goodput: float = 0.0
    overall_throughput_rps: float = 0.0

    normal_request_count: int = 0
    normal_avg_latency: float = 0.0
    normal_avg_ttft: float = 0.0
    normal_avg_tpot: float = 0.0
    normal_ttft_p99_ms: float = 0.0
    normal_tpot_p99_ms: float = 0.0
    normal_error_rate: float = 0.0
    normal_itl_mean_ms: float = 0.0
    normal_itl_p75_ms: float = 0.0
    normal_itl_p99_ms: float = 0.0
    normal_waiting_ratio: float = 0.0

    baseline_request_count: int = 0
    baseline_avg_latency: float = 0.0
    baseline_avg_ttft: float = 0.0
    baseline_avg_tpot: float = 0.0
    baseline_ttft_p99_ms: float = 0.0
    baseline_tpot_p99_ms: float = 0.0
    baseline_error_rate: float = 0.0
    baseline_itl_mean_ms: float = 0.0
    baseline_itl_p75_ms: float = 0.0
    baseline_itl_p99_ms: float = 0.0
    baseline_waiting_ratio: float = 0.0

    malicious_request_count: int = 0
    malicious_avg_latency: float = 0.0
    malicious_avg_ttft: float = 0.0
    malicious_avg_tpot: float = 0.0
    malicious_error_rate: float = 0.0
    malicious_waiting_ratio: float = 0.0

    preemption_count: int = 0
    preempted_normal_count: int = 0
    preempted_baseline_count: int = 0
    preempted_malicious_count: int = 0

    malicious_user_cost: float = 0.0
    cost_metric_type: str = "tokens"

    status: str = "pending"
    start_time: str = ""
    end_time: str = ""
    actual_duration_seconds: float = 0.0


def _mean_or_nan(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.mean())


def _percentile_or_nan(series: pd.Series, q: float) -> float:
    if series.empty:
        return float("nan")
    return float(np.nanpercentile(series, q))


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def load_config_yaml(config_path: Path) -> Optional[Dict]:
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def extract_config_info(run_dir: Path) -> Dict:
    cfg = load_config_yaml(run_dir / "config.yaml")
    if not cfg:
        return {}

    users = cfg.get("users", {})
    datasets = cfg.get("datasets", {})
    sweep = cfg.get("sweep", {})
    server = cfg.get("server", {})

    total = users.get("total", 20)
    ratios = users.get("ratios", {})

    baseline_count = int(total * ratios.get("baseline", 0))
    attack_count = int(total * ratios.get("attack", 0))
    malicious_count = baseline_count + attack_count
    normal_count = total - malicious_count

    normal_ds = datasets.get("normal", {})
    if isinstance(normal_ds, dict):
        normal_dataset = normal_ds.get("file", "unknown")
    else:
        normal_dataset = str(normal_ds) if normal_ds else "unknown"

    attack_strength = sweep.get("special_dataset", "none")

    model_name = server.get("model_path", "unknown")
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    return {
        "run_id": run_dir.name,
        "combo_id": sweep.get("combo_id", run_dir.name),
        "model_name": model_name,
        "duration": sweep.get("duration", 0),
        "total_users": total,
        "normal_user_count": normal_count,
        "baseline_user_count": baseline_count,
        "malicious_user_count": malicious_count,
        "normal_dataset": normal_dataset,
        "attack_method": sweep.get("special_type", "") or "none",
        "attack_strength": attack_strength,
        "malicious_ratio": ratios.get("baseline", 0) + ratios.get("attack", 0),
    }


def load_experiment_meta(run_dir: Path) -> Dict:
    meta_file = run_dir / "logs" / "experiment_meta.json"
    if not meta_file.exists():
        return {}
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def parse_user_statistics(results_csv: Path) -> Dict:
    if not results_csv.exists():
        return {}

    df = pd.read_csv(results_csv)
    if df.empty:
        return {}

    required_cols = ["user_type", "latency"]
    if not all(col in df.columns for col in required_cols):
        return {}

    df = df[df["user_type"].astype(str).str.lower() != "warmup"]

    for col in ["latency", "ttft", "tpot", "prompt_tokens", "completion_tokens", "total_tokens"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "error" in df.columns:
        err = df["error"].fillna("").astype(str).str.lower().str.strip()
        success_mask = (err == "") | (err == "none") | (err == "nan")
    else:
        success_mask = pd.Series([True] * len(df), index=df.index)

    df_success = df[success_mask].copy()
    df_failed = df[~success_mask].copy()

    total_count = len(df)
    success_count = len(df_success)

    goodput = _safe_div(success_count, total_count)
    throughput_rps = 0.0
    if "latency" in df_success.columns:
        if "num_run" in df_success.columns:
            throughput_rps = total_count / df_success["latency"].sum() if df_success["latency"].sum() > 0 else 0.0
        else:
            throughput_rps = success_count / df_success["latency"].sum() if df_success["latency"].sum() > 0 else 0.0

    overall_stats = {
        "total_requests": total_count,
        "successful_requests": success_count,
        "overall_avg_latency": _mean_or_nan(df_success.get("latency", pd.Series(dtype=float))) * 1000,
        "overall_goodput": goodput,
        "overall_throughput_rps": throughput_rps,
    }
    if "prompt_tokens" in df_success.columns:
        overall_stats["overall_avg_prompt_tokens"] = _mean_or_nan(df_success["prompt_tokens"].dropna())
    if "total_tokens" in df_success.columns:
        overall_stats["overall_avg_total_tokens"] = _mean_or_nan(df_success["total_tokens"].dropna())

    result = {"overall": overall_stats}
    user_types = df["user_type"].astype(str).str.lower().unique()

    for user_type in user_types:
        type_df = df[df["user_type"].astype(str).str.lower() == user_type]
        type_success = type_df[success_mask[type_df.index]]
        type_total = len(type_df)
        type_success_count = len(type_success)

        type_key = f"{user_type}_"
        type_stats = {
            f"{type_key}request_count": type_total,
            f"{type_key}error_rate": _safe_div(type_total - type_success_count, type_total),
        }

        if "latency" in type_success.columns:
            type_stats[f"{type_key}avg_latency"] = _mean_or_nan(type_success["latency"]) * 1000

        if "ttft" in type_success.columns:
            type_stats[f"{type_key}avg_ttft"] = _mean_or_nan(type_success["ttft"])
            type_stats[f"{type_key}ttft_p99_ms"] = _percentile_or_nan(type_success["ttft"], 99)

        if "tpot" in type_success.columns:
            type_stats[f"{type_key}avg_tpot"] = _mean_or_nan(type_success["tpot"])
            type_stats[f"{type_key}tpot_p99_ms"] = _percentile_or_nan(type_success["tpot"], 99)

        result[user_type] = type_stats

    return result


def parse_itl_stats(itl_csv: Path) -> Dict:
    if not itl_csv.exists():
        return {}

    df = pd.read_csv(itl_csv)
    if df.empty or "itl_ms" not in df.columns:
        return {}

    df["itl_ms"] = pd.to_numeric(df["itl_ms"], errors="coerce")
    df["user_type"] = df["user_type"].astype(str).str.lower()

    df = df.dropna(subset=["itl_ms"])
    df = df[df["itl_ms"] > 0]

    result = {}
    user_types = df["user_type"].unique()

    for user_type in user_types:
        type_df = df[df["user_type"] == user_type]
        type_key = f"{user_type}_"
        result[f"{type_key}itl_mean_ms"] = _mean_or_nan(type_df["itl_ms"])
        result[f"{type_key}itl_p75_ms"] = _percentile_or_nan(type_df["itl_ms"], 75)
        result[f"{type_key}itl_p99_ms"] = _percentile_or_nan(type_df["itl_ms"], 99)

    return result


def parse_preemption_events(log_file: Path, start_time, end_time) -> Dict:
    if not log_file.exists():
        return {
            "preemption_count": 0,
            "preempted_normal_count": 0,
            "preempted_baseline_count": 0,
            "preempted_malicious_count": 0,
        }

    import re
    from datetime import datetime

    log_pattern = re.compile(
        r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*'
        r'\[(?P<event>preempted|preempt_recompute|preempt_swap)\]'
        r'.*'
        r'\[type: (?P<user_type>\w+)\]'
    )

    preempted_normal = 0
    preempted_baseline = 0
    preempted_malicious = 0
    malicious_types = {"baseline", "attack", "malicious"}

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                try:
                    log_time = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S,%f")
                except ValueError:
                    continue

                if log_time < start_time or log_time > end_time:
                    continue

                user_type = match.group("user_type")
                if user_type == "normal":
                    preempted_normal += 1
                elif user_type == "baseline":
                    preempted_baseline += 1
                elif user_type in malicious_types:
                    preempted_malicious += 1

    return {
        "preemption_count": preempted_normal + preempted_baseline + preempted_malicious,
        "preempted_normal_count": preempted_normal,
        "preempted_baseline_count": preempted_baseline,
        "preempted_malicious_count": preempted_malicious,
    }


def _get_user_counts_from_results(results_csv: Path) -> Dict[str, int]:
    if not results_csv.exists():
        return {}
    
    try:
        df = pd.read_csv(results_csv)
        if df.empty or "user_type" not in df.columns or "user_id" not in df.columns:
            return {}
        
        df["user_type"] = df["user_type"].astype(str).str.lower().str.strip()
        df = df[df["user_type"] != "warmup"]
        
        user_counts = {}
        for user_type in df["user_type"].unique():
            type_df = df[df["user_type"] == user_type]
            unique_users = type_df["user_id"].nunique()
            user_counts[user_type] = unique_users
        
        return user_counts
    except Exception:
        return {}


def calculate_waiting_ratio(log_file: Path, run_dir: Path) -> Dict:
    if not log_file.exists():
        return {}

    config_info = extract_config_info(run_dir)
    user_counts = {
        "normal": config_info.get("normal_user_count", 0),
        "baseline": config_info.get("baseline_user_count", 0),
        "malicious": config_info.get("malicious_user_count", 0),
    }

    results_csv = run_dir / "logs" / "results.csv"
    if results_csv.exists():
        results_user_counts = _get_user_counts_from_results(results_csv)
        for user_type, count in results_user_counts.items():
            if user_counts.get(user_type, 0) == 0:
                user_counts[user_type] = count

    data_structures = initialize_data_structures()
    parse_log_file(str(log_file), data_structures)

    times = data_structures["times"]
    wc_data = data_structures["wc_data"]

    if not times or len(times) < 2:
        return {}

    intervals = []
    for i in range(len(times) - 1):
        interval = (times[i + 1] - times[i]).total_seconds()
        intervals.append(interval)

    total_duration = sum(intervals)
    if total_duration <= 0:
        return {}

    waiting_time = {}
    for i in range(len(wc_data)):
        if i < len(intervals):
            interval = intervals[i]
            for user_type, count in wc_data[i].items():
                if user_type == "warmup":
                    continue
                waiting_time[user_type] = waiting_time.get(user_type, 0) + count * interval

    result = {}
    for user_type, time_sum in waiting_time.items():
        user_count = user_counts.get(user_type, 0)
        
        if user_type == "attack" and user_count == 0:
            user_count = max(0, user_counts.get("malicious", 0) - user_counts.get("baseline", 0))
        
        if user_count > 0:
            avg_waiting_time = _safe_div(time_sum, user_count)
            result[f"{user_type}_waiting_ratio"] = _safe_div(avg_waiting_time, total_duration)
        else:
            result[f"{user_type}_waiting_ratio"] = _safe_div(time_sum, total_duration)

    return result


def calculate_malicious_cost(results_csv: Path) -> float:
    if not results_csv.exists():
        return 0.0

    df = pd.read_csv(results_csv)
    df["user_type"] = df["user_type"].astype(str).str.lower()
    df = df[df["user_type"] != "warmup"]

    malicious_df = df[df["user_type"].isin(["baseline", "attack", "malicious"])]

    if len(malicious_df) == 0:
        return 0.0

    # INPUT_PRICE_PER_1K = 0.00025  # Input token price
    # OUTPUT_PRICE_PER_1K = 0.002   # Output token price
    
    INPUT_PRICE_PER_1K = 0.00015
    OUTPUT_PRICE_PER_1K = 0.0006

    input_tokens = malicious_df["prompt_tokens"].sum()
    output_tokens = malicious_df["completion_tokens"].sum()
    total_cost = (input_tokens / 1000) * INPUT_PRICE_PER_1K + (output_tokens / 1000) * OUTPUT_PRICE_PER_1K
    return total_cost


def summarize_experiment(run_dir: Path, log_file: Optional[Path] = None) -> Optional[SweepSummaryRow]:
    config_info = extract_config_info(run_dir)
    if not config_info:
        print(f"Could not read configuration: {run_dir}")
        return None

    meta = load_experiment_meta(run_dir)
    actual_duration = meta.get("actual_duration_seconds", 0.0)

    results_csv = run_dir / "logs" / "results.csv"
    user_stats = parse_user_statistics(results_csv)

    itl_csv = run_dir / "logs" / "itl.csv"
    itl_stats = parse_itl_stats(itl_csv)

    preemption_stats = {"preemption_count": 0, "preempted_normal_count": 0,
                        "preempted_baseline_count": 0, "preempted_malicious_count": 0}
    if log_file and log_file.exists():
        start_time = pd.to_datetime(meta.get("start_time", ""))
        end_time = pd.to_datetime(meta.get("actual_end_time", ""))
        if pd.notna(start_time) and pd.notna(end_time):
            preemption_stats = parse_preemption_events(log_file, start_time, end_time)

    waiting_stats = {}
    if not log_file:
        sweep_dir = run_dir.parent
        log_files = list((sweep_dir / "logs").glob("*.log"))
        if log_files:
            log_file = log_files[0]
    if log_file and log_file.exists():
        waiting_stats = calculate_waiting_ratio(log_file, run_dir)

    cost = calculate_malicious_cost(results_csv)

    start_time = ""
    end_time = ""
    if results_csv.exists():
        stat = results_csv.stat()
        from datetime import datetime
        end_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        start_time = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")

    all_stats = {
        **config_info,
        **user_stats.get("overall", {}),
    }

    for user_type in ["normal", "baseline", "malicious"]:
        type_stats = user_stats.get(user_type, {})
        for key, value in type_stats.items():
            all_stats[key] = value

    all_stats.update(itl_stats)

    all_stats.update(waiting_stats)

    all_stats.update(preemption_stats)

    all_stats["malicious_user_cost"] = cost
    all_stats["status"] = "completed"
    all_stats["start_time"] = start_time
    all_stats["end_time"] = end_time
    all_stats["actual_duration_seconds"] = actual_duration

    valid_fields = set(SweepSummaryRow.__dataclass_fields__.keys())

    filtered_stats = {k: v for k, v in all_stats.items() if k in valid_fields}

    return SweepSummaryRow(**filtered_stats)


def generate_summary_table(root_dir: Path, output_path: Optional[Path] = None,
                          recursive: bool = False) -> pd.DataFrame:
    rows: List[SweepSummaryRow] = []

    if (root_dir / "config.yaml").exists():
        log_file = list((root_dir.parent / "logs").glob("*.log")) if (root_dir.parent / "logs").exists() else None
        if not log_file:
            log_file = list((root_dir / "logs").glob("*.log")) if (root_dir / "logs").exists() else None
        row = summarize_experiment(root_dir, log_file[0] if log_file else None)
        if row:
            rows.append(row)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame([vars(row)])
            df.to_csv(output_path, index=False)
            print(f"Summary table saved: {output_path}")
            print(f"Summarized {len(rows)} experiments")
        return pd.DataFrame([vars(row)]) if row else pd.DataFrame()

    sweep_root = root_dir
    log_file = None

    if (root_dir / "logs").exists():
        log_files = list((root_dir / "logs").glob("*.log"))
        if log_files:
            log_file = log_files[0]

    experiment_dirs = []
    for item in sorted(root_dir.iterdir()):
        if item.is_dir():
            if (item / "config.yaml").exists():
                experiment_dirs.append(item)
            elif (item / "logs").exists() and recursive:
                sub_experiments = [d for d in item.iterdir() if d.is_dir() and (d / "config.yaml").exists()]
                experiment_dirs.extend(sub_experiments)

    if not experiment_dirs and recursive:
        for item in sorted(root_dir.rglob("*")):
            if item.is_dir() and (item / "config.yaml").exists():
                experiment_dirs.append(item)

    experiment_dirs = sorted(set(experiment_dirs), key=lambda x: x.name)

    print(f"Found {len(experiment_dirs)} experiment directories")

    for exp_dir in experiment_dirs:
        print(f"  Processing: {exp_dir.name}")
        row = summarize_experiment(exp_dir, log_file)
        if row:
            rows.append(row)

    if not rows:
        print("No valid experiment data found")
        return pd.DataFrame()

    data = [vars(r) for r in rows]
    df = pd.DataFrame(data)

    if not df.empty:
        sort_cols = []
        if "attack_method" in df.columns:
            sort_cols.append("attack_method")
        if "malicious_ratio" in df.columns:
            sort_cols.append("malicious_ratio")
        if "combo_id" in df.columns:
            sort_cols.append("combo_id")
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Summary table saved: {output_path}")

    print(f"Summarized {len(df)} experiments")
    return df


def update_existing_table(run_dir: Path, table_path: Path) -> None:
    if not table_path.exists():
        return

    df = pd.read_csv(table_path)

    sweep_dir = run_dir.parent
    log_file = None
    log_files = list((sweep_dir / "logs").glob("*.log"))
    if log_files:
        log_file = log_files[0]

    row = summarize_experiment(run_dir, log_file)
    if not row:
        return

    row_dict = vars(row)
    run_id = row_dict["run_id"]

    if "run_id" in df.columns and run_id in df["run_id"].values:
        idx = df[df["run_id"] == run_id].index[0]
        for col, val in row_dict.items():
            df.at[idx, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)

    df.to_csv(table_path, index=False)
    print(f"Updated: {run_id}")


def get_field_names() -> List[str]:
    row = SweepSummaryRow(
        run_id="example",
        combo_id="example",
        model_name="example",
        duration=1800,
        total_users=20,
        normal_user_count=15,
        baseline_user_count=5,
        malicious_user_count=0,
        normal_dataset="example",
        attack_method="none",
        attack_strength="none",
        malicious_ratio=0.0,
    )
    return list(vars(row).keys())


def create_empty_template(output_path: Path, run_ids: Optional[List[str]] = None) -> None:
    fields = get_field_names()
    df = pd.DataFrame(columns=fields)

    if run_ids:
        for run_id in run_ids:
            row = {f: "" for f in fields}
            row["run_id"] = run_id
            row["status"] = "pending"
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Template table created: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep experiment summarization script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", type=Path, help="Sweep root directory or experiment directory")
    parser.add_argument("-o", "--output", type=Path, help="Output file path")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Recursively search for experiments in subdirectories")
    parser.add_argument("--update", action="store_true",
                        help="Update the existing table (append/update rows)")
    parser.add_argument("--template", action="store_true",
                        help="Create an empty template table")

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Path does not exist: {args.path}")
        sys.exit(1)

    output_path = args.output
    if not output_path:
        if args.template:
            output_path = args.path / "sweep_summary_template.csv"
        elif (args.path / "sweep_summary.csv").exists():
            output_path = args.path / "sweep_summary.csv"
        else:
            output_path = args.path / "sweep_summary.csv"

    if args.template:
        run_ids = None
        if (args.path / "config.yaml").exists():
            run_ids = [args.path.name]
        else:
            run_ids = [d.name for d in args.path.iterdir() if d.is_dir() and (d / "config.yaml").exists()]
        create_empty_template(output_path, run_ids)
    elif args.update:
        if (args.path / "config.yaml").exists():
            update_existing_table(args.path, output_path)
        else:
            print("--update mode requires a single experiment directory")
            sys.exit(1)
    else:
        df = generate_summary_table(args.path, output_path, args.recursive)
        if not df.empty:
            print(f"\nTable fields ({len(df.columns)}):")
            for i, col in enumerate(sorted(df.columns), 1):
                print(f"  {i:2d}. {col}")


if __name__ == "__main__":
    main()
