#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import yaml
import re
from datetime import datetime

try:
    from .paths import load_config
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paths import load_config


@dataclass
class SweepSummaryRow:
    run_id: str
    combo_id: str
    model_name: str
    duration: int
    total_users: int
    normal_user_count: int
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
    normal_request_count: int = 0
    normal_avg_latency: float = 0.0
    normal_avg_ttft: float = 0.0
    normal_avg_tpot: float = 0.0
    malicious_request_count: int = 0
    malicious_avg_latency: float = 0.0
    malicious_avg_ttft: float = 0.0
    malicious_avg_tpot: float = 0.0
    preemption_count: int = 0
    preempted_normal_count: int = 0
    preempted_malicious_count: int = 0
    malicious_user_cost: float = 0.0
    cost_metric_type: str = "tokens"
    status: str = "pending"
    start_time: str = ""
    end_time: str = ""


LEGACY_SWEEP_RESULTS_COLUMNS = [
    "run_id",
    "normal_user_count",
    "normal_dataset",
    "malicious_user_count",
    "attack_method",
    "total_requests",
    "avg_latency_s",
    "avg_prompt_tokens",
    "avg_total_tokens",
    "normal_request_count",
    "normal_avg_latency_s",
    "normal_avg_ttft_s",
    "normal_avg_tpot_s",
    "malicious_request_count",
    "malicious_avg_latency_s",
    "malicious_avg_ttft_s",
    "malicious_avg_tpot_s",
    "preemption_count",
    "preempted_normal_count",
    "preempted_malicious_count",
    "malicious_user_cost",
    "status",
    "start_time",
    "end_time",
]


def extract_config_info(run_dir: Path) -> Dict:
    cfg = load_config(run_dir)
    if not cfg:
        return {}

    users = cfg.get("users", {})
    datasets = cfg.get("datasets", {})
    sweep = cfg.get("sweep", {})
    server = cfg.get("server", {})

    total = users.get("total", 20)
    ratios = users.get("ratios", {})
    special_type = sweep.get("special_type", "")
    special_ratio = ratios.get(special_type if special_type in ratios else "attack", 0)

    malicious_count = 0
    if "baseline" in ratios:
        malicious_count = int(total * ratios["baseline"])
    elif "attack" in ratios:
        malicious_count = int(total * ratios["attack"])

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
        "malicious_user_count": malicious_count,
        "normal_dataset": normal_dataset,
        "attack_method": special_type or "none",
        "attack_strength": attack_strength,
        "malicious_ratio": special_ratio,
    }


def _mean_or_nan(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return series.mean()


def _safe_div(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def calculate_user_statistics(results_csv: Path) -> Dict:
    if not results_csv.exists():
        return {}

    df = pd.read_csv(results_csv)

    # Ensure the columns exist
    required_cols = ["user_type", "latency"]
    if not all(col in df.columns for col in required_cols):
        return {}

    if "error" in df.columns:
        err = df["error"].fillna("").astype(str).str.lower().str.strip()
        success_mask = (err == "") | (err == "none") | (err == "nan")
        df_success = df[success_mask].copy()
    else:
        df_success = df.copy()

    df_success["user_type"] = df_success["user_type"].astype(str).str.lower()
    df_success = df_success[df_success["user_type"] != "warmup"]

    for col in ["latency", "ttft", "tpot", "prompt_tokens", "total_tokens"]:
        if col in df_success.columns:
            df_success[col] = pd.to_numeric(df_success[col], errors="coerce")

    normal_df = df_success[df_success["user_type"] == "normal"]
    normal_stats = {
        "normal_request_count": len(normal_df),
        "normal_avg_latency": _mean_or_nan(normal_df.get("latency", pd.Series(dtype=float))),
    }
    if "ttft" in normal_df.columns:
        normal_stats["normal_avg_ttft"] = _mean_or_nan(normal_df["ttft"])
    if "tpot" in normal_df.columns:
        normal_stats["normal_avg_tpot"] = _mean_or_nan(normal_df["tpot"])

    malicious_df = df_success[df_success["user_type"].isin(["baseline", "attack", "malicious"])]
    malicious_stats = {
        "malicious_request_count": len(malicious_df),
        "malicious_avg_latency": _mean_or_nan(malicious_df.get("latency", pd.Series(dtype=float))),
    }
    if "ttft" in malicious_df.columns:
        malicious_stats["malicious_avg_ttft"] = _mean_or_nan(malicious_df["ttft"])
    if "tpot" in malicious_df.columns:
        malicious_stats["malicious_avg_tpot"] = _mean_or_nan(malicious_df["tpot"])

    overall_stats = {
        "total_requests": len(df_success),
        "successful_requests": len(df_success),
        "overall_avg_latency": _mean_or_nan(df_success.get("latency", pd.Series(dtype=float))),
        "overall_goodput": _safe_div(len(df_success), len(df)),
    }
    if "prompt_tokens" in df_success.columns:
        overall_stats["overall_avg_prompt_tokens"] = _mean_or_nan(
            df_success["prompt_tokens"].dropna()
        )
    if "total_tokens" in df_success.columns:
        overall_stats["overall_avg_total_tokens"] = _mean_or_nan(
            df_success["total_tokens"].dropna()
        )

    return {**normal_stats, **malicious_stats, **overall_stats}


def _resolve_experiment_time_range(results_csv: Path) -> tuple[datetime, datetime] | tuple[None, None]:
    meta_file = results_csv.parent / "experiment_meta.json"
    if meta_file.exists():
        import json

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        start_time = datetime.fromisoformat(meta["start_time"])
        end_time = datetime.fromisoformat(meta["actual_end_time"])
        return start_time, end_time

    if results_csv.exists():
        try:
            df = pd.read_csv(results_csv)
            if "time" in df.columns:
                times = pd.to_datetime(df["time"], errors="coerce")
                times = times.dropna()
                if not times.empty:
                    return times.min().to_pydatetime(), times.max().to_pydatetime()
        except Exception:
            pass

        stat = results_csv.stat()
        return (
            datetime.fromtimestamp(stat.st_ctime),
            datetime.fromtimestamp(stat.st_mtime),
        )

    return None, None


def parse_preemption_events(log_file: Path, results_csv: Path) -> Dict:
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return {
            "preemption_count": 0,
            "preempted_normal_count": 0,
            "preempted_malicious_count": 0,
        }

    start_time, end_time = _resolve_experiment_time_range(results_csv)
    if start_time is None or end_time is None:
        print(f"Could not determine the experiment time range, skipping preemption statistics: {results_csv}")
        return {
            "preemption_count": 0,
            "preempted_normal_count": 0,
            "preempted_malicious_count": 0,
        }

    from .summarize_sweep import parse_preemption_events as parse_preemption_events_in_window

    stats = parse_preemption_events_in_window(log_file, start_time, end_time)
    return {
        "preemption_count": stats.get("preemption_count", 0),
        "preempted_normal_count": stats.get("preempted_normal_count", 0),
        "preempted_malicious_count": stats.get("preempted_baseline_count", 0)
        + stats.get("preempted_malicious_count", 0),
    }


def calculate_malicious_cost(results_csv: Path) -> float:
    if not results_csv.exists():
        return 0.0

    df = pd.read_csv(results_csv)
    df["user_type"] = df["user_type"].astype(str).str.lower()
    df = df[df["user_type"] != "warmup"]

    malicious_df = df[df["user_type"].isin(["baseline", "attack", "malicious"])]

    if len(malicious_df) == 0:
        return 0.0

    INPUT_PRICE_PER_1K = 0.00025
    OUTPUT_PRICE_PER_1K = 0.002

    input_tokens=  malicious_df["prompt_tokens"].sum()
    output_tokens= malicious_df["completion_tokens"].sum()
    total_cost = (input_tokens / 1000) * INPUT_PRICE_PER_1K + (output_tokens / 1000) * OUTPUT_PRICE_PER_1K
    return total_cost


def initialize_sweep_table(sweep_dir: Path, rows_data: List[Dict]) -> Path:
    output_path = sweep_dir / "sweep_results.csv"

    columns = LEGACY_SWEEP_RESULTS_COLUMNS

    df = pd.DataFrame(columns=columns)

    for row_data in rows_data:
        df.loc[len(df)] = {
            "run_id": row_data.get("run_id", ""),
            "normal_user_count": row_data.get("normal_user_count", 0),
            "normal_dataset": row_data.get("normal_dataset", ""),
            "malicious_user_count": row_data.get("malicious_user_count", 0),
            "attack_method": row_data.get("attack_method", ""),
            "total_requests": "",
            "avg_latency_s": "",
            "avg_prompt_tokens": "",
            "avg_total_tokens": "",
            "normal_request_count": "",
            "normal_avg_latency_s": "",
            "normal_avg_ttft_s": "",
            "normal_avg_tpot_s": "",
            "malicious_request_count": "",
            "malicious_avg_latency_s": "",
            "malicious_avg_ttft_s": "",
            "malicious_avg_tpot_s": "",
            "preemption_count": "",
            "preempted_normal_count": "",
            "preempted_malicious_count": "",
            "malicious_user_cost": "",
            "status": "pending",
            "start_time": "",
            "end_time": "",
        }

    df.to_csv(output_path, index=False)
    print(f"Sweep table initialized: {output_path}")
    return output_path


def _legacy_sweep_results_path(sweep_dir: Path) -> Path:
    return sweep_dir / "sweep_results.csv"


def _load_or_init_legacy_table(table_path: Path) -> pd.DataFrame:
    if table_path.exists():
        return pd.read_csv(table_path)
    return pd.DataFrame(columns=LEGACY_SWEEP_RESULTS_COLUMNS)


def update_sweep_table_row(run_dir: Path) -> None:
    sweep_dir = run_dir.parent
    table_path = _legacy_sweep_results_path(sweep_dir)
    legacy_tmp_path = sweep_dir / "sweep_results_tmp.csv"

    df = _load_or_init_legacy_table(table_path)
    run_id = run_dir.name

    config_info = extract_config_info(run_dir)
    if not config_info:
        return

    results_csv = run_dir / "logs" / "results.csv"
    stats = calculate_user_statistics(results_csv)

    log_files = list((sweep_dir / "logs").glob("*.log"))
    log_file = log_files[0] if log_files else None
    if log_file and log_file.exists():
        preemption_stats = parse_preemption_events(log_file, results_csv)
    else:
        preemption_stats = {"preemption_count": 0, "preempted_normal_count": 0, "preempted_malicious_count": 0}

    cost = calculate_malicious_cost(results_csv)

    start_time = ""
    end_time = ""
    if results_csv.exists():
        import os
        stat = results_csv.stat()
        end_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        start_time = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")

    update_data = {
        "total_requests": stats.get("total_requests", 0),
        "avg_latency_s": round(stats.get("overall_avg_latency", 0), 3),
        "avg_prompt_tokens": round(stats.get("overall_avg_prompt_tokens", 0), 3),
        "avg_total_tokens": round(stats.get("overall_avg_total_tokens", 0), 3),
        "normal_request_count": stats.get("normal_request_count", 0),
        "normal_avg_latency_s": round(stats.get("normal_avg_latency", 0), 3),
        "normal_avg_ttft_s": round(stats.get("normal_avg_ttft", 0), 3),
        "normal_avg_tpot_s": round(stats.get("normal_avg_tpot", 0), 3),
        "malicious_request_count": stats.get("malicious_request_count", 0),
        "malicious_avg_latency_s": round(stats.get("malicious_avg_latency", 0), 3),
        "malicious_avg_ttft_s": round(stats.get("malicious_avg_ttft", 0), 3),
        "malicious_avg_tpot_s": round(stats.get("malicious_avg_tpot", 0), 3),
        "preemption_count": preemption_stats.get("preemption_count", 0),
        "preempted_normal_count": preemption_stats.get("preempted_normal_count", 0),
        "preempted_malicious_count": preemption_stats.get("preempted_malicious_count", 0),
        "malicious_user_cost": round(cost, 6),
        "status": "completed",
        "start_time": start_time,
        "end_time": end_time,
    }

    # Update the matching row - delete the old row first, then reinsert (to avoid type conversion issues)
    df = df[df["run_id"] != run_id]

    new_row = {
        "run_id": run_id,
        "normal_user_count": config_info.get("normal_user_count", 0),
        "normal_dataset": config_info.get("normal_dataset", ""),
        "malicious_user_count": config_info.get("malicious_user_count", 0),
        "attack_method": config_info.get("attack_method", ""),
    }
    new_row.update(update_data)

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(table_path, index=False)
    if legacy_tmp_path.exists():
        df.to_csv(legacy_tmp_path, index=False)
    print(f"Updated table row: {run_id}")


def generate_row_for_run(run_dir: Path) -> Optional[SweepSummaryRow]:
    config_info = extract_config_info(run_dir)
    if not config_info:
        return None

    results_csv = run_dir / "logs" / "results.csv"
    stats = calculate_user_statistics(results_csv)

    sweep_dir = run_dir.parent
    log_files = list((sweep_dir / "logs").glob("*.log"))
    log_file = log_files[0] if log_files else None
    if log_file and log_file.exists():
        preemption_stats = parse_preemption_events(log_file, results_csv)
    else:
        preemption_stats = {"preemption_count": 0, "preempted_normal_count": 0, "preempted_malicious_count": 0}

    cost = calculate_malicious_cost(results_csv)

    start_time = ""
    end_time = ""
    if results_csv.exists():
        stat = results_csv.stat()
        end_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        start_time = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")

    return SweepSummaryRow(
        **config_info,
        **stats,
        **preemption_stats,
        malicious_user_cost=cost,
        status="completed",
        start_time=start_time,
        end_time=end_time,
    )


def generate_sweep_summary(sweep_dir: Path) -> pd.DataFrame:
    rows = []

    for item in sorted(sweep_dir.iterdir()):
        run_dir = item
        if run_dir.is_dir() and (run_dir / "config.yaml").exists():
            row = generate_row_for_run(run_dir)
            if row:
                rows.append(row)

    if not rows:
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

    return df


def save_sweep_summary(df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sweep summary table saved: {output_path}")


def refresh_sweep_summary(sweep_dir: Path):
    df = generate_sweep_summary(sweep_dir)
    if df.empty:
        print(f"No valid experiment data found: {sweep_dir}")
        return

    output_path = sweep_dir / "sweep_summary.csv"
    save_sweep_summary(df, output_path)
    print(f"Summarized {len(df)} sub-experiments")


def update_sweep_summary_row(run_dir: Path):
    sweep_dir = run_dir.parent
    summary_path = sweep_dir / "sweep_summary.csv"

    new_row = generate_row_for_run(run_dir)
    if not new_row:
        return

    new_row_dict = vars(new_row)

    if summary_path.exists():
        df = pd.read_csv(summary_path)
        run_id = new_row_dict["run_id"]

        if run_id in df["run_id"].values:
            idx = df[df["run_id"] == run_id].index[0]
            for col, val in new_row_dict.items():
                df.at[idx, col] = val
        else:
            df = pd.concat([df, pd.DataFrame([new_row_dict])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row_dict])

    save_sweep_summary(df, summary_path)


def update_sweep_tables(run_dir: Path) -> None:
    update_sweep_table_row(run_dir)
    update_sweep_summary_row(run_dir)


def rebuild_sweep_tables(sweep_dir: Path) -> None:
    sweep_dir = sweep_dir.resolve()
    run_dirs = sorted(
        item for item in sweep_dir.iterdir()
        if item.is_dir() and (item / "config.yaml").exists()
    )

    if not run_dirs:
        print(f"No valid experiment directories found: {sweep_dir}")
        return

    rows_data = []
    for run_dir in run_dirs:
        config_info = extract_config_info(run_dir)
        if not config_info:
            continue
        rows_data.append(
            {
                "run_id": run_dir.name,
                "normal_user_count": config_info.get("normal_user_count", 0),
                "normal_dataset": config_info.get("normal_dataset", ""),
                "malicious_user_count": config_info.get("malicious_user_count", 0),
                "attack_method": config_info.get("attack_method", ""),
            }
        )

    initialize_sweep_table(sweep_dir, rows_data)
    for run_dir in run_dirs:
        update_sweep_table_row(run_dir)

    refresh_sweep_summary(sweep_dir)
