#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

try:
    from .paths import load_config, parse_run_id, normalize_ratio_string
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paths import load_config, parse_run_id, normalize_ratio_string


@dataclass
class SummaryRow:
    data: Dict[str, object]


def _is_success(series: pd.Series) -> pd.Series:
    if "error" not in series:
        return pd.Series([True] * len(series), index=series.index)
    err = series.fillna("").astype(str).str.lower().str.strip()
    return (err == "") | (err == "none") | (err == "nan")


def _mean_or_nan(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return series.mean()


def _safe_div(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def summarize_run(run_dir: Path) -> Optional[SummaryRow]:
    results_csv = run_dir / "logs" / "results.csv"
    if not results_csv.exists():
        return None

    df = pd.read_csv(results_csv)
    if "user_type" not in df.columns:
        return None

    df["user_type"] = df["user_type"].astype(str).str.lower()
    df = df[df["user_type"] != "warmup"].copy()
    if df.empty:
        return None

    success_mask = _is_success(df["error"]) if "error" in df.columns else pd.Series(True, index=df.index)
    df_success = df[success_mask].copy()

    cfg = load_config(run_dir)
    users_cfg = cfg.get("users") or {}
    sweep_cfg = cfg.get("sweep") or {}
    meta_cfg = cfg.get("_meta") or {}

    total_users = users_cfg.get("total")
    malicious_ratio = users_cfg.get("malicious_ratio")
    malicious_users = None
    normal_users = None
    if total_users is not None and malicious_ratio is not None:
        malicious_users = int(total_users * malicious_ratio)
        normal_users = int(total_users - malicious_users)

    run_id = meta_cfg.get("config_name", run_dir.name)
    run_meta = parse_run_id(run_id)

    attack_method = sweep_cfg.get("attack_method") or run_meta.get("payload") or ""
    attack_strength = sweep_cfg.get("attack_strength") or ""
    attack_dataset = sweep_cfg.get("attack_dataset") or run_meta.get("dataset") or ""
    sweep_ts = sweep_cfg.get("sweep_timestamp")
    ratio_value = run_meta.get("ratio")
    ratio_numeric = normalize_ratio_string(ratio_value)

    def per_user_metrics(user_type: str) -> Dict[str, object]:
        sub = df[df["user_type"] == user_type]
        sub_success = df_success[df_success["user_type"] == user_type]
        return {
            "request_count": len(sub),
            "goodput": _safe_div(len(sub_success), len(sub)),
            "latency": _mean_or_nan(sub_success.get("latency", pd.Series(dtype=float))),
            "ttft": _mean_or_nan(sub_success.get("ttft", pd.Series(dtype=float))),
            "tpot": _mean_or_nan(sub_success.get("tpot", pd.Series(dtype=float))),
            "completion_tokens": _mean_or_nan(sub_success.get("completion_tokens", pd.Series(dtype=float))),
        }

    overall_goodput = _safe_div(len(df_success), len(df))
    overall_latency = _mean_or_nan(df_success.get("latency", pd.Series(dtype=float)))

    normal = per_user_metrics("normal")
    malicious = per_user_metrics("malicious")

    row = {
        "Attack Method": attack_method,
        "Attack Strength": attack_strength,
        "Attack Dataset": attack_dataset,
        "Normal Users": normal_users,
        "Malicious Users": malicious_users,
        "Total Requests": len(df),
        "Average Latency (s)": overall_latency,
        "Overall Goodput Rate": overall_goodput,
        "Normal User Requests": normal["request_count"],
        "Normal User Average Latency (s)": normal["latency"],
        "Normal User Average TTFT (s)": normal["ttft"],
        "Normal User Average TPOT (s)": normal["tpot"],
        "Normal User Average Complete Tokens": normal["completion_tokens"],
        "Normal User Goodput Rate": normal["goodput"],
        "Malicious User Requests": malicious["request_count"],
        "Malicious User Average Latency (s)": malicious["latency"],
        "Malicious User Average TTFT (s)": malicious["ttft"],
        "Malicious User Average TPOT (s)": malicious["tpot"],
        "Malicious User Average Complete Tokens": malicious["completion_tokens"],
        "Malicious User Goodput Rate": malicious["goodput"],
        "sweep_timestamp": sweep_ts,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "Malicious User Ratio": ratio_numeric,
    }

    return SummaryRow(data=row)


def summarize_runs(run_dirs: List[Path]) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        summary = summarize_run(run_dir)
        if summary:
            rows.append(summary.data)
    return pd.DataFrame(rows)
