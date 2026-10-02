#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import os
import re
import yaml


@dataclass(frozen=True)
class RunMeta:
    run_dir: Path
    config: dict


VIS_DIR = Path(__file__).resolve().parent
PERF_DIR = VIS_DIR.parent
OUTPUT_ROOT = PERF_DIR / "output"
RUN_DIR_FILE = OUTPUT_ROOT / ".vllm_run_dir"


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (PERF_DIR / p).resolve()
    return p


def read_run_dir_from_env_or_file() -> Optional[Path]:
    env_dir = os.environ.get("RUN_OUTPUT_DIR")
    if env_dir:
        return _resolve_path(env_dir)
    if RUN_DIR_FILE.exists():
        content = RUN_DIR_FILE.read_text(encoding="utf-8").strip()
        if content:
            return _resolve_path(content)
    return None


def find_latest_scheduler_log(run_dir: Path) -> Optional[Path]:
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        return None
    candidates = []
    for log_path in logs_dir.glob("*.log"):
        if log_path.name.endswith("_server.log"):
            continue
        candidates.append(log_path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_config(run_dir: Path) -> dict:
    for name in ("config.yaml", "config.yml"):
        cfg_path = run_dir / name
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


def collect_run_dirs(root: Path) -> List[RunMeta]:
    root = root.resolve()
    metas: List[RunMeta] = []
    for cfg_path in root.glob("**/config.yaml"):
        run_dir = cfg_path.parent
        cfg = load_config(run_dir)
        metas.append(RunMeta(run_dir=run_dir, config=cfg))
    return metas


def get_sweep_timestamp(cfg: dict) -> Optional[str]:
    return (cfg.get("sweep") or {}).get("sweep_timestamp")


def pick_latest_sweep_timestamp(metas: Iterable[RunMeta]) -> Optional[str]:
    timestamps = [get_sweep_timestamp(m.config) for m in metas if get_sweep_timestamp(m.config)]
    if not timestamps:
        return None
    return sorted(timestamps)[-1]


def filter_by_sweep_timestamp(metas: Iterable[RunMeta], sweep_ts: Optional[str]) -> List[RunMeta]:
    if not sweep_ts:
        return list(metas)
    selected = []
    for meta in metas:
        if get_sweep_timestamp(meta.config) == sweep_ts:
            selected.append(meta)
    return selected


def collect_results_csvs(path: Path, sweep_ts: Optional[str] = None) -> List[Path]:
    path = path.resolve()
    if path.is_file():
        return [path]
    metas = collect_run_dirs(path)
    metas = filter_by_sweep_timestamp(metas, sweep_ts)
    results = []
    for meta in metas:
        candidate = meta.run_dir / "logs" / "results.csv"
        if candidate.exists():
            results.append(candidate)
    return results


def extract_run_id_from_path(path: Path) -> Optional[str]:
    if path.name == "results.csv":
        run_dir = path.parent.parent
        return run_dir.name
    return None


def parse_run_id(run_id: str) -> dict:
    # Try to parse the new format: 1800s_engorgio_qwen_8b_0p
    new_format_match = re.match(r'(\d+)s_(\w+)_(.+?)_(\d+(?:p\d+)?)p', run_id)
    if new_format_match:
        duration = new_format_match.group(1)
        strategy = new_format_match.group(2)   # engorgio
        model = new_format_match.group(3)      # qwen_8b
        ratio = new_format_match.group(4).replace('p', '.')
        return {
            "duration": duration,
            "strategy": strategy,
            "model": model,
            "ratio": ratio,
            "payload": f"{strategy}_attack",
            "dataset": model,
        }

    # Legacy format: payload_x__file_y__ratio_0p5_rep_1
    parts = run_id.split("__")
    meta = {}
    for part in parts:
        if part.startswith("payload_"):
            meta["payload"] = part.replace("payload_", "", 1)
        elif part.startswith("file_"):
            meta["dataset"] = part.replace("file_", "", 1)
        elif part.startswith("ratio_"):
            meta["ratio"] = part.replace("ratio_", "", 1)
        elif part.startswith("rep_"):
            meta["repeat"] = part.replace("rep_", "", 1)
    return meta


def load_experiment_meta(run_dir: Path) -> Optional[dict]:
    meta_path = run_dir / "logs" / "experiment_meta.json"
    if not meta_path.exists():
        return None
    try:
        import json
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_actual_duration(run_dir: Path) -> Optional[float]:
    meta = load_experiment_meta(run_dir)
    if meta and "actual_duration_seconds" in meta:
        return float(meta["actual_duration_seconds"])
    return None


def normalize_ratio_string(ratio_str: Optional[str]) -> Optional[float]:
    if not ratio_str:
        return None
    cleaned = ratio_str.replace("p", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None
