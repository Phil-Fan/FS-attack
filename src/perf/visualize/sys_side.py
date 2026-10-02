import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import logging
import sys
import os
import csv
import yaml
import json
from pathlib import Path

try:
    from .paths import read_run_dir_from_env_or_file, find_latest_scheduler_log, load_config as load_run_config
except ImportError:
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from paths import read_run_dir_from_env_or_file, find_latest_scheduler_log, load_config as load_run_config

try:
    from utils.logging_utils import setup_logging
except ImportError:
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

KV_THRESHOLD = 97.5
PRED_CLASS_PATTERN = re.compile(r"Predicted class:\s*(\d+)")
QUEUE_USER_TYPE_MAP = {
    "normal": "normal",
    "baseline": "baseline",
    "malicious": "malicious",
    "attack": "malicious",
}
QUEUE_USER_TYPE_ORDER = ["normal", "baseline", "malicious"]


def _parse_log_timestamp(ts_str: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None


def _parse_experiment_log_line(line: str) -> tuple[datetime | None, str | None]:
    line = line.strip()
    if not line:
        return None, None

    if " - " in line:
        parts = line.split(" - ", 3)
        if len(parts) == 4:
            ts = _parse_log_timestamp(parts[0])
            return ts, parts[3]

    old_match = re.match(
        r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[.,]\d{3}) "
        r"\[(?P<level>[A-Z]+)\] (?P<name>[^:]+): (?P<msg>.*)$",
        line,
    )
    if old_match:
        ts = _parse_log_timestamp(old_match.group("ts"))
        return ts, old_match.group("msg")

    return None, None


def parse_probe_predictions(log_path: Path) -> list[tuple[datetime, int]]:
    if not log_path.exists():
        return []

    predictions: list[tuple[datetime, int]] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if "Predicted class" not in line:
                continue
            ts, msg = _parse_experiment_log_line(line)
            if not ts or not msg:
                continue
            match = PRED_CLASS_PATTERN.search(msg)
            if not match:
                continue
            predictions.append((ts, int(match.group(1))))
    return predictions


def parse_probe_predictions_csv(csv_path: Path) -> list[tuple[datetime, int]]:
    if not csv_path.exists():
        return []

    predictions: list[tuple[datetime, int]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = (row.get("time") or "").strip()
            pred_str = (row.get("predict_class") or "").strip()
            if not ts_str or not pred_str:
                continue
            ts = _parse_log_timestamp(ts_str)
            if not ts:
                continue
            try:
                pred_class = int(float(pred_str))
            except ValueError:
                continue
            predictions.append((ts, pred_class))
    return predictions


def canonicalize_queue_user_type(user_type: str) -> str | None:
    return QUEUE_USER_TYPE_MAP.get(str(user_type).strip().lower())


def aggregate_queue_counts(counts: dict[str, int]) -> dict[str, int]:
    aggregated: dict[str, int] = {}
    for raw_user_type, value in counts.items():
        canonical_type = canonicalize_queue_user_type(raw_user_type)
        if canonical_type is None:
            continue
        aggregated[canonical_type] = aggregated.get(canonical_type, 0) + value
    return aggregated


def _find_nearest_step(sorted_steps: list[tuple[int, datetime]], timestamp: datetime) -> int | None:
    if not sorted_steps:
        return None
    min_diff = float("inf")
    nearest_step = None
    for step, step_time in sorted_steps:
        diff = abs((timestamp - step_time).total_seconds())
        if diff < min_diff:
            min_diff = diff
            nearest_step = step
    return nearest_step


def derive_user_time_path(log_file_path):
    # Extract the file name (e.g., 20251113_143300.log)
    filename = os.path.basename(log_file_path)
    log_dir = os.path.dirname(log_file_path)
    if "scheduler_v" in log_dir:
        logs_dir = os.path.dirname(log_dir)
        user_time_path = os.path.join(logs_dir, "user_time", filename)
    else:
        logs_dir = log_dir
        user_time_path = os.path.join(logs_dir, "user_time", filename)
    return user_time_path


def parse_user_time_csv(csv_file_path, step_to_time):
    user_data = {
        "sessions": [],
        "requests": [],
    }

    if not os.path.exists(csv_file_path):
        logger.warning(
            "User time CSV file '%s' not found. Skipping user timeline plot.",
            csv_file_path,
        )
        return user_data

    logger.info("Parsing user time CSV file: %s", csv_file_path)

    sorted_times = sorted(step_to_time.items(), key=lambda x: x[1])

    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse the timestamp (format: "2025-11-13 16:36:33,256")
            timestamp_str = row["timestamp"].strip('"')
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
            except ValueError:
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    logger.warning(
                        "Could not parse timestamp '%s', skipping row", timestamp_str
                    )
                    continue

            user_id = int(row["id"])
            user_type = row["type"]
            message = row["message"]

            if user_type == "warmup":
                continue

            step = _find_nearest_step(sorted_times, timestamp)
            if step is None:
                continue

            if message == "start":
                user_data["sessions"].append(
                    {"user_id": user_id, "user_type": user_type, "start_step": step}
                )
            elif message == "completed":
                for session in reversed(user_data["sessions"]):
                    if session["user_id"] == user_id and "end_step" not in session:
                        session["end_step"] = step
                        break
            elif message == "req_start":
                user_data["requests"].append(
                    {"user_id": user_id, "user_type": user_type, "req_start_step": step}
                )
            elif message == "req_end":
                for req in reversed(user_data["requests"]):
                    if req["user_id"] == user_id and "req_end_step" not in req:
                        req["req_end_step"] = step
                        break

    user_data["sessions"] = [s for s in user_data["sessions"] if "end_step" in s]
    user_data["requests"] = [r for r in user_data["requests"] if "req_end_step" in r]

    logger.info(
        "Parsed %d user sessions and %d requests",
        len(user_data["sessions"]),
        len(user_data["requests"]),
    )
    return user_data


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Generate scheduler analysis visualization"
    )
    parser.add_argument(
        "--log-file", required=False, help="Path to the log file to analyze"
    )
    parser.add_argument(
        "--run-dir", required=False, help="Run directory (output/{timestamp}__*)"
    )
    parser.add_argument(
        "--log-dir", required=False, help="Logs directory under run dir"
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Path to save the output image (auto-generated if not provided)",
    )
    parser.add_argument(
        "--config",
        required=False,
        help="Path to config YAML file (auto-detected if not provided)",
    )
    return parser.parse_args()


def load_config(config_path=None, run_dir: str | None = None):
    if config_path:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        logger.warning("Config file '%s' not found, using defaults", config_path)
        return None

    if run_dir:
        cfg = load_run_config(Path(run_dir))
        if cfg:
            return cfg

    script_dir = os.path.dirname(os.path.abspath(__file__))
    perf_dir = os.path.dirname(script_dir)
    default_config_path = os.path.join(perf_dir, "config", "strategy_config.yml")
    if os.path.exists(default_config_path):
        try:
            with open(default_config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            return None

    return None


def generate_output_filename(log_file: str) -> str:
    log_path = Path(log_file)
    run_dir = log_path.parent.parent
    output_dir = run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / "3_sys_side.png")


step_pattern = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - (?:scheduler - )?DEBUG - "
    r"\[step (?P<step>\d+)\] r_c: (?P<r_c>\{.*?\}) w_c: (?P<w_c>\{.*?\})(?: s_c: (?P<s_c>\{.*?\}))??; "
    r"(?:(?:kv:(?P<kv>[\d\.]+)%)|(?:gpu_kv:(?P<gpu_kv>[\d\.]+)%, cpu_kv:(?P<cpu_kv>[\d\.]+)%)); "
    r"computed: (?P<computed>\{.*?\}); cached: (?P<cached>\{.*?\})"
)

event_pattern = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - (?:scheduler - )?DEBUG - "
    r"\[step (?P<step>\d+)\] \[(?P<event>preempted|retracted|scheduled|queued|finished|swap_in|swap_out|preempt_recompute|preempt_swap)\] \[type: (?P<user_type>\w+)\](?:\s+\[request: (?P<request_id>[^\]]+)\])?$"
)


def initialize_data_structures():
    steps, times = [], []
    rc_data, wc_data, sc_data = [], [], []
    kv_usage = []
    gpu_kv_usage = []
    cpu_kv_usage = []
    computed_data, cached_data = [], []

    event_data = {
        "preempted": {},
        "scheduled": {},
        "queued": {},
        "finished": {},
    }

    event_counts = {
        "preempted": {},
        "scheduled": {},
        "queued": {},
        "finished": {},
    }

    request_tracking = {}

    all_user_types = set()

    return {
        "steps": steps,
        "times": times,
        "rc_data": rc_data,
        "wc_data": wc_data,
        "sc_data": sc_data,
        "kv_usage": kv_usage,
        "gpu_kv_usage": gpu_kv_usage,
        "cpu_kv_usage": cpu_kv_usage,
        "computed_data": computed_data,
        "cached_data": cached_data,
        "event_data": event_data,
        "event_counts": event_counts,
        "request_tracking": request_tracking,
        "all_user_types": all_user_types,
    }


def parse_log_file(log_file_path, data_structures):
    steps = data_structures["steps"]
    times = data_structures["times"]
    rc_data = data_structures["rc_data"]
    wc_data = data_structures["wc_data"]
    sc_data = data_structures["sc_data"]
    kv_usage = data_structures["kv_usage"]
    gpu_kv_usage = data_structures["gpu_kv_usage"]
    cpu_kv_usage = data_structures["cpu_kv_usage"]
    computed_data = data_structures["computed_data"]
    cached_data = data_structures["cached_data"]
    event_data = data_structures["event_data"]
    event_counts = data_structures["event_counts"]
    request_tracking = data_structures["request_tracking"]
    all_user_types = data_structures["all_user_types"]

    logger.info("Parsing log file: %s", log_file_path)

    with open(log_file_path, "r") as f:
        for line in f:
            step_match = step_pattern.match(line)
            event_match = event_pattern.match(line)

            if step_match:
                ts = datetime.strptime(step_match.group("time"), "%Y-%m-%d %H:%M:%S,%f")
                step = int(step_match.group("step"))
                r_c = eval(step_match.group("r_c"))
                w_c = eval(step_match.group("w_c"))
                s_c = eval(step_match.group("s_c") or "{}")
                kv = step_match.group("kv")
                gpu_kv = step_match.group("gpu_kv")
                cpu_kv = step_match.group("cpu_kv")

                if kv is not None:
                    kv = float(kv)
                    kv_usage.append(kv)
                    gpu_kv_usage.append(0)
                    cpu_kv_usage.append(0)
                else:
                    gpu_kv = float(gpu_kv)
                    cpu_kv = float(cpu_kv)
                    kv_usage.append(max(gpu_kv, cpu_kv))
                    gpu_kv_usage.append(gpu_kv)
                    cpu_kv_usage.append(cpu_kv)

                computed = eval(step_match.group("computed"))
                cached = eval(step_match.group("cached"))

                steps.append(step)
                times.append(ts)
                rc_data.append(r_c)
                wc_data.append(w_c)
                sc_data.append(s_c)
                computed_data.append(computed)
                cached_data.append(cached)

            elif event_match:
                ts = datetime.strptime(
                    event_match.group("time"), "%Y-%m-%d %H:%M:%S,%f"
                )
                step = int(event_match.group("step"))
                event_type = event_match.group("event")
                user_type = event_match.group("user_type")
                request_id = event_match.group("request_id")

                # In sglang, retracted is the effective preemption signal.
                if event_type == "retracted":
                    event_type = "preempted"

                if user_type == "warmup":
                    continue

                if user_type not in event_data[event_type]:
                    event_data[event_type][user_type] = []
                    event_counts[event_type][user_type] = {}
                    all_user_types.add(user_type)

                event_data[event_type][user_type].append((step, ts))

                if step not in event_counts[event_type][user_type]:
                    event_counts[event_type][user_type][step] = 0
                event_counts[event_type][user_type][step] += 1

                if request_id:
                    if request_id not in request_tracking:
                        request_tracking[request_id] = []
                    request_tracking[request_id].append(
                        {
                            "step": step,
                            "time": ts,
                            "event": event_type,
                            "user_type": user_type,
                        }
                    )

    logger.info("Parsed %d step records and %d user types", len(steps), len(all_user_types))
    return data_structures


def generate_event_statistics(data_structures):
    steps = data_structures["steps"]
    event_counts = data_structures["event_counts"]
    all_user_types = data_structures["all_user_types"]

    all_steps = sorted(set(steps))

    event_stats = {}
    for event_type in event_counts:
        event_stats[event_type] = {}
        for user_type in all_user_types:
            event_stats[event_type][user_type] = []

        for step in all_steps:
            for user_type in all_user_types:
                if user_type in event_counts[event_type]:
                    count = event_counts[event_type][user_type].get(step, 0)
                else:
                    count = 0
                event_stats[event_type][user_type].append(count)

    return all_steps, event_stats


def export_kv_cache_csv(data_structures, output_csv_path: str):
    import csv

    steps = data_structures["steps"]
    times = data_structures["times"]
    kv_usage = data_structures["kv_usage"]
    gpu_kv_usage = data_structures["gpu_kv_usage"]
    cpu_kv_usage = data_structures["cpu_kv_usage"]
    rc_data = [aggregate_queue_counts(d) for d in data_structures["rc_data"]]
    wc_data = [aggregate_queue_counts(d) for d in data_structures["wc_data"]]
    computed_data = data_structures["computed_data"]
    cached_data = data_structures["cached_data"]

    is_v0 = any(x > 0 for x in gpu_kv_usage)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = ["step", "datetime", "kv_usage"]
        if is_v0:
            header.extend(["gpu_kv_usage", "cpu_kv_usage"])
        header.extend(["running_count", "waiting_count"])
        all_user_types = [
            user_type
            for user_type in QUEUE_USER_TYPE_ORDER
            if any(d.get(user_type, 0) for d in rc_data + wc_data)
        ]
        for user_type in all_user_types:
            header.extend([f"running_{user_type}", f"waiting_{user_type}"])
        header.extend(["computed_tokens", "cached_tokens"])
        writer.writerow(header)

        for i, step in enumerate(steps):
            row = [
                step,
                times[i].isoformat(),
                kv_usage[i],
            ]
            if is_v0:
                row.extend([gpu_kv_usage[i], cpu_kv_usage[i]])

            row.extend([
                sum(rc_data[i].values()),
                sum(wc_data[i].values()),
            ])

            for user_type in all_user_types:
                row.extend([
                    rc_data[i].get(user_type, 0),
                    wc_data[i].get(user_type, 0),
                ])

            row.extend([
                sum(computed_data[i].values()),
                sum(cached_data[i].values()),
            ])

            writer.writerow(row)

    logger.info("Exported KV cache CSV: %s (%d records)", output_csv_path, len(steps))
    return output_csv_path


COLORS = {
    "running": "#2E86C1",
    "waiting": "#E74C3C",
    "swapped": "#9B59B6",
    "normal": "#2E86C1",
    "baseline": "#27AE60",
    "malicious": "#C0392B",
    "kv_usage": "blue",
    "user_type_1": "red",
    "user_type_2": "gray",
    "user_type_3": "green",
    "user_type_4": "#FFEAA7",
    "user_type_5": "purple",
    "user_type_6": "cyan",
}


def collect_experiment_metadata(sweep_root: Path):
    experiments = []

    # Find all experiment directories (matching the 1800s_* format)
    for run_dir in sorted(sweep_root.glob("1800s_*")):
        if not (run_dir / "logs").exists():
            continue

        meta_file = run_dir / "logs" / "experiment_meta.json"
        if not meta_file.exists():
            continue

        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
        except Exception as e:
            logger.warning("Could not read %s: %s", meta_file, e)
            continue

        # Format: 1800s_{strategy}_{model}_{ratio}p
        dir_name = run_dir.name
        match = re.match(r'1800s_(\w+)_[^_]+_(\d+(?:p\d+)?)p', dir_name)
        if match:
            strategy = match.group(1)
            ratio = match.group(2).replace('p', '.')
            label = f"{strategy}+{ratio}p"
        else:
            strategy = dir_name
            ratio = "unknown"
            label = dir_name

        start_time = datetime.fromisoformat(meta["start_time"])
        
        end_time = None
        if "actual_end_time" in meta:
            end_time = datetime.fromisoformat(meta["actual_end_time"])
        elif "expected_end_time" in meta:
            end_time = datetime.fromisoformat(meta["expected_end_time"])
        elif "actual_duration_seconds" in meta:
            duration_seconds = meta["actual_duration_seconds"]
            end_time = start_time + timedelta(seconds=duration_seconds)
        elif "configured_duration_seconds" in meta:
            duration_seconds = meta["configured_duration_seconds"]
            end_time = start_time + timedelta(seconds=duration_seconds)
        else:
            logger.warning("Experiment %s is missing end-time information, skipping", run_dir.name)
            continue

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


def create_visualization(
    data_structures,
    event_stats,
    all_event_steps,
    output_file,
    user_time_data=None,
    duration=None,
    start_time=None,
    experiments=None,
    probe_predictions=None,
):
    steps = data_structures["steps"]
    times = data_structures["times"]
    rc_data = data_structures["rc_data"]
    wc_data = data_structures["wc_data"]
    sc_data = data_structures["sc_data"]
    kv_usage = data_structures["kv_usage"]
    event_data = data_structures["event_data"]
    all_user_types = data_structures["all_user_types"]
    filtered_rc_data = [aggregate_queue_counts(d) for d in rc_data]
    filtered_wc_data = [aggregate_queue_counts(d) for d in wc_data]
    filtered_sc_data = [aggregate_queue_counts(d) for d in sc_data]

    logger.info("Creating visualization with %d user types...", len(all_user_types))

    step_to_time = dict(zip(steps, times))

    if start_time is None:
        start_time = times[0] if times else None

    fig, axes = plt.subplots(5, 1, figsize=(20, 22), sharex=True)

    def step_formatter(x, p):
        step = int(round(x))
        if step in step_to_time:
            return step_to_time[step].strftime("%H:%M:%S")
        return f"{step}"

    for ax in axes:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(step_formatter))
        ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    axes[-1].set_xlabel("Time")

    # Assign colors to user types (defined before all subplots to ensure consistent colors)
    all_user_types_combined = set(all_user_types)
    if user_time_data:
        for session in user_time_data.get("sessions", []):
            all_user_types_combined.add(session["user_type"])
        for req in user_time_data.get("requests", []):
            all_user_types_combined.add(req["user_type"])

    user_type_colors = {}
    sorted_user_types = sorted(all_user_types_combined)

    available_colors = ["gray", "green", "#FFEAA7", "purple", "cyan"]

    color_index = 0
    for user_type in sorted_user_types:
        if user_type in COLORS:
            user_type_colors[user_type] = COLORS[user_type]
        elif user_type == "malicious":
            user_type_colors[user_type] = "red"
        elif user_type == "normal":
            user_type_colors[user_type] = "#2E86C1"
        else:
            user_type_colors[user_type] = available_colors[
                color_index % len(available_colors)
            ]
            color_index += 1

    def lighten_color(color, factor=0.3):
        import matplotlib.colors as mcolors

        if isinstance(color, str):
            color = mcolors.to_rgba(color)
        r, g, b, a = color
        return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor, a)

    def darken_color(color, factor=0.3):
        import matplotlib.colors as mcolors

        if isinstance(color, str):
            color = mcolors.to_rgba(color)
        r, g, b, a = color
        return (r * (1 - factor), g * (1 - factor), b * (1 - factor), a)

    if user_time_data and (user_time_data["sessions"] or user_time_data["requests"]):
        axes[0].set_title("User Activity Timeline", fontsize=12, fontweight="bold")

        all_user_ids = set()
        user_types_in_data = set()
        for session in user_time_data["sessions"]:
            if session["user_type"] != "warmup":
                all_user_ids.add(session["user_id"])
                user_types_in_data.add(session["user_type"])
        for req in user_time_data["requests"]:
            if req["user_type"] != "warmup":
                all_user_ids.add(req["user_id"])
                user_types_in_data.add(req["user_type"])
        sorted_user_ids = sorted(all_user_ids)

        for session in user_time_data["sessions"]:
            if session["user_type"] == "warmup":
                continue
            user_id = session["user_id"]
            user_type = session["user_type"]
            start_step = session["start_step"]
            end_step = session["end_step"]
            base_color = user_type_colors.get(user_type, "gray")
            light_color = lighten_color(base_color, factor=0.5)

            axes[0].fill_betweenx(
                [user_id - 0.4, user_id + 0.4],
                start_step,
                end_step,
                color=light_color,
                alpha=0.6,
                zorder=1,
            )

        for req in user_time_data["requests"]:
            if req["user_type"] == "warmup":
                continue
            user_id = req["user_id"]
            user_type = req["user_type"]
            req_start_step = req["req_start_step"]
            req_end_step = req["req_end_step"]
            base_color = user_type_colors.get(user_type, "gray")
            dark_color = darken_color(base_color, factor=0.2)

            axes[0].fill_betweenx(
                [user_id - 0.4, user_id + 0.4],
                req_start_step,
                req_end_step,
                color=dark_color,
                alpha=0.9,
                zorder=2,
            )

        if sorted_user_ids:
            axes[0].set_yticks(sorted_user_ids)
            yticklabels = [""] * len(sorted_user_ids)
            yticklabels[0] = f"User {sorted_user_ids[0]}"
            yticklabels[-1] = f"User {sorted_user_ids[-1]}"
            axes[0].set_yticklabels(yticklabels)
        axes[0].set_ylabel("User ID", fontsize=10)

        legend_handles = []
        legend_labels = []
        for user_type in sorted(user_types_in_data):
            color = user_type_colors.get(user_type, "gray")
            handle = axes[0].scatter(
                [], [], color=color, marker="s", s=100, label=user_type
            )
            legend_handles.append(handle)
            legend_labels.append(user_type)
        axes[0].legend(
            legend_handles,
            legend_labels,
            frameon=True,
            fancybox=True,
            shadow=True,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )

        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].set_visible(False)

    # Subplot 1: Event timeline - different events at different y positions, different user types shown with different colors in the legend, and the same type uses different shapes
    axes[1].set_title("Event Timeline", fontsize=12, fontweight="bold")
    event_types = [
        "preempted",
        "scheduled",
        "queued",
        "finished",
    ]
    event_markers = {
        "preempted": "o",
        "scheduled": "^",
        "queued": "s",
        "finished": "D",
    }

    event_y_positions = {
        "preempted": 2,
        "scheduled": 4,
        "queued": 3,
        "finished": 1,
    }

    legend_handles = []
    legend_labels = []

    for user_type in sorted(all_user_types):
        if user_type == "warmup":
            continue
        user_color = user_type_colors[user_type]

        for i, event_type in enumerate(event_types):
            if (
                user_type in event_data[event_type]
                and event_data[event_type][user_type]
            ):
                user_steps = [step for step, ts in event_data[event_type][user_type]]
                y_pos = event_y_positions[event_type]

                if i == 0:
                    handle = axes[1].scatter(
                        [],
                        [],
                        color=user_color,
                        marker="o",
                        s=100,
                        alpha=0.8,
                        edgecolors="white",
                        linewidth=2,
                        label=user_type,
                    )
                    legend_handles.append(handle)
                    legend_labels.append(f"{user_type}")

                axes[1].scatter(
                    user_steps,
                    [y_pos] * len(user_steps),
                    color=user_color,
                    marker=event_markers[event_type],
                    s=60,
                    alpha=0.8,
                    edgecolors="white",
                    linewidth=1,
                )

    axes[1].set_yticks([event_y_positions[event] for event in event_types])
    axes[1].set_yticklabels(
        [f"{event.capitalize()}\n({event_markers[event]})" for event in event_types]
    )
    axes[1].set_ylim(0.5, 4.5)

    axes[1].legend(
        legend_handles,
        legend_labels,
        frameon=True,
        fancybox=True,
        shadow=True,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    axes[1].text(
        0.02,
        0.98,
        "Event Types: preempted(o), scheduled(^), queued(□), finished(◆)",
        transform=axes[1].transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Total Running/Waiting/Swapped Count", fontsize=12, fontweight="bold")
    axes[2].plot(
        steps,
        [sum(d.values()) for d in filtered_rc_data],
        label="running",
        color=COLORS["running"],
        linewidth=2,
        alpha=0.8,
    )
    axes[2].plot(
        steps,
        [sum(d.values()) for d in filtered_wc_data],
        label="waiting",
        color=COLORS["waiting"],
        linewidth=2,
        alpha=0.8,
    )
    if any(d for d in filtered_sc_data):
        axes[2].plot(
            steps,
            [sum(d.values()) for d in filtered_sc_data],
            label="swapped",
            color=COLORS["swapped"],
            linewidth=2,
            alpha=0.8,
        )
    axes[2].legend(
        frameon=True,
        fancybox=True,
        shadow=True,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    axes[2].grid(True, alpha=0.3)

    axes[3].set_title("Running/Waiting/Swapped Queues", fontsize=12, fontweight="bold")
    queue_user_types = [
        user_type
        for user_type in QUEUE_USER_TYPE_ORDER
        if any(d.get(user_type, 0) for d in filtered_rc_data + filtered_wc_data + filtered_sc_data)
    ]
    for u in queue_user_types:
        color = user_type_colors.get(u, COLORS["user_type_1"])
        axes[3].plot(
            steps,
            [d.get(u, 0) for d in filtered_rc_data],
            label=f"{u} running",
            color=color,
            linewidth=2,
            alpha=0.8,
            linestyle="-",
        )
        axes[3].plot(
            steps,
            [d.get(u, 0) for d in filtered_wc_data],
            label=f"{u} waiting",
            color=color,
            linewidth=2,
            alpha=0.8,
            linestyle="--",
        )
        if any(d for d in filtered_sc_data):
            axes[3].plot(
                steps,
                [d.get(u, 0) for d in filtered_sc_data],
                label=f"{u} swapped",
                color=color,
                linewidth=2,
                alpha=0.8,
                linestyle=":",
            )
    axes[3].legend(
        frameon=True,
        fancybox=True,
        shadow=True,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    axes[3].grid(True, alpha=0.3)

    above_threshold = [x > KV_THRESHOLD for x in kv_usage]
    threshold_ranges = []
    if any(above_threshold):
        start_idx = None
        for i, is_above in enumerate(above_threshold):
            if is_above and start_idx is None:
                start_idx = i
            elif not is_above and start_idx is not None:
                threshold_ranges.append((start_idx, i))
                start_idx = None
        if start_idx is not None:
            threshold_ranges.append((start_idx, len(above_threshold)))

    duration_step = None
    if duration is not None and start_time and times:
        duration_end_time = start_time + timedelta(seconds=duration)
        min_time_diff = float("inf")
        for step, step_time in step_to_time.items():
            time_diff = abs((step_time - duration_end_time).total_seconds())
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                duration_step = step

    for ax in axes:
        for start, end in threshold_ranges:
            ax.axvspan(
                steps[start],
                steps[end - 1],
                color="gray",
                alpha=0.15,
                label=f"KV>{KV_THRESHOLD}%" if start == threshold_ranges[0][0] else "",
            )

    axes[4].set_title("KV cache usage %", fontsize=12, fontweight="bold")

    is_v0 = any(x > 0 for x in data_structures["gpu_kv_usage"])

    if is_v0:
        axes[4].plot(
            steps,
            data_structures["gpu_kv_usage"],
            label="GPU KV usage",
            color="blue",
            linewidth=2,
            alpha=0.8,
        )
        axes[4].plot(
            steps,
            data_structures["cpu_kv_usage"],
            label="CPU KV usage",
            color="green",
            linewidth=2,
            alpha=0.8,
        )
    else:
        axes[4].plot(
            steps,
            kv_usage,
            label="KV usage",
            color=COLORS["kv_usage"],
            linewidth=2,
            alpha=0.8,
        )

    if probe_predictions:
        sorted_steps = sorted(step_to_time.items(), key=lambda x: x[1])
        mapped = []
        for ts, pred_class in probe_predictions:
            step = _find_nearest_step(sorted_steps, ts)
            if step is not None:
                mapped.append((step, pred_class))

        if mapped:
            mapped.sort(key=lambda x: x[0])
            pred_steps = [step for step, _ in mapped]
            pred_vals = [pred_class * 10.0 + 5.0 for _, pred_class in mapped]
            axes[4].plot(
                pred_steps,
                pred_vals,
                label="Probe KV bin (pred)",
                color="orange",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
            )

    axes[4].axhline(
        y=KV_THRESHOLD,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label=f"{KV_THRESHOLD}% threshold",
        linewidth=1.5,
    )

    if experiments:
        experiment_labels = []
        for exp in experiments:
            exp_start_step = None
            exp_end_step = None

            for step, step_time in step_to_time.items():
                time_diff_start = abs((step_time - exp["start_time"]).total_seconds())
                time_diff_end = abs((step_time - exp["end_time"]).total_seconds())

                if exp_start_step is None or time_diff_start < abs((step_to_time[exp_start_step] - exp["start_time"]).total_seconds()):
                    exp_start_step = step

                if exp_end_step is None or time_diff_end < abs((step_to_time[exp_end_step] - exp["end_time"]).total_seconds()):
                    exp_end_step = step

            if exp_start_step is not None and exp_end_step is not None:
                experiment_labels.append({
                    "start_step": exp_start_step,
                    "end_step": exp_end_step,
                    "label": exp["label"]
                })

        y_max = axes[4].get_ylim()[1]
        y_text = y_max * 0.95

        used_x_positions = set()
        for exp_label in experiment_labels:
            start_step = exp_label["start_step"]
            end_step = exp_label["end_step"]
            label = exp_label["label"]

            mid_step = (start_step + end_step) / 2

            # Avoid overlapping labels
            x_pos = mid_step
            offset = 0
            while x_pos in used_x_positions and offset < 5:
                offset += 1
                x_pos = mid_step + offset * 10

            if abs(x_pos - mid_step) > (end_step - start_step) * 2:
                continue

            used_x_positions.add(x_pos)

            axes[4].axvline(
                x=start_step,
                color="gray",
                linestyle=":",
                alpha=0.3,
                linewidth=1,
            )
            axes[4].axvline(
                x=end_step,
                color="gray",
                linestyle=":",
                alpha=0.3,
                linewidth=1,
            )

            axes[4].text(
                x_pos,
                y_text,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7, edgecolor="gray"),
            )

    axes[4].legend(
        frameon=True,
        fancybox=True,
        shadow=True,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    axes[4].grid(True, alpha=0.3)

    if duration_step is not None:
        for ax in axes:
            ax.axvline(
                x=duration_step,
                color="gray",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Duration end" if ax == axes[0] else "",
            )

    plt.style.use("default")
    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.set_facecolor("#FAFAFA")

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Visualization saved to: %s", output_file)


def main():
    setup_logging()
    args = parse_command_line()

    log_file = args.log_file
    run_dir = None

    if not log_file:
        if args.log_dir:
            log_dir = Path(args.log_dir)
            if log_dir.exists():
                candidates = [p for p in log_dir.glob("*.log") if not p.name.endswith("_server.log")]
                if candidates:
                    log_file = str(max(candidates, key=lambda p: p.stat().st_mtime))
                    run_dir = str(log_dir.parent)
        elif args.run_dir:
            run_dir = args.run_dir
            latest_log = find_latest_scheduler_log(Path(run_dir))
            if latest_log:
                log_file = str(latest_log)
        else:
            auto_run_dir = read_run_dir_from_env_or_file()
            if auto_run_dir:
                run_dir = str(auto_run_dir)
                latest_log = find_latest_scheduler_log(auto_run_dir)
                if latest_log:
                    log_file = str(latest_log)

    if not log_file or not os.path.exists(log_file):
        logger.error("No usable scheduler log file found")
        sys.exit(1)

    config = load_config(args.config, run_dir=run_dir)

    # Use --output if provided, otherwise generate from config
    if args.output:
        if os.path.isdir(args.output):
            output_file = os.path.join(args.output, "3_sys_side.png")
        else:
            output_file = args.output
    else:
        output_file = generate_output_filename(log_file)
    
    logger.info("Output file: %s", output_file)

    try:
        run_analysis(log_file, output_file, config)
        logger.info("Analysis completed successfully!")
    except Exception as e:
        logger.error("Error during analysis: %s", e)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def run_analysis(
    log_file: str,
    output_file: str,
    config: dict | None = None,
    data_structures: dict | None = None,
) -> None:
    duration = None
    if config:
        if "experiment" in config:
            duration = config["experiment"].get("duration")
        elif "global" in config:
            duration = config["global"].get("duration")

    if data_structures is None:
        data_structures = initialize_data_structures()
        data_structures = parse_log_file(log_file, data_structures)

    all_event_steps, event_stats = generate_event_statistics(data_structures)

    csv_output_path = output_file.rsplit(".", 1)[0] + ".csv"
    export_kv_cache_csv(data_structures, csv_output_path)

    step_to_time = dict(zip(data_structures["steps"], data_structures["times"]))

    start_time = data_structures["times"][0] if data_structures["times"] else None

    user_time_csv_path = derive_user_time_path(log_file)
    user_time_data = parse_user_time_csv(user_time_csv_path, step_to_time)

    experiments = None
    log_path = Path(log_file)
    # Experiment directories are at: run_dir/1800s_*/
    sweep_root = log_path.parent.parent
    if sweep_root.exists():
        try:
            experiments = collect_experiment_metadata(sweep_root)
            logger.info("Collected time-period information for %d experiments", len(experiments))
            for exp in experiments:
                logger.info(
                    "  - %s: %s ~ %s",
                    exp["label"],
                    exp["start_time"].strftime("%H:%M:%S"),
                    exp["end_time"].strftime("%H:%M:%S"),
                )
        except Exception as e:
            logger.warning("Could not collect experiment metadata: %s", e)

    probe_predictions = None
    log_path = Path(log_file)
    run_dir = log_path.parent.parent if log_path.parent.name == "logs" else log_path.parent
    exp_pred_csv = run_dir / "logs" / "prober_predictions.csv"
    exp_log = run_dir / "logs" / "experiment.log"
    if exp_pred_csv.exists():
        probe_predictions = parse_probe_predictions_csv(exp_pred_csv)
    elif exp_log.exists():
        probe_predictions = parse_probe_predictions(exp_log)

    create_visualization(
        data_structures,
        event_stats,
        all_event_steps,
        output_file,
        user_time_data,
        duration=duration,
        start_time=start_time,
        experiments=experiments,
        probe_predictions=probe_predictions,
    )


def generate_per_experiment_sys_plots(
    log_file: str,
    sweep_root: Path,
    config: dict | None = None,
    data_structures: dict | None = None,
) -> None:
    experiments = collect_experiment_metadata(sweep_root)
    if not experiments:
        logger.warning("No experiments found")
        return

    logger.info("Found %d experiments; generating an independent sys-side visualization for each...", len(experiments))

    if data_structures is None:
        data_structures = initialize_data_structures()
        data_structures = parse_log_file(log_file, data_structures)

    times = data_structures["times"]
    steps = data_structures["steps"]

    for exp in experiments:
        exp_dir = exp["run_dir"]
        exp_label = exp["label"]
        exp_start = exp["start_time"]
        exp_end = exp["end_time"]

        logger.info("Processing experiment: %s (%s)", exp_label, exp_dir.name)

        exp_figures_dir = exp_dir / "figures"
        exp_figures_dir.mkdir(parents=True, exist_ok=True)

        exp_data = filter_data_by_time_range(
            data_structures, exp_start, exp_end
        )

        if exp_data is None or len(exp_data["steps"]) == 0:
            logger.warning("No data within this experiment's time range, skipping")
            continue

        logger.info("Filtered to %d records", len(exp_data["steps"]))

        all_event_steps, event_stats = generate_event_statistics(exp_data)

        csv_output_path = exp_figures_dir / "sys_side.csv"
        export_kv_cache_csv(exp_data, str(csv_output_path))

        step_to_time = dict(zip(exp_data["steps"], exp_data["times"]))

        user_time_data = None
        user_time_csv_path = exp_dir / "logs" / "user_time" / f"{exp_dir.name}.log"
        if user_time_csv_path.exists():
            user_time_data = parse_user_time_csv(str(user_time_csv_path), step_to_time)

        probe_predictions = None
        exp_pred_csv = exp_dir / "logs" / "prober_predictions.csv"
        exp_log = exp_dir / "logs" / "experiment.log"
        if exp_pred_csv.exists():
            probe_predictions = parse_probe_predictions_csv(exp_pred_csv)
        elif exp_log.exists():
            probe_predictions = parse_probe_predictions(exp_log)

        output_file = exp_figures_dir / "sys_side.png"

        # Cache check: skip if the output file exists and the source files are unchanged
        log_path = Path(log_file)
        meta_file = exp_dir / "logs" / "experiment_meta.json"
        
        if output_file.exists():
            output_mtime = output_file.stat().st_mtime
            should_skip = True
            
            if log_path.exists():
                log_mtime = log_path.stat().st_mtime
                if output_mtime < log_mtime:
                    should_skip = False
            
            if meta_file.exists():
                meta_mtime = meta_file.stat().st_mtime
                if output_mtime < meta_mtime:
                    should_skip = False
            
            if should_skip:
                logger.info("  Skipping: %s already exists and the source files are unchanged", output_file.name)
                continue

        duration = None
        if config:
            if "experiment" in config:
                duration = config["experiment"].get("duration")
            elif "global" in config:
                duration = config["global"].get("duration")

        exp_start_time = exp_data["times"][0] if exp_data["times"] else None

        create_visualization(
            exp_data,
            event_stats,
            all_event_steps,
            str(output_file),
            user_time_data,
            duration=duration,
            start_time=exp_start_time,
            experiments=None,
            probe_predictions=probe_predictions,
        )

        logger.info("Generated visualization: %s", output_file)


def filter_data_by_time_range(
    data_structures: dict,
    start_time: datetime,
    end_time: datetime,
) -> dict | None:
    times = data_structures["times"]
    steps = data_structures["steps"]

    indices = [
        i for i, t in enumerate(times)
        if start_time <= t <= end_time
    ]

    if not indices:
        return None

    # Note: rc_data, wc_data, and sc_data are lists of dicts, each element being {user_type: count}
    filtered = {
        "steps": [steps[i] for i in indices],
        "times": [times[i] for i in indices],
        "rc_data": [data_structures["rc_data"][i] for i in indices],
        "wc_data": [data_structures["wc_data"][i] for i in indices],
        "sc_data": [data_structures["sc_data"][i] for i in indices],
        "kv_usage": [data_structures["kv_usage"][i] for i in indices],
        "gpu_kv_usage": [data_structures["gpu_kv_usage"][i] for i in indices],
        "cpu_kv_usage": [data_structures["cpu_kv_usage"][i] for i in indices],
        "computed_data": [data_structures["computed_data"][i] for i in indices],
        "cached_data": [data_structures["cached_data"][i] for i in indices],
        "event_data": {
            "preempted": {},
            "scheduled": {},
            "queued": {},
            "finished": {},
        },
        "event_counts": {
            "preempted": {},
            "scheduled": {},
            "queued": {},
            "finished": {},
        },
        "request_tracking": {},
        "all_user_types": data_structures["all_user_types"].copy(),
    }

    for event_type in filtered["event_data"]:
        for user_type, events in data_structures["event_data"][event_type].items():
            filtered_events = [
                (step, ts) for step, ts in events
                if start_time <= ts <= end_time
            ]
            if filtered_events:
                filtered["event_data"][event_type][user_type] = filtered_events

                filtered["event_counts"][event_type][user_type] = {}
                for step, ts in filtered_events:
                    filtered["event_counts"][event_type][user_type][step] = \
                        filtered["event_counts"][event_type][user_type].get(step, 0) + 1

    return filtered


if __name__ == "__main__":
    main()
