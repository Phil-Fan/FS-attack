#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
SUMMARY_SCRIPT="${SCRIPT_DIR}/visualize/summarize_sweep.py"

if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "[summary] Output directory not found: $OUTPUT_DIR" >&2
    exit 1
fi

if [[ ! -f "$SUMMARY_SCRIPT" ]]; then
    echo "[summary] summarize_sweep.py not found: $SUMMARY_SCRIPT" >&2
    exit 1
fi

shopt -s nullglob
subdirs=("$OUTPUT_DIR"/*/)
shopt -u nullglob

if [[ ${#subdirs[@]} -eq 0 ]]; then
    echo "[summary] No subdirectories found in $OUTPUT_DIR" >&2
    exit 0
fi

if command -v nproc >/dev/null 2>&1; then
    MAX_JOBS_DEFAULT=$(nproc)
else
    MAX_JOBS_DEFAULT=4
fi

MAX_JOBS=${MAX_JOBS:-$MAX_JOBS_DEFAULT}
if ! [[ $MAX_JOBS =~ ^[0-9]+$ ]] || [[ $MAX_JOBS -lt 1 ]]; then
    echo "[summary] Invalid MAX_JOBS=$MAX_JOBS, fallback to 1" >&2
    MAX_JOBS=1
fi

declare -a pid_queue=()
declare -A pid_to_dir=()
overall_status=0

wait_for_pid() {
    local pid=$1
    local dir_name=${pid_to_dir[$pid]}
    if wait "$pid"; then
        echo "[summary] Completed $dir_name"
    else
        echo "[summary] Failed $dir_name" >&2
        overall_status=1
    fi
    unset 'pid_to_dir[$pid]'
}

wait_until_slot_available() {
    while [[ ${#pid_queue[@]} -ge $MAX_JOBS ]]; do
        local oldest_pid=${pid_queue[0]}
        wait_for_pid "$oldest_pid"
        pid_queue=("${pid_queue[@]:1}")
    done
}

for dir_with_slash in "${subdirs[@]}"; do
    wait_until_slot_available

    dir="${dir_with_slash%/}"
    basename="$(basename "$dir")"
    echo "[summary] Processing $basename"
    python3 "$SUMMARY_SCRIPT" "$dir" --recursive &
    pid=$!
    pid_queue+=("$pid")
    pid_to_dir["$pid"]="$basename"
done

for pid in "${pid_queue[@]}"; do
    wait_for_pid "$pid"
done

exit $overall_status
