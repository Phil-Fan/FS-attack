#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_OUTPUT_DIR:-${1:-}}"

if [[ -z "$RUN_ROOT" ]]; then
    echo "Missing root directory argument, and RUN_OUTPUT_DIR is not set"
    echo "Usage: $0 <root_dir>"
    exit 1
fi

echo "Starting post-processing..."
python "$SCRIPT_DIR/utils/postprocess.py" --root "$RUN_ROOT"

echo "Generating visualization charts..."
python "$SCRIPT_DIR/visualize/cli.py" batch --root "$RUN_ROOT"

echo "Generating per-experiment segmented ITL visualizations..."
python "$SCRIPT_DIR/visualize/cli.py" itl-per-exp --run-dir "$RUN_ROOT" --time-limit ${TIME_LIMIT:-1200}

echo "Generating per-experiment segmented sys visualizations..."
python "$SCRIPT_DIR/visualize/cli.py" sys-per-exp --run-dir "$RUN_ROOT"

echo "Post-processing complete!"
