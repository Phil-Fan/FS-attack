#!/bin/bash
set -e

# ------------------------------
# Minimal SGLang server launcher
# ------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SGLANG_PATCH_DIR="${REPO_SRC_DIR}/sglang"

# Default configuration
PORT=${PORT:-8080}
GPU_MEMORY=${GPU_MEMORY:-0.20}
MAX_SEQS=${MAX_SEQS:-20}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-10240}
HOST=${HOST:-0.0.0.0}
LOG_REQUESTS=${LOG_REQUESTS:-false}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
SGLANG_BIN=${SGLANG_BIN:-$(command -v sglang 2>/dev/null || true)}
SGLANG_PYTHON=${SGLANG_PYTHON:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)}
SGLANG_SCHEDULE_POLICY=${SGLANG_SCHEDULE_POLICY:-fcfs}
APPLY_SGLANG_PATCH=${APPLY_SGLANG_PATCH:-true}
SGLANG_PATCH_VARIANT=${SGLANG_PATCH_VARIANT:-l40s}
SKIP_SERVER_WARMUP=${SKIP_SERVER_WARMUP:-false}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port) PORT="$2"; shift 2 ;;
    -g|--gpu-memory) GPU_MEMORY="$2"; shift 2 ;;
    -m|--max-seqs) MAX_SEQS="$2"; shift 2 ;;
    -l|--log-requests) LOG_REQUESTS=true; shift ;;
    -h|--help)
      echo "Usage: $0 [-p port] [-g gpu_memory] [-m max_seqs] [-l]"
      echo ""
      echo "Extra env vars:"
      echo "  SGLANG_BIN=/path/to/sglang"
      echo "  SGLANG_PYTHON=/path/to/python"
      echo "  TARGET_DIR=/path/to/site-packages/sglang"
      echo "  SGLANG_SCHEDULE_POLICY=fcfs|lpm|dfs-weight|lof|random|priority"
      echo "  APPLY_SGLANG_PATCH=true|false"
      echo "  SGLANG_PATCH_VARIANT=default|l40s"
      echo "  SKIP_SERVER_WARMUP=true|false"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ ! -x "${SGLANG_BIN}" ]; then
  echo "ERROR: SGLang binary not found: ${SGLANG_BIN}"
  exit 1
fi

if [ ! -x "${SGLANG_PYTHON}" ]; then
  echo "ERROR: SGLang python not found: ${SGLANG_PYTHON}"
  exit 1
fi

# Model path (required)
MODEL_ROOT="${MODEL_PATH:-}"
MODEL_NAME="${MODEL_NAME:-$(cat config/model_name 2>/dev/null || echo "")}"
if [ -z "${MODEL_NAME}" ]; then
  echo "ERROR: MODEL_NAME not set"
  exit 1
fi
if [ -z "${MODEL_ROOT}" ]; then
  echo "ERROR: MODEL_PATH not set"
  exit 1
fi

if [[ "${MODEL_NAME}" = /* ]]; then
  RESOLVED_MODEL_PATH="${MODEL_NAME}"
elif [ -d "${MODEL_ROOT}/${MODEL_NAME}" ]; then
  RESOLVED_MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"
else
  MATCHED_MODEL_PATHS=()
  while IFS= read -r -d '' candidate; do
    MATCHED_MODEL_PATHS+=("${candidate}")
  done < <(find "${MODEL_ROOT}" -mindepth 2 -maxdepth 2 -type d -name "${MODEL_NAME}" -print0 2>/dev/null)

  if [ "${#MATCHED_MODEL_PATHS[@]}" -eq 1 ]; then
    RESOLVED_MODEL_PATH="${MATCHED_MODEL_PATHS[0]}"
  elif [ "${#MATCHED_MODEL_PATHS[@]}" -gt 1 ]; then
    echo "ERROR: Multiple model paths match ${MODEL_NAME}:"
    printf '  %s\n' "${MATCHED_MODEL_PATHS[@]}"
    echo "Set MODEL_NAME to an absolute path or set MODEL_PATH to the exact parent directory."
    exit 1
  else
    echo "ERROR: model path not found: ${MODEL_ROOT}/${MODEL_NAME}"
    echo "Also searched one level below ${MODEL_ROOT}."
    exit 1
  fi
fi
MODEL_PATH="${RESOLVED_MODEL_PATH}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${SCRIPT_DIR}/output/sglang}"
if [ ! -d "${RUN_OUTPUT_DIR}/logs" ]; then
  mkdir -p "${RUN_OUTPUT_DIR}/logs"
fi
SCHEDULER_LOG="${RUN_OUTPUT_DIR}/logs/${TIMESTAMP}.log"

if [ "${APPLY_SGLANG_PATCH}" = "true" ]; then
  echo "Applying local SGLang patch from ${SGLANG_PATCH_DIR} (variant: ${SGLANG_PATCH_VARIANT})"
  if [ -n "${TARGET_DIR}" ]; then
    "${SGLANG_PYTHON}" "${SGLANG_PATCH_DIR}/patch.py" --target-dir "${TARGET_DIR}" --variant "${SGLANG_PATCH_VARIANT}"
  else
    "${SGLANG_PYTHON}" "${SGLANG_PATCH_DIR}/patch.py" --python "${SGLANG_PYTHON}" --variant "${SGLANG_PATCH_VARIANT}"
  fi
fi

echo ""
echo "=========================================="
echo "SGLang Server Configuration"
echo "=========================================="
echo "SGLang schedule policy: ${SGLANG_SCHEDULE_POLICY}"
echo "Model path:             ${MODEL_PATH}"
echo "Listen address:         ${HOST}:${PORT}"
echo "GPU memory utilization: ${GPU_MEMORY}"
echo "Max concurrent seqs:    ${MAX_SEQS}"
echo "Max model length:       ${MAX_MODEL_LEN}"
echo "TP parallel size:       ${TENSOR_PARALLEL_SIZE}"
echo "Chunked Prefill:        ${MAX_NUM_BATCHED_TOKENS}"
echo "SGLang bin:             ${SGLANG_BIN}"
echo "Python:                 ${SGLANG_PYTHON}"
echo "Target dir:             ${TARGET_DIR:-<auto-detect>}"
echo "Patch variant:          ${SGLANG_PATCH_VARIANT}"
echo "Skip warmup:            ${SKIP_SERVER_WARMUP}"
echo "Scheduler log:          ${SCHEDULER_LOG}"
echo "=========================================="
echo ""

export SGLANG_SCHEDULER_LOG_PATH="${SCHEDULER_LOG}"
SGLANG_CMD=(
  "${SGLANG_BIN}" serve --model-path "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --dtype auto
  --max-running-requests "${MAX_SEQS}"
  --context-length "${MAX_MODEL_LEN}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --trust-remote-code
  --mem-fraction-static "${GPU_MEMORY}"
  --chunked-prefill-size "${MAX_NUM_BATCHED_TOKENS}"
  --schedule-policy "${SGLANG_SCHEDULE_POLICY}"
)

if [ "${SKIP_SERVER_WARMUP}" = "true" ]; then
  SGLANG_CMD+=(--skip-server-warmup)
fi

if [ "${LOG_REQUESTS}" = "true" ]; then
  SGLANG_CMD+=(--log-requests)
fi

printf 'Running:'
printf ' %q' "SGLANG_SCHEDULER_LOG_PATH=${SCHEDULER_LOG}" "${SGLANG_CMD[@]}"
printf '\n'
"${SGLANG_CMD[@]}"
