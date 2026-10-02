#!/bin/bash
set -e

# ----------------------------
# Minimal vLLM server launcher
# ----------------------------

# Default configuration
PORT=${PORT:-8080}
GPU_MEMORY=${GPU_MEMORY:-0.20}
MAX_SEQS=${MAX_SEQS:-20}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-10240}
HOST=${HOST:-0.0.0.0}
SCHEDULER=${SCHEDULER:-v1}
LOG_REQUESTS=${LOG_REQUESTS:-false}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1} 
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}

# Parse command-line arguments (keeping common vLLM options)
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port) PORT="$2"; shift 2 ;;
    -g|--gpu-memory) GPU_MEMORY="$2"; shift 2 ;;
    -m|--max-seqs) MAX_SEQS="$2"; shift 2 ;;
    -l|--log-requests) LOG_REQUESTS=true; shift ;;
    -s|--scheduler) SCHEDULER="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [-p port] [-g gpu_memory] [-m max_seqs] [-s v0|v1] [-l]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Model path (required)
MODEL_NAME="${MODEL_NAME:-$(cat config/model_name 2>/dev/null || echo "")}"
if [ -z "$MODEL_NAME" ]; then
  echo "ERROR: MODEL_NAME not set"
  exit 1
fi
MODEL_PATH="${MODEL_PATH}/${MODEL_NAME}"


# scheduler
VLLM_USE_V1=$([ "$SCHEDULER" = "v1" ] && echo "1" || echo "0")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Create RUN_OUTPUT_DIR if it does not exist
if [ ! -d "${RUN_OUTPUT_DIR}/logs" ]; then
  mkdir -p "${RUN_OUTPUT_DIR}/logs"
fi
SCHEDULER_LOG="${RUN_OUTPUT_DIR}/logs/${TIMESTAMP}.log"


echo ""
echo "=========================================="
echo "vLLM Server Configuration"
echo "=========================================="
echo "Scheduler version:       $SCHEDULER (VLLM_USE_V1=${VLLM_USE_V1})"
echo "Model path:              $MODEL_PATH"
echo "Listen address:          ${HOST}:${PORT}"
echo "GPU memory utilization:  ${GPU_MEMORY}"
echo "Max concurrent seqs:     ${MAX_SEQS}"
echo "Max model length:        ${MAX_MODEL_LEN}"
echo "TP parallel size:        ${TENSOR_PARALLEL_SIZE}"
echo "Max batched tokens:      ${MAX_NUM_BATCHED_TOKENS}"
echo "=========================================="
echo ""


# Build the vLLM command (keeping options)
VLLM_CMD="VLLM_SCHEDULER_LOG_PATH=${SCHEDULER_LOG} VLLM_USE_V1=${VLLM_USE_V1} vllm serve ${MODEL_PATH}"
VLLM_CMD+=" --host ${HOST}"
VLLM_CMD+=" --port ${PORT}"
VLLM_CMD+=" --dtype auto"
VLLM_CMD+=" --max-num-seqs ${MAX_SEQS}"
VLLM_CMD+=" --max-model-len ${MAX_MODEL_LEN}"
VLLM_CMD+=" --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}"
VLLM_CMD+=" --trust-remote-code"
VLLM_CMD+=" --gpu-memory-utilization ${GPU_MEMORY}"
VLLM_CMD+=" --enable-chunked-prefill"
VLLM_CMD+=" --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS}"

# Launch
echo "Running: ${VLLM_CMD}"
eval ${VLLM_CMD}
