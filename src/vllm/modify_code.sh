#!/bin/bash
USER_NAME=${USER:-$(whoami)}
BASE_DIR="${BASE_DIR:-/home/${USER_NAME}/latency-attack}"
WORK_DIR=${BASE_DIR}/src/perf
TARGET_DIR="${TARGET_DIR:-${BASE_DIR}/.venv}"
# # move code to the environment 复制文件夹中修改的源码文件
DEVICE=$(cat ${BASE_DIR}/src/vllm/.device)
if [ "$DEVICE" = "dgx" ]; then
    python ${BASE_DIR}/src/vllm/file_operations.py --base-dir ${BASE_DIR} --csv-path ${BASE_DIR}/src/vllm/file_list_dgx.csv --target-base-dir /home/${USER_NAME}/exp/dgx-spark-vllm-setup/vllm-install/vllm/vllm
else
    python ${BASE_DIR}/src/vllm/file_operations.py --base-dir ${BASE_DIR} --target-base-dir ${TARGET_DIR} --csv-path ${BASE_DIR}/src/vllm/file_list.csv
fi