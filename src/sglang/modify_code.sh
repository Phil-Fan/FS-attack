#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TARGET_DIR="${TARGET_DIR:-/root/miniconda3/lib/python3.10/site-packages/sglang}"
PATCH_VARIANT="${SGLANG_PATCH_VARIANT:-l40s}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)}"

if [ -z "${PYTHON_BIN}" ]; then
  echo "ERROR: python3/python not found in PATH"
  exit 1
fi

"${PYTHON_BIN}" "${REPO_DIR}/src/sglang/patch.py" --target-dir "${TARGET_DIR}" --variant "${PATCH_VARIANT}"
