#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_ASCEND_SOURCE="/vllm-workspace/vllm-ascend"
JOBS=8

while [ "$#" -gt 0 ]; do
    case "$1" in
        --vllm-ascend-source) VLLM_ASCEND_SOURCE="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

: "${ASCEND_HOME_PATH:=/usr/local/Ascend/cann-9.1.0}"
TORCH_NPU_PATH="$(python3 -c 'import pathlib, torch_npu; print(pathlib.Path(torch_npu.__file__).resolve().parent)')"
TORCH_CMAKE_PREFIX="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
BUILD_DIR="$REPO_ROOT/build/ascendc_torch_binding"
OUTPUT_DIR="$REPO_ROOT/build/ascendc"

cmake -S "$REPO_ROOT/ascendc/torch_binding" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX" \
    -DVLLM_ASCEND_SOURCE="$VLLM_ASCEND_SOURCE" \
    -DTORCH_NPU_PATH="$TORCH_NPU_PATH" \
    -DASCEND_HOME_PATH="$ASCEND_HOME_PATH"
cmake --build "$BUILD_DIR" --parallel "$JOBS"
mkdir -p "$OUTPUT_DIR"
cp "$BUILD_DIR/liboscar_ascend_torch.so" "$OUTPUT_DIR/"
echo "[oscar-ascendc] torch binding: $OUTPUT_DIR/liboscar_ascend_torch.so"
