#!/usr/bin/env bash
# 档案 #75/#94/#95/#125/#140-147：一键实验服务，保留精度门与退出码，跳过配对性能验收。
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
exec "${OSCAR_PYTHON:-python3}" -m tools.aisbench_serve "$@"
