#!/usr/bin/env bash
# 档案 #34/#36/#94/#95/#125/#148：使用原生预算、保留候选算子，编译失败即停止，输出实时到终端。
# 基于4defe206；不设置--max-num-batched-tokens，不启用whole32K调度；直启不运行probe。
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
export PYTHONUNBUFFERED=1
exec "${OSCAR_PYTHON:-python3}" -m tools.install_serve --variant candidate --native-token-budget "$@"
