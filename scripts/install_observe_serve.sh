#!/usr/bin/env bash
# 档案 #75/#94/#95/#125/#133/#140-145：一键启动受管服务，仅观察用户AISBench。
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
exec "${OSCAR_PYTHON:-python3}" -m tools.observe_serve "$@"
