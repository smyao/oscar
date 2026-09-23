#!/usr/bin/env bash
# 档案 #70-#73/#94/#95/#125/#129-#139：一次命令配对跑原生和OSCAR，完整日志落盘，仅打印关键结论。
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
exec "${OSCAR_PYTHON:-python3}" -m tools.paired_concurrency_probe "$@"
