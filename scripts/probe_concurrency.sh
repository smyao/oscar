#!/usr/bin/env bash
# 档案 #75/#94/#95/#138：shell仅exec固定Python入口；并发阶梯诊断专用快路径。
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
export OSCAR_PROGRESS_INTERVAL_SECONDS="${OSCAR_PROGRESS_INTERVAL_SECONDS:-1}"
STAMP="$(date -u +%Y%m%dT%H%M%S.%6NZ)"
exec "${OSCAR_PYTHON:-python3}" -m tools.service_probe --ladder \
  --log-dir "logs/ladder-${STAMP}" --output "logs/ladder-${STAMP}/ladder.json" "$@"
