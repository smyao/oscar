#!/usr/bin/env bash
# 档案 #75/#94/#95/#138/#139：只跑4路20/23/27/30K synthetic streaming诊断。
# --native 显式选原生基线；绝不因OSCAR失败自动切换。用户32路真实负载走正式服务被动观察。
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
export OSCAR_PROGRESS_INTERVAL_SECONDS="${OSCAR_PROGRESS_INTERVAL_SECONDS:-1}"
STAMP="$(date -u +%Y%m%dT%H%M%S.%6NZ)"
exec "${OSCAR_PYTHON:-python3}" -m tools.service_probe --synthetic \
  --log-dir "logs/synthetic-mixed-${STAMP}" --output "logs/synthetic-mixed-${STAMP}/report.json" "$@"
