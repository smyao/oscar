#!/usr/bin/env bash
# 档案 #75/#94/#95/#125/#133/#140-145：一键启动受管服务，仅观察用户AISBench。
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
export PYTHONUNBUFFERED=1
# This entry is dedicated to the rear-card task. Hide cards 0-3 before any
# Python/CANN/NPU import; all probe children receive the same effective config.
export ASCEND_RT_VISIBLE_DEVICES="4,5,6,7"
exec "${OSCAR_PYTHON:-python3}" -m tools.observe_serve --rear-cards "$@"
