#!/usr/bin/env bash
# 档案 #75/#94/#95/#116/#117/#120：shell仅exec固定Python入口，不吞相位退出码或用heredoc管道。
# --rear-cards 由 deploy 生成本次有效配置，所有子探针与正式服务共享卡4-7/端口7878。
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
exec "${OSCAR_PYTHON:-python3}" -m tools.deploy "$@"
