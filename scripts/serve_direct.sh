#!/usr/bin/env bash
# 档案 #75/#94/#95/#116/#117/#120：shell仅exec固定Python入口，不吞相位退出码或用heredoc管道。
# 直拉正式服务：跳过安装/编译/全部探针，仅用于已成功跑通 install_probe_serve 的真机；服务输出实时打印到当前终端。
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
export PYTHONUNBUFFERED=1
exec "${OSCAR_PYTHON:-python3}" -m tools.target_cli "$@"
