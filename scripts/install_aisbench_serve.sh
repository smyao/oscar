#!/usr/bin/env bash
# 档案 #75/#94/#95/#116/#117/#120/#125：shell 仅 exec 固定 Python
# 入口，实时保留安装、编译和服务输出及真实退出码。
# 当前分支 AISBench 入口固定使用物理卡 4,5,6,7 和端口 7878。
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
export PYTHONUNBUFFERED=1
exec "${OSCAR_PYTHON:-python3}" -m tools.install_serve --rear-cards "$@"
