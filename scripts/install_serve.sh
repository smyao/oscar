#!/usr/bin/env bash
# 档案 #75/#94/#95/#116/#117/#120/#125：shell仅exec固定Python入口，不吞相位退出码或用heredoc管道。
# 一键编译安装后直接启动正式服务：不运行任何测试或probe；输出实时打印到当前终端。
# --variant candidate 显式启用已实现的C4/q1优化；省略则遵循输入配置。
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
export PYTHONUNBUFFERED=1
exec "${OSCAR_PYTHON:-python3}" -m tools.install_serve "$@"
