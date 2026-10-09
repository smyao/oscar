#!/usr/bin/env bash
# 档案 #75/#94/#95/#116/#117/#120/#125：shell仅exec固定Python入口，不吞相位退出码或用heredoc管道。
# 一键编译安装后直接启动正式服务：不运行任何测试或probe；输出实时打印到当前终端。
# --variant candidate 使用tools/serving_variants.py统一预设：组合decode、current-only、MTP精简、混合分流及整段prefill。
# 默认及兼容参数 --rear-cards 均使用物理卡4,5,6,7和端口7878。
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
export PYTHONUNBUFFERED=1
exec "${OSCAR_PYTHON:-python3}" -m tools.install_serve "$@"
