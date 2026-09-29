#!/usr/bin/env bash
# 档案 #75/#94/#95/#116/#117/#120/#125：shell仅exec固定Python入口，不吞相位退出码或用heredoc管道。
# 一键编译安装后直接启动正式服务：不运行任何测试或probe；输出实时打印到当前终端。
# --variant candidate 显式启用C4/q1及fast unpack；省略则遵循输入配置。
# --rear-cards 本次使用物理卡4,5,6,7和端口7878；默认配置不变。
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
export PYTHONUNBUFFERED=1
# --rear-cards must hide cards 0-3 before Python, CANN, torch_npu or vLLM is
# imported. Cleanup remains scoped to child process groups created by this
# launcher; no process-name or machine-wide NPU kill is permitted.
for arg in "$@"; do
  if [[ "${arg}" == "--rear-cards" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="4,5,6,7"
    break
  fi
done
exec "${OSCAR_PYTHON:-python3}" -m tools.install_serve "$@"
