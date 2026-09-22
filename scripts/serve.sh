#!/usr/bin/env bash
# 档案 #27/#94：统一正式入口，未完成的运行时不得以原生服务冒充。
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
exec bash scripts/install_probe_serve.sh "$@"
