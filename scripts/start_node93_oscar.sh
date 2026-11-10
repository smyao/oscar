#!/usr/bin/env bash
# 档案 #74/#75/#94/#125/#175/#176：保留已安装SDK，绑定node93配置，失败不启动正式服务。
set -euo pipefail
for argument in "$@"; do
    if [[ "${argument}" == "--help" || "${argument}" == "-h" ]]; then
        printf '%s\n' \
            'Usage: bash scripts/start_node93_oscar.sh [install_serve arguments]' \
            'Sources the four installed SDKs, then starts the node93 candidate.' \
            '--plan is passed to install_serve and still requires the installed SDK environment.'
        exit 0
    fi
done
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
if source scripts/env_node93_memcache.sh; then
    :
else
    exit "$?"
fi
exec bash scripts/install_serve.sh --config configs/benchmark_node93.json --variant candidate "$@"
