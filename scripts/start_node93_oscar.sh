#!/usr/bin/env bash
# 档案 #27/#29/#75/#94/#125/#148：仅对齐卸载环境，保留candidate与原配置的其余参数。
set -euo pipefail
for argument in "$@"; do
    if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
        printf '%s\n' \
            'Usage: bash scripts/start_node93_oscar.sh [install_serve arguments]' \
            'Loads the installed CANN, ATB, MemFabric and MemCache SDKs and enables KV offload.' \
            'All other model/service settings come from the existing OSCAR configuration.' \
            'Requires an already running MemCache MetaService; does not start or reconfigure it.' \
            '--plan still loads the SDK environment; for a host-only plan use install_serve.sh --plan.'
        exit 0
    fi
done
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "$TASK_ROOT"
if source scripts/env_node93_memcache.sh; then
    :
else
    exit "$?"
fi
exec bash scripts/install_serve.sh --variant candidate --kv-offload "$@"
