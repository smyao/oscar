#!/usr/bin/env bash
# 档案 #27/#29/#75/#94/#125/#148/#182：加载卸载环境，默认同时开启卸载和prefix caching。
set -euo pipefail
for argument in "$@"; do
    if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
        printf '%s\n' \
            'Usage: bash scripts/start_node93_oscar.sh [install_serve arguments]' \
            'Loads the installed CANN, ATB, MemFabric and MemCache SDKs; enables KV offload and prefix caching.' \
            'Use --no-enable-prefix-caching only to explicitly disable prefix caching.' \
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
exec bash scripts/install_serve.sh --variant candidate --kv-offload --enable-prefix-caching "$@"
