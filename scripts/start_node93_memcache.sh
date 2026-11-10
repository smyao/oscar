#!/usr/bin/env bash
# 档案 #27/#29/#74/#75/#94/#125/#175/#176：复用已安装SDK；显式主机准备失败即停止，前台保留真实退出码。
set -euo pipefail
PREPARE_HOST=0
for argument in "$@"; do
    case "${argument}" in
        --prepare-host) PREPARE_HOST=1 ;;
        --help|-h)
            printf '%s\n' \
                'Usage: bash scripts/start_node93_memcache.sh [--prepare-host]' \
                '--prepare-host writes nr_hugepages=200000, remounts /dev/shm at 8G, and creates /home/kvcache.' \
                'Inspect those operations without executing: python3 -m tools.node93_host --plan'
            exit 0 ;;
        *) printf '[oscar] unknown argument: %s\n' "${argument}" >&2; exit 2 ;;
    esac
done
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
if (( PREPARE_HOST )); then
    if "${OSCAR_PYTHON:-python3}" -m tools.node93_host; then
        :
    else
        exit "$?"
    fi
fi
if source scripts/env_node93_memcache.sh; then
    :
else
    exit "$?"
fi
"${OSCAR_PYTHON:-python3}" -m tools.memcache_config prepare-node93
exec "${OSCAR_PYTHON:-python3}" -m tools.memcache_config serve-meta --config "${MMC_META_CONFIG_PATH}"
