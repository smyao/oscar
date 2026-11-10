#!/usr/bin/env bash
# 档案 #27/#29/#74/#75/#94/#125/#175/#176/#177：先识别并复用已有服务，冷启动才准备主机/配置。
set -euo pipefail
PREPARE_HOST=0
for argument in "$@"; do
    case "${argument}" in
        --prepare-host) PREPARE_HOST=1 ;;
        --help|-h)
            printf '%s\n' \
                'Usage: bash scripts/start_node93_memcache.sh [--prepare-host]' \
                'Reuses an existing MetaService after an SDK metadata check; no host/configuration changes on reuse.' \
                '--prepare-host applies nr_hugepages=200000, /dev/shm=8G and mkdir /home/kvcache only for a new service.' \
                'Inspect those operations without executing: python3 -m tools.node93_host --plan'
            exit 0 ;;
        *) printf '[oscar] unknown argument: %s\n' "${argument}" >&2; exit 2 ;;
    esac
done
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
if source scripts/env_node93_memcache.sh; then
    :
else
    exit "$?"
fi
START_ARGS=(node93 --config-dir "${MMC_META_CONFIG_PATH%/*}")
if (( PREPARE_HOST )); then
    START_ARGS+=(--prepare-host)
fi
exec "${OSCAR_PYTHON:-python3}" -m tools.memcache_service "${START_ARGS[@]}"
