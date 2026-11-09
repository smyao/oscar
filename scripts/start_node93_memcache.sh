#!/usr/bin/env bash
# 档案 #27/#29/#94/#125：复用现场 SDK 环境，配置失败即停止，MetaService 前台输出并保留退出码。
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
source scripts/env_node93_memcache.sh
"${OSCAR_PYTHON:-python3}" -m tools.memcache_config prepare-node93
exec "${OSCAR_PYTHON:-python3}" -m tools.memcache_config serve-meta --config "${MMC_META_CONFIG_PATH}"
