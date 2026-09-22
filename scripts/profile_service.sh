#!/usr/bin/env bash
# 档案 #70–73/#131/#132：一键有界设备trace窗口，禁止手工改配置或手工调端点。
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
export OSCAR_PROFILER=1
export OSCAR_PROFILE_DIR="${OSCAR_PROFILE_DIR:-reports/npu_profile}"
exec bash scripts/install_probe_serve.sh "$@"
