#!/usr/bin/env bash
# Archive #94/#95/#125/#133/#143: preserve exits and owned lifecycle; no profiler.
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
export PYTHONUNBUFFERED=1
exec "${OSCAR_PYTHON:-python3}" -m tools.c16_breakdown "$@"
