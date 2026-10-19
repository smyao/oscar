#!/usr/bin/env bash
# Archive #94/#95/#125/#133/#140: bounded owned diagnostic and durable logs;
# exact exit code, no HTTP profiler or change to formal serving defaults.
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
export PYTHONUNBUFFERED=1
exec "${OSCAR_PYTHON:-python3}" -m tools.timer32k "$@"
