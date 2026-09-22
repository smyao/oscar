#!/usr/bin/env bash
# Archive #79-85/#96/#117: independent Lima workspace and reproducible CANN CPU validation.
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
exec "${OSCAR_PYTHON:-python3}" -m tools.vm_validate "$@"
