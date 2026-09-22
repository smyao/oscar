#!/usr/bin/env bash
# Archive #51/#68/#70-73/#129: explicit eager prefill checkpoints, same target/probes/graph configuration.
set -euo pipefail
TASK_ROOT="${BASH_SOURCE[0]%/*}/.."
cd -- "${TASK_ROOT}"
export OSCAR_DEBUG_SYNC=1
export OSCAR_DEBUG_MIN_TOKENS="${OSCAR_DEBUG_MIN_TOKENS:-1024}"
exec bash scripts/install_probe_serve.sh "$@"
