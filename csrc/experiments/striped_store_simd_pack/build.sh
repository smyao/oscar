#!/usr/bin/env bash
# Archive #75/#94/#125/#145/#154, D.4: one exclusive research CPU/CANN build.
# No install/deploy, no automatic probe, no retries or timeout expansion.
set -eo pipefail
trial_source="$(cd -- "${BASH_SOURCE[0]%/*}" && pwd)"
trial_build="${1:?usage: build.sh NEW_BUILD_DIRECTORY}"
trial_cann="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}"
if [[ -e "${trial_build}" ]]; then
  echo "Refuse existing research build directory: ${trial_build}" >&2
  exit 2
fi
exec 9>/tmp/oscar-cann-validation.lock
flock -n 9
source "${trial_cann}/set_env.sh"
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cmake -S "${trial_source}" -B "${trial_build}/cpu" -DSOC_VERSION=ascend910b4 \
  -DASCEND_HOME_PATH="${trial_cann}" -DOSCAR_SIMD_PACK_CPU_DEBUG=ON
cmake --build "${trial_build}/cpu" -j1
cmake -S "${trial_source}" -B "${trial_build}/device" -DSOC_VERSION=ascend910b4 \
  -DASCEND_HOME_PATH="${trial_cann}"
cmake --build "${trial_build}/device" -j1
