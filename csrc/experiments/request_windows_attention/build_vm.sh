#!/usr/bin/env bash
# Archive #79-85/#94-95/#125/#145/#154/#155: owned tmpfs build, real rc/logs.
# Invoke under flock /tmp/oscar-cann-validation.lock. No kernels run here.
set -eo pipefail
reader_root="/dev/shm/oscar-request-window-reader-20261009"
cd "$reader_root"
mkdir -p logs tmp
df -h /tmp /dev/shm
# This build and all compiler temporaries use its own tmpfs directory.
# No cleanup of prior experiments is needed to run this scoped build.
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
set +u
source "$ASCEND_HOME_PATH/set_env.sh"
set -u
export TMPDIR="$reader_root/tmp"
reader_phase() {
  local reader_name="$1"
  shift
  set +e
  "$@" 2>&1 | tee "logs/$reader_name.log"
  local reader_codes=("${PIPESTATUS[@]}")
  set -e
  printf '%s\n' "${reader_codes[0]}" > "logs/$reader_name.rc"
  printf 'READER_BUILD_PHASE %s rc=%s tee_rc=%s\n' "$reader_name" "${reader_codes[0]}" "${reader_codes[1]}"
  [[ "${reader_codes[0]}" == 0 && "${reader_codes[1]}" == 0 ]]
}
reader_phase cann-config cmake -S csrc/experiments/request_windows_attention -B build/cann -DSOC_VERSION=ascend910b4 -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DOSCAR_REQUEST_WINDOWS_ATTENTION_CPU_DEBUG=OFF
reader_phase cann-build cmake --build build/cann --parallel 2
reader_phase cpu-config cmake -S csrc/experiments/request_windows_attention -B build/cpu -DSOC_VERSION=ascend910b4 -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DOSCAR_REQUEST_WINDOWS_ATTENTION_CPU_DEBUG=ON
reader_phase cpu-build cmake --build build/cpu --parallel 2
df -h /tmp /dev/shm
