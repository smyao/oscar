#!/usr/bin/env bash
# Build the optional OSCAR AscendC operator against the pinned vLLM-Ascend ABI.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_ASCEND_SOURCE="/vllm-workspace/vllm-ascend"
SOC="ascend910b"
JOBS=8

usage() {
    echo "usage: $0 [--vllm-ascend-source DIR] [--soc ascend910b] [--jobs N]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --vllm-ascend-source) VLLM_ASCEND_SOURCE="$2"; shift 2 ;;
        --soc) SOC="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$JOBS" in
    ''|*[!0-9]*) echo "--jobs must be a positive integer" >&2; exit 2 ;;
esac
[ "$JOBS" -gt 0 ] || { echo "--jobs must be positive" >&2; exit 2; }
[ "$SOC" = "ascend910b" ] || {
    echo "OSCAR AscendC currently supports only ascend910b/910B4" >&2
    exit 2
}

: "${ASCEND_HOME_PATH:=/usr/local/Ascend/cann-9.1.0}"
export ASCEND_HOME_PATH
export ASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH:-$ASCEND_HOME_PATH}"

python3 "$REPO_ROOT/tools/diag_ascendc_env.py"
[ -d "$VLLM_ASCEND_SOURCE/.git" ] || {
    echo "vLLM-Ascend source tree not found: $VLLM_ASCEND_SOURCE" >&2
    exit 2
}
[ -x "$VLLM_ASCEND_SOURCE/csrc/build.sh" ] || {
    echo "vLLM-Ascend csrc/build.sh is missing or not executable" >&2
    exit 2
}

EXPECTED_COMMIT="5cb98caaadeff42b5b62b996e34bb2aaa29d20fd"
ACTUAL_COMMIT="$(git -C "$VLLM_ASCEND_SOURCE" rev-parse HEAD)"
[ "$ACTUAL_COMMIT" = "$EXPECTED_COMMIT" ] || {
    echo "unsupported vLLM-Ascend commit: $ACTUAL_COMMIT" >&2
    echo "expected: $EXPECTED_COMMIT" >&2
    exit 2
}

OP_SOURCE="$REPO_ROOT/ascendc/oscar_int2_paged_attention"
for required in \
    CMakeLists.txt \
    op_host/CMakeLists.txt \
    op_host/oscar_int2_paged_attention_def.cpp \
    op_host/oscar_int2_paged_attention_infershape.cpp \
    op_host/oscar_int2_paged_attention_tiling.cpp \
    op_host/oscar_int2_paged_attention_tiling.h \
    op_kernel/oscar_int2_paged_attention.cpp; do
    [ -f "$OP_SOURCE/$required" ] || {
        echo "incomplete OSCAR AscendC source: $OP_SOURCE/$required" >&2
        exit 2
    }
done

# Build from a disposable copy.  Never patch or clean the user's dirty source
# tree: target images commonly carry local vLLM-Ascend fixes.
BUILD_ROOT="$(mktemp -d /tmp/oscar-ascendc-build.XXXXXX)"
cleanup() { rm -rf "$BUILD_ROOT"; }
trap cleanup EXIT

echo "[oscar-ascendc] staging isolated source in $BUILD_ROOT"
git -C "$VLLM_ASCEND_SOURCE" archive HEAD | tar -x -C "$BUILD_ROOT"
# git archive intentionally excludes submodule contents.  The pinned build
# needs CATLASS headers, so reuse the already checked-out read-only dependency
# from the target source tree without importing unrelated dirty files.
CATLASS_SOURCE="$VLLM_ASCEND_SOURCE/csrc/third_party/catlass"
[ -d "$CATLASS_SOURCE/include" ] || {
    echo "CATLASS submodule is not initialized: $CATLASS_SOURCE" >&2
    exit 2
}
mkdir -p "$BUILD_ROOT/csrc/third_party/catlass"
cp -R "$CATLASS_SOURCE/include" "$BUILD_ROOT/csrc/third_party/catlass/"
mkdir -p "$BUILD_ROOT/csrc/attention/oscar_int2_paged_attention"
cp -R "$OP_SOURCE"/. "$BUILD_ROOT/csrc/attention/oscar_int2_paged_attention/"

python3 "$REPO_ROOT/tools/patch_vllm_ascendc_build.py" \
    --source "$BUILD_ROOT" \
    --operator oscar_int2_paged_attention

echo "[oscar-ascendc] compiling operator for $SOC"
(
    cd "$BUILD_ROOT/csrc"
    bash build.sh \
        --ops=oscar_int2_paged_attention \
        --soc="$SOC" \
        --ophost --opkernel --opapi --pkg \
        -O3 "-j$JOBS"
)

OUTPUT_DIR="$REPO_ROOT/build/ascendc"
mkdir -p "$OUTPUT_DIR"
find "$BUILD_ROOT/csrc" -type f \
    \( -name 'custom_opp_*.run' -o -name 'CANN-custom_ops*.run' \) \
    -exec cp {} "$OUTPUT_DIR/" \;

PACKAGE="$(find "$OUTPUT_DIR" -maxdepth 1 -type f \
    \( -name 'custom_opp_*.run' -o -name 'CANN-custom_ops*.run' \) \
    -print -quit)"
[ -n "$PACKAGE" ] || {
    echo "build completed but no custom-op package was produced" >&2
    exit 3
}
echo "[oscar-ascendc] package: $PACKAGE"
