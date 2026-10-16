set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
set -u
cd /tmp/oscar-bf16-clusters-20261016
unset LD_PRELOAD
case "$1" in
device) cmake -S csrc/experiments/bf16_clusters_20261016 -B build-rne-device -DSOC_VERSION=ascend910b4 -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" && cmake --build build-rne-device --parallel 2 ;;
cpu-build) cmake -S csrc/experiments/bf16_clusters_20261016 -B build-rne-cpu -DSOC_VERSION=ascend910b4 -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DOSCAR_BF16_CLUSTERS_CPU_DEBUG=ON && cmake --build build-rne-cpu --parallel 2 ;;
cpu) build-rne-cpu/oscar_bf16_clusters_cpu c4 fixtures/c4_C1_hkv2_s3 oracle/c4_C1_hkv2_s3 ;;
esac
