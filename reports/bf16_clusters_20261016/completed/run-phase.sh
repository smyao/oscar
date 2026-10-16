set -euo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd /tmp/oscar-bf16-clusters-20261016
unset LD_PRELOAD
case "$1" in
configure-cpu) cmake -S csrc/experiments/bf16_clusters_20261016 -B build-cpu -DSOC_VERSION=ascend910b4 -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DOSCAR_BF16_CLUSTERS_CPU_DEBUG=ON ;;
compile-cpu) cmake --build build-cpu --parallel 2 ;;
prepare) mkdir -p "oracle/$3"; build-cpu/oscar_bf16_clusters_cpu "$2" "fixtures/$3" "oracle/$3" prepare-only ;;
cpu) build-cpu/oscar_bf16_clusters_cpu "$2" "fixtures/$3" "oracle/$3" ;;
*) exit 2 ;;
esac
