#!/usr/bin/env bash
# 档案 #27/#29/#94/#125：用户指定 node93 前四卡、SDK 顺序与配置路径；不改 SDK 文件、不启动服务。
# Source from the OSCAR project. SDK final failures stop setup with their real status.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    printf '%s\n' '[oscar] source this file in the current Bash: source scripts/env_node93_memcache.sh' >&2
    exit 2
fi

_oscar_node93_source_sdk() {
    local oscar_sdk_path="$1" oscar_sdk_flags="$-" oscar_sdk_status oscar_sdk_pipefail=0
    shift  # Do not pass this helper's path argument to the SDK script.
    if [[ -o pipefail ]]; then oscar_sdk_pipefail=1; fi
    # SDK environment scripts may read unset LD_LIBRARY_PATH/PYTHONPATH.
    # Capture their final status explicitly; a conditional source does not
    # enforce errexit on each command inside an external SDK script.
    set +eu
    if source "$oscar_sdk_path"; then
        oscar_sdk_status=0
    else
        oscar_sdk_status=$?
    fi
    if (( oscar_sdk_pipefail )); then set -o pipefail; else set +o pipefail; fi
    if [[ "$oscar_sdk_flags" == *u* ]]; then set -u; else set +u; fi
    if [[ "$oscar_sdk_flags" == *e* ]]; then set -e; else set +e; fi
    return "$oscar_sdk_status"
}

_oscar_node93_environment() {
    local oscar_sdk_path oscar_sdk_status
    export PYTHONHASHSEED=0 || return $?
    export MMC_LOCAL_SERVICE_BACKEND_ID=1 || return $?
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 || return $?
    export CUDA_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" || return $?
    for oscar_sdk_path in \
        /usr/local/Ascend/ascend-toolkit/set_env.sh \
        /usr/local/Ascend/nnal/atb/set_env.sh \
        /usr/local/memfabric_hybrid/set_env.sh \
        /usr/local/memcache_hybrid/set_env.sh; do
        printf '[oscar] source %s\n' "$oscar_sdk_path"
        if _oscar_node93_source_sdk "$oscar_sdk_path"; then
            :
        else
            oscar_sdk_status=$?
            printf '[oscar] FAILED source=%s rc=%s\n' "$oscar_sdk_path" "$oscar_sdk_status" >&2
            unset -f _oscar_node93_source_sdk _oscar_node93_environment
            return "$oscar_sdk_status"
        fi
    done
    # SDK scripts may change these values; the user's explicit selection wins.
    export PYTHONHASHSEED=0 || return $?
    export MMC_LOCAL_SERVICE_BACKEND_ID=1 || return $?
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 || return $?
    export CUDA_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" || return $?
    # SDK defaults must not replace these user-selected final bindings.
    # Exporting a path does not assert that its file exists or is SDK-valid.
    export MMC_META_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-meta.conf || return $?
    export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf || return $?
    export MMC_DIR_STORE_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/dir_store.conf || return $?
    unset -f _oscar_node93_source_sdk _oscar_node93_environment
}

# Explicitly capture the function result before returning from a sourced file;
# this also avoids older Bash errexit skipping the caller's failure handler.
if _oscar_node93_environment; then
    return 0
else
    return $?
fi
