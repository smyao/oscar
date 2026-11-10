#!/usr/bin/env bash
# 档案 #27/#29/#75/#94/#125：只加载用户指定SDK与卸载环境；不启动中央服务、不写SDK配置。
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '%s\n' '[oscar] source this file: source scripts/env_node93_memcache.sh' >&2
    exit 2
fi

_oscar_node93_source_sdk() {
    local oscar_sdk_path="$1" oscar_sdk_flags="$-" oscar_sdk_status oscar_sdk_pipefail=0
    shift
    if [[ -o pipefail ]]; then oscar_sdk_pipefail=1; fi
    # External SDK scripts may read unset variables. Preserve their final
    # status and restore the caller's shell options after sourcing.
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
            return "$oscar_sdk_status"
        fi
    done
    # Keep the explicitly selected values if an SDK script supplied defaults.
    export PYTHONHASHSEED=0 || return $?
    export MMC_LOCAL_SERVICE_BACKEND_ID=1 || return $?
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 || return $?
    export CUDA_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" || return $?
    export MMC_META_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-meta.conf || return $?
    export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf || return $?
    # The user already installed these SDKs; do not resolve generic pip SDKs.
    export OSCAR_OFFLOAD_SDK_SOURCE=installed || return $?
}

if _oscar_node93_environment; then
    unset -f _oscar_node93_source_sdk _oscar_node93_environment
else
    _oscar_node93_env_status=$?
    unset -f _oscar_node93_source_sdk _oscar_node93_environment
    return "$_oscar_node93_env_status"
fi
