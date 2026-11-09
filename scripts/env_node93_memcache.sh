#!/usr/bin/env bash
# 档案 #27/#29/#94/#125：本轮用户显式指定 node93 前四卡和已有 SDK 配置；不改设备配置文件、不启动服务。
# Source this file from the OSCAR project before using configs/benchmark_node93.json.
set -euo pipefail
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf
export MMC_META_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-meta.conf
export PYTHONHASHSEED=0
