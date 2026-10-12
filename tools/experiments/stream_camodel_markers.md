# 当前 bundle 的有界 marker FIFO 采集

档案 #132–135：诊断不能改变数学或把异步等待当裸算子时间；#145：不动完成事件及缓冲所有权；#150：保留真实 R32/N128/S3、所有任务及 60 actor，actor 时间不得相加成设备 wall。

## 已实现及 host 证据

`stream_camodel_markers.py` 在没有 `LD_PRELOAD` 的 supervisor 中先创建并读开 60 个 FIFO，再 exec 具有 SDK preload 的 worker。没有运行中文件截断、FD handoff 或首次打开竞态。每个 FIFO 保存完整字节 SHA、字节数和 4 KiB 首尾；仅保留 magic MOV marker，全部 actor 的 marker JSONL 总量最多 32 MiB，转成最终 JSON 时最多暂留第二份。超限仍排空生产者，但结果失败。

worker 导入当前项目的 `window_batch_decode/replay_camodel.py`，仅适配两处：已创建目录必须保持原 inode；`quiet_dumps` 仅略过本轮预创建 FIFO 的精确 device/inode，其余 dump 继续原来的 `/dev/null` 路由。原核、SDK、cache model、全局 raw trace cap 均不变。原 replayer 数值、status、GM guards、profile/actor/task 身份检查照常运行；原最终 passed 在 worker 中临时改成 `numeric_passed_pending_stream_markers`。

supervisor 必须再核对真实 `tasks.bin` 和 `profile.bin` 的 SHA、形状与每 actor 工作计数，从当前 pinned bundle 的遍历/调用关系推导每个 source/field 的调用数。所有 marker 必须闭合、cycle 单调、exclusive 时间守恒且调用数相等；60 个 FIFO 都到达 EOF 且 inode 未被 SDK 替换后才把报告置为 passed。这里的 passed 仅指该采样运行完整，仍不是 NPU/模型/整轮性能验收。

宿主测试：

```bash
.venv/bin/python -m unittest tests.test_stream_camodel_markers -v
```

测试使用普通真实 subprocess producer 写全部 60 FIFO，覆盖超过 pipe buffer、碎片行、SHA 完整、缺/错闭合、数量不符、保留预算耗尽、持续写或缺 EOF、owned inode 例外、worker 目录适配以及真实 supervisor deadline 强制结束拒绝 SIGTERM 的 child。测试 producer 不是 SDK，测试字典不是实际任务计数证据。

已完成 C1025 两路真实 `tasks.bin` + `profile.bin` 的 hash/工作量核对：均匹配 60 actor；按当前调用关系分别应有 q4 12,764 / q1 13,948 对 scope。这是待采集 marker 的数量判据，**不是尚未采集的 marker 数量已通过**。见 `reports/stream_camodel_marker_host_20261009.json`。

`stream_owner.json` 记录 supervisor 与独立 runtime child 的 PID/PGID、命令和实际退出码。外部 SIGTERM/SIGINT 在轮询点转发到本轮 runtime group，最多等 5 秒再 SIGKILL；finally 同样清除 leader 已退出后的同组 writer。FIFO drain 最多再等 5 秒，外部监管器须留 **至少 12 秒 cleanup grace**，不能只杀 collector 而漏掉独立 SDK group。测试包含实际外部 SIGTERM + 拒绝 SIGTERM 的 child，确认 child 被回收后 supervisor 返回失败。

## 首次 SDK 接入

**SDK 对 FIFO 的 fopen/seek/rotate 行为尚未验证。** 不能直接把 host 测试视为兼容证据。首轮需由 root 在排他 slot 中使用已完成数值的 C1025 profile、同库/输入/配置拓扑，仅打开明确的 instruction logging；不要先上更大 context。调用结构如下（所有路径与 deadline 由执行前明确的本轮计划填写）：

```bash
# 在 VM 中，已 source 当前 CANN，LD_LIBRARY_PATH 含 dav_2201 simulator lib。
# SDK preload 只传给 worker；不能提前 export 到 supervisor。
env -u LD_PRELOAD flock --no-fork --exclusive --nonblock /tmp/oscar-cann-validation.lock \
  python3 tools/experiments/stream_camodel_markers.py \
  --replayer csrc/experiments/window_batch_decode/replay_camodel.py \
  --runtime-preload /usr/local/Ascend/cann-9.1.0/aarch64-linux/simulator/dav_2201/lib/libruntime_camodel.so \
  --timeout-seconds 600 -- \
  --root "$RUN_ROOT" --output "$NEW_OUTPUT" \
  --case q4_r32_n128_c1025_s3 --variant candidate \
  --fixture-root "$FIXTURE_ROOT" --export-root "$EXPORT_ROOT" \
  --library "$SAME_PROFILE_LIBRARY" --simulator-config-dir "$LOGGING_CONFIG" \
  --candidate-q4 oscar_bundle_profile_q4_kernel \
  --candidate-q1 oscar_bundle_profile_q1_kernel \
  --candidate-workspace-bytes-per-core 1146880 \
  --baseline-workspace-bytes-per-core 1146880 \
  --cpu-reference optional --instruction-budget-bytes 0 --profile-counters
```

`CAMODEL_CONFIG_PATH` 必须在 worker 启动、SDK 加载前设置成同一个 `$LOGGING_CONFIG`。使用新 output 目录，拒绝覆盖。保持 cache/model/topology 参数，与已完成 arm 的差异只能明确记录为 logging；不能用日志开关差异通过要求相同配置 SHA 的优化 A/B 门。数值输出必须与已完成相同 profile 核逐位相同，并比较 logging 开启后的 task ticks；host 耗时单列。首次成功后再用至少 C2049 检查实际预取稳态，不将 C1025 单 history tile 的份额外推到 64K。

任何 FIFO 替换、seek 导致的 SDK 错误、deadline、输出不完整、计数不合或 task tick 扰动不可接受，都保留为失败；不改变 SDK、不扩大原 trace cap、不重启同一次运行去掩盖失败。
