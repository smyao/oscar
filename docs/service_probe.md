# 全服务探针与资源回收

档案依据：G21/G22（NPU free memory 与错误的无限等待），G31/#50–52/#68（超时和 worker 挂起），#27（只有 readiness 不等于服务实现），#72/#94–95（失败结果和日志不能丢失），#74–78/#84（原生组成、导入与父进程线程池），#117（固定 cwd）。本流程使用当前 `configs/target.json`，不会从环境或旧任务继承卡号。

临时完整探针：

```bash
python -m tools.service_probe --config configs/target.json \
  --output reports/service_probe.json --log-dir logs/service-probe
```

正式服务使用同一个 supervisor，并在短请求与必需证据通过后保持前台运行：

```bash
python -m tools.service_probe --config configs/target.json \
  --output reports/formal_service.json --log-dir logs/formal-service --serve
```

部署入口负责先完整 probe、核验本轮原生树未改动，再启动正式服务。两种模式都通过 `tools.target_cli` 拉起 vLLM，保留 TP4、DP1、FULL_DECODE_ONLY、三 token Qwen3.5 MTP、仅 draft 的 eager、原生 GDN 和全部目标参数。未明确选择四张物理 NPU 时，流程在任何资源观察或服务启动前失败。

## 实际请求与证据

等待 `/health` 成功后，探针用模型本地 `AutoTokenizer` 编码固定句子，不访问下载端点；重复并裁切 token ID，构造精确长度为 128、16384、32768、50000 的 `/v1/completions` 请求。每个请求要求至少 16 个输出 token，并核对服务返回的 prompt token 数。串行请求后，再提交包含这四种长度的并发请求，以覆盖 prefix 重用、混合长度和 decode。正式模式至少执行真实 128-token 请求与 16-token 输出。

HTTP 200 只说明请求返回。最终通过还必须满足：

- 每个 TP rank 0–3 的 `cache_layout` 与真实模型配置声明的所有 FULL 层一致，包含 MTP FULL 层；GDN reshape 标记为原生，物理布局尺寸有效。
- 每个 cache 层均有 `attention_dispatched` 且 `route=ascendc_int2_cv`；四个 rank 的层集合相同。
- 四个 rank 均有 FULL 模式的 `graph_capture_return` 与 `graph_replay_launch_return`，并有相同的 captured/replayed descriptor。
- 原生 `/metrics` 的 speculative drafts 和 draft tokens 确实增长，同时记录 accepted tokens、接受率及平均接受长度。
- 所属服务进程组已退出，并且所选 NPU 的 free memory 按下面规则恢复。

每次服务启动使用唯一 `OSCAR_TRACE_DIR=.../trace-<uuid>`，不会把旧 trace 当成此次执行的证据。图回放事件只声称 launch 返回；真实请求完成记录为另一个状态。HTTP 输出不建立 kernel 精度门、logits/任务质量、MTP 质量对照或性能通过，这些字段仍为 `not_run`。

`attention_dispatched`/`cache_layout` 等 `emit_once` 证据按签名去重，长 prefill 或 decode 期间不再重复写入。为让 REQUEST_WAIT 的 worker 进展不冻结在旧签名上，每个 FULL 层还按 `OSCAR_PROGRESS_INTERVAL_SECONDS`（默认 5 秒）写入 `attention_progress` 心跳，`wall_time` 与 `max_seq_len` 随 chunk 推进刷新。FULL_DECODE_ONLY 图回放绕过 Python attention 路径，decode 阶段的存活心跳由图回放包装器以同间隔写入 `graph_replay_progress`。心跳只证明宿主侧存活与推进，不是设备完成或性能证据。

## 进程与资源边界

服务在固定项目 cwd 和独立 session/process group 中启动。配置端口被占用时明确失败。SIGINT、SIGTERM、HTTP 失败、早退和超时都会进入 `finally`，仅向 supervisor 自己创建的进程组发送 SIGTERM；有界 grace 后仍未退出才对同组 SIGKILL。不会按进程名杀人、重置 NPU 或处理其他服务。

启动前和清理后，通过独立短进程执行 `torch_npu` 的 `mem_get_info()`，分别记录四个所选物理卡对应的 logical device、free bytes 与 total bytes；观察进程退出后不在 supervisor 内残留 NPU context。不解析 `npu-smi` 的文本列。

资源释放默认等待最多 30 秒，允许每卡 free memory 比启动前少 **256 MiB**，total bytes 必须一致。该容差是明确的驱动/观察上下文资源检查容差，不是性能或量化质量门。可在目标配置中预先设置 `resource_release_timeout_seconds` 和 `resource_release_tolerance_bytes`。外部作业同时占用卡导致 free memory 未恢复时，报告失败并保留每次读数，不擅自判定是谁占用了显存，更不会释放别人的资源。观察工具错误立即报告，不用零值替代、不无限重试。

最终报告的顶层 `status=passed` 仅在完整临时 probe 与清理结束后写入；`resource_release=passed|failed|not_run` 是独立闸。详细前后资源读数和有界等待证据位于 `resource_evidence`。`server_lifecycle.json`、`server.log`、原生 metrics、逐条 trace、请求响应和 `probe_result.log` 均保留。中间 `requests_passed_cleanup_pending` 不可解锁正式服务。

启动、请求和整个临时相位有有限期限；原生 metrics 的异步更新最多观察 15 秒。正式服务验证完成后的前台运行是用户要求的常驻状态，不受临时 probe 的相位期限终止。

## 本地验证范围

`tests/test_service_workflow.py` 启动真实的测试 HTTP 子进程，注入错误 token 计数、无路由证据、无 MTP、早退、启动挂起、SIGTERM、SIGINT 和显存未回收，验证 supervisor 的故障处理。测试中的 HTTP、trace 和资源数据明确是 fixture，不能作为任何 NPU、TP4、图、MTP 或性能验收记录。没有 NPU 的 VM 只能验证这些流程与 CANN CPU debugger，不会产出真实服务通过。
