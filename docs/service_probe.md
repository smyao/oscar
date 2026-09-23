# 全服务探针与资源回收

档案依据：G21/G22（NPU free memory 与错误的无限等待），G31/#50–52/#68（超时和 worker 挂起），#27（只有 readiness 不等于服务实现），#70–73/#130–139（设备与客户端计时分离、并发工作量和真实负载边界），#72/#94–95（失败结果和日志不能丢失），#74–78/#84（原生组成、导入与父进程线程池），#117（固定 cwd）。本流程使用当前 `configs/target.json`，不会从环境或旧任务继承卡号。

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

上述功能证据门全部通过后（非 serve 模式），探针在同一只托管服务上执行 **4 路 20K/23K/27K/30K 不等长 synthetic streaming 诊断**。输入由固定句子重复生成，报告明确标为 `synthetic_repeated_sentence`，**不代表用户的真实 32 并发数据集**。四条请求同时释放，每条用不同的 `cache_salt` 请求前缀隔离，输出 16 token，保留 300 秒请求截止。逐请求记录精确 prompt ID hash、SSE 接收时钟的 TTFT/TPOT/ITL/E2E、输出 token ID、失败/超时；同时记录每秒 Running/Waiting、原生 `/metrics` 前后快照和 prompt/generation 计数器差值。4/4 请求完成、usage 一致、至少一条有效调度采样、计数器可读且单调才记为 `measured`；否则 `failed`，**一键部署不进入正式服务**。这是性能诊断门，不是与原生追平的验收结论；客户端时间不是设备时间，算子精度仍按冻结门记录。

**4 路 synthetic 诊断快路径**（不跑安装/编译/功能相位）：

```bash
git pull --ff-only && bash scripts/probe_concurrency.sh
```

它把心跳间隔收紧到 `OSCAR_PROGRESS_INTERVAL_SECONDS=1`（生产时序、不插同步），拉起托管服务后只跑四条 synthetic streaming 请求，输出 `SYNTHETIC_MIXED` 与逐请求 JSON，并对 trace 目录运行 `tools/summarize_progress` 打印 `PROGRESS_BUCKET`。后者是 **host 心跳间隔**（`rank_seconds` 为 TP rank 累计、`mean_rank_s` 为按 rank 均值），不是单步 device 算子时间。`bash scripts/probe_concurrency.sh --native` 才显式启动原生服务取同样的受控基线，OSCAR 失败时绝不自动切换。快路径不执行完整功能门或资源释放验收，不能代替正式部署。

两次快路径的 `report.json` 可用 `python3 -m benchmarks.mixed compare --native <原生报告> --oscar <OSCAR报告> --output <对照报告>` 配对；工具先核对 target 配置、模型配置/权重索引指纹、四条精确 prompt ID、输出长度与到达方式，再逐请求列出 TTFT/TPOT/E2E 比值和两端吞吐比。对照状态仅为 `diagnostic_measured`，不解锁真实性能或精度验收。

**用户自己的 32 并发 20–30K 压测**：`bash scripts/install_probe_serve.sh` 全门通过并拉起正式服务后，等待终端的 `OBSERVER_READY`，再运行原有压测程序向配置端口（当前 `8989`）发流量。正式 supervisor 自带被动观察器，只发 `/metrics` GET，**不发送任何压测请求**。它按同一负载窗口保存每秒 Running/Waiting、prompt/generation 累计计数器、MTP 增量、原始 metrics 前后快照，并在服务停止后把 OSCAR trace 按窗口汇总。观察结果在本轮 `logs/<时间戳>/serve/external-load.json`，trace 汇总在同目录的 `progress-summary.json`；服务端日志实时打印到当前终端。若用户仍以 `scripts/serve_direct.sh` 直拉服务，可另开终端运行 `python3 -m benchmarks.passive --variant oscar --url http://127.0.0.1:8989 --output reports/external-oscar.json` 接入相同的被动 metrics 观察。

被动 `/metrics` 无法得知客户端恰好提交了 32 条、每条实际 token 长度、请求级 TTFT/ITL/E2E 或失败率；这些要从用户原压测程序的结果、请求清单与终态统计并入配对分析。首次采样已在流量中、metrics 缺计数器或采样有断点时，窗口标 `partial`，不报告完整窗口吞吐。原生对照须显式拉起同配置的 native 服务、跑**同一客户端与数据集**，再比较相同窗口；设备相位成本另需独立 device timing。健康、图回放、CPU 测试或 synthetic 请求均不推出性能追平。

每次服务启动使用唯一 `OSCAR_TRACE_DIR=.../trace-<uuid>`，不会把旧 trace 当成此次执行的证据。图回放事件只声称 launch 返回；真实请求完成记录为另一个状态。HTTP 输出不建立 kernel 精度门、logits/任务质量、MTP 质量对照或性能通过，这些字段仍为 `not_run`。

`attention_dispatched`/`cache_layout` 等 `emit_once` 证据按签名去重，长 prefill 或 decode 期间不再重复写入。为让 REQUEST_WAIT 的 worker 进展不冻结在旧签名上，每个 FULL 层还按 `OSCAR_PROGRESS_INTERVAL_SECONDS`（默认 5 秒）写入 `attention_progress` 心跳，`wall_time` 与 `max_seq_len` 随 chunk 推进刷新。FULL_DECODE_ONLY 图回放绕过 Python attention 路径，decode 阶段的存活心跳由图回放包装器以同间隔写入 `graph_replay_progress`。心跳只证明宿主侧存活与推进，不是设备完成或性能证据。终端上的 REQUEST_WAIT/REQUEST_ERROR 行是每 rank 一段的紧凑摘要（event、层、kv、age）；完整 worker 记录仍保存在逐请求 `status.json` 与 trace 目录，host 相位打点写入 `timing-<pid>.jsonl`，不向终端刷屏。

## 进程与资源边界

服务在固定项目 cwd 和独立 session/process group 中启动。配置端口被占用时明确失败。SIGINT、SIGTERM、HTTP 失败、早退和超时都会进入 `finally`，仅向 supervisor 自己创建的进程组发送 SIGTERM；有界 grace 后仍未退出才对同组 SIGKILL。不会按进程名杀人、重置 NPU 或处理其他服务。

启动前和清理后，通过独立短进程执行 `torch_npu` 的 `mem_get_info()`，分别记录四个所选物理卡对应的 logical device、free bytes 与 total bytes；观察进程退出后不在 supervisor 内残留 NPU context。不解析 `npu-smi` 的文本列。

资源释放默认等待最多 30 秒，允许每卡 free memory 比启动前少 **256 MiB**，total bytes 必须一致。该容差是明确的驱动/观察上下文资源检查容差，不是性能或量化质量门。可在目标配置中预先设置 `resource_release_timeout_seconds` 和 `resource_release_tolerance_bytes`。外部作业同时占用卡导致 free memory 未恢复时，报告失败并保留每次读数，不擅自判定是谁占用了显存，更不会释放别人的资源。观察工具错误立即报告，不用零值替代、不无限重试。

最终报告的顶层 `status=passed` 仅在完整临时 probe 与清理结束后写入；`resource_release=passed|failed|not_run` 是独立闸。详细前后资源读数和有界等待证据位于 `resource_evidence`。`server_lifecycle.json`、`server.log`、原生 metrics、逐条 trace、请求响应和 `probe_result.log` 均保留。中间 `requests_passed_cleanup_pending` 不可解锁正式服务。

启动、请求和整个临时相位有有限期限；原生 metrics 的异步更新最多观察 15 秒。正式服务验证完成后的前台运行是用户要求的常驻状态，不受临时 probe 的相位期限终止。

## 本地验证范围

`tests/test_service_workflow.py` 启动真实的测试 HTTP 子进程，注入错误 token 计数、无路由证据、无 MTP、早退、启动挂起、SIGTERM、SIGINT 和显存未回收，验证 supervisor 的故障处理。测试中的 HTTP、trace 和资源数据明确是 fixture，不能作为任何 NPU、TP4、图、MTP 或性能验收记录。没有 NPU 的 VM 只能验证这些流程与 CANN CPU debugger，不会产出真实服务通过。
