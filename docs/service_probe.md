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

一键入口先启动原生服务，对 **4 路 20K/23K/27K/30K 不等长 synthetic streaming 请求**建立本轮基线，确认原生 worker 和 NPU 资源释放，再运行上述 OSCAR 完整功能探针与同一组性能请求。输入由固定句子重复生成，报告标为 `synthetic_repeated_sentence`，不代表用户自有负载。四条请求同时释放，每条使用不同的确定性 `cache_salt`，性能诊断每条输出 **64 token**、请求截止仍为 300 秒；原有串行与混合功能探针继续输出16 token。报告保留精确 prompt ID hash、SSE 接收时钟的 TTFT/TPOT/ITL/E2E、输出 token ID、失败/超时、每秒 Running/Waiting、`/metrics` 前后快照和累计 token 差值。原生或 OSCAR 请求不完整、资源未释放、指标缺失，或 OSCAR 任一已测请求延迟/吞吐未达冻结的原生比值门时，**一键部署停止，不进入正式服务**。这是单批客户端速度诊断门；预热状态不完全配对，也不构成完整多工况性能验收或设备算子计时。

**4 路 synthetic 诊断快路径**（不跑完整安装/功能相位；同一命令自动校验算子并跑原生→OSCAR）：

```bash
git pull --ff-only && bash scripts/probe_concurrency.sh
```

快路径也只需执行一次；它先按当前源码/CANN/SOC签名检查算子产物，签名不变则复用，变更则自动重编，并在新构建后跑冻结oracle的真NPU CV/旋转数值探针及资源释放门，任何失败都会阻止服务启动。只有构建产物、选卡、验收配置和oracle指纹完全一致时，快路径才标注`reused_prior_evidence`复用前次NPU数值证据；正式一键流程始终要求本轮新鲜数值探针。随后依次拉起两只托管服务，对相同四条 synthetic streaming 请求测量，确认每轮进程与NPU资源释放，再自动比较。原生只是显式配对基线，OSCAR失败不会切换执行路由。终端只保留少量阶段/资源状态、错误和最多8行配对 `PERF_*` 摘要；完整日志与JSON在本轮`logs/paired-concurrency-*/`。`PROGRESS_BUCKET`若在详细报告中出现，只是跨TP rank的 **host 心跳间隔**，不是单步设备算子时间。快路径跳过完整功能门，不能代替正式部署。

配对在脚本内自动完成：先核对target配置、模型指纹、精确prompt ID、输出长度、cache salt和到达方式，再列出TTFT/TPOT/E2E与吞吐比；SSE单次批量到达导致TPOT无法计时时明确报`needs_evidence`。终端的`PERF_STATUS`、`PERF_QUEUE`、`PERF_TTFT_MS`、`PERF_TPOT_MS`、`PERF_E2E_MS`、`PERF_THROUGHPUT`、`PERF_VERDICT`、`PERF_EVIDENCE`就是直接复制给维护者的重点行。完整证据路径在最后一行。

**用户自己的 32 并发 20–30K 压测**：`bash scripts/install_probe_serve.sh` 全门通过并拉起正式服务后，等待终端的 `OBSERVER_READY`，再运行原有压测程序向配置端口（当前 `8989`）发流量。正式 supervisor 自带被动观察器，只发 `/metrics` GET，**不发送任何压测请求**。它按同一负载窗口保存每秒 Running/Waiting、prompt/generation 累计计数器、MTP 增量、原始 metrics 前后快照，并在服务停止后把 OSCAR trace 按窗口汇总。观察结果在本轮 `logs/<时间戳>/serve/external-load.json`，trace 汇总在同目录的 `progress-summary.json`；常规INFO保存在完整日志，终端只实时显示阶段结果和错误。若用户仍以 `scripts/serve_direct.sh` 直拉服务，可另开终端运行 `python3 -m benchmarks.passive --variant oscar --url http://127.0.0.1:8989 --output reports/external-oscar.json` 接入相同的被动 metrics 观察。

被动 `/metrics` 无法得知客户端恰好提交了 32 条、每条实际 token 长度、请求级 TTFT/ITL/E2E 或失败率；这些要从用户原压测程序的结果、请求清单与终态统计并入配对分析。首次采样已在流量中、metrics 缺计数器或采样有断点时，窗口标 `partial`，不报告完整窗口吞吐。原生对照须显式拉起同配置的 native 服务、跑**同一客户端与数据集**，再比较相同窗口；设备相位成本另需独立 device timing。健康、图回放、CPU 测试或 synthetic 请求均不推出性能追平。

每次服务启动使用唯一 `OSCAR_TRACE_DIR=.../trace-<uuid>`，不会把旧 trace 当成此次执行的证据。图回放事件只声称 launch 返回；真实请求完成记录为另一个状态。HTTP 输出不建立 kernel 精度门、logits/任务质量、MTP 质量对照或性能通过，这些字段仍为 `not_run`。

`attention_dispatched`/`cache_layout` 等 `emit_once` 证据按签名去重，长 prefill 或 decode 期间不再重复写入。为让 REQUEST_WAIT 的 worker 进展不冻结在旧签名上，每个 FULL 层还按 `OSCAR_PROGRESS_INTERVAL_SECONDS`（默认 5 秒）写入 `attention_progress` 心跳，`wall_time` 与 `max_seq_len` 随 chunk 推进刷新。FULL_DECODE_ONLY 图回放绕过 Python attention 路径，decode 阶段的存活心跳由图回放包装器以同间隔写入 `graph_replay_progress`。心跳只证明宿主侧存活与推进，不是设备完成或性能证据。逐请求的 REQUEST_WAIT/REQUEST_ERROR 和 worker 明细保存在相位日志、`status.json` 与 trace 目录；默认一键终端仅显示故障、`PERF_*` 和每60秒的阶段进度，不刷常规心跳。host相位打点写入`timing-<pid>.jsonl`。

## 进程与资源边界

服务在固定项目 cwd 和独立 session/process group 中启动。配置端口被占用时明确失败。SIGINT、SIGTERM、HTTP 失败、早退和超时都会进入 `finally`，仅向 supervisor 自己创建的进程组发送 SIGTERM；有界 grace 后仍未退出才对同组 SIGKILL。不会按进程名杀人、重置 NPU 或处理其他服务。

启动前和清理后，通过独立短进程执行 `torch_npu` 的 `mem_get_info()`，分别记录四个所选物理卡对应的 logical device、free bytes 与 total bytes；观察进程退出后不在 supervisor 内残留 NPU context。不解析 `npu-smi` 的文本列。

资源释放默认等待最多 30 秒，允许每卡 free memory 比启动前少 **256 MiB**，total bytes 必须一致。该容差是明确的驱动/观察上下文资源检查容差，不是性能或量化质量门。可在目标配置中预先设置 `resource_release_timeout_seconds` 和 `resource_release_tolerance_bytes`。外部作业同时占用卡导致 free memory 未恢复时，报告失败并保留每次读数，不擅自判定是谁占用了显存，更不会释放别人的资源。观察工具错误立即报告，不用零值替代、不无限重试。

最终报告的顶层 `status=passed` 仅在完整临时 probe 与清理结束后写入；`resource_release=passed|failed|not_run` 是独立闸。详细前后资源读数和有界等待证据位于 `resource_evidence`。`server_lifecycle.json`、`server.log`、原生 metrics、逐条 trace、请求响应和 `probe_result.log` 均保留。中间 `requests_passed_cleanup_pending` 不可解锁正式服务。

启动、请求和整个临时相位有有限期限；原生 metrics 的异步更新最多观察 15 秒。正式服务验证完成后的前台运行是用户要求的常驻状态，不受临时 probe 的相位期限终止。

## 本地验证范围

`tests/test_service_workflow.py` 启动真实的测试 HTTP 子进程，注入错误 token 计数、无路由证据、无 MTP、早退、启动挂起、SIGTERM、SIGINT 和显存未回收，验证 supervisor 的故障处理。测试中的 HTTP、trace 和资源数据明确是 fixture，不能作为任何 NPU、TP4、图、MTP 或性能验收记录。没有 NPU 的 VM 只能验证这些流程与 CANN CPU debugger，不会产出真实服务通过。
