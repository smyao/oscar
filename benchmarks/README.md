# 配对 HTTP 测量与性能证据比较

档案 #70/#71/#130–#139 要求区分 host launch 时间与 device 时间、逐相位定位慢点；#72 要求自动创建
输出目录；#73 的 31897-token/4.6s 历史数字不能替代本轮同环境配对基线。本工具只审查输入
报告与证据。`benchmarks.measure` 向已启动的本地服务生成测量请求；`benchmarks.compare`
只审查报告，不启动模型。两者都不能用 HTTP 时间或历史日志冒充 NPU 设备证据。

当前目标负载分两种：`bash scripts/probe_concurrency.sh` 发 4 条明确标注 **synthetic** 的
20K/23K/27K/30K 请求；正式服务自带 `benchmarks.passive`，等用户原压测程序发送真实
32 并发 20–30K 请求，仅采 `/metrics`、Running/Waiting 和同窗口 OSCAR trace。
前者能记录客户端 SSE TTFT/TPOT/ITL/E2E、失败率，后者不能从服务端计数器重建这些
请求级数据。被动报告的 `client_request_count`、`prompt_length_distribution` 与
`ttft_itl_e2e_completion` 会保持未观测，直到用户提供原客户端产物；synthetic 报告
不作为真实负载证据。详见[服务探针](../docs/service_probe.md)。

## 真实 HTTP 测量入口

先用当前 target 配置分别拉起明确的 native 或 OSCAR 服务，再测量同一个本地 URL。
测量客户端不启动或修改服务，不会因 OSCAR 失败自动改用 native。两轮应使用同一模型、
配置、文本 dataset、输出 token 数、软件/设备身份文件；`--variant` 仅声明本轮身份，
实际服务 route 仍需要独立证据。原生基线入口由 `tools.target_cli --native` 提供，
保留原生图、GDN、TP 和 MTP 参数。

```bash
python -m benchmarks.measure --variant native --config configs/target.json \
  --output reports/native/http.json --identity reports/server_identity.json
python -m benchmarks.measure --variant oscar --config configs/target.json \
  --output reports/oscar/http.json --identity reports/server_identity.json
```

URL 默认 `http://127.0.0.1:<target.port>`，也可通过 `--url` 指定本地 loopback 地址。
客户端只连接该地址，不接受公网 URL、不经过继承的 HTTP proxy。

默认输入长度明确采用启动文档 §10.1 的标准输入 **16384、32768、50000**，并在报告标注
`default_workload_scope=required_standard_inputs`；默认并发取冻结 `configs/acceptance.json`。
完整验收要求的五个长度和四个设备 q_len 仍原样记录，因此默认并不宣称完整矩阵已完成。
每个 HTTP 工况固定执行 **2 次 warmup、5 次正式 repeat**。
每次 repeat 同时提交指定并发数的请求并等待本批结束，再开始下一批。
`--lengths 16384 32768 --concurrency 1 4` 可缩小开发检查范围；报告明确标记完整矩阵未完成。
`--output-tokens` 默认 32，至少为 2；`--timeout` 默认每请求 300 秒，`--run-timeout`
默认整轮 7200 秒，超时与每个失败请求原样记录。

输入来自本地 tokenizer 编码的固定句子，或 `--dataset <本地UTF-8文本>`。重复并裁切
token ID 得到精确长度，向 `/v1/completions` 直接提交 ID，禁用额外 special tokens。
真实输入 ID、dataset 文本及其 SHA256、模型 config/weight-index 指纹、完整 target
配置与 workload 指纹均落盘。每个请求使用唯一 `cache_salt`，让两轮的冷前缀条件一致，
不会为清缓存而触碰其他请求。warmup 仍会预热计算路径，但不会给正式 repeat 偷留命中前缀。
若最大输入长度加输出长度超过服务的 `max_model_len`，该工况保存为 `not_run` 并解释原因，
不会把 262144 的输入悄悄截成更短的数。

每个请求开启 native protocol 的 `return_token_ids` 与 streaming usage。客户端记录：

- 收到第一个非空 delta token-ID SSE 事件的 TTFT，以及收到 `[DONE]` 的完整延迟；
- 每个 token 的实际客户端接收时间、SSE burst 大小、TPOT 和客户端 ITL P95；同一 MTP
  burst 的 token 共享到达时间，内部间隔为零，不能伪造细分的设备发送时间；
- 服务终态 usage、完整输出 ID、每个请求的失败/超时，以及完成请求数和本批吞吐；
- 每个工况开始和结束的原生 `/metrics` 文本及哈希、MTP drafts/draft/accepted counters
  的增量。metrics 缺失保留 `needs_evidence`，counter 倒退标记失败。

TTFT/TPOT/ITL/e2e 的 clock 都是 `client_monotonic_receive`，不是 NPU device 时间。
吞吐为完成请求的 prompt/generated token 总量除以本批从同时释放到全部请求结束的时长。
各 repeat 的原始请求与指标都保留，不用跨工况平均掩盖慢点。

## HTTP 报告与验收报告的边界

客户端报告采用 `schema_version=1`、`kind=native|oscar`、唯一 `run_id`、冻结策略 hash
以及与比较器相同的 `pair`/`warmup`/`samples` 字段。`measurement_type=http_client`，
正常采集结束的状态为 `needs_evidence`，**不写 `target_npu` 或性能 `passed`**。
CLI 返回 0 只表示此次选择范围的 HTTP 请求数据采集完整；报告中的性能验收仍是未完成。
失败、超时、部分未执行或输入配置错误返回非零。

HTTP 无法指定或观察 kernel 的 MTP verify `q_len`。因此真实数据存入 `http_cases`，
其中 `q_len=null`；要求已知设备 q_len 的正式 `cases` 保持空数组，绝不从一次 HTTP
请求复制出 1/2/3/4 四个所谓已测工况。`phase_ms={}`、`phase_status=needs_evidence`，
也不会自动填入 accuracy、graph、route、memory 或 profiler 证据。

`--identity` 是可选的已核实服务器身份 JSON，包含 `software` 的全部字段（见下）和
`npu_model`，可带与 target 一致的 `devices`、`tp`。其文件路径与 digest 被记录；
客户端不把这个声明当作硬件执行证明。缺身份时也能收集 HTTP 数据，但会明确列入
`needs_evidence`。模型指纹当前范围是本地 config 与 weight-index，未宣称遍历权重字节。

把完整配对身份的 HTTP 报告直接交给比较器，结果应为 `not_run`：仍缺真正的硬件、
设备 q_len、数值和相位证据。身份字段自身缺失或不一致属于结构性无效报告，现有比较器
可能先报 `failed`；两种状态都不能解锁性能验收。

## 证据比较入口

```bash
python3 -m benchmarks.compare \
  --native reports/native/run.json \
  --oscar reports/oscar/run.json \
  --acceptance configs/acceptance.json \
  --target configs/target.json \
  --output reports/paired/comparison.json
```

仅 `status=passed` 返回 0；`failed` 或 `not_run` 返回 2。输出父目录自动创建。
标准库即可运行，不导入 torch/vLLM/NPU。相对 evidence 路径按各自 run.json 所在目录解析。

`examples/run.template.json` 是未测量模板，字段保留 null、空数组与 `template_not_measured`，
不能通过验收。复制并填写真实结果时，native/oscar 分别产生自己的 run_id 与证据文件。
没有真实数据时不要填入示例数字。

## 冻结策略与覆盖

`configs/acceptance.json` 是阈值来源。`acceptance_sha256` 定义为整个策略文件解析后的
canonical JSON（键排序、紧凑分隔符、UTF-8、不允许 NaN）的 SHA256，可调用
`benchmarks.compare.canonical_sha256` 得到。更改策略而不重新取得绑定该策略的报告会失败。

- 每个 case 恰好 2 个已完成 warmup 记录、5 个实际 measurement，repeat id 依次 0–4。
- 比较统计量为 median；同时报告 nearest-rank P95、min/max 与全部 5 个原值。
- latency/phase 使用 `max_latency_ratio`；throughput 使用 `min_throughput_ratio`。
  `noise_allowance=0`，没有隐藏噪声裕量。
- 必测 case 是 `required_input_lengths × required_concurrency × required_q_lens`。
  未提供 `required_q_lens` 时协议固定为 `[1,2,3,4]`，覆盖 MTP verify 可出现的短 query。
- 每个 case 单独通过/失败。没有跨长度、并发或 query 长度的总平均分，也没有“多数通过”。
- `model_quality` 的三个阈值仍为 null 或状态不是 `frozen_before_measurement` 时，
  整体验收为 `not_run/requires_premeasurement_definition`。工具不替使用者制定质量标准。
  这些阈值必须在真机测量前确定，不能根据结果修改。

case 的 `input_tokens` 是真实 tokenizer token 数；`q_len` 是该工况实际测量的 decode/verify
query 数。真实 workload 必须明确测量阶段与计时范围。最大输入长度与输出 token 的总和
不得超过服务可处理的上下文；极限长度若不可测，应保留该 case 的缺失/失败状态并说明，
不能静默改小 required_input_lengths 来制造通过。

## 输入 report schema v1

| 字段 | 契约 |
| --- | --- |
| `schema_version` | 整数 1 |
| `kind` | `native` 或 `oscar`，与 CLI 位置匹配 |
| `run_id` | 本轮唯一非空字符串 |
| `measurement_type` | 真实目标测量才能声明 `target_npu` |
| `acceptance_sha256` | 上述冻结策略 digest |
| `pair` | 两份报告必须完全相等的比较条件，见下 |
| `phase_definitions` | 相位名到 `clock=device`、非空 `scope`、`definition_sha256` 的映射；两端须相同 |
| `cases` | 每个 `(input_tokens, concurrency, q_len)` 仅出现一次 |

`pair` 必须包含：

- `model_fingerprint`、`dataset_sha256`、`workload_sha256`：完整小写 64 位 SHA256；
- `software`：非空 `vllm_commit`、`vllm_ascend_commit`、`torch`、`torch_npu`、`cann`、
  `driver`、`firmware`、`compiler`，可附原生源码状态 hash；
- `devices` 为本轮选择的 4 个唯一非负 physical ID、`tp=4`、真实 `npu_model`；
  不根据历史日志推断可用卡。当 `--target` 配置中的 `devices` 已明确时，还必须精确匹配；
  当前 target 为 null 只表示尚未选择，不把它变成合成协议测试的假硬件结论；
- `mtp={"method":"qwen3_5_mtp","num_speculative_tokens":3}`（附录 A 的实际方法名）；
- `graph="FULL_DECODE_ONLY"`、`draft_graph_scope="eager"`；
- `workload`：`output_tokens`、`sampling`、`arrival`、`prefix_cache`、`max_model_len`、
  `max_num_seqs`、`async_scheduling`，以及实际需要的启动/计时/诊断开关。
  `workload_sha256` 必须等于该完整 workload 字典的 canonical digest。

每个 case 包含真实 `input_sha256`（两端相同）、2 个 `warmup` 状态记录、5 个 `samples`，
以及 `evidence` 路径表。每个 sample 必须包含：

- `repeat`、`completed_requests`、`failed_requests=0`、`timeouts=0`；
- `latency_ms`：正的有限 `ttft`、`tpot`、`itl_p95`、`e2e`；
- `throughput_tps`：正的有限 `prompt`、`generation`；
- `phase_ms`：相位名映射到 4 个 TP rank 的非负、有限 device 毫秒值。

所有相位、每个 rank 都独立比较；任何一个 rank 退化均失败。新的 OSCAR 相位没有可证明
等价的 native 相位，或两端的相位 scope 不同，则该相位为 `needs_baseline`，case/整体为
`not_run`。不能将 native 整段 prefill 与 OSCAR 某个任意小 kernel 错配成相位加速。
phase 定义 hash 标识事先冻结的测量定义，真实 trace 仍须展示该定义确实被执行。

## 必须存在且校验内容的 evidence

每个 case 的 `evidence.<kind>` 包含 `{ "path": "...json", "sha256": "..." }`。
SHA256 基于原始文件字节，工具实际读取并核对，文件不存在时不是通过。
每个证据文件的公共字段如下；这里仅展示未执行结构：

```json
{
  "schema_version": 1,
  "evidence_kind": "npu",
  "run_id": null,
  "measurement_type": "not_run",
  "acceptance_sha256": null,
  "pair_sha256": null,
  "case": {"input_tokens": 32768, "concurrency": 1, "q_len": 4},
  "status": "not_run",
  "checks": {}
}
```

`case`、`run_id`、策略 hash、pair hash 都须绑定该测量，防止误用别轮、别工况证据。
使用真实结果时必须有下列信息：

| 证据 kind | 额外字段及必要状态 |
| --- | --- |
| `npu` | `checks` 中 build/load/device_completion/graph_capture/graph_replay/tp4/mtp 全部 `passed` |
| `accuracy` | native 的 `checks.native_output=passed`；OSCAR 的 pack_unpack/store_dequant/fused_attention/prefill_window/model_quality 全部 `passed` |
| `route` | `backend` 精确等于 native/oscar；`ranks=[0,1,2,3]`，`all_full_layers=true`，`gdn_native=true`；native 还需 `oscar_plugin_disabled=true` |
| `memory` | OSCAR 必需；`full_history_bf16_restore_bytes=0`、`full_history_bf16_shadow_bytes=0` 必须为实测整数零 |

任何 CPU 证据、缺数值验证、graph 未回放、route 并非 OSCAR、全历史 BF16 恢复/副本都会阻止
通过。先验证这些证据，再计算该 case 的性能比值。数值容差由绑定的 acceptance 配置约束，
相应精度 probe 负责实际验证，比较器不重新运行数值计算。

文件 hash 证明内容完整性与配对绑定，**不能独立证明测量生产者诚实或硬件确实执行**。
必须保留原始日志、trace、profiler、二进制/配置 hash 供人工复核。本工具没有签发硬件证明
或把人为填的 `passed` 变成真机证据的能力。

## 本机验证范围

`python -m pytest -q tests/test_benchmark_measure.py` 使用本地真实 HTTP server fixture，
验证 SSE 首 token/burst/usage、HTTP 错误、坏 JSON、缺 `[DONE]`、超时、固定 2+5 次数、
冷热前缀隔离、配对 hash 和上下文边界。这些是客户端契约测试，不是真实 NPU 测量。

`python -m pytest -q tests/test_benchmark_compare.py` 测试只在临时目录创建明确标识为
synthetic 的 fixtures，验证缺证据、hash 漂移、错误 route、遗漏 case、慢 rank、缺 baseline
等情况不会通过。这是比较器逻辑验证，不是本项目的 NPU 性能验收。
