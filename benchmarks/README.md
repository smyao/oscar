# 配对性能证据比较

档案 #70/#71 要求区分 host launch 时间与 device 时间、逐相位定位慢点；#72 要求自动创建
输出目录；#73 的 31897-token/4.6s 历史数字不能替代本轮同环境配对基线。本工具只审查输入
报告与证据，不启动模型、不生成测试流量，也不生成任何“实测通过”报告。

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

`python -m pytest -q tests/test_benchmark_compare.py` 测试只在临时目录创建明确标识为
synthetic 的 fixtures，验证缺证据、hash 漂移、错误 route、遗漏 case、慢 rank、缺 baseline
等情况不会通过。这是比较器逻辑验证，不是本项目的 NPU 性能验收。
