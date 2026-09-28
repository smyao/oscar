# 2026-09-28 混合 prefill/decode 片段判读

**范围与状态。** 本文只分析用户已完成的 AISBench 日志片段，不要求重跑。附件原文逐字节保存于 `reports/target_mixed_prefill_20260928.txt`（241 行，SHA256 `1333518e91dbd67fe632687d482ed2674f27e6649f338bfdc60d1295ca71f47d`）；全部 51 组 Engine/SpecDecoding 采样及原文行号在 `reports/mixed_prefill_20260928.json`。OSCAR 片段为 2026-09-28 07:26:11–07:30:31（27 组），原生片段为 2026-09-07 03:10:47–03:14:37（24 组）。用户报告完整运行约 16 分钟与 8 分钟；这些片段不能重建两轮全程积分、请求级延迟或质量分数。依据档案 #130/#139/#143/#148–#151；特别是 #148 的模型质量回退与 #151 的 q4 设备热点，均不可用 HTTP 200 或局部速度替代验收。

## 同相对窗口的观测

以每轮首次 `Waiting≥28` 且 `Running+Waiting≥30` 的采样作 `t=0`：OSCAR 07:26:11（2/30，原文 48 行），原生 03:11:47（3/28，168 行）。从随后 `t=10…170 s` 的 17 个完整十秒区间，把每条日志的周期速率乘以 10 秒后相加。**这是日志采样积分近似，不是逐请求 token 账；各 rank 不相加。** 原生起点少一个运行请求，日期也不同，因此不是严格配对实验。

| 窗口或时点 | OSCAR | 原生 | 可读出的边界 |
| --- | ---: | ---: | --- |
| `t=0…170 s` 输入采样积分 | ≈719,763 token；平均 4,233.9/s | ≈1,032,710 token；平均 6,074.8/s | OSCAR/原生≈0.697；仅本窗口。 |
| `t=0…170 s` 生成采样积分 | ≈1,834 token；平均 10.8/s | ≈17,812 token；平均 104.8/s | OSCAR/原生≈0.103；不能外推 16/8 分钟。 |
| 同窗 Running/Waiting 采样均值 | 17.5 / 14.4 | 20.1 / 11.7 | 均值是调度状态采样，不是完成数。 |
| 同窗 MTP Accepted/Drafted 计数加权 | 1,162/1,941 = **59.9%**；推算平均接受长度≈2.80 | 12,298/16,488 = **74.6%**；推算≈3.24 | 每轮 3 个 draft 下，接受长度约差 14%；不能单独量化解释约 9.7 倍生成速率缺口，更不是模型精度。 |
| `t=140 s`，双方均 `Running=25, Waiting=7` | 07:28:31 输入 4,141.8/s、生成 **17.2/s**、KV 45.9%（80–81 行） | 03:14:07 输入 3,786.2/s、生成 **125.3/s**、KV 91.4%（213–214 行） | 相同队列计数和接近的输入速率仍伴约 7.3 倍生成速率差；请求阶段/长度组成未知，不能据此断定单一算子根因。 |

同窗三个分段的生成均值依次为 OSCAR/原生 **5.0/53.0**（`0–60 s`）、**11.6/128.3**（`60–120 s`）、**16.8/138.7 token/s**（`120–170 s`）；缺口贯穿所贴混合片段，不是单个起始采样造成。KV 使用率的容量分母不同；不能把 45.9% 与 91.4% 换算为物理字节或已处理 token。两轮均有 ArgSort AiCPU 与 rejection sampler 告警（原文 11、44–47、117、163–166 行），不足以归因差异。

## 高峰出现和消退的同轮证据

| OSCAR 时段（相对起点） | Running/Waiting | 平均输入/生成速率 | MTP Accepted/Drafted | 证据 |
| --- | --- | --- | --- | --- |
| `t=190 s` | 32/0 | 5,082.1 / 37.2 token/s | 250/363 | 原文 91–92 行；队列已空，但仍有大量 prefill。 |
| `t=210–230 s` 三个十秒采样 | 每次 32/0 | **0 / 296.6 token/s**；单次峰 315.5/s | 5,635/9,786 = 57.6% | 原文 95–101 行；输入速率为零时短时生成很高。 |
| `t=240–260 s` 三个十秒采样 | 每次 32/0 | **2,071.5 / 116.2 token/s** | 2,193/3,894 = 56.3% | 原文 103–111 行；输入速率再起，生成采样回落，MTP 接受率接近上一段。 |

这组时间顺序支持“高生成峰值发生在前段输入工作退去后，输入工作再起时回落”的**相关性**，并说明 `315.5 token/s` 不能代表混合负载稳态或整轮 16 分钟表现。它不证明前段低速全由 prefill CV 造成：`Running=32` 不告诉我们各请求当步处于 prefill、decode 还是 MTP；新请求、上下文长度和输出剩余量也会变。HTTP `200` 仅说明该 HTTP 响应状态，不能当作请求完成、质量通过或 127 条全量进度。

## 代码边界：需要计时，不先改结论

| 代码证据 | 对本次现象的限定 |
| --- | --- |
| 原生调度器 `references/vllm/vllm/v1/core/sched/scheduler.py:342–350,376–410,562–566` | 调度器没有专门的 decode-first 阶段；它先遍历 Running，再在 token 预算允许时取 Waiting。`configs/target.json:12` 的每步 16,384 token 预算使不同阶段竞争同一步工作量。日志没有每步 scheduled tokens，不能宣称“调度器只做 prefill”或断定纯调度问题。 |
| `configs/target.json:15`；原生 `references/vllm/vllm/config/compilation.py:617–619` | 目标是 `FULL_DECODE_ONLY`：纯 decode 可图回放，混合 prefill/decode 不走 full graph。这是目标配置的共同模式；附件没有原生启动参数快照，不能把图模式当作本项目独有根因。 |
| `oscar_ascend/ops/cv_dispatch.py:8`、`oscar_ascend/runtime.py:65–78`、`oscar_ascend/integration/impl.py:82–98` | `source_splits` 由**整个 padded token 数 N** 选取；以 128 行 query tile、GQA6、20 Cube 为例，N128 可取 S3，N16384 取 S1。whole-batch S3→S1 是现有代码路径，实际混合步骤的 N、S 与耗时仍待短诊断核对。 |
| `oscar_ascend/integration/current_attention.py:26–46,74–98,160–165`、`oscar_ascend/integration/impl.py:118–163` | 主模型 prefill/mixed 可用原生 BF16 current FIA，history/window 仍为 OSCAR CV；此时必须使 CV 的 source2 当前段为空，再把原生 FIA output/LSE 写回 source2 split0。诊断必须保留该抑制，不能重复计算 current 或将 native-current 时间错计给 history CV。 |

## 短诊断与下一项优化的判据

入口：`git pull --ff-only && bash scripts/install_observe_serve.sh --variant candidate --probe-only --diagnose-mixed`。先完成已有真实算子门，再用独立合成张量测四个attention形状，每形状2次预热、5次NPU Event；不加载模型、不跑AISBench。

1. 31条q4 decode，上下文20K/23K/27K/30K循环，N128/S3。
2. 同一31条decode，保留相同有效Q/K/V和历史，仅改padding为N16384、S1。这是隔离形状开销的对照，不能冒称完整服务step。
3. 同一逻辑decode cohort加一条q16260、无旧历史的prefill，合计N16384/S1，使用生产CV＋native current FIA＋merge。
4. 同一逻辑decode cohort加一条q16260、旧历史13740的prefill，最终长度30000；路径同上。

每个形状核对冻结独立oracle，mixed验证source2抑制/写回；不要求数学实现不同的FIA与CV逐位一致。终端输出四条`PERF_MIXED_ATTENTION`及`PERF_MIXED_RESULT`，完整数据在`mixed-attention-report.json`。每相位无同步，结束Event确认设备完成；准备/旋转在测量区间外，current包含FIA及source2写回。这只覆盖单层主模型FULL attention，不包含其余模型算子、MTP draft、图回放或调度器。

若对照2相对1明显变慢，才能定量支持padding/split工作分配优化；若形状4的CV主导、形状3较轻，则优先处理长prefill反复读取/解包历史；若CV较轻而current/其它相位主导，就不能用CV加速比例外推整步。不同形状中位数之差不是严格的独占prefill耗时，最终仍须保留同一步证据。旧mixed同一步的MTP draft0 CV约236ms、整步5732ms，也不足以独自解释16→8分钟（`reports/target_observation_20260928/snippet-2.txt`）。

目前不直接缩小默认chunk：这可能增加后续chunk需要反复扫描的INT2历史，即使token间隔更短，总耗时仍可能变差。档案 #133 的HTTP profiler曾导致worker失败，本诊断沿用有界Event。常用`install_serve.sh --variant candidate`仍安装并启用C4/q1/fast unpack；本轮新增的是诊断，没有未经测量的新生产调度策略。完整模型精度、8分钟目标仍未新增验收。

本地主机验证：`PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_probe_mixed_attention.py tests/test_observe_serve.py tests/test_probe_history_reuse.py tests/test_install_serve.py tests/test_serving_variants.py`，60 passed。覆盖空历史fixture、同输入padding对照、生产相位顺序、报告与阶段状态不重名、失败退出码及资源释放失败不掩盖首错。Mac没有NPU，本轮四组新形状的设备数值与时间未测，不能把这60项测试当成设备或性能通过。
