# 2026-09-23 长上下文并发性能实测判读

**状态：现有证据复核；尚未进行本轮新 NPU 测量，性能问题尚未解决，精度与性能均未验收。** 本文区分客户端墙钟、vLLM 周期日志、worker 心跳和带同步的设备归因。20–30K 输入长度、32 并发与“同一数据集”均是用户对其自有压测程序的陈述；项目没有该数据集或其逐请求记录。历史故障约束见档案 #70–#73、#130–#139；旧标题和索引行号有漂移，下列行号按当前条目标题定位。

## 证据位置与适用范围

| 简称 | 本地证据（行号） | 范围 |
| --- | --- | --- |
| A | `/Users/sunao2000/.codex/attachments/d5c076c5-4ce0-4e28-90f4-2d3666826cb5/已粘贴的文本.txt:7–82` | 2026-09-23 同一托管服务的 16,384 token、K=1/4 并发阶梯；是诊断运行，不是 20–30K、K=32 验收负载。 |
| B | `/Users/sunao2000/.codex/attachments/490cd93b-8770-42f2-ab96-4a5bca96ed0f/已粘贴的文本.txt:1–179` | 用户粘贴的 OSCAR 2026-09-23 与原生 2026-09-07 服务片段；用户称卡型、配置、数据集相同。附件没有数据集正文、可核对的版本、请求级 usage 或客户端时间戳。 |
| 档案 | `issue1_full_record.md:27980–28351`（#130–#139）；`issue1_full_record.md:21717–22156`（#70–#73） | 此前串行/混合服务、host 与 debug-sync 归因、profiler 故障、真实负载和阶梯探针的记录。 |
| 配置 | `configs/target.json:9–16`；`configs/acceptance.json:3–15` | TP=4、`max_num_batched_tokens=16384`、MTP/图模式与冻结的数值、性能验收边界。 |

## 可直接确认的数值

| 运行 | 观察值 | 证据 |
| --- | --- | --- |
| 阶梯 K=1 | 16,384 输入、19.0 s、完成 1/1；探针报告 prompt 860.1 token/s、generation 0.8 token/s。 | A:7–17 |
| 阶梯 K=4 | 4×16,384 输入、76.5 s、完成 4/4；探针报告聚合 prompt 856.6 token/s、generation 0.8 token/s，makespan/K=1 为 4.016。运行请求数逐次达到 2、3、4，等待数最终为 0；KV 使用率最终 6.4%，prefix hit 0。 | A:21–63 |
| OSCAR 真实负载片段 | 32 请求涌入后，00:24:06 为 Running=2/Waiting=30，00:25:46 为 Running=4/Waiting=28，KV 使用率 3.5%→7.0%；该段每次上报的生成吞吐为 0–0.7 token/s，非零输入吞吐上报为 1,667.5–2,535.1 token/s。此片段没有全部请求的完成时间。 | B:49–66 |
| 原生真实负载片段 | 03:11:47 为 Running=3/Waiting=28，03:13:17 为 Running=23/Waiting=9，03:13:37 为 Running=26/Waiting=6，KV 使用率到 95.0%；末次上报生成吞吐 156.4 token/s；03:11:47–03:13:37 的非零输入吞吐上报为 3,632.9–8,979.9 token/s。 | B:144–178 |
| 旧混合请求 | #130/#131 的 128/16K/32K/50K 串行请求完成；混合阶段 3/4 长请求在 300 s 请求死线超时，worker 心跳显示推进。#138 的旧 32 并发测量记录完成 3/32、prompt 499.0 token/s、generation 0.2 token/s，随后被阶梯探针替代。 | 档案 #130–#131 `issue1_full_record.md:27980–28167`；#138 `issue1_full_record.md:28325–28344` |
| 旧设备归因 | #134 的独立 debug-sync 运行：四 rank 累加的 FIA 同步区间 2,463.859 s/868 次，约占该运行所记录相位设备时间的 99.3%；#135 微调后 1,873.305 s/912 次、p95 从 7.166 s 到 5.043 s。同步运行改变时序，不能作为当前生产模式的绝对设备耗时或原生性能对照。 | 档案 #132–#135 `issue1_full_record.md:28168–28283` |

API 的 `Avg prompt throughput` 是报告周期内的速率，单个非零周期不能当成整次请求吞吐。例如 A:28 的 1,638.4 token/s 与 A:17 的整次 860.1 token/s 口径不同。`PROGRESS_BUCKET wall_s` 也是各 TP worker 的**心跳间隔之和**，不是单次算子的设备耗时；A:66–80 中 K=1 的 73.531 rank-s 与 19.0 s 客户端墙钟不能直接相除以求算子速度。`tools/summarize_progress.py:24–30,61–86` 明确说明间隔可能包含 GDN、调度、图等待及 host I/O，跨臂间隔单列。A:75–76 的 `between_arms kv=256K` 不属于任一 K 臂的 256K 请求。

### 静态工作量模型（仅代码推导，非 NPU 实测）

当前目标 TP rank 的 GQA=6；CV 的 query tile 为 `floor(64/6)=10` 个 query token，KV tile 为 32 token（`csrc/kernels/attention_cv.cpp:55,137–145`；`docs/cv_implementation.md:47–53`）。对 16,384-token 首次 prefill，仓库的源代码调度分析报告给出 **420,761 个 current-source query×KV tile 对/主模型 FULL 层**（`reports/prefill_work_analysis.json`；`docs/design.md:149–151`）。每对在当前 Cube 路径执行 QK 与 PV 两次 matmul（`csrc/kernels/attention_cv.cpp:206–216`）；按主模型 16 个 FULL 层计算约 **6,732,176 对、13,464,352 次 matmul 调用**，不包括其他 source、旋转、store、GDN 或 MTP draft。20–30K 输入的因果扫描工作量更高，但实际分块、窗口和并发形状需逐步记录。这个数量级说明 CV 是值得优先验证的**结构性候选**；每对真实设备时延、CV 总占比以及优化后的加速比均不能从静态计数推出。

## 当前可下的判断及边界

1. **单请求总耗时已可量化，排队会放大长服务时间。** K=1 没有并发队列时 16K 请求仍耗 19.0 s（含短生成）；真实负载中 OSCAR 的 Running 长时间仅 2–4，Waiting 仍为 27–30。原生在用户声称相同负载的片段中能逐步达到 26 Running。队列堆积本身不是根因定位；还需逐请求与逐调度步时间。A:7–63；B:53–66,149–178。
2. **CV attention 是最强的既有设备侧线索。** 当前 `impl.py` 中相位名 `fia` 包住的是自研 `attention_cv_out`，不是原生 FIA。#132 的 host enqueue 背压不能分清设备内核；#134/#135 的独立 debug-sync 将主要同步时间归于该相位。生产运行是否仍由同一内核主导、其成本如何随请求数和历史长度变化，需本轮配对设备测量。档案 `issue1_full_record.md:28168–28283`；历史禁止全历史恢复见 #70–#73 与启动文档附录 D.4。
3. **K=1→4 的约 4 倍 makespan 不能证明 CV 串行、调度器只处理一个请求或解码无扩展。** 当前 `max_num_batched_tokens=16384` 正好等于每条阶梯输入长度；四条输入至少需要四个满额 prefill token 预算的调度步，即使每步执行完全正常，总输入工作量也为四倍。A:7–27,62–63；`configs/target.json:12`。K=4 日志明确已有 2–4 个 Running（A:39,45,52）。要区分原因，必须记录每步实际调度的 prefill/decode token、请求数、设备执行与 host 等待。
4. **不能据现有日志给出单请求 25K prefill 倍率、解码倍率、精度结论或内核级绝对时间。** 两份真实负载日志没有对应 request ID、到达/首 token/结束时间、实际 prompt/output usage；阶梯使用 `/v1/completions`（A:16,58–61），用户负载使用 `/v1/chat/completions`（B:7,14–47），且两次用户负载测试相隔约 16 天，版本和请求发送节奏未在附件中核对。两侧均出现 ArgSort AiCPU 告警（B:49–52,144–147）及原生 rejection sampler 的 fallback 警告（B:9,98），不能据此归因 OSCAR 增量。两侧 KV 使用率的容量分母不同，也不能把 7% 与 95% 直接解释为处理 token 比例。OSCAR 的 MTP 接受率日志仅含每段几次 draft（B:56,63,66），样本太小；HTTP 200 亦不证明模型质量。档案 #130/#137 `issue1_full_record.md:27980–28135,28303–28324` 中更强的归因表述应以本段边界为准。

## 配对真机诊断决策树

被动观测直接配合用户现有压测程序：项目采集服务端日志、`/metrics` 累计计数器、Running/Waiting 与 worker 相位信息，不生成压测输入。若该客户端可导出同一请求序列、到达时刻、token usage 与流式时间戳，才做严格逐请求配对；否则只比较可观测的服务端聚合量及其条件差异，逐请求 TTFT/TPOT 标“未测”。对照运行需保持同一模型、TP=4、MTP、图模式、`max_num_batched_tokens=16384` 与 API 路径，并记录代码版本、设备、OSCAR 旋转文件和 prefix hit。预热和重复次数按 `configs/acceptance.json:10` 的 2/5 及逐工况中位数要求执行。正式验收仍覆盖 `configs/acceptance.json:11–13` 的全部必测长度、并发和 q_len。#136 `issue1_full_record.md:28284–28302` 提醒探针报告与相位账本不能同名；#139 `issue1_full_record.md:28345–28351` 的阶梯快路径仅供诊断，不代替全服务与性能验收。

| 问题与控制工况 | 同时采集 | 如果出现的现象 → 下一步 |
| --- | --- | --- |
| **Prefill-only**：用户客户端若支持控制输出长度，可在其 20–30K 请求中取 K=1、4、16、32 并将输出设短；否则只对现有请求的 prefill 调度步按实际长度分桶。 | 服务端每步 prefill token、chunk 数、Running/Waiting、KV 与 prefix；host enqueue/同步墙钟、可验证的设备事件时间。客户端若提供时间戳，再加提交、首 token 和逐请求 usage。 | 同长度的 K=1 prefill 已慢：与原生配对后拆 CV/FIA、store、rotate、merge、GDN 与其他模型段。仅高 K 慢：查 16K 预算的步数/填充率、admission、queue、KV 容量。host 时间大且设备空闲则查 metadata/同步/调度；FIA 设备时间随历史或 `num_reqs` 暴涨再优化该内核。 |
| **Decode-only**：用户客户端若可配置，先完成长上下文 prefill，再在 K=1、4、16、32 持续生成足量 token；否则仅按服务端可辨认的 decode/verify 步归类，不能声称逐请求纯 decode 吞吐。 | 服务端每步 decode/verify token、MTP draft/accepted、graph replay、host 与设备相位时间；客户端有流式时间戳时才计算相邻 token 间隔。 | 仅 decode 段慢：定位 CV 读取、MTP verify/accept 或图回放；decode 段接近原生而 mixed 慢：优先查 prefill 对 decode 的阻塞和调度公平性。 |
| **Mixed**：被动观察用户现有程序的 32 并发、用户称 20–30K 输入的负载，不在项目中构造替代数据集。 | 一秒级累计 token 计数器差值、Running/Waiting、KV、prefix、服务端错误与超时；客户端提供逐请求记录时才计算队列时间、TTFT、TPOT、总时长分布和 usage。 | 队列增长且设备持续忙：按 prefill/decode 占用时间分配归因；设备空闲而队列增长：查 host/调度；KV 接近容量才判断内存约束。保留所有可见超时，不用已完成请求均值掩盖失败。 |

生产模式计时不得插入同步并把该结果当作吞吐对照。`OSCAR_DEBUG_SYNC=1` 只在单独的定位运行中使用；档案 #133 `issue1_full_record.md:28198–28225` 已记录目标栈通过 `/start_profile`、`/stop_profile` 导出时四 rank worker 崩溃，不能默认恢复该测量路径。设备事件若无可验证的完整记录，应标记为“未测”，不能用 host enqueue 或心跳墙钟代替。

## 精度门与交付状态

任何性能优化须保持真实 OSCAR INT2、CV、旋转、窗口、MTP 与图路径，并先通过独立 oracle 和 `configs/acceptance.json:5–9` 的冻结门：pack/unpack 位级精确、store↔dequant `atol/rtol=0.002`、fused attention `0.005`、prefill/window relative L2 `<0.02`、参考 staging eviction `0.0001`。逐工况比较原生性能必须遵守 `configs/acceptance.json:10–15`，不能放宽阈值或以均值掩盖慢项。模型级 logits/任务指标/MTP 接受率容差目前仍为 `null`（`configs/acceptance.json:14`）；必须在测量前定义并冻结，当前不能宣称模型质量已通过。图捕获、图回放、设备完成、质量、容量与性能分别记录。

本文只复核已有附件和档案。**未完成新的配对 NPU 运行，未证明任何性能修复，也未证明最终精度。**
