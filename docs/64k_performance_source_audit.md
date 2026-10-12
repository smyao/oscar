# 65,100 输入路径的源码性能审计

日期：2026-10-09。审计基线：`4defe20`。目标负载仅为 65,100 输入、512 输出、120 请求、K32；用户给出的 client wall 为 OSCAR 2467.1275737 s、native 1342.4437259 s。本文不使用 LongBenchv2 性能日志，也不将旧负载的相位占比迁移到本轮。

本轮只读当前工程及指定 native references，仅新增本文。未运行 VM、CPU-debug、CAModel、NPU 或测试；未修改生产配置、算子或 GDN，未推送。以下流量和次数是明确形状假设下的源码账，不是设备测量。

## 1. 结论

1. **whole-prefill 的现有门槛不覆盖 65,100。** candidate 的预算为 32,768，扣除 128 个序列的四 token decode 预留后，whole_limit 只有 32,256。65,100 仍按最多 16,384 token 分块，后续块进入压缩历史 CV。仅把预算改为 65,536 仍不够：whole_limit=65,024。
2. **current-only 只消除真正零历史的尾部请求。** 理想四块执行中，首块可以走 current-only；后三块分别带约 16K、32K、48K 历史，仍执行 INT2 history/window CV，再执行原生 current FIA 并合并。首 MTP 的 current FIA 同样只替换 current source。
3. **C16 的解包复用没有消除历史 attention 的主要数学与状态处理次数。** 16 个 query 组共享一次 KV256 恢复，但仍分别 QK、softmax、PV、在线累加。每组每 KV256 都把 FP32 在线状态搬入、搬出 GM。理想四块情形，后三块每 FULL 层、每 rank 的 FP32 K/V 发布约 13.683 GiB，簇内状态读写约 70.295 GiB。
4. **mixed split 不把短请求从完整模型的长批次延迟中独立出来。** 它在每个 FULL attention 内顺序调用短前缀与长后缀，保留原大 N 的 split 数。大 padding later-MTP 已有 compact 路径；常规 N128/B32 decode 不满足 compact 门槛，不能再把旧 N16K padding 问题当成所有 decode 的现状。
5. 源码支持“长 history prefill 与 decode 同批造成服务等待”的机制，但不能证明当前日志的零吞吐区间就是设备死锁、纯 prefill 或单一 attention stall。当前负载须补最小 step 级时间关联，才能给上述机制排序。

## 2. 当前有效路由

`configs/target.json:12–16`：基础预算 16,384，max_num_seqs=128，FULL_DECODE_ONLY，MTP 三个 speculative token、eager draft。`tools/serving_variants.py:13–21,45–58,81–93` 的 candidate 同时启用 current-only、first-MTP current FIA、later-MTP compact、mixed split、decode bundle、whole-prefill，并将预算提升至至少 32,768。不能根据某些文件头仍写着 experimental/unenabled 判定实际未启用。

| 阶段 | 当前实际入口与边界 |
|---|---|
| 零历史 fresh suffix | `impl.py:88–96` 优先 current-only；`current_only_plan.py:36–44` 要求 CPU seq upper bound 等于本次 query 长度，且必须是连续尾部。`current_only_dispatch.py:54–79` 再检查设备 task context 和 slots。 |
| 有历史的主模型 prefill | `impl.py:143–181` 建三源 task，置空健康 source2 的 CV 范围，执行 history/window CV；`:182–205` 用 native current FIA 填 source2 partial 后 merge。 |
| 首 MTP | `metadata.py:118–137` 只允许明确 qualified 的 draft_index=0；`current_attention.py:26–50` 支持当前 source 的原生 FIA。历史、window 保留 CV；零历史 prefill suffix 才能 current-only。 |
| 后两轮 MTP | `references/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:1213–1219,1264–1284,1563–1570`：原生保留 input_batch_size，实际 q1 token 为 batch_size。外部 compact 有严格大 padding 条件，见第 5 节。 |
| 长 query 的历史 | `cv_dispatch.py:48–54,89–93`：max_query_len≥8192 的 candidate 映射到 `attention_cv_window_range_cluster16_out`。 |
| 短 q4 / later q1 | `cv_dispatch.py:74–88` 加 `:23–29` 映射到 `attention_cv_bundle_decode_out` / `attention_cv_bundle_q1_out`，不是旧 scalar/旧 SIMD 实验臂。 |

所有成功路径仍发布 INT2 和精确 window：普通路径 `impl.py:206–224`，current-only 路径 `current_only_dispatch.py:100–112`。发布发生在旧 window 读取后；不能为了快提前覆盖 ring。

## 3. whole32k 在 64K 上的缺口与代价

`whole_prefill_scheduler.py:18–58` 定义：

```text
whole_limit = budget - max_sequences * (speculative_tokens + 1)
is_whole = waiting_request.num_tokens <= whole_limit
non-whole long_prefill_token_threshold <= 16384
```

本配置为 `32768 - 128*4 = 32256`。资格检查的是请求总 token 数，不是本次剩余长度；policy 每轮新建且只在 WAITING admission 选择请求（`:64–82`）。原生先调度 RUNNING 请求，随后 WAITING；分别受 threshold 限制，见 `references/vllm/vllm/v1/core/sched/scheduler.py:376–410,680–699`。因此不能期待同一 65,100 请求在后续剩余长度变短时自动进入 current-only whole 路由。

理想单请求、无 prefix cache/抢占/对齐切分、没有别的请求侵占预算时：

| 块 | 当前 query Q | 已有 context C | history 路由 |
|---|---:|---:|---|
| 1 | 16,384 | 0 | 可 current-only |
| 2 | 16,384 | 16,384 | C16 history + window + current FIA |
| 3 | 16,384 | 32,768 | C16 history + window + current FIA |
| 4 | 15,948 | 49,152 | C16 history + window + current FIA |

真实 mixed 调度可能产生更小块；native Mamba 对齐、剩余预算、prefix cache 也会改变块边界。上表不是当前 120 请求实际轨迹。

忽略 sink/recent 的小修正，跨块 query-key 对数量为：

```text
16384*16384 + 32768*16384 + 49152*15948 = 1,589,182,464
全部 causal 对数量 = 65100*65101/2 = 2,119,037,550
```

约四分之三的 causal 对落在跨块部分。whole fresh 65,100 的结构价值是把这些对也交给已有的原生 current FIA；历史压缩 attention 数学并未被改写或转成原生 BF16 history。它不能降低 attention 的固有二次工作量，也不会消除后续长上下文 decode。

### 预算与固定 workspace 必须同时处理

仅改 budget=65,536 得到 whole_limit=65,024，仍差 76。按当前保守预留，至少需要 budget=65,612；128 对齐可取 65,664。这只是准入算术，并非已验证配置建议。

`runtime.py:270–283` 按预算分配固定 workspace，`:97–104` 给出完整字节公式；`:107–135` 各 tensor 实际预分配。Hq=6/Hkv=1/D=256、S容量=1、20 Cube、C16 时：

```text
workspace(N) = 35388*N + 99943680 bytes
N=32768: 1201.188721 MiB / rank
N=65664: 2311.383545 MiB / rank
增量:    1110.194824 MiB / rank
```

workspace 是同 geometry 的 FULL 层顺序复用，不应再乘 FULL 层数。增量会在 KV profiling 前被占用，并可能减少可驻留请求数；原生激活/GDN 内存也不能忽略。当前 32K workspace 的容量成本已存在，但 65,100 不享受 whole 路由。应联合考虑 phase 所需 arena 分配，保留 graph 地址与顺序复用契约，而非盲目翻倍所有 arena。

## 4. C16 的真实复用与剩余放大

Hq/Hkv=6 时 query tile 是 `floor(128/6)=21`。`attention_cv_window_range_cluster16.cpp:406–435` 要求：kind0、qcount=21、task 有效且无错、非空 history、`kvend==context`、request-relative query 对齐；同簇 16 个成员的 request/head/split/context/requestBegin/kvbegin/kvend 必须完全相同。没有跨 request/head/split 复用。

物理页不要求连续：`:539–570` 在 virtual128 边界重新寻址。不能为扩大簇放宽地址、有限值预检查或错误域。

S1 的 history 范围来自 `attention_tasks.cpp:134–142`：

```text
begin = min(sink, C)
end   = min(C, max(sink, C+x+qcount-recent))
```

sink=64、recent=256、qcount=21 时，前 12 组的 end<C，且各不相同。按 16 组 anchor 成簇，整个首簇的 16 组无法组成 C16；尾部不足 16 个完整组也不能成簇。S>1 时 `kvend==context` 还排除了通常更早的 history split，即使其组间域一致；但理想长 chunk 是 S1，不能把该排除夸成此负载主收益。

| 当前 Q | query 组数 | 合格 C16 簇 | 单独组数 | 完整 history tile 的等效恢复扫描数 |
|---|---:|---:|---:|---:|
| 16,384 | 781 | 47 | 29 | 76 |
| 15,948 | 760 | 46 | 24 | 70 |

设 U 为未成簇组、C 为合格簇，长度为 `L=end-begin`：

```text
恢复 KV256 单位数 = Σu ceil(Lu/256) + Σc ceil(Lc/256)
QK/PV 单位数      = Σu ceil(Lu/256) + 16*Σc ceil(Lc/256)
packed payload    = 136*(Σu Lu + Σc Lc) bytes   [D256/Hkv1]
```

`:305–374` 两阶段处理未成簇组与簇；`:920–954` 每 KV256 只执行一次 LoadPacked/Unpack/Publish，却对 16 个成员逐个 RestoreState/Softmax/PV/SpillState。`:478–501` 也保留独立 QK/PV 和最终逆旋转。因此“16 组复用”不是 attention 16 倍加速。

在第 3 节的理想四块形状下，后三块 C=16384/32768/49152，H=C−64。前 12 组分别少 235,214,…,4 行，合计 1434；packed 恢复行数为 `76H−1434,76H−1434,70H−1434`，对应 KV256 恢复次数 **4861、9725、13437，共 28023**。

`:694–701` K/V 即使尾 tile 也按完整 tile 发布 FP32，因此源码 GM 发布量：

```text
28023 * (2*256*256*4) = 13.683105 GiB / FULL层 / rank
```

`:885–904` 每个 member 的 RestoreState + SpillState 覆盖 128 行的 D 维 accumulator 及 max/sum；两 AIV 合计每簇每 KV256：

```text
16 * 2 * 128 * (256+2) * 4 = 4,227,072 bytes
簇-KVtile 总数 = 47*64 + 47*128 + 46*192 = 17,856
仅簇内循环 state 读写 = 70.294922 GiB / FULL层 / rank
```

以上不含首末状态、score/P/PV、Q 加载、window/current、未成簇组及其它层的流量，也不是测得 HBM 事务。初始化与最终 Restore/Spill 还会增加小项。反复 DMA/fence 的指令代价同样待测。

## 5. mixed split / MTP 的当前边界

`mixed_decode_split.py:23–51` 只拆连续短前缀：每请求 q≤4，短前缀总 token≤128，后缀包含 query≥84（GQA6）的请求；capture/dummy 排除。短请求不在前缀时不会任意重排。`:82–86` 顺序执行两次 FULL attention；这不把一个 decode step 从长后缀所属的完整模型 forward 中解耦。

`:49,65–66` 将原大 N 保存为 `cv_shape_tokens`；`impl.py:114–125` 依它计算 S。`runtime.py:74–78`：

```text
groups = ceil(N/21) * Hkv
S = max(1, min(32, ceil(CubeCores/groups), arena_capacity/N))
```

20 Cube、Hkv1 时 N128→S3，而 N≥16K→S1。拆出的 short prefix 保留 S1；later-MTP compact 同样保留原 N（`compact_later_mtp.py:145–147`）。这会保留原分段数学和 FP32 合并顺序；不能直接删掉字段后声称无精度影响。K32 的 32 个独立 history leader 也已能覆盖 20 核，S1 不是严格三倍浪费证明。

later-MTP 的 compact 条件是 `N>=1024 && N>=8*B`（`:25–28`），并严格限定真实 eager、后续 draft、模型/通信契约；`:148–162` 只把模型输入缩到 B，输出无需再扩 N。因此大 mixed prefill padding 已有处理。常规 decode N128/B32 不 compact，仍做两轮 q1 padded 模型调用；这是潜在小范围专化机会，不能引用旧 16K padding 成本把它夸成当前大头。

长 history 在主模型 q4、首 MTP 以及后两轮 MTP q1 都真实存在。三个 MTP 调用不是由 first-MTP current FIA 消掉了整个 attention。MTP layer 明确用 identity（`runtime.py:390–406`），也不能把所有 hadamard=False 当可任意省略的旋转。

若本轮 acceptance 较低，完成固定 512 输出需要更多完整 target+draft 迭代，长历史成本随之重复。应直接记录每个完成 step 的接受 token 数和实际输出数；打印的聚合 acceptance 窗口不能单独还原迭代次数，更不能据此判断量化是否超出冻结精度门。

## 6. 结构候选与最小验证

| 优先候选 | 具体作用 | 最小有效验证（由主代理统一排他安排） |
|---|---|---|
| 65,100 fresh whole admission + workspace 联合设计 | 消除后三块的压缩历史 prefill，扩大现有合法 current-only 域；保留后续 INT2 store/decode。先算真实激活、GDN、KV 与 scratch 总预算，避免额外约 1,110 MiB/rank 的简单扩容损害并发。 | 纯调度契约覆盖65100、65024/65025、剩余预算、等待/恢复、K32与128保守预留；单个65100 fresh的完整FULL/current+store/首MTP短探针，逐级记录峰值内存与精度。再与decode并存的短服务probe比较TTFT/ITL，不能仅报prefill变快。 |
| C16 state/DMA 成本下降 | 保留每原KV256数学顺序与错误域，消除重复资格扫描；探索可变尾簇和更合适的 query 微批，联合优化 KV 复用、state 搬运和核占用。 | 当前 C16 的同源 phase 计数：unpack/publish、state restore/spill、QK/PV等待、softmax/acc、最终写回；记录 actor，不相加重叠相位。先同shape独立oracle与逐位旧新，再短同核/同input A/B。禁止全历史恢复或仅换更漂亮shape。 |
| 长历史 prefill 的服务准入/分块策略 | 控制长块插入造成的完整step延迟；whole 的吞吐收益与decode等待必须同时评估。mixed attention split 本身不等于请求级隔离。 | 仅针对65100上下文：采样连续几步scheduler输出（请求、Q、C、N、S、路由、实际接受/输出token）、CPU提交与NPU completion时间；覆盖decode-only、混合、fresh admission。不要再次跑AISBench。 |
| short/compact 后独立 S 与 normal-decode MTP compact | 在较低实际B时改善核分布或减少padding模型工作。K32不能预设有大收益。 | 同N128/B32、q4与两轮q1、context65100；保持冻结容差并核对原生MTP逻辑、slot、position、draft结果。若重新选S改变reduction顺序，必须重新独立oracle，不能只看字节文件或耗时。 |

最先需要的是当前负载的 step 关联证据：native scheduler 本身同时调度 RUNNING 与 WAITING（`scheduler.py:340–351,376–410`），没有独立的“pure decode 时间段”保证。现有代码能解释等待放大，但没有把当前客户端的停顿定位到某个设备 actor。应先对齐这些小样本与该负载服务日志，再决定 whole、C16、decoder kernel 或 MTP 哪条线能提供所需大幅收益。

## 7. 归档约束

已按启动文档 §0、档案索引、#129/#140/#142/#145/#148/#151–155 及完整 D.4 原始日志审阅。D.4 的全历史恢复、host 准备和逐 token 调度失败不应重现；#142 的 warmup/capture 判定、#145 的异步 UB 所有权、#148/#154 的独立 oracle 与冻结精度、#155 的 NaN/window 字节语义均须保留。

本审计不提出原生 GDN 修改、不调用独立测试 oracle 到生产、不恢复整段 BF16 history、不放宽 `configs/acceptance.json`。编译、CPU、CAModel、真 NPU、图、短服务与完整负载性能是分别记录的门；本文只提供源码结论和待验证的结构方案。

## 8. 后续独立 bounded workspace/scheduler 的原生 host 资格验证

本节是后续开发证据，前述 `4defe20` 当前路由审计保持原口径。新能力尚未注册到 candidate，未修改原生 references、GDN、CANN 或模型数学，也未运行 VM/CPU-debug/CAModel/NPU。

### 8.1 两种 query 容量

`runtime.py` 新增可选 `history_token_capacity`，默认 None 保持原有每个缓冲的形状及字节数。此参数限制进入历史 CV 的**本轮 query 行数**，不是 KV context 长度。

- query_rot、partial、partial_lse、output 按 history query capacity 分配。
- 当前 Q/K/V projection、positions/slots、LSE 和各 status 保持完整 input capacity；current-only 也需要它们。
- task/status 行数覆盖 `max(input_capacity, history_capacity*splits)*3*Hkv`，因此大 fresh S1 仍逐行检查设备 context，不能只分配小历史任务表。
- 普通 `validate()`、`partial_views()` 严格限制历史容量；独立 `validate_current_only()` 只检查输入容量，不授予路由资格。current-only 保留已有 eager/origin/context/slot/padding guards，并在执行 prefix 前检查其容量。

Hq6/Hkv1/D256、20 Cube、C16、S容量1：旧 input32768 工作区为 **1,259,537,664 B / rank**；新 input66048/history16896 为 **923,761,920 B / rank**，净减 **335,775,744 B（320.221 MiB）**。固定 arena 仍在模型加载、KV profiling 前分配，运行中更改新的容量配置会报错。主机容量与旧路由回归为 99 passed、8 项既有 skip；不代表图或设备通过。

`cv_shape_tokens` 可以继续保留 original padded N 来选 S，但真实 live CV 行数必须受限。若 history 容量过小，S 的 arena 限制会与原来变化，不能泛称所有参数下分片数学相同。本次目标16896仍覆盖N128/S3。`mixed_decode_split.py:71` 对 parent 总 N 的普通校验未放宽，未经 fresh-suffix 证明的超大历史 batch 仍会拒绝。

### 8.2 执行了哪些原生代码

新 `tests/test_bounded_whole_prefill_native.py` 直接 AST 抽取并执行当前 references 的：

- `Scheduler.schedule` 完整方法、`_make_cached_request_data`、`_update_after_schedule`、`_select_waiting_queue_for_scheduling`；
- `_preempt_request`、`reset_prefix_cache`；
- 完整 `AsyncScheduler` 类与原生 RequestQueue/FCFSRequestQueue；
- Ascend `_should_build_dummy_attn_metadata` 和 native `_warmup_and_capture`。

仅初始化配置、request payload、KV allocator、输出 transport 对象由小型 host fixture 提供；不加载 vLLM/NPU，不构造模型或大 tensor。没有用第二份手写 clipping 循环代替被验证的原生 schedule。

通过的关键形状与事件：

| 用例 | 原生执行结果 |
|---|---|
| 31 个已有 q4 + fresh65100 | 同步输出 scheduled tokens=65,224，历史 query=124；fresh 整段65,100；下一异步 step 为32×q4=128，没有因 placeholders 累加而扩成大 query。 |
| 四个 fresh100000 | 首步4×16384=65,536，全为零 context；下一步历史总行数16,896，分配16,384+512。 |
| 31 q4 + 两个历史续算 + fresh65100 | 历史总行数16,896，含124 decode +16,384+388；fresh保持排队。 |
| whole边界 | 65,100和65,536整段准入；65,537按16,384开始。 |
| 当前历史步 + 大fresh | 大fresh不够剩余预算时保持在等待队列，不偷偷切成残余小块。 |
| native allocator抢占 | `_preempt_request`释放并将computed清零；下一步用原生 resumed_req_ids，零context恢复。 |
| forced reset | 原生reset清prev-step缓存并预留stale async丢弃计数；mixin保留同输出preempt+resume通知；实际AsyncScheduler丢弃旧frame。 |

合并命令：

```text
.venv/bin/python -m pytest -q --tb=short tests/test_bounded_whole_prefill_native.py tests/test_bounded_whole_prefill.py tests/test_whole_prefill_scheduler.py tests/test_request_windows.py
43 passed, 10 subtests passed
```

### 8.3 此次发现并修复的资格缺口

1. **EC connector 漏拒绝。** 原构造器只检查 KV connector；原生 `scheduler.py:142–146` 独立创建 `ec_connector`。现在两者均明确拒绝，不让外部 cache arrival 改写 zero-context 前提。
2. **WAITING 不必然 fresh。** 原生 `_update_request_as_session`（`scheduler.py:1042`附近）与 `_handle_stopped_request`（`:1677`附近）可让 streaming session 保留正 computed 后重回等待队列，即使 prefix/connector 都关闭。新独立策略在任何 native allocation/mutation 前拒绝 WAITING/skipped WAITING 中有历史的未资格请求；普通 preemption/resume 因 native 清零仍合法。该策略尚未支持 streaming continuation，不应注册为覆盖所有请求形式的默认路径。
3. **动态调度可能替换自定义类。** 指定 Ascend `platform.py:713–725` 确实会按 `additional_config.SLO_limits_for_dynamic_batch != -1` 或 `profiling_chunk_config.enabled` 替换 scheduler_cls。新配置验证明确拒绝两项；未来接入入口还必须在平台改写前执行等价检查。仅在 bounded 类构造器里检查不能保护一个已被替换、从未调用的构造器。

### 8.4 大 dummy 与启动范围

- native profile 在 `gpu_model_runner.py:6228` 用 `self.max_num_tokens, is_profile=True` 调 `_dummy_run`。Ascend `model_runner_v1.py:3352–3364` 的真实判定是 `force_attention or cudagraph_runtime_mode==FULL`，单凭 `is_profile=True` 不构造 attention metadata。现有 `impl.py:66–69` 对 metadata=None 的 profiling 行为未改动。
- MTP dummy 只有 FULL graph mode 才建多步 attention metadata（`llm_base_proposer.py:586`）；普通 eager profiling保持None，最终原生forward context仍显式标记profile（`:689–698`）。此处没有新增跳过或把大N伪装成小N。
- `gpu_model_runner.py:6589`附近的 `_warmup_and_capture` 会先用 force_attention=True 做 FULL 预热。按当前显式 capture 列表，最大512，低于16,896；host fixture执行了此原生方法与Ascend predicate。
- **大 forced dummy 确实存在于可选动态profiling路径**：Ascend `worker.py:847–852` 的 `profile_prefill_latency` 调 `_dummy_run(num_tokens, force_attention=True, profile_cpp=True)`，num_tokens只按整个input budget限幅。动态profiling须禁用；另外目标PP1本身也被原生 `ascend_config.py:69–74` 的profiling配置门排除。任何额外外部forced dummy超过history容量，仍必须显式失败，不能借本节的普通profile结论视为已通过。

这些 host gates 建立了指定原生调度控制流的容量与事件契约。后续仍需实际worker batch重排后的current-only资格、TP4/MTP、设备graph地址、完整current/store精度、峰值内存及短服务验证，才能启用。不得从此结果宣称65,100请求性能或120请求总时长已经改善。
