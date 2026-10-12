# 当前局部优化的整轮 ROI 与 decode 跨任务流水条件

2026-10-10，只读源码/已完成证据审计；没有新 kernel、VM 执行或生产启用。档案依据：#126/#129/#140/#145/#148/#151/#154–156，启动文档 §0.7 与 D.4。机器可读账：`reports/roi_decode_overlap_20261010.json`。

## 1. 当前证据不能证明整轮目标成立

用户最新目标是相对原生 2–3 倍。已有 64K/120 请求日志的 native=1342.4437259s、OSCAR=2467.1275737s 是历史实测锚点；**当前省略 token-budget 参数后的入口没有新的整轮实测**，不能把该锚点当作当前版本计时。

| 目标 | 目标秒数 | 相对历史 OSCAR 尚需提升 | 尚需减少历史 wall |
| --- | ---: | ---: | ---: |
| native 2× | 671.221863 | 3.675577× | 72.7934% |
| native 3× | 447.481242 | 5.513365× | 81.8623% |

| 已完成局部证据 | old/new ticks | old/new | 边界 |
| --- | ---: | ---: | --- |
| 完整 writer，N128/Hkv1/D256 | 173419 / 96566 | 1.795860× | 同库 CAModel；不是服务时间 |
| 有界 unpack helper，short8 | 98478 / 41974 | 2.346167× | helper 单独计时；193 CPU 数值例通过 |
| **完整 C4**，Q336/R1/C113/S1/20core | 1355015 / 1346984 | **1.005962×** | 全 1008 tasks、source0+1；完整数值/逐位/60actor，source2 按生产 prefill 抑制 |

对应报告依次为 `reports/quant_store_simd_pack_20261010/reports/store-camodel-n128/comparison.json`、`reports/quant_unpack32_20261010/comparison.json`、`reports/quant_unpack32_20261010/full_c4_camodel_comparison.json`。完整 C4 的极小收益不能写成 2.346×；同样，C113 不能证明长历史下收益必然很小。H32 仍留在独立研究目录，不据此单点升级生产。

固定调度、各阶段局部比例可迁移且没有其他效应时，Amdahl 形式为 `T'/T = 1-fw-fu + fw/1.79586 + fu/2.34617`，其中 `fw+fu≤1`，二者不能相乘。即使极不现实地给**全部**历史 wall 套上最大 helper 比例，仍是 1051.557s，距两个目标仍差 1.56663×/2.34995×；全部套完整 writer 比例则是 1373.786s。这只是明确条件下的反事实边界，**不是当前版本预测，也不是通用硬件性能上界**。现有证据没有给出完整服务正收益的保证下界。

当前默认预算若最终为 2048，`cv_dispatch.py:92–103` 选择 C4；不会因 workspace 是 C16 就选择 C16。65100-token prompt 理想分块的跨 chunk 历史 pair 占比 96.869%，是工作量比例而非 wall 比例。不得再计入已废止 whole66048 移除历史计算的收益。

## 2. 显存字节比单列

按当前源码账，输出末 FULL 物理 KV 比为 3.58333×，全部 KV 为 2.72222×。研究中的双段页＋128 request-window 槽按 K32 摊销，FULL 物理比 6.30883×、全部 KV 3.82339×；`experimental_request_windows` 默认未启用，实际 HBM 峰值未验收。引用 `reports/memory_64k_accounting_20261009.json`、`reports/forecast_64k_memory_20261009.json`。

这些数值不能转成吞吐倍数。旧 whole 工作区/额外 activation 账已失效，不加入本轮速度预测。FULL 6–7×目标和全部 KV 报告仍保留。

## 3. 当前 q4/q1 确实还存在的串行边界

当前 `csrc/kernels/attention_cv_decode_bundle.cpp`，SHA256 `d8171f0403cf176087e1c76c64e454e6fb1e74fd79252aab3b59e848cca19401`：

- `Process:124–154` 对本 core 原 owner 的每个完整 task 执行 `HistoryCubeTask/HistoryVectorTask`，完成后才取下一个；padding、非法 task 均有原发布路径。
- `HistoryCubeTask:681–699` 每 KV 为 QK→PReady→PV；只允许同请求下一 QK 与上一 PV 消费重叠。
- `HistoryVectorTask:700–745` 已有同请求 next-KV K/V 预取。`PrefetchKeys/Values:634–657` 和 `CommitPrefetchErrors:658–665` 保留未来错误在上一 PV 后提交的优先级。
- q4/q1 当前已经使用 HalfKv64 padded unpack；C4 H16→H32 helper 的局部倍率不适用于它们。

因此剩余候选是**同 core 不同请求之间的有限流水**，不能把已实现的 next-KV 预取再记一次收益。原 `CvTaskSchedule` 公式位于 `csrc/include/oscar_cv_schedule.h:10–16`。对 R32/N128/Hkv1/S3，q4 与 padded q1 均为 96 个活跃 source0 tasks，16 核各 5 个、4 核各 4 个；每核这些 tasks 的 request ID 均不同。具体 owner/request 列表保存在 JSON。低并发或其他 S/shape 不保证具备这样的可重叠工作。

## 4. 从当前闭合 q4 事件得到的条件包预算

证据为 `reports/request_window_decode_completion_20261009/q4_r32_n128_c1025_s3-candidate/` 的完整 60actor trace。这里只取 core0 五个历史 task 的分量中位数；marker SHA、全部五个样本保存在 JSON。这个点每 split 只有一个 KV256，**没有执行 hasNext**，因此下表的稳态递推是条件推算，不是已测长循环。

| 包或依赖延迟 | AIC / AIV0 | AIV1 |
| --- | ---: | ---: |
| QK | 7039 | — |
| PV | 11252 | — |
| QReady 跨 actor 延迟 | 7818 | 7818 |
| PVReady 跨 actor 延迟 | 9618 | 9618 |
| PReady / KVReady 延迟 | 445 / 1799 | — |
| 有界 K load/unpack/publish 包 | 7551 | 8505 |
| 有界 V unpack/publish 包 | 5763 | 5782 |
| Softmax | 11259 | 8036 |
| PV 消费 | 457 | 457 |

单位是该 CAModel actor 时间域的 cycles，没有换成 NPU 毫秒。K/V 包沿用实测有界阶段，不能把所有 wait 都当作可删除成本。

候选每对任务 Cube 顺序为 `QA, QB, PVA, PVB`；每个 AIV 顺序为 `K_A_next, S_A, S_B, V_A_next, K_B_next, C_A, V_B_next, C_B`。共享 packed UB 只在该任务 V 已用完后交给另一个任务。每请求下一 softmax 必须晚于自己的上一 PV 消费；PV 必须等待两 lane 的 PReady；下一 QK 等待自己的完整 KVReady。Cube 串行、各 AIV 串行，跨 actor 依赖保留上述实测延迟。查询加载、最终归一化/旋转/写回、任务切换和新增上下文开销另计，不能藏入重叠。

以这些固定包重复递推 30 回合：原单请求单位周期 40582 cycles，两个请求一对周期 51841 cycles，相当于每请求 25920.5 cycles，history 条件比例 **1.565633×**。这只是这个候选顺序的估值，不是不可超越的上界。两请求 AIV 包的资源下界为 50060 cycles，说明这份估值对新增向量/调度成本很敏感。

要得到 **1.5×完整 decode kernel**，在新增成本为零、其余成本不变时，旧 kernel 的可受益 history 占比必须至少 **92.2643%**；只看 history 时，每对额外成本最多 **2268.33 cycles**。一般条件为：

`Tnew/Told = (1-h) + h/1.565633 + e ≤ 2/3`

其中 `e` 包括所有新增 prologue/context/planner/park/flag/固定成本除以原完整 kernel 时间。当前短点不能证明长64K满足该 h；q1 尚无当前完整分相位 trace，**不给 q1 填 q4 的比例**。

## 5. 奇数任务决定是否还值得实施

多数核实际有 5 个任务。两对重叠后让剩余一个任务全程串行，在每任务 85 个 KV 单位的示例下，history 只有 **1.406519×**，即使未加任何其他 source/固定成本也不到 1.5。

同一结构方向需要两活跃上下文＋一个有界 GM 暂停上下文，处理三任务的半程交错。例如等长 A/B/C：先 A/B 半程，暂停 A；B/C 半程至 B 完成，再恢复 A，与 C 完成后半程。其余 D/E 正常配对。每请求的 KV256 归约顺序不变，原 task/输出 owner 不变。长度为奇数时最多留下一个 KV 单位，而不是一整条历史任务。

五任务各85单位的示例，未计新成本时约 **1.563552×**；整个 odd-triad 额外成本容许量为467359 cycles，仍只适用于 history≥1.5这一较弱门。不同长度时必须按实际剩余单位形成分段，不能假定等长或把多余的短/非法 tasks 消失掉。该具体85单位仅为长历史工作量示例，不是已生成的65100全部真实任务表。

### 有界空间与正确性条件

- 当前 AIV 显式 UB 158656B，增加第二份 FP32 accumulator/stat 数组约16640B，总175296B，距184KiB保留13120B；需真实编译确认临时寄存器/栈/SDK缓冲没有超出。
- 当前 GM 每 core 1146880B。保留三个独立 GM 上下文为3440640B；额外暂停 acc/max/sum 快照33024B，保存＋恢复66048B，仅在奇数三任务分段一次。算术上可容于已有 C16 arena4997120B/core，但地址/stride/graph与运行绑定仍需实现验证。
- 同一时间只有两套 UB acc/stats。暂存保留 FP32 位值及每 lane 错误、cursor、buffer、pending error；不按每 KV 做全状态 spill，不恢复全历史。
- **不能只在 Vector 消费后随手暂停。** 原 Cube 可能已经发出下一 QK；分段终点必须在两侧一致地禁止越段发射，清楚消费/rebind readiness flag，并保留已预取 GM tile。旧 KVReady token 不能被新任务误消费。
- `CommitPrefetchErrors` 原优先级必须保留；不能因预取下一任务而提前覆盖本任务的 score/nonfinite 错误。所有 padding/非法/source1/source2 状态照常发布，不按“对性能无用”过滤。
- 单个可用任务、短历史、不能配对、低并发使用原完整 OSCAR task 路径；没有 native BF16 回退。分支不得依赖 CPU device readback，并需验证无明显短请求延迟退化、固定地址改输入 graph、q4/q1及不等长等场景。

## 6. 本轮判定

局部 writer 与 helper 证据不足以证明447–671秒。C4短核已实证1.006×，不单独提升生产。跨任务 decode 方向在源码上存在且有明确可行性条件，但简单配对已被真实 odd owner 数量否定；加入有限三任务分段后也只有较窄的1.5整核余量。下一步应先验证这个单一结构方向的上下文/同步合法性和完整长循环收益，不能先声称它能完成整轮2–3倍目标。

### D.4 四问

1. 对应 history 的 INT2 dequant＋QK/softmax/PV，window/current 与 merge数学保持。
2. D.4 的全历史恢复约6499.8–6655.1ms是明确失败样本；不是本版相位占比。
3. 本方案最多三个固定 KV256 上下文，两活跃 FP32 acc＋一份暂停状态；空间不随历史长度增加，保留每请求原256归约次序，禁止全量反量化。
4. 当前只能给 CAModel 条件周期预算，不能将它换算到 D.4/native 的 NPU 毫秒。完整算子与整轮目标都尚未验收。

## 7. 原生2048预算下，prefill补入与decode驻留的耦合

本节2026-10-10补充，仅只读工作量分析，不改变预算。指定参考 `references/vllm/vllm/engine/arg_utils.py:2316–2335` 对低于70GiB设备的OPENAI_API_SERVER默认值是2048；TP4不把它乘4，只有另一个CPU分支才乘world_size。`2480–2545`仍有throughput模式、禁用chunked prefill等条件分支，因此2048是当前所用分支的源码推断，实际最终值须以引擎解析为准。当前Ascend platform未找到batched-token覆盖；其 `platform.py:1111–1119` 将不同的max_num_partial_prefills重置为1。这里不通过任何显式或隐式配置提高预算。

原生Scheduler `scheduler.py:103–109,343–410,512–516,562–566,684–699,843–849` 的行为是：每step一个总token_budget；先遍历RUNNING（其中也包含尚未完成的prefill），再用剩余预算接WAITING。RUNNING并不等于正在decode；源码没有为每个请求分别给2048，也不是无条件先decode后prefill。正的long_prefill阈值、mamba对齐、抢占/恢复、异步eligibility等都会再改变实际分块。

无prefix命中/重算、每请求65100输入时：

- 每个请求至少参与 `ceil(65100/2048)=32` 个prefill调度步；120请求至少3840次**请求级参与**。这些参与可在一个全局step内交接或并存，不能当成3840个必然不同的全局step。
- 仅输入token量就要求至少 `ceil(7812000/2048)=3815` 个全局预算step。decode占掉预算、预算未满或重算只会增加这个数。
- 如果一个step有d个先被调度的q4请求，每个实际占4个target token，prefill最多剩 `2048−4d`。d=0/4/12/16/31持续不变时，单请求最少32/33/33/33/34块；这是说明竞争关系的假设，不是固定混合调度预测。
- 512输出在每次有效验证输出1–4token的简化条件下需约128–512个**请求级验证事件**。首token在最后prefill步生成会有1token边界修正；invalid-draft、停止裁剪、重算及eligible间隔使真实事件与全局step不同。三轮MTP是同一次服务迭代内的额外模型调用，不据此另给三份2048预算。

因此不能把“32个客户端在途请求”和“32个持续执行decode的请求”划等号。用一个明确的条件模型说明：假设每个已进入decode的请求在每个全局step都得到一次验证，平均每次输出α，target固定q4收费，prefill填满剩余预算，忽略爬坡/排空/重算。令 `L=512/α`、λ为每全局step完成prefill的请求数、d为该step域内平均decode请求数，则：

```text
λ = (2048 − 4d) / 65100
d = λ L
d = 2048 L / (65100 + 4L)
```

| 假设α | L（请求验证事件） | 条件平均d |
| ---: | ---: | ---: |
| 4 | 128 | 3.995 |
| 2 | 256 | 7.929 |
| 1 | 512 | 15.616 |

把历史已打印接受均值2.91357/3.78898代入，只分别得到5.469/4.216这样的**条件示例**；接受日志有warmup/缺失、当前预算也没有新实测，不能把这些值写成当前OSCAR/native的预测，更不是32并发的硬上限。不同step时长会改变wall加权并发；批量准入、decode被跳过/阻塞或临近排空都可产生不同瞬时值。它说明FULL空间变大后，decode并发仍可能受长prefill补入速度限制。

对671.222/447.481秒目标，仅输入量要求完成批次平均输入吞吐≥11638.48/17457.72 token/s。在没有模型执行重叠的简化服务时间口径下，至少3815个必要step对应整轮平均≤175.943/117.295ms，且这个预算已包括这些step中的decode、MTP及所有其他模型工作，还未给额外纯decode尾段留时间。这是可否证的必要工作量预算，不是单C4核的时间门或硬件毫秒预测。

**ROI判断**：C4长history与q4/q1必须共同优化；只让decode kernel更快会缩短其驻留，使同时decode的请求更少，并不保证提高prefill补入量。6.309× FULL字节比可提供容量空间，但仍不能自动提供 `32/12=2.667×` 吞吐；原生12只是旧日志Running峰值，也不是稳态decode batch。当前2–3×整轮目标仍需完整mixed路径成本，而不是把内存比、并发比和局部算子比相乘。
