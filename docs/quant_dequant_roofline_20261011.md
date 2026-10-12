# OSCAR 组合优化：64K 容量、roofline 与 decode 吞吐

> 2026-10-12有效门：完整单项相对当前基底至少1.15倍即可组合评估，最终组合相对原生2–4倍；默认prefix caching必须开启。本文历史2–3倍/none估算不可覆盖总MD的新目标与APC约束。

> **文档归属**：本文件是[《OSCAR 量化优化闭环》总 MD](../skills/oscar-quant-optimization/SKILL.md)下的 **roofline 专项子 MD**，用于总流程的“组合收益与瓶颈评估”环节。先按总 MD 核对约束和[统一台账](optimization_ledger.md)，再用本文件核算；结果回填台账，由总 MD 的验证与交付流程裁决。本文件不是第二个总入口，不独立修改目标或放行交付。

2026-10-11，按用户本轮补充明确设计机制。依据当前源码、`docs/optimization_ledger.md`、`docs/64k_120_optimization.md`、`docs/roi_decode_overlap_20261010.md`、档案 #140/#148/#154–156 与启动文档 D.4。下述公式是成本模型；未把本机模拟写成真 NPU roofline 测量。

## 0. 容量收益为何没有变成吞吐：本轮首先要解决的问题

遵循总 MD 的第一指示：找出当前 OSCAR 哪些执行成本抵消了更大 batch 的收益，并消除主要成本。当前不能把“INT2占用小、能多放请求”写成“执行已更快”。

### 已有实测说明什么

同一65100输入/512输出/120请求/K32历史对照，OSCAR/native墙钟2467.13/1342.44秒；平均TTFT208.93/230.37秒，而平均首token后时间约440.39/105.55秒，相差4.172倍。容量/准入已有好处，后续执行代价却吞掉了收益。Running采样峰值27/12支持驻留差异，但不是实际decode batch。[原日志汇总](../reports/evidence_64k_120_20261009.json)

完整打印Spec窗口的平均接受长度条件值约2.914/3.789，含warmup和缺失边界；它们不是严格配对的全局step计数。接受效率变化必须计入，但不能单独解释、也不能精确扣除上述4.172倍客户端时间。首token后时间仍含混合prefill干扰和调度等待，不冒充纯decode kernel时间。历史两轮也不是当前省略token-budget参数后的新测量。

### 当前代码确实增加的执行成本

以每rank Hkv1、D256、一个历史token为单位：

| 项目 | 当前普通decode路径的逻辑字节/工作 |
| --- | --- |
| INT2 K/V及FP16 scale/zero | `2*256/4 + 8 = 136 B`持久有效payload |
| 有界反量化后的K/V | `2*256*4 = 2048 B`，在UB计算后写GM工作区，再被Cube读取 |
| 临时K/V GM写＋读 | 至少4096 B/token，不含Q/P/score/PV、窗口、metadata与额外读取 |
| 同batch原生BF16 K/V的有效读取payload | `2*256*2 = 1024 B/token`；不据此声明实际总HBM流量 |
| 算术与依赖 | QK/PV采用FP32、HF32关闭；KV256循环内保留Vector→Cube QK→Vector softmax→Cube PV→Vector累积的依赖链，已有next-KV预取只隐藏其中一部分 |

源码入口：[`PipelinePublishKv/PrefetchKeys/PrefetchValues`](../csrc/kernels/attention_cv_decode_bundle.cpp:626)、[`Matmul`](../csrc/kernels/attention_cv_decode_bundle.cpp:186)、[`HistoryCubeTask/HistoryVectorTask`](../csrc/kernels/attention_cv_decode_bundle.cpp:681)。因此当前是**INT2持久存储＋浮点展开后的执行路径**；压缩率没有直接变成执行访存缩减率。K32、65536历史token时，仅这一层的临时K/V写＋读约8GiB，packed有效payload约272MiB；65100实际输入和sink/window分域另按真实任务计算。

这里的8GiB是累计GM地址访问，不是常驻内存或已测HBM。官方A2代码确认当前默认NORMAL、未显式L2 bypass；PERSISTENT枚举也不提供本路径可用的新驻留策略。临时工作区重复使用，实际L2命中、回写与资源竞争尚未测得；cache-off CAModel不能代替真机roofline。[缓存策略审计](../reports/quant_cache_policy_audit_20261011.json)

### 要闭合的因果判断与行动

`有效输出吞吐 = B*a/T_step`。如果上述反量化、临时流量、FP32计算与同步使`T_step`随B近线性增加，更大batch的吞吐收益会被吃掉；若每步增长更快或a下降，还会变慢。**这解释了机制，但目前未证明其中哪项占真实整步的主要比例。** 下一步的决定性证据必须区分这几项的暴露成本，并在完整相关路径验证删除该成本后的净变化。

当前优先主线是压缩域到计算单元的数据流和解包/消费成本。仅改变格式或隐藏局部等待已不能据此宣称解决：双请求流水长形约1.137倍，Kt/Vt-u32完整C4仍约6.1%回退；它们保留为局部证据，没有进入新增组合。已有writer/旋转/C4组合保持各自验收边界；不将小比例相乘包装成2–4倍。整数/仿射BF16分量路线已有完整原空间反例；先保留真实dequant值的BF16积方案在有限CPU样例通过，但成本不支持当前纳入。

## 1. 明确采用组合方式提速

本轮采用组合优化：量化写入、反量化与数据布局、历史复用/任务执行、紧凑 FULL KV 协同减少完整路径成本。单项有收益仍须核实它在当前组合中的**新增净贡献**；过小单项保留记录和重开条件，不装入当前组合。组合必须在同一母源、缓存格式、ABI、调度预算和冻结精度下成立，不能将不同实验的倍率相乘。

| 组合组成 | 要解决的问题 | 必须计入的代价 |
| --- | --- | --- |
| writer pack＋Init、Q Init | 降低量化打包和旋转准备成本 | 完整 writer/rotate，不用 helper 倍率代替 |
| 有界解包、复用、流水；布局研究另行裁决 | 减少反量化、格式转换和等待；保持数据及时到达计算单元 | Gather、padding、临时数据读写、索引、屏障、额外 UB/GM，及失去的重叠 |
| 紧凑 FULL 页＋请求级精确窗口 | 减少每页重复窗口占用，释放可用 KV 容量 | 全部 GDN/MTP 状态、sidecar、allocator padding、固定 workspace 与实际 HBM 峰值 |
| 依据真实任务工作的调度候选 | 避免部分核排长队、其他核空闲 | 规划、增加的解包/读写、窗口任务和短请求成本；不改变每请求归约顺序 |

表中包含研究方向，**不是全部已启用或已验收**。实际状态以统一台账和对应报告为准；当前PV K-major、完整B-NZ与Vᵀ-NZ均已测回退，不加入组合；仅K/V的L1复用仍独立审查。

## 2. 用户指定的主要提速机制

64K 场景按以下因果链设计：

**INT2 降低 KV 容量与实际访存 → 可驻留更多请求 → 调度/prefill供给允许时形成更大的实际 decode batch → OSCAR 与原生都受访存限制，且反量化和 MTP 未形成新的计算瓶颈 → 每个有效输出 token 的平均成本下降，decode 吞吐提高。**

原生是访存受限，按用户提供的现场判断作为设计前提；当前复制日志并未给出原生和新组合在不同 batch 下的完整带宽/算力测量，因此具体 roofline 位置仍须分范围证明。不能只凭 INT2 存储位宽宣布 OSCAR 已访存受限。

必须区分三个量：

- **可驻留请求数**由实际 KV、GDN/MTP 状态、workspace 和其他显存共同决定。
- **实际 decode batch**是这一轮被调度并执行 decode/verify 的请求数，不能用 Running 或客户端在途上限替代。原生小 token budget 下，prefill 补入速度也可能限制它。
- **每步有效输出数**取决于 MTP 接受、停止裁剪与真实执行。drafted token 数不是有效输出数，平均接受长度也不能直接当全局 step 数。

服务仍省略 `--max-num-batched-tokens`，不抬高预算、不关 MTP、不改 GDN 来制造该机制。

## 3. roofline 要计算完整成本

对同一轮 decode/verify 的实际工作，以设备层级分别记账：

```text
D_HBM = weights + 实际KV读取/写入 + 精确窗口 + metadata
        + 到达HBM的临时数据 + MTP所有实际轮次的相应流量
D_L2、D_L1分别统计经过这些层级的流量，不与D_HBM重复相加。
F_Cube[精度] = target + 所有实际draft轮次 + OSCAR额外Cube工作（按精度分列）
F_Vector = 量化/反量化/旋转/softmax/归约等实际Vector工作

单阶段资源时间下界：
T_min = max(D_HBM/BW_HBM, D_L2/BW_L2,
            T_Cube服务下界, F_Vector/P_Vector, 其他独立资源下界)
T_actual >= T_min
实际step时间由各阶段依赖、资源竞争、通信、标量/发射与同步共同决定。
```

带宽和计算上限必须注明来自哪里、适用什么形状和精度；不可用无法达到的峰值给出乐观结论。串行阶段按顺序累加，重叠阶段按依赖图核算，不将 actor 或嵌套相位时间相加成 wall。只检查 Cube FLOPs/byte 不足以排除 Vector、标量或同步瓶颈。

尤其不能把 W8A8、BF16 和 FP32/HF32-off 的运算量混在一起除以同一个 Cube 峰值；同一共享计算单元上的各精度服务成本分别核算。TP4使用每rank的实际工作、带宽和关键路径，另计通信；不能把全模型字节/运算量与单卡带宽或算力混用。

**OSCAR 必须避免压缩省下的访问被其他成本抵销**：当前有界 FP32 K/V 发布、Cube 重复读取、score/P/accState 中间流量都要入账；其中多少命中 L2、多少落到 HBM 不能猜。cache-off CAModel 可以比较其固定模型内两臂，不能直接证明真机访存瓶颈或实际 HBM 压缩倍率。

公开SDK的UB→L1接口也要按实际实现计费：当前A2的`DataCopyUB2L1Impl`及ND2NZ变体均通过GM workspace软件模拟，包含UB→GM与GM→L1消息，不能按无GM直通删掉这段流量。[SDK源码123–213行](../artifacts/cann_sdk_cube_public_20261010/impl/basic_api/dav_c220/kernel_operator_data_copy_impl.h)。其中多少访问落到HBM仍须分层测量。仅K/V的L1复用是独立的驻留/生命周期问题，当前未验证、未否决。

每个 batch 下重新核算 MTP。batch 增大后，verify/draft 的计算量随工作增长，weight 摊销虽然改善，也可能撞上计算上限。若反量化、Cube/Vector、MTP 或同步成为关键瓶颈，就不能再用“都访存受限”的简化模型解释加速，必须继续降低对应的 OSCAR 成本或使用包含该瓶颈的实际时间。**不能通过增加无用访存把算子人为变成 memory-bound。**

## 4. decode 吞吐倍数的量化口径

在稳态、同类请求、每个活跃请求每轮参与一次验证的条件下，记 `B` 为实际 decode batch，`a` 为每请求每轮产生的平均有效输出 token，`T` 为包含实际 MTP 的 step 时间：

```text
Throughput = B * a / T
X_decode = (B_oscar / B_native) * (a_oscar / a_native)
           * (T_native / T_oscar)
```

更一般的变 batch 情形用 `总有效输出token / 对应完成窗口wall`，不能把三个独立平均数拼在一起。只有两侧 `a` 相同且 step 时间近似相同时，batch 比才近似等于吞吐比。MTP 接受效率变化必须保留在计算中。

即使两侧都是访存受限，也应使用各自真实流量和可达到带宽；条件式是：

```text
X_decode ≈ (B_oscar*a_oscar)/(B_native*a_native)
           * (D_native/BW_native)/(D_oscar/BW_oscar)
```

这里的 `D` 包含上述完整对应流量，不是只取 INT2 payload。容量压缩比、HBM 流量比、batch 比和吞吐比必须分别报告。

## 5. 当前证据和组合验收

- 已有真实整轮锚点：native 1342.4437259s、旧 OSCAR 2467.1275737s；输入65100、输出512、120请求、客户端K32。当前目标是相对原生2–4倍，即335.61–671.22s。它们不是当前省略 token-budget 参数后的新实测。
- 请求窗口方案的条件字节账为 FULL 约6.309倍、全部KV约3.823倍；未用该容量比预测同倍吞吐。两侧旧日志 Running 峰值27/12也不是实际稳态decode batch。
- writer约1.796倍、Q Init约1.621倍有各自局部证据；C4成员流水1.148704倍低于新1.15门，已移至显式research且默认不启用。专用H16→H32组合gate固定要求至少1.15倍，尚无同形完整通过证据；不能把局部倍率相乘得到整轮或decode倍率。
- 完整B-NZ **1608607→1743703 ticks（慢8.40%）**、Vᵀ-NZ **1610898→1734415（慢7.67%）**均退出组合。两项各自同库、600秒期限内完成，CAModel逐位/独立oracle/1008任务/60actor通过；真实K/V发布已计费，Rv一次256KiB位pack未计时。[B配对](../reports/c4_b_nz_20261011/completed/reports/camodel-valid/comparison.json)、[Vᵀ配对](../reports/c4_b_nz_vt_20261011/reports/vt-camodel-valid/comparison.json)。更少发布burst或更快的独立consumer没有转成完整核提速，但本轮没有相位证据可将回退唯一归因。
- 两项publisher CPU通过；B完整candidate CPU仍为120秒超时，Vᵀ完整candidate CPU未运行。完整CAModel数值完成不改写这两个CPU边界，也不等于目标NPU或graph通过。[B裁决](../reports/c4_b_nz_20261011/decision.json)、[Vᵀ裁决](../reports/c4_b_nz_vt_20261011/decision.json)
- 下一轮组合需同时回答：实际batch是否提高；每有效输出的分层访存是否下降；计算/Vector/MTP/同步是否仍有余量；完整kernel与step是否缩短。先用既有日志、源码账和必要短合成实验筛选，用户最终真机认证，不再要求重跑AISBench。
- 稳态decode提速还须加上prefill、混合阶段、爬坡与排空的时间，才能判断整轮目标。若组合的可信估算不满足目标，回到剩余OSCAR主成本继续优化；不得改模型参数来拟合目标。

交付始终同步到 `install_serve.sh --variant candidate` 与 `install_observe_serve.sh --variant candidate`；共享候选开关、实际选核和缓存格式一致，真实探针失败则阻断observe正式服务。CPU/CANN/CAModel、真NPU、图、质量、显存、decode吞吐和整轮性能分别记录。

## 2026-10-12 新增完成证据与默认 prefix 约束

小型当前 d817 R2/q4/C1341（每history任务4完整tile+尾行）的同库normal/profile全部完成，1072361/1093991 ticks，插桩扰动2.017%；partial/LSE/status逐位、独立oracle、3actor及全部任务通过。证据为 `reports/decode_completion_20261012/completed/reports/` 中comparison、analysis及两臂6份完整completion/popped日志。

两个history任务中间完整轮：PVReady→下一ScoreReady约17064–17083，ScoreReady→PReady约5910–5988，PReady→PVReady约20969–20975 ticks。当前稳态关键交付链约86.5%在QK/PV、约13.5%在softmax与P交接；后续KVReady已就绪。这不等于Cube纯算力占比，也不代表64K/K32或NPU。需进一步减少AIC侧暴露的ND2NZ块装载/控制；只优化已隐藏的解包不能兑现该小形的主周期收益。

用户最新要求所有服务入口默认显式prefix caching=true，走原生align和现有page-owned精确快照；旧none/request-private布局容量数值不再是默认配置的容量证据。研究request-private window尚缺持久exact checkpoint与native/GDN共同命中接缝，不能简单删除guard。兼容设计与容量成本见 `docs/prefix_caching_compatibility.md`，完整协议未实现之前不宣称该研究路径兼容。
