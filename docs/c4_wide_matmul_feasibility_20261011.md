# C4 四成员宽 Matmul：只读可行性裁决（2026-10-11）

**结论：保留结构候选，暂不写独立核、不入组合。** 四次调用合并有减少 B 搬入的机会，但尚未证明 SDK M512 的实际复用；等待四段 P 全部完成会撤掉当前已测成员流水，现有条件账没有证明明显净收益。本次只读当前源码和官方 SDK 快照，复用已亲读的档案 #148/#154 与 D.4；未读撤销实现，未运行新实验。

**B 是否真四读。** 当前 [c4_candidate.cpp](../csrc/experiments/c4_member_pipeline/c4_candidate.cpp) 的 `CubeCluster`（499–525行）每 KV tile 调四次 QK、四次 atomic PV；K/V 由 `LoadClusterKv` 只发布一次，但每次 `Matmul/MatmulAtomic` 都重新 `SetTensorB→IterateAll→End`（448–477行），没有公开的跨调用 B 驻留契约。因此可确认四次独立消费，**尚不能把完整 GM→L1 的四遍字节量冒充已测事实**。官方 [scheduler_norm.h](../artifacts/cann_sdk_matmul_move_20261010/scheduler/base/scheduler_norm.h) 的 B 加载键为 `(kOuterIdx,nInnerIdx)`，不含 M，支持跨 M 复用的结构；但 [IsBFullLoad](../artifacts/cann_sdk_matmul_move_20261010/utils/matmul_config_utils.h)（349–355行）还要求 `stepN×baseN≥N`、`stepKb×baseK≥K`。仅改 `singleCoreM=512` 不能保证满足，现有快照缺缓存生命周期模块和 M512 实际静态 tiling。D256 若确实四遍变一遍，K+V 的理论搬入量才是 2 MiB→512 KiB/簇/tile；不是吞吐倍率。

**数学与状态必须按当前基底定义。** 当前不是“PV临时输出后 AIV Add”：AIV 按 member 执行 `RestoreState→Softmax内部alpha×acc→SpillState`，Cube 再 `MatmulAtomic` 直接累加到该 member 的 `accState`；`RoundDone` 后才允许下一轮恢复（936–999行）。宽 PV 应保持这条链，不能偷偷改成 deferred-alpha 或新 consume 算法。四份 Q/acc 已按128行连续存放；M512是504有效行加8条内部 padding。原 M64/M62 变成八个 M64，不改变各有效输出的 K128 顺序，但仍需逐位资格；尤其当前 Softmax **只发布 activeRows**（864–871行），每段 P 的126/127行必须在正确 MTE3 所有权下显式写 +0，不能消费旧 score/poison。finite 检查、mask、memberErrors、预取下一 KV 的错误归属保持每成员原顺序，不合并错误域；#154说明 CPU 顺序通过不能代替这些跨管道证明。

**容量有条件可容。** 逐 member 复用现有 UB 时，D256 的声明缓冲合计157760 B，不随 M512增长。保持当前双 KV 缓冲和 atomic 输出，只将 score/P 两槽改为四槽，按当前 workspace 公式为 **2494464→2756608 B/core**（增加256 KiB），必须同步 allocator 与所有偏移。若另设四份 PV 临时输出，相对现有单份还要393216 B，并新增回读/数学接缝。Cube不能同时把 A512×256和完整B256×256放入512 KiB L1（共768 KiB）；保留B256 KiB、分块A可容，但必须核生成的 L1深度和 L0配置，不能凭 GM arena 足够推定。

**条件预算首先要补回丢失的重叠。** [旧闭合循环预算](../reports/c4_phase_profile_q336_20261009/partial_c4_pipeline_budget.json) 的同一完整256-KV轮中，四次QK/PV合计192432 ticks，较慢AIV逐成员soft合计99966。按“四段QK完成→四段soft完成→宽PV”的新依赖，沿用这些服务成本、甚至忽略旗标和新增成本，需292398 ticks；原成员交错模型为220248。矩阵部分须先省 **72150 ticks，即37.5%** 才持平；尾轮同口径门槛约49.9%。这是**旧闭合片段的条件DAG门槛，不是当前C337实测预测或硬件下界**，不能与当前完整核1849738→1610282的已测收益重复计账，也不能把 wait 全当可删除。

**重开条件：** 先取得 M512 的实际 B 缓存/搬入证据，并让包含填零、状态通信、完整同步和丢失重叠的预算显示明显净贡献，再考虑单独研究；随后必须与当前 pipeline 完整配对。若只剩1–5%的增量或仅证明A/B加载次数下降，继续暂缓。#148要求保护数学与模型精度边界；D.4对应融合FIA内有界数据流，旧全历史dequant为6499.8–6655.1ms，本方向只能用固定tile共享规避重复，禁止全历史恢复；现阶段没有完整算子净提速上界或整轮达标承诺。

源码绑定：`c4_candidate.cpp` SHA256 `da59a923a44cdb7ac0916596cd89f802a24fc1ffef4426952402d90b5d2e7b31`；SDK来源与哈希见 `artifacts/cann_sdk_matmul_move_20261010/provenance.json`。
