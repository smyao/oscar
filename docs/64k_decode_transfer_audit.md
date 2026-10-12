# 64K decode：A2 FP32 Vector→Cube 传输审计

日期：2026-10-09。范围：当前 `attention_cv_decode_bundle.cpp`、本项目指定的只读 `references/vllm-ascend` 原生实现。已按启动文档 §0、完整 D.4 与档案 #126/#129/#140/#145/#148/#151/#154/#155 核对。§1–9为只读源码与已有证据审计，当时未启动编译/模拟；§10另列后续明确授权的独立CANN/CAModel首轮结果。未修改生产实现，未跑目标NPU或服务；源码推导不能作为设备能力、正确性或速度验收。

## 1. 先排除什么

1. **目前没有找到可直接复用的 A2 AIV→AIC 共享 L1/TSCM FP32 实例。** 找到了同名通用模板，但没有调用；找到了真实的 UB→L1 float 实例，却由 `__DAV_C310__` 独占选择。不能把它移植到 910b4 后声称可用，也不能用搜索未找到反向证明硬件绝不支持。
2. **跨 q4 与后续 q1 保留 L1 不能成为当前实现的小改动。** 它们是不同模型调用；target 与 MTP 还有不同层的 KV。后续 draft 的 Q 依赖前一步模型输出。当前 API 没有跨 launch 的 L1 所有权、存活或核映射协议。
3. **21.875 MiB workspace 不是已证实的 L2 溢出，更不是 HBM 字节数。** 需要实际 cache/GM/HBM 计数；减少 workspace 可减少容量压力，也可能破坏现有流水。
4. **将逻辑 KV256 改为 KV128 会改变 online softmax 的归约边界。** 本轮冻结数学要求下不作为“仅搬运优化”。可以研究物理 K128 分两次 QK 列块、仍收齐 score256 后做一次原 softmax、一次原 PV256，但 API 的 FP32 累加次序和互联能力仍须证明。

优先顺序：先做一个独立的 A2 FP32 UB→候选 L1→Cube 最小能力证伪实验；失败即停止该方向。仅当它通过后，才值得尝试 480 KiB 的逻辑 KV256 方案。原生已有的 GM→L1 复用也可研究，但 decode q4 已共享同请求的四个 query；没有可凭空合并的相邻 query 组。

## 2. 当前每个工作单元实际做了什么

### 2.1 类型与数学边界

`csrc/kernels/attention_cv_decode_bundle.cpp:33–56` 固定 M32、KV256、每 AIV 的 HalfKv64；两个 AIV 对应一个 Cube。`Matrix` / `TransposedMatrix` 的 A/B/C 都是 `TPosition::GM`、ND、float。`:183–194` 每次设置 GM A/B、`IterateAll<true>` 写 GM C，然后 `mm.End()`；`:106/:185` 明确 `SetHF32Mode(false)`。

`:279–305` 读 INT2 packed 行到 UB，按实际虚拟128页边界分段 DMA；D256 每行 `D/2+8=136 B`，UB 行补齐到160 B。`:620–656` 在 UB 恢复 float K/V，`:627–632` 发布到 GM 的两个 K/V ping-pong tile。这里只恢复有界 tile，未恢复整段历史。

`:681–744` 的 source0 协议：

| 阶段 | AIV | AIC | 保护的存活期 |
|---|---|---|---|
| 初始 | 解包并写 K0/V0；MTE3→`KvReady` | 等待 | 两个 AIV 的行区不重叠 |
| QK(i) | 预取/解包/写 K(i+1) 到另一 buffer | 读 K(i)，QK→GM score；FIX→`ScoreReady` | K 当前与下一份同时存活 |
| softmax(i) | 读完整 score256；FP32 online update；写 P；`PReady` | 等待 | score/P 同址，QK 已写完 |
| PV(i) | 解包/写 V(i+1) 到另一 buffer，发下一 `KvReady` | 读 P、V(i)，PV→GM；`PvReady` | V 当前与下一份同时存活 |
| 累加 | 读 PV，加到 UB accumulator | 可开始下一 QK | 下一 `PReady` 必须晚于本次 PV 消费 |
| 最后 | FP32 normalize，写 GM accumulator | FP32 inverse rotation，再交 AIV | K/V 已死，可另行复用空间 |

额外约束：`:658–665` 在前一个 PV 已累加后，按 K0,V0,K1,V1 原顺序提交预取错误；不能让下一 tile 的错误覆盖更早的 score 错误。`:547–554` 的 P 写回使用 **MTE3→MTE2**，防止下一次 score DMA 提前复用 UB；不能只换成 MTE3→V。这些是档案 #145/#154 同类问题的生产防线。

### 2.2 可复算的字节与运算账

按一个同请求、同 KV head、同 source0 split 的完整 KV256 单元，D256；不乘 TP4，因为以下均为单 rank：

| 项目 | 公式 | 字节/次数 |
|---|---|---:|
| INT2 K+V+metadata 逻辑读取 | `256×136` | 34,816 B |
| FP32 K/V 写 GM | `2×256×256×4` | 524,288 B |
| Cube 的 K/V 逻辑读需求 | 同上；实际 SDK tile 重读另计 | 至少 524,288 B |
| K/V GM 接口写+读 | 上两项之和 | 1 MiB |
| q4 QK+PV 有效 FLOP | `4×(4×6)×256×256` | 6,291,456 |
| q1 QK+PV 有效 FLOP | `4×(1×6)×256×256` | 1,572,864 |

这里的“1 MiB”是已写出的 GM 接口工作量，不等于 1 MiB HBM 访问；L2 可命中，SDK 内部也可能重复读。q4/q1 相对这部分字节的有效计算强度分别仅 6 / 1.5 FLOP/B；不能据此直接认定设备瓶颈，因为解包、softmax、同步、Cube 内部 padding 和 native 模型时间还在。

另有 Q、score/P、PV、最终旋转及 metadata/status 流量。其精确值不能统一按活跃行计算：例如 `:453` 每个活跃 AIV 始终读16行 score，而 `:550` 只写实际活跃 P 行；q1 只有6行，q4有24行。K/V 的上述完整 tile 发布仍与 qcount 无关。

上下文 C=65100、S=64、R=256、同请求 q4 从当前上下文起始时，历史候选区间为 `[64,64848)`，长度64784。生产 task 公式在 `csrc/kernels/attention_tasks.cpp:128–145`。S1 需要 `ceil(64784/256)=254` 单元；S3 按原生任务等宽划分需要 `3×ceil(ceil(64784/3)/256)=255` 单元。因此单 FULL 层/单请求/q4 的 K/V GM 交接为254或255 MiB；窗口/current、头尾 padding、Q/score/P 和旋转另外计算。S3 的活跃 packed 行仍只读64784×136 B；FP32 tail 仍按完整256发布。后续 token 的实际区间应代入对应 context，不把这个例子当整轮实测。

单 core workspace 是：

```text
((2×M + 4×B)×D + M×B)×4
= ((64+1024)×256 +8192)×4
= 1,146,880 B = 1.09375 MiB
20 core = 21.875 MiB
```

四个 B 平面是两份 K 和两份 V。其余三个 M×D/M×B 区域用于 Q、score/P、PV/rotation；UB accumulator 跨256单元保留。权威公式也在 `csrc/include/oscar_attention_launch.h:24–27`。20 core 是本次设备形状假设，不应覆盖 runtime 实际 core 数。

## 3. A2 与其他架构的证据边界

### 3.1 无调用的通用模板，不能充当共享 L1 实例

`references/vllm-ascend/csrc/common/include/kernel/l1_to_ub_iterator.h:24–50` 定义 `l1_to_ub<ArchTag,DataType>` 与 `ub_to_l1`，主体仅 `DataCopy(LocalTensor,LocalTensor,DataCopyParams)`；`ArchTag` 没有约束或特化。全指定 native C++ 树的同名检索只找到此定义、MLA preprocess 的重复定义与 include，未找到实际实例化调用。也未找到 `TPosition::TSCM` 的实例。

特别是 `common/include/kernel/mem.h:52–58`：`__DAV_C220_VEC__` 只初始化 UB；`__DAV_C220_CUBE__` 才初始化 A1/L0A/L0B/L0C。这个封装不提供一个两边同时拥有的共享 L1 对象。CPU fallback 分支 `:59–65` 同时建立多类 local tensor，也不能证明 A2 device 访问资格。

`common/include/kernel/hardware.h:21–33` 的通用常量给 L1=512 KiB、L0A/B=64 KiB、L0C=128 KiB，但该**未按 SKU 特化的模板**还填入 L2=192 MiB、supportMix=0。只能作为该模板的资源假设，不能当成910b4设备查询，更不能用其中 L2 常量做性能承诺。

### 3.2 原生 ascend910b 的 float 传输确实经 GM

`moe/hc_pre/op_host/hc_pre_def.cpp:65–66` 注册 ascend910b / ascend910_93。入口 `op_kernel/hc_pre.cpp:19–29,49–88` 用 `__DAV_C310__` 选择 arch35，其余选择 `hc_pre_m_k_split_core.h`；后者注释虽写“A3”，仍有上述910b注册，不能仅凭注释排除它。

非310路径中：

- `hc_pre_m_k_split_core.h:48–63` 创建每 Cube 的 `GlobalTensor<float> xCastFp32WsGm`；`:151–163` 从 UB float 经 `CopyOut` 写这个 GM buffer，MTE3 跨核 flag 发布。
- `:122–131` Cube 等待后读取该 GM buffer；`hc_pre_cube_compute.h:146–165` 分配 A1/B1 各256 KiB 与 float L0。
- `hc_pre_cube_compute.h:178–207` 用 `Nd2NzParams` 做 GM float→L1 float；`:360–413` 做 float L1→L0 的格式搬运。

**数学不可以照搬：** `hc_pre_cube_compute.h:462–476` 的 `MmadAB` 明确打开 HF32/HF32Trans。它证明该 native 路径的 float 存储/传输与 L1 管理，不证明 OSCAR 禁 HF32 的 QK/PV 数学等价。

### 3.3 真正的 UB→L1 float 例子属于 arch35

`hc_pre_m_k_split_core_arch35.h:42–65` 在该路径建立 L1/UB对象，用 `SYNC_MODE4`；`:175–198` 先 `VFTransND2NZ`，再由两个 AIV 写 L1 的不同位置，MTE3 发布。`hc_pre_base_arch35.h:1084–1087` 的 `CopyToL1` 最终也是 LocalTensor→LocalTensor DataCopy。

但这个头只在 `hc_pre.cpp:19–22` 的 `__DAV_C310__` 分支被选中。其 VF 转换、同步模式、AIV L1 可见性均没有在本次 A2 路径获得证实。**这是要排除误用的对照，不是910b4可用方案。** 同理不能自行引入 arch32/35 的接口并当作 A2 能力。

## 4. 原生 arch22 的 L1 复用能借鉴到哪里

`attention/sparse_flash_attention/op_kernel/sparse_flash_attention.cpp:18–22` 非 `__CCE_AICORE__==310` 走 arch22；`:93–99` 的实际 Q/K/V 模板是 half 或 bfloat16，绝非 FP32 输入。float intermediate/L0 accumulation 不改变输入类型。

`arch22/sparse_flash_attention_service_cube_mla.h` 有完整可定位的复用：

- `:597–614` 外层 N、K 切片先取 K 的 L1 buffer；`:726–746` 从 GM 载入/发布；`:750–840` 内层多个 M 子块反复使用同一 K L1 内容；`:841` 最后释放。
- `:766–784` Q 在第一个 N 切片载入，后续 N 复用，`:817–819` 才释放。
- PV 对称：`:899–977` 载 V 到 L1，`:980` 开始多个 M 的循环；`:990–995` P 也按 N 复用。
- `:790–815` 展示 MTE1/M 保护 L0 消费；`:822–835` 的 float score 经 Fixpipe 写 GM。它没有提供 AIV→AIC 直接传 P 的先例。

这些说明“同 kernel、同 core、明确所有权、多个已就绪 query 组共享同 K/V tile”的 L1 复用可以设计。当前 decode 却不满足多 query 组条件：

- `oscar_ascend/ops/cv_dispatch.py:74–88` 后续 draft q1 与 target≤4 路由不同；`attention_cv_decode_bundle.cpp:818–821` 分别 OwnerTile1/4，`:168–177` 限制同 task 的 qcount、head、request/source/split。
- q4 已将四个 query×6个 GQA heads 合成24行，仅一组 M32；q1只有6行。不同 request、KV head、layer、source/split 不能因长度相同就共享 KV。metadata 中不同请求行可映射不同物理页，prefix=false 时更无已声明共享关系。
- q4 task结束后再取下个 task，按 `:130–151` 的 workId 跨 source/head/split 分配；邻接 workId 不是同请求相邻 query 的充分条件。当前 Matmul 是否在一个调用中重载某些 GM tile，必须由生成代码/设备计数证明，不能假定每个 basicN 都重载同一 K/V。
- MTP 后续迭代见 `vllm_ascend/spec_decode/llm_base_proposer.py:1219–1229,1276–1289,1305–1331`：取上一步 draft token/hidden states，执行模型、采样，再写下一步 token。未来 Q 未就绪。target FULL 与 draft FULL 的 KV还是不同层，不能共享它们的值。

完整65100历史 FP32 K+V 为 `65100×256×2×4=127.1484375 MiB`，远大于上述单 core L1 假设。既不能跨整次历史扫描保住所有 K/V，也不能为了将来的 q1 恢复整段 FP32 历史；D.4 已禁止这种退化。跨 launch L1驻留还需要被其他层算子覆盖、流次序和core映射的正式协议，本项目没有。

## 5. 保留逻辑 KV256 的候选与生存账

### 5.1 480 KiB L1 方案：空间成立，通路待证

假设 A2 可从两个 AIV 正确发布 float NZ 到同一 AIC L1，且资源可由专用 kernel 独占；候选容量：

| 内容 | FP32 形状 | 大小 | 活到何时 |
|---|---|---:|---|
| Q | 32×256 | 32 KiB | 本 task 全部 KV 单元 |
| K 半块 | 128×256 | 128 KiB | 对应 QK128 列块的 L0 已消费 |
| V 完块 | 256×256 | 256 KiB | 一次 PV256 的 L0 已消费 |
| P | 32×256 | 32 KiB | 本次 PV 已消费 |
| PV 预留区 | 32×256 | 32 KiB | AIV 已取回/累加 |
| 合计 | | 480 KiB | 剩32 KiB只是容量余量 |

流程：发布 K前128→QK前128；确认消费后复用 K区发布后128→QK后128；收齐原 score256 后调用一次现有 softmax；发布完整 P与V，执行一次原 PV256。最后的 Rv 可复用已死 K/V区。QK拆的是 N 列，不拆每个 score 的 D256 点积；PV仍沿同一个256归约。**这只是尽量保住数学边界，尚未证明 SDK 实际累加顺序不变。** 若改写原生手工 Mmad，尤其不能照搬 HC_pre 的 HF32开关或不同K分块而不验证。

剩余阻断：

1. 两个 AIV是否能寻址同一个 AIC L1、float LocalTensor→L1资格、producer/consumer对应关系、跨核同步模式；无当前 A2实例。
2. 当前 unpack输出是 ND；普通 LocalTensor DataCopy不会自动证明 ND→NZ转换。需独立 bit-preserving layout transform；arch35 VF函数不能照搬。其额外 UB和指令账尚未知。
3. `MatmulType<TPosition::GM>` 不能仅换一个指针就读 L1。指定树没有 TSCM float Matmul实例；须证明 A1/B1或TSCM的合法type/布局，或另写正确的FP32 L0加载。L1的“PV预留区”也没有证明 AIC L0C→L1或L1→AIV通路；初版可保留 score/PV GM结果，仅验证K/V输入方向。
4. 该布局容不下原来的 K/V双缓冲；保留部分重叠、延迟发布V或重新分配K死区需要新的完整协议。省字节不保证省时间。
5. 每个替换/覆写必须等 L0 的最终消费确认，不只是AIV写完；score/PV GM仍须保留原flag与MTE3→MTE2保护，错误提交顺序也不得变化。

所以不能凭480 KiB加法立即实施生产 kernel。最小能力实验通过前，此方案状态是**未证明可落地**。

### 5.2 保持 GM 通路，只缩短 buffer 存活

下表只是显式调度候选，逻辑 KV256、unpack、FP32 matmul/softmax仍可保持原边界：

| GM K/V平面数 | 单core/20core容量 | 必须付出的存活约束 |
|---:|---:|---|
| 当前4 | 1,146,880 B /21.875 MiB | K/V均能预取下一tile |
| 3 | 884,736 B /16.875 MiB | K或V单缓冲；必须等其当前QK或PV消费后才能写下一份 |
| 2 | 622,592 B /11.875 MiB | K/V各一份，原来的两个预取窗口不能照搬 |
| 1，K/V别名 | 360,448 B /6.875 MiB | QK读完K后才写V；PV读完V后才写下一K，显著收紧流水 |

容量公式统一 `((2×32+p×256)×256+32×256)×4`。**每KV256仍写512 KiB float K/V，Cube仍有对应读需求**；这不是减少GM接口字节的方案。3平面可先作“容量/流水”的对照，避免直接改分块数学：例如K双缓冲、V单缓冲必须把下一V发布移到当前 `PvReady` 之后，再发下一个 `KvReady`。预取错误的提交仍在原PV累加之后。

没有真实L2瓶颈证据时，不优先投入这个方向；原生产专门增加第二tile就是为了与QK/PV重叠。若单缓冲仅减少静态地址范围、却让Cube等待更久，它应被淘汰。

## 6. 最小可否证实验；本轮未执行

### 门A：只验证 A2通路，不写完整 attention

独立文件、独立入口、显式 ascend910b4/C220 架构限定；不接受310编译产物。每个混合核组的两个 AIV各生成一个含lane/core/generation特征的 float32小tile。尝试候选 LocalTensor UB→L1 发布；AIC读取同一tile，做HF32关闭的最小 FP32 恒等矩阵乘，输出GM。先验证FP32位模式可恢复，再扩到64×256行区与实际NZ布局。小tile应覆盖正负、非整数量化值、0、边界finite；位模式搬运检验与FP32算术比较分开。

要求：编译器接受**AIV端**L1寻址与float DMA；证明输出对两个lane/core/generation都匹配；汇编/访问记录不能偷偷经GM staging；反向消费ack能阻止下一代覆盖；多core重复、改变输入的图回放均正确。编译失败、仅Cube本地拷贝通过、只单lane成功或只CPU debug成功均不算门A通过。若此门不成立，直接停止共享L1主线，保留GM路径。

不能预设 `SYNC_MODE4` 在A2可用；必须从该架构API编译和真实事件行为确定协议。也不能只验证“CPU分支同时构造UB/L1对象”就宣布跨核共享成立。

### 门B：一次逻辑KV256，保住数学与错误契约

在门A通过后才做K128两次QK列块、原score256/softmax/PV。与当前生产输出逐项比较：INT2恢复的K/V bit patterns、QK各score、P/FP32 max/sum、PV、partial、LSE、两AIV status、最终旋转；不要只比最终token。q1/q4、D256、GQA6、全零/极端finite/NaN/Inf注入、空源、KV尾1/127/128/129/255/256/257、非连续页、S1/S3、padding孔、两个不同请求、MTP拒绝回退均需覆盖。验收阈值取冻结配置；不为新分块放宽。

### 门C：有限decode设备基准，确认收益确实来自传输

仅固定shape短算子/图probe：同工作量的GM原版、门B候选，另加3平面GM对照；涵盖q1/q4、并发1/8/32、C=65100附近、S1/S3、20core实值。报告设备事件时间、AIV解包/等待、AIC QK/PV/等待、GM/L2/HBM计数（如工具能提供）、workspace占用和完整status。先用相同输入算子比，再用changed-input图回放核验资格；不能把模拟器cycle或workspace减少换算成模型加速。

## 7. 对整轮速度目标的意义

该方向可能减少source0的GM中转或片上重复加载；它不减少GDN、MLP、通信、native current attention、真实INT2数学和MTP依赖。当前证据不能推出消除1 MiB交接就能减少对应比例墙钟，也不能推出相对native的4–5倍。

整轮目标仍为native约22.4分钟的四至五分之一，即约268–336秒；所需预算与不可识别部分见 `docs/64k_latency_budget.md`。FULL KV 6–7倍容量收益与这里的速度比是两种口径。本文淘汰的是无A2资格的整段重写与跨launch复用假设，保留的是可用很小实验判定的片上传递能力问题。

## 8. 补审：M32 workspace 是否偷偷计算 M64/M128

**结论：已有实际 MMAD 记录排除了这个解释；没有“将 M64 改成 M24/M6”可以再领取的算量收益。** 本节只读已存本工程算子编译/模拟证据，未重新编译或运行。

### 8.1 能证明什么，哪些文件不能证明

当前 `attention_cv_decode_bundle.cpp:48–56` 的 `basicM=64`、`singleCoreM=128` 是 SDK 编译 tiling 配置；`:183–194` 仍调用 `SetOrgShape(m,...)` 和 `SetSingleShape(m,n,k)`。source0 的实际参数来自 `:681–689`：`m=qcount×GQA`，所以 q4/GQA6 是24，q1是6，N/K均256。normal config 的 SetTail 机制在本工程已有 SDK 核查记录 `docs/cv_implementation.md:24`；不能只按常量64/128推算运行M。

三份 `reports/decode_cube_tail_baseline_aic{,_c220,_decoded}_20261006.asm` 是同一个 SHA256 `3c17fc6a7557a28d0565e1e3e9df53abaac88b7fa6a2bcaedcc75288d1b56db1`；每份5124条地址行均为 `<not available>`。文件名含 decoded 并不表示成功反汇编，**这些文件没有提供 MMAD 形状证据**。

可用证据是 CAModel 实际指令：

| 样本 | 指令记录 | 观察 |
|---|---|---|
| q4，C1025，原SIMD诊断克隆 | `reports/decode_marker_v2_camodel_20261007/q4_three_tiles_1025-baseline/instruction_samples/core0.cubecore0.instr_log.dump.txt:85,99` | `MMAD dtype:F32F32`，XT分别 `0x8100000040080018` / `0x0180000040080018` |
| q1，C773，window-batch的baseline | `reports/window_batch_decode_camodel_20261006/q1_c773_ring259-baseline/instruction_samples/core0.cubecore0.instr_log.dump.txt:90,101` | 同类型，XT分别 `0x8100000040080006` / `0x0180000040080006` |
| q1，同case candidate | 同路径 `-candidate/...:90,101` | 相同 M6 编码 |

将XT低36位按连续12位字段展开，分别为 `(24,128,64)` 与 `(6,128,64)`；结合已知QK/PV的M/N/K调用、basicN64/basicK128和后继累加对，可对应 **M24/M6、K128、N64**。高位的初始化/累加控制不同。完整q4 `core0.cubecore0.instr_log.dump.gz` 有104条MMAD，全部都是 `(24,128,64)`，其中两类控制值各52条；它对应6个KV单元的QK/PV共12次matmul，加一次Rv旋转，即 `13×(4个N64×2个K128)=104`。core1没有MMAD。该actor原始gzip完整，不能用只保留selected lines的txt去数总次数。

q1的现存core0样本明确是 `prefix_capped=true / complete_for_actor=false`，只支持所见指令M6，不支持完整kernel的MMAD计数或时间占比。上述样本是当前bundle之前、具有同一GM float Matmul配置与真实M参数的已存克隆；当前bundle的history保留这些参数及K128基础分块，新增tail逻辑只改变precise source的N/K。**本轮没有当前bundle的新反汇编或NPU MMAD trace**，因此不把旧样本冒充当前二进制逐指令身份认证。

### 8.2 padding 有三种口径

| 口径 | q4：24行 | q1：6行 | 能否称为可删除Cube FLOP |
|---|---:|---:|---|
| GM固定Q/score/PV容量M32 | 8/32=25%空行 | 26/32=81.25%空行 | 否；PV/写回等已按活跃行裁剪 |
| 观察到的L1→L0A行extent | 32行，8/32=25%空行 | 16行，10/16=62.5%空行 | 仅能证明格式/搬运对齐 |
| MMAD编码的逻辑M | 24，额外0行 | 6，额外0行 | 没有M64/M128逻辑行浪费 |

L0A证据：q4同instruction txt `:80–81` 的 FMATRIX=`0x10020`、LOAD3D XM=`0x200080`；q1 `:82–83` 是 `0x10010`、`0x100080`，与原生float LoadData按16行对齐的结构相符。q4同一MMAD完成与popped记录首对相差277 cycle，q1首对相差149 cycle，说明这两个样本的实际执行时长也有区别；它们不能精确解出设备内部乘加阵列有效lane数，不能把L0A填充比例直接当墙钟可省比例。

`basicM16`若使q4拆为M16+M8，会把原单M轮变成两轮；每行D256仍需原来K128+K128的累加，N64四轮也还在，MMAD数可能从8增到16/次matmul。若仍按16行搬运，总行extent依然是16+16=32。即使假设能消掉全部32→24的内部填充，仅该项理想上限也是4/3，未达到1.5，更不用说整个kernel。q1已经发布M6；把配置改成M8并不会把6行再减半，强制执行M8反而增加逻辑空行。指定证据没有建立A2可用的独立M8搬运/执行模式。缩小tiling资源或控制开销可以另有收益，但不能再以虚构M64算量为依据。

### 8.3 已经试过与已经进入生产的部分

`csrc/experiments/decode_static_matmul/generate.py:1–7,27–39` 已明确记录“当前MMAD是M6/M24”，实验只把 `singleCoreM` 固定成 `OwnerTile×6`，关闭runtime SetTail/SetOrgShape，保留FP32 K128累加。`attention_static.cpp:48–67,182–191` 可直接核对。对应 `reports/decode_static_long_camodel_20261007/comparison.json`：完整q4/R1/N128/C20000/S3算子，baseline **2,464,632**、candidate **2,426,161** tick，比值 **1.0158567×**；partial/LSE/status逐位相同、两臂独立oracle通过，结论 `rejected_insufficient_gain`，未启用生产，NPU/graph未跑。这是已有局部否决证据，不应重复作为新候选，也不能把1.016外推到64K整轮。

`decode_tail_range`针对的是KV尾部N/K零填充，**不是M64→M24**。`csrc/experiments/decode_bundle/generate.py:34–40` 已将其Matmul物理stride/真实shape与TailColumns整合；当前生产 `attention_cv_decode_bundle.cpp:196–220` 对window/current按16列向上取整，source0明确仍为KV256以保护流水共享score/P。故precise tail裁剪也已经进入当前bundle，不能再次作为未实现的大收益。

本次可作出的判断：**没有证据支持 M16/q1-M8 能带来 >1.5×完整decode kernel加速；“M64/M128浪费”方向已被现有指令与静态特化实验否定。** 若未来仅研究更紧的L1/L0资源分配，必须保持每行K128累加顺序、全部输出/状态与现有尾部规则，并先证明它改变了实际指令或等待；无需为同一虚假前提重跑大实验。

## 9. 补审：packed DMA 是否值得改成单burst+Gather

**本轮筛选结果：暂时淘汰这个实现方向。** 已存相位数据没有显示packed DMA是足够大的热点；当前bundle的残余相位占比又没有被记录。不能把64×136 B写法直接翻译成“大量低效小拷贝”或完整kernel两倍收益。

### 9.1 哪份相位数据有效，属于哪个版本

- `reports/decode_phase_profile_camodel_20261007/.../phase_profile_clock_audit.json` 的 `timing_status=unavailable_zero_SYS_CNT`、`timing_usable=false`；其计时值不能使用。
- 可用marker记录是 `reports/decode_marker_v2_camodel_20261007/q4_three_tiles_1025-baseline/marker_phase_analysis.json`，对应kernel **`oscar_decode_simd_profile_q4_kernel`**，不是bundle；q4、C1025、N8、R1、S1、2个Cube，完整算子1,451,744 tick，带诊断marker。其clone由 `csrc/experiments/decode_phase_profile/manifest.json` 绑定原SIMD源码SHA `c67219ce...`，数学helper hash和可逆生成机制保留。
- 当前bundle已存的R1/C20000/N128/S3与R4/C20000–30000/N128/S3报告分别在 `reports/decode_bundle_long_camodel_20261007/.../report.json` 与 `reports/decode_bundle_batch4_camodel_20261007/.../report.json`：`profile_counters=false`、`instruction_samples=[]`。它们有完整算子结果及总tick，没有packed/unpack/Cube/softmax残余份额。不能从两个总tick作相位分摊。
- 本轮重新对照 `reports/decode_bundle_packaging_20261007.json`：当前实验bundle与生产bundle、padded unpack、range-mask这三对文件的SHA都仍匹配归档；所以可以可靠地列出源码已经改了什么，但不能因此补造相位测量。

### 9.2 旧SIMD的数量，仅用于筛选旧瓶颈

下表采用同一actor的exclusive cycles；百分比的分母写明。**不相加AIC/AIV，不把等待重复算成producer成本，不移用为当前bundle占比。** AIV0 history span330,890；AIV1 history span324,971；它们整个actor分别1,442,239 /1,436,032 tick。

| 原SIMD history相位 | AIV0 cycles /history占比 | AIV1 cycles /history占比 |
|---|---:|---:|
| packed DMA，6次 | 10,247 /3.097% | 9,472 /2.915% |
| unpack，12次K/V | 54,243 /16.393% | 53,602 /16.494% |
| FP32 KV publish | 714 /0.216% | 714 /0.220% |
| softmax+P | 98,131 /29.657% | 52,730 /16.226% |
| 等待QK | 54,540 /16.483% | 62,307 /19.173% |
| 等待PV | 64,035 /19.352% | 109,433 /33.675% |

packed DMA只占原actor总span的0.710% /0.660%。AIV0六次调用为630、2705、767、2525、1086、2534 tick；其范围包含buffer清零、Physical页表查询、DMA提交与MTE2完成等待，并非纯DMA引擎占用。没有逐burst服务时间、MTE排队、实际HBM/L2计数来证明瓶颈来自136 B行长。即使将旧AIV0该phase完全删除，单纯串行账的history理想值也仅 `1/(1−0.03097)=1.032×`，整个actor约1.007×；这只是该旧样本的条件上界，**不是bundle或64K预测**。两个AIV是并行生产者，不能把两者DMA cycles相加再声称翻倍的可省时间。

其AIC history span317,698：QK25,773（8.112%）、PV33,751（10.624%）、rotation7,277（2.291%）；其余主要是等KV84,279、等P122,150、等AIV累加31,157及控制开销。QK/PV标记覆盖SDK提交、内部搬运和已有同步，并非裸MMAD执行份额。KV publish只有714 cycle也不代表512 KiB/单元的后端搬运免费：异步完成和排队可能由下游wait承受，不能把各类等待全部归因给GM写或“小行DMA”。

### 9.3 大相位中哪些已经改掉

| 原SIMD现象 | 当前bundle源码 | 不能重复领取的收益 |
|---|---|---|
| window `precise_load`：AIV0 607,197、AIV1 413,230 tick | `attention_cv_decode_bundle.cpp:307–370` 的批量tag读取、连续有效run K/V DMA | 已消除旧逐row窗口搬运模式；不是当前window成本 |
| AIV0 history三次softmax为7,178 /5,537 /85,416；最后tail明显不同 | `:471–510` 的 `MaskComplement` 取代旧SIMD `:409–415` 逐column scalar SetValue | 旧tail的大标量mask代价已改；不能将98,131整体视为剩余SoftmaxFlashV2成本 |
| history K/V解包与Cube串行、AIC每tile等AIV累加 | `:620–744` 预取K/V与QK/PV重叠；没有旧 `aic_acc_wait` 握手 | 旧KV/P/acc wait份额不能沿用；完成依赖仍必须保留 |
| natural int16行stride512 B | padded helper `attention_striped_unpack_padded.h:75,90–105` 使用544 B行stride | 旧unpack 54k不能当新的16%残余；它仍做完整INT2恢复，但bank/向量代价已变 |
| window/current的不足256列仍计算完整tile | `attention_cv_decode_bundle.cpp:196–220` 的precise TailColumns | 已实现；history保持256是流水/数学约束 |
| M64字面配置 | 本文§8的真实M24/M6 MMAD | 无新增“裁M64”算量红利 |

因此当前仍有资格称“存在”的工作是INT2恢复、FP32 K/V交接、QK/PV、真实softmax与必要依赖；但**没有当前相位证据可以把其中任何一个排序为足以令完整kernel显著提速的剩余大头**。本轮不重新实现这些已经交付的优化。

### 9.4 对8704 B burst候选的具体判定

当前 `LoadPackedInto`（`:279–299`）每个64行subtile按虚拟128页边界发一到两次 `DataCopyPad`。一次调用的 `blockCount=rows`、`blockLen=136`，属于硬件二维搬运；**不是host或device循环提交64条独立DMA指令**。目标Hkv1时同页GM有效字节连续，但UB必须变成64×160 B，以符合现有32 B对齐的向量解包stride。Hkv>1、跨页、tail或无效物理映射时不能无条件读一个连续8704 B区间。

“同页Hkv1一次8704 B→紧凑UB，再Gather到160 B stride”的方案不会减少有效GM字节；它把padding变换转给Vector，增加UB读写、索引/临时空间与同步。当前两个packed UB缓冲还要一直保留到后续V解包，不能在K处理完后破坏内容。它可能减少MTE二维搬运的行开销，但已存数据没有证明该行开销够大，也没有证明新Gather比MTE padding便宜。

**停止条件已满足：** 原样本packed phase不足history的3.1%，且不是当前bundle profile；不据此启动新实现或长模拟。如果未来已有短设备诊断能证明当前bundle的packed-DMA关键路径占比超过所需收益门槛，再重新考虑。仅改此phase，想得到1.5×完整kernel，即使该phase免费也至少要占原关键路径1/3；想得到2×至少1/2。若候选只能把该phase加速2×，达到1.5×完整kernel更要求它原先占关键路径2/3。当前证据远未达到这些资格条件。

## 10. 当前bundle分相位缺口：独立原型与首轮结果

新增 `csrc/experiments/request_window_decode_profile/`，生成当前已交付bundle的normal/profile q4/q1四入口；原生产文件与data path不变。目录名不代表新dual-region/request-window ABI：被测版本明确为旧per-page striped bundle，不能将source1占比套到新窗口实现。

`generate.py`固定源/依赖/helper SHA，逆向每项计数编辑后恢复全部生产源码字节，保留原fences、flags、INT2/FP32数学与错误顺序。复用已有schema2、SYS_CNT和显式MOV机制；相位嵌套与并集、async含义见该目录README。`normal.cpp`无诊断计数，作为同形插桩扰动分母。

计划固定R32/N128/S3/Hq6/Hkv1/D256/20Cube，全1152任务槽：先C1025的q4/q1，之后才决定第二context。C1025与C4097的history KV256工作分别96与480单位/完整请求批次（q4/q1均如此），window/current保留。现有高核数trace cap不变；60actor完整trace的资源预算需独立成立，截断必须失败，不能降成R1补百分比。

原型阶段完成源码生成/`--check`、Python语法检查与3项stdlib证据门测试（拒绝R1伪装、错误symbol/library/status、过大正负插桩扰动）。后经root授权排他执行，CANN编译/链接rc0；真实C1025/R32/N128/S3/20core的normal/profile四臂均完整数值通过：q4为1,229,822/1,242,723 ticks，扰动+1.0490%；q1为1,033,238/1,065,947 ticks，扰动+3.1657%。两对partial/LSE/status逐位相同，均通过预设5%研究扰动门；每臂均在原600秒预算内完成。

两profile全部actor工作计数吻合，但SYS_CNT为0，未采marker，因此相位占比仍未知。C1025在S3下每history split仅一块KV256，没有触发hasNext稳定流水；不能用本轮解释64K。纯计数表明C2049每split三块足够首/中/末迭代，history全批288单位，比C4097的480更小；资源与时间预算仍须分别审计，不能把单位比当墙钟比。当前plan未改、第二context未运行。详细记录：`reports/request_window_decode_profile_20261009/summary.md`。NPU、graph、模型质量、整轮未验收，本原型也未增加新优化。
