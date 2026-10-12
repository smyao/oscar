# 当前量化与反量化路径优化

档案：G26–G34、#13–22、#70–73、#126、#145、#148、#151、#154–156；启动文档完整 D.4。本轮按用户要求优先处理 OSCAR 自身的量化、反量化及其必要搬运。所有服务继续省略 `--max-num-batched-tokens`，原性能、FULL KV 与全部 KV 报告目标不变。

## 当前源码确认

| 路径 | 当前真实实现 | 本轮重点 |
| --- | --- | --- |
| candidate 新 KV 写入 | `integration/cache_storage_ops.py` 选择 `rotate_clip_store_striped_out` | Hadamard 准备、旋转、FP16 metadata 舍入、FP32 量化、INT2 打包 |
| request-window 研究写入 | 独立 packed/window 两个算子；packed 复制同一量化流程 | 合格的新量化实现必须覆盖此格式，不只改 probe |
| q4/q1 decode | `attention_cv_decode_bundle.cpp`，64 行 padded unpack 与有界预取 | 已有优化不能再次计入收益；检查剩余 metadata/位操作/等待 |
| 小 chunk C4 | `attention_cv_window_range_cluster4.cpp`，16 行 striped unpack | 扩大反量化批次、去掉已不用的 UB 缓冲；保持 source1/current 与 softmax 不变 |

`active_identity_store` 仍为研究代码，未进入 CMake/capabilities/实际 selector；不能把它的局部收益写成已部署优化。当前配置 clip 缺省为 0，不能把没有执行的 percentile Sort 当作线上瓶颈。

## 已确认的浪费与独立原型

### 1. Hadamard 索引初始化

当前 `RotationEngine::Init` 在每个核、每次启动时按标量写 `D×log2(D)` 个索引与同样数量的符号。D256 为 **4096 次 SetValue**，与本次有效 token 数无关。

研究实现 `csrc/experiments/hadamard_indices/vector_indices.h` 使用整数等差序列与按位运算生成同一索引；符号仍精确为 FP32 的 +1/-1。只在输入 Load 之前借用已有 input/work/red 缓冲，不增加工作区、参数或 launch，不改变旋转的 Gather/Mul/Add 顺序。

- 官方 CPU-debug：D64/D128/D256 的旧新字节、独立公式、padding 与 guards 全部通过。
- CANN 9.1 / ascend910b4 编译通过。
- CAModel 单 AIV、D256、相同库与配置：原路径 **33393 ticks**，向量路径 **4695 ticks**，这个元数据原型约 **7.11×**。两臂均完成独立公式/guard/profile 检查，无超时。
- 两臂包含相同的输出拷贝。**这不是完整 store、目标 NPU、模型或整轮 7.11×**；完整写路径组合比较尚未完成。
- 原始结果：`reports/quant_hadamard_indices_20261009/reports/`；VM 原始指令 dump 仍保留，宿主回收数值、guard、profile 与监管报告。

### 2. INT2 写入位打包

当前 `ClipQuantize::Process` 每个 D256 K/V 行通过 256 次 `codes.GetValue` 和 64 次字节 `packed.SetValue` 拼装 32 个 word。独立 SIMD pack 原型正在实现；目标是保持原量化码、LSB 位序和 metadata 字节，只替换位打包。

D64 的 V 段只有16B且地址不满足32B向量对齐，必须保留其邻域；不能用宽写覆盖 metadata 或另一侧。完整 writer 的输入、raw window、tags、status、非法输入与 guards 需要同时对照。

2026-10-10 本地结果：两个完整writer CPU病例（N128/D256含padding、N16/D64含非零clip）三臂均通过全部字节/独立oracle/guards。D256紧凑域检查覆盖每侧65536种word，加每个位置0/ffff/16个单bit，共66112个word/侧，通过；原每位置全域枚举120秒超时、临时日志占满的失败已完整压缩保存，未改写为通过。

同库、同输入、同模拟配置、grid16的完整N128 writer CAModel三臂：

| 实现 | 完整task周期 | 相对当前writer |
| --- | ---: | ---: |
| 当前writer | 173419 | 1.000× |
| SIMD位打包 | 136213 | 1.273× |
| SIMD位打包 + 精确Hadamard初始化 | 96566 | **1.796×** |

三臂保存的全部输出及输入回读逐位一致，独立oracle、前后guards、完整16个AIV和task周期、180秒监管/清理均通过。完整writer包含旋转、量化和窗口发布，未用单独pack或Init时间替代。结果在`reports/quant_store_simd_pack_20261010/comparison.json`。这是完整writer局部CAModel收益；目标NPU、图和整轮性能仍未验收。

已将同一实现接入本地 candidate 的 striped writer 与研究 request-packed writer，共享 `striped_store_pack.h` / `hadamard_indices.h`。保持原算子名/ABI/selector，因此两个固定candidate入口安装后使用同一实现，无新增参数。源还原检查确认正式striped与已测C臂只差符号、include与说明，request-packed完整旋转/量化前缀逐字一致。旧基线已冻结，原canonical及主rotate_out未改。相关17+7项主机检查通过；实际生产源CANN与request-packed CPU接缝仍在验证，未推送。

### 3. C4 扩大反量化批次

当前 C4 仍为16行；已有模拟的 core3/AIV0 历史解包96次共212598 cycles。decode 的64行路径在另一个短形中20次共46527 cycles。两者不是同形 A/B，**只能支持检验固定开销摊薄的方向，不能直接报告加速倍数**。

独立32行方案显式 UB 为157760B（154.0625KiB），原C4为157632B，只多128B；仍保留独立64KiB accumulator和32KiB score。仅扩大 source0 的 LoadPacked/Unpack/Publish；source1/current/PV消费仍按原16行，Softmax scratch view 固定为原16KiB，避免本次反量化实验改变 softmax 分块。

2026-10-10：193个数值/尾行/非法metadata病例通过官方CPU逐位旧新对照及独立oracle；CANN编译通过。同库、同配置、同8个32行块的CAModel：原16行helper **98478 ticks**，32行helper **41974 ticks**，局部 **2.346×**，输出/status逐位一致，guards与完整task/AIV记录通过。结果`reports/quant_unpack32_20261010/comparison.json`。首次调用方多传CLI参数在解析时退出、未运行kernel，失败记录保留。

完整C4候选保留原每lane的两段16行归属，用两段DMA恢复自然KV顺序，防止扩大批次后把非法metadata错误移到另一个AIV。不能宣称DMA段数减半。完整C4数值及净周期继续验证，尚未写入生产C4。64行直接扩到C4会超出当前有效UB容量，不能照搬decode参数。

完整C4后续结果：Q336/C113、保留1008任务和20Cube/40AIV，旧新partial/LSE/status/counters逐位、独立oracle及全部actor完成通过，周期 **1355015→1346984，只有1.005962×**。三个CPU配对还覆盖valid、bad80、bad96，两侧错误归属均相同。该短形不能代表64K收益，H32不会凭2.346×微核结果单独进入生产。完整证据`reports/quant_unpack32_20261010/full_c4_camodel_comparison.json`。

## 后续大结构资格（未进生产）

- 整体ROI见`docs/roi_decode_overlap_20261010.md`：writer/helper是不同耗时份额，不能相乘；当前两项不足以推出整体native2–3倍。
- 已否决新的M32/N128/K64矩阵排布：实际指令确认8个small-M Cube BAR消失，但同输入单AIC QK的完整诊断周期 **20724→27856（慢34.41%）**；live输出逐位相同。compact M32/N64/K128为20546，收益也不足。记录`reports/decode_mmad_barrier_20261010/decision.json`，不追加无收益的完整decoder改造；scalar marker结束不等于最后MMAD完成，裁决使用完整task周期。
- C4跨成员流水已形成独立原型，保留原FP32/K128和task归属，使用双KV/score槽与FP32 atomic state；资格仍以独立即时回读、atomic→普通模式恢复及完整算子检查为前提。没有将其条件预算作为性能通过项。

## 交付条件

合格组件需要组合后测完整算子，核对实际 selector，再接入两个固定 candidate 入口。局部微核倍数不能当整轮目标达到；未合格研究代码不进入默认生产。FP16 metadata 先舍入、FP32 Mul→Add、量化公式、INT2 位序、原生GDN与冻结精度门保持不变。

## C4 跨成员流水的完整双轮结果（2026-10-10）

同输入 Q336/R1/C337/S1/20core，两臂均保留全部1008任务及生产source2抑制，所有partial/LSE/status/cluster counters逐位相同，独立oracle及60actor完成门通过。完整CAModel周期1849738→1610282，**1.148704倍**；两臂均在原600s期限内正常结束。该形状有一个full256和17行tail，不是65K稳态或整轮性能。预取错误bad336的CPU边界门仍在进行，生产未启用；不凭该局部倍数推送达标版本。证据：`reports/c4_member_pipeline_20261010/reports/c4-pipeline-camodel337/comparison.json`。

## 紧凑Batch8与Q旋转完整对照（2026-10-10）

完整writer N128/Hkv1/D256/grid16：当前pack+Init 95930ticks，Batch8 92403，只有1.03817倍；D256与D64/clip CPU两组全输出字节相同，不合入生产。Q旋转N128/H6/D256/grid32：旧110343，单独Init68082，Batch8 64537；相对旧分别1.62074/1.70976倍，Batch8在Init之后只增加1.05493倍。三臂CAModel完整32actor、输入/guards、output/status全位一致及独立oracle通过。7个小CPU边界、21臂均逐位通过；主CPU形状因调试日志耗尽tmpfs失败，原错误/3.8GB原始记录已完整压缩归档，不能计作CPU主形状通过。源/设备模型/180s期限相同。以上仍是量化和旋转局部结果，未建立相对native整轮2–3倍依据；生产、NPU、graph未升级/验收。完整证据`reports/hadamard_batch8_20261010/summary.json`。
