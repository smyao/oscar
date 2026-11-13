# C4/C16 的执行含义与当前数据流 roofline 账

> 2026-11-13增量：本文保留当时基底的字节账。当前派生源已新增C4单B1复用和C16四槽score环，见[实现、成本与验证边界](cv_reuse_20261113.md)；下方“C4无显式B1”等描述属于该历史基底，不能再当作最新源码。

本报告以 `de8d55f` 的生产生成源为核算基底。机器可读公式、源码 SHA 和数字在 [work_model.json](../reports/cv_roofline_20261112/work_model.json)。已阅读启动文档 D.4 完整原始打点、档案 #145/#150/#151/#154/#164/#172，以及 [2026-10-11 roofline 文档](quant_dequant_roofline_20261011.md)。旧文档的 136B 元数据布局和 FP32 K/V 发布账属于历史版本；当前使用 144B 物理槽、BF16 Cube 操作数及 FP32 累加，不能沿用旧数值。

这是源码工作量模型，不是实测硬件 roofline。全文的字节账、源码减少的指令/传输与设备运行时间分别陈述，不据此宣称编译、NPU 数值或速度已经验收。

## C4/C16 共享什么

`C` 表示一个历史复用簇内的 query tile 数量。当前 `D=256`、GQA 比例 `G=Hq/Hkv=6`，每个 tile 容纳 `floor(128/6)=21` 个 query token，即 `21×6=126` 个有效 query-head 行，预留 128 行存储。

| 路径 | 一个成员的 query token | 成员数量 | 一个完整簇的 query token |
| --- | ---: | ---: | ---: |
| q1 | 1 | 1 | 1 |
| q4，按 qcount=4 核算 | 4 | 1 | 4 |
| C4 | 21 | 4 | 84 |
| C16 | 21 | 16 | 336 |

C4/C16 不是 batch size、INT 位宽或 NPU 数量。同一簇的所有成员必须属于同一个请求、KV head、split、context、requestBegin 和历史 `[kvbegin,kvend)`；每个成员都必须是完整的 21-token tile，`kind==0`、无 task error、`kvbegin<kvend` 且 `kvend==context`。不同请求不能因为 batch 同为 8 就共享这一份 KV。

[主机选择](../oscar_ascend/ops/cv_dispatch.py)首先区分后续 MTP q1、`max_query_len≤4` 的 decode/verify，再按现有规则选择 C4/C16 入口。C16 的主机条件为 `max_query_len≥8192`；这是本轮未改变的 query 长度路由条件，不等于设备最低成簇条件 336。设备逐成员验证域一致性，未成簇任务继续执行该入口保留的普通路径。C4/C16 入口还处理 current/window 域，因此 profile 中出现该名称不表示每次调用的全部工作都成功组成历史簇。

两族均保持每次 Matmul 的有效 `M=126`，分别执行 4/16 次 QK 和 PV；没有合并成一次 `M=504/2016` 的矩阵乘法。它们共享展开历史的成本，QK/PV 对每个 query 的工作量仍然存在。

当前 C16 还有 C4 没有的显式 K/V L1 驻留：

- [C4 CubeCluster](../csrc/kernels/attention_cv_official_cluster4_active.cpp)对每个成员调用 GM-B Matmul，QK/PV按两槽交错执行。
- [C16 CubeCluster](../csrc/kernels/attention_cv_official_cluster16_active.cpp)把 K 的 128KiB BF16 tile 加载到外部 B1，一次供 16 个 QK 使用；所有 P/alpha 就绪后，将 B1 换为 V 的 128KiB tile，供 16 个 PV 使用。

因此 C16 的一个完整簇只显式读取一次 K 和一次 V 到 L1；它同时需要更多 score 槽，并保留所有成员的 FP32 online state。C4 的 SDK 可能利用缓存，但源码没有跨成员外部 B1 的同等驻留保证。

## 核算范围与公式

以下按**单 rank、单 KV head、一个完整历史 KV256 tile**核算。`F` 是 QK/PV 有效形状的乘加 FLOPs，乘加算 2 FLOPs；它不包含硬件补齐产生的额外运算，也不把 Cast/整数解包/Exp 当成 Cube FLOPs。完整 tile 即使含 masked 列也仍按当前 QK/PV 的完整形状计算；不足 KV256 的尾块和 source1/source2 需另算。

`B` 是显式 GM 地址传输与 Matmul 最小操作数读取、原子读改写的**语义工作量**。显式 DMA 长度可直接从源码得到；SDK 内部实际读取次数、cache line、L2 命中、原子事务和写回均未由该模型测量，不能把 `B` 命名为 HBM 流量。

```text
D=N=256, G=6, M=G*q
R = Vector 实际整块读取的 score 行数：q1=16, q4=32, C4/C16=128
U = query/state 槽行数：q1/q4=32, C4/C16=128
C = cluster 成员数

F_member = 4*M*N*D                         # QK + PV

B_decode = 144*N + 8*N*D + 10*M*D + 8*M*N + 4*R*N

B_cluster = 144*N + 4*N*D + B_KV_Cube
          + C*(10*M*D + 8*M*N + 4*R*N + 8*U*D + 16*U)

B_KV_Cube(C4)  = 4*C*N*D                 # 每次 GM-B Matmul 的最小消费
B_KV_Cube(C16) = 4*N*D                   # 明确一次 K/一次 V 的 B1 加载
```

物理 packed 槽为 `D/2+16=144B`：K/V INT2 各 64B，FP32 scale/normalized-zero 共 16B。UB 中行 stride 对齐为 160B，其中 16B padding 不能计成从 GM 读取的有效槽字节。

## 完整历史 tile 主数据流

| 项目，字节 | q1 | q4 | C4 整簇 | C16 整簇 |
| --- | ---: | ---: | ---: | ---: |
| packed KV 读取 | 36,864 | 36,864 | 36,864 | 36,864 |
| BF16 K/V 发布到 GM | 262,144 | 262,144 | 262,144 | 262,144 |
| K/V Cube 最小语义读取 | 262,144 | 262,144 | 1,048,576 | 262,144 |
| Q Cube 最小语义读取 | 3,072 | 12,288 | 258,048 | 1,032,192 |
| FP32 score 写 | 6,144 | 24,576 | 516,096 | 2,064,384 |
| Vector 整块 score 读 | 16,384 | 32,768 | 524,288 | 2,097,152 |
| BF16 P 发布＋最小语义读取 | 6,144 | 24,576 | 516,096 | 2,064,384 |
| FP32 PV 写＋Vector 读 | 12,288 | 49,152 | — | — |
| FP32 accState 原子读改写 | — | — | 1,032,192 | 4,128,768 |
| FP32 state Spill＋Restore | — | — | 1,056,768 | 4,227,072 |
| **主数据流合计** | **605,184** | **704,512** | **5,251,072** | **16,175,104** |
| **QK/PV FLOPs** | **1,572,864** | **6,291,456** | **132,120,576** | **528,482,304** |
| 每有效 query token 字节 | 605,184 | 176,128 | 62,512.762 | 48,140.190 |
| **语义算术强度 FLOP/B** | **2.599** | **8.930** | **25.161** | **32.673** |

score 的写入只需有效 M 行，Vector 按固定块读取，因此 q1/q4/C4 包含不同数量的无效行读取。随后既有 padding 初始化保护软最大计算；不能为了账面流量直接删掉这些行的处理。

cluster 的 PV 直接原子加到成员 GM accState，不再走普通 decode 的独立 PV 写回后由 Vector 加入 UB accumulator。必须同时计入该原子更新以及每轮 Spill/Restore，不能只删去普通 PV 行而漏掉替代成本。

每个成员每次 state transfer 为：

```text
FP32 accumulator = 128*256*4 = 131,072 B
FP32 max + sum   = 2*128*4   =   1,024 B
合计                              132,096 B
每轮 Spill + Restore              264,192 B/member
```

state transfer 与原子 acc 更新占上表 C4/C16 合计的 39.78%/51.66%。这是本模型内的字节占比，不能视为设备时间或 HBM 带宽占比。仅统计 packed＋K/V 发布/读取会把 C16 算术强度写成 941.78 FLOP/B；计入其他主数据流后为 32.67，说明只看压缩位宽会严重漏账。

## 一次性成本、其他域与 Vector/同步屋顶

`de8d55f` 每 history task/cluster 另有以下一次性成本；有 L 个完整历史 tile 时可按 `L*B_tile+B_once` 汇总。下表不能直接用于其他源域或不完整簇。

```text
每成员 Q 源读取＋完整 Q 槽发布 = 2*M*D + 2*U*D
cluster 初始 state 写＋最终读 = 2*(4*U*D + 8*U)
每成员最终 partial＋LSE       = 4*M*(D+1)
每成员两 AIV status           = 8 B
```

块表每次 `Physical` 有 4B 显式读，调用数受 run 和虚拟128页边界影响；task 表扫描还依赖 padded tokens、任务数和成员验证，均需按真实调度另加。以上未计 writer、Q 旋转、merge/最终逆旋转、GDN、模型权重、HCCL、MemCache 搬运及调度。query token 也不等于 MTP 最终 accepted output token。

当前 mode0 解包每完整 KV tile 重建 `2ND=131,072` 个 K/V 元素，单是 FP32 subtract/multiply 就有 `4ND=262,144` 次；另有 ShiftRight/And、int16→FP32 Cast、metadata Gather/检查/Brcb 和 BF16 发布转换。它们由共享簇摊销，但应使用 Vector/SFU 对应吞吐评估，不能除以 BF16 Cube 峰值。mode1 还存在额外舍入；使用哪个 mode 必须按实际配置确认。

softmax、score scale、finite certificate、alpha accumulator 更新和 P Cast 随有效 query 行继续增长。C4/C16 减少重复展开，不会把这些每成员工作消除。两族每 KV tile 的 Cube 侧均有 `C+1` 次跨核 wait 和 `C+1` 次 set 调用，每个 AIV 也维持对应协议；普通 history decode 每 tile 是 2 次 wait/set，另有任务结束交接。这是源级调用计数，不能把多个 actor 的等待时间相加当 wall。

C16 的 K 驻留阶段必须在所有 P/alpha 发布后才能切到 V，C4采用交错 QK/PV。这种依赖与定形 M/N/K、标量发射、DMA 及资源竞争共同构成 roofline 之外的延迟约束。正确的多资源下界应分别保留：

```text
T >= max(D_HBM/BW_HBM, D_L2/BW_L2,
         F_BF16_Cube/P_Cube, Vector/SFU 服务下界,
         DMA、标量及依赖链下界)
```

HBM/L2 的实际流量和相应可达到带宽、Cube/Vector 的实测利用率尚未给出。模型允许解释可能的抵销机制，不能唯一宣布当前路径属于 memory-bound 或 compute-bound。D.4 的整历史反量化 6499.8–6655.1ms 是应避免的历史结构，不能作为本版本相位耗时。

## 用户 BS8 profile 能支持什么

原始数据及计数差异见 [BS8 对照报告](bs8_profile_20261112.md)。用户确认两侧都是同一批11条、32K输入、50输出、max BS8、MTP3且端到端完成。算子表中 OSCAR/base 累计时长分别为43.84264448/29.18894154秒；C4＋C16累计13.78287564秒，是优先审查这两族的依据。

部署提交、实际 argv、profile 起止和 timeline 未给出；两侧 slot mapping、sample/MTP 等调用计数存在未解释差异。不能把算子累计比1.502当成端到端慢1.502倍，也不能用配置不同解释已经确认同为MTP3的两侧。该表不含本报告后续改动的实测效果，不能证明这些改动已经产生收益。

## 本轮两处实现改动的工作量

以下源码变换已接入 candidate 实际编译的 C4/C16 同 ABI 派生核；测试、CANN、NPU及速度状态由[验证记录](../reports/cv_roofline_20261112/validation.json)单独报告，工作量不授予性能验收结论。

**current K/V 去掉被完整覆写的零填充。** 仅对 C4/C16 的合法、非空 `kind==2` 载入跳过初始 FP32 `dst` 清零，因为后续原样 Cast 会写完整 `16×256` 元素；只有满16行 DMA 时才同时跳过 BF16 staging 清零。partial仍完整清零BF16 staging以保护Cast尾部；empty/error继续原FP32清零；window/tag处理保持。

完整 source2 KV256 的K/V、两AIV、8个subtile合计32次载入，每次可去掉16KiB FP32零写和8KiB BF16零写：

```text
FP32 UB零写减少 = 32*16KiB = 512KiB
BF16 UB零写减少 = 32* 8KiB = 256KiB
总计                          768KiB（786,432 B）
```

这是 UB 逻辑零写，GM字节减少为0；新增少量整数资格判断。不能把768KiB换算成已测HBM节省或服务速度。

**cluster 首次使用时初始化 state。** 初始 query 发布不消费 online state；将相同 acc/max/sum 初始化移至每成员首次 Softmax 前，可删掉初始 Spill 和首轮 Restore。首次真实 Spill仍在PReady之前，Cube原子PV由该信号保护；后续每轮state和最终Restore保持。每完整簇净少一个 state roundtrip：

```text
264,192 B/member
C4 : 1,056,768 B/cluster
C16: 4,227,072 B/cluster
```

这是整段历史循环的一次性减少，不能再乘历史tile数量。上面的 `de8d55f` 成本分解把初始Spill＋最终Restore放在一次性项中；新实现保留最终Restore、删除初始Spill＋首轮Restore，总state传输从 `(2L+2)` 次降为 `2L` 次。

余下state传输的max/sum相邻UB块映射到GM两个跨lane块，通过一次 `DataCopyParams{2,8,0,8}` 写出、`{2,8,8,0}` 读入（32B单位）替代两次独立stats DMA。每次少一条提交，地址、payload和有效传输的所有权屏障不变，因此合并本身节省字节为0。它也不删除每轮必须保存的FP32状态，不改变softmax累计顺序或精度。

合计两AIV，`C`个成员、`L`个历史tile，state的DMA API调用从 `12C(L+1)` 降至 `8CL`，有效payload burst从 `12C(L+1)` 降至 `12CL`；两块合并不会把两段有效payload变成一段。状态首次往返的字节节省比例为 `1/(L+1)`，长历史下摊薄。初始化算术次数不变，删去的初始往返同时少 `8C` 个硬事件调用，新增至多 `2CL` 次 `start==first` 标量判定。current的额外资格计算、保留的屏障也计入净收益边界。

## 验证与交付

261项主机测试通过：196项生成链/有限性/零填/原数值保护/功能probe计划，44项build/variant/runtime接缝，21项state专项。零填控制运行真实提取的LoadPrecise、PublishKv及finite helper，8568组固定地址换元数据，检查FP32结果、BF16发布、输入/canary、错误、DMA/Cast/fence序列；state控制运行真实提取的资格及生命周期方法，114组C4/C16、两lane、1/2/3/129历史tile，检查未初始化GM读取、状态与错误bits、跨核控制和DMA地址账。底层原语是主机桩，不是官方CPU-debug或CANN异步执行。

所有原始母源保留；派生源可完整逆变换，生成工具/源/hash接入构建缓存和probe绑定。新增4个单历史tile、mode0/1、固定地址换query/context的NPU功能图用例，已经核实fixture分配不变，尚未在设备执行。正常candidate及node93入口编译同一新源，无额外开关；q4/q1、PT/clip/旋转、PC/MemCache、GDN/MTP、启动参数、timer保持。

本机没有CANN SDK，按本轮限制未连接真机、未测速；CANN编译、NPU正确性、真实图、模型质量和吞吐均未获得本轮通过证据。每轮FP32 state搬运、固定小Matmul形状和精确域独立工作仍存在，不能把这次消除冗余描述为已消除全部13.78秒成本。
