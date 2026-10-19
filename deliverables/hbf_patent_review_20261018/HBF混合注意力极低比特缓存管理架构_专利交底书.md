# 一种基于 HBF 的混合精度 KV 缓存分层布局及控制装置

**技术交底书 · 2026 年 10 月 19 日修订**

本发明面向同时包含全注意力和循环状态层的模型，研究 **HBM 中的混合精度 KV 如何组织，以及迁入二级内存时如何转换槽位布局、保存地址对应关系，并在恢复时重新定位**。本稿以当前 OSCAR BF16 分支的真实 HBM 布局为实现基础，二级内存布局及其控制装置为拟议扩展。

周期检查点、前缀续算和历史原精度窗口复用是本方案可以采用的技术手段，不单独作为本稿拟保护的发明点。DeepSeek-V4 报告 §3.5.2 已公开周期保存 SWA 末窗、命中后恢复检查点并补算尾部的策略。本稿据此将保护重点转向具体的跨层级数据布局及其映射控制。[DeepSeek-V4 技术报告](https://arxiv.org/html/2606.19348v1#S3.SS5.SSS2)

## 一、现状：原生 BF16 KV 与二级缓存

### 1.1 现有分层缓存的基本做法

全注意力层通常将历史 K、V 保存在 HBM，并在容量紧张或请求暂停时将选定缓存转存到二级内存。以 BF16、head dimension＝256 为例，一个 token、一个 KV head 的 K 和 V 各占 512 字节，合计 1024 字节。长前缀和多会话会同时扩大 HBM 保存量及跨层级搬运量。

将数据转到容量更大的介质可以延长保留时间，但如果历史始终按 BF16 保存和搬运，单位 token 的载荷并没有因此缩小。另一方面，混合模型还包含卷积状态和循环状态，这些数据与全注意力 K/V 的形状、精度、有效范围及生命周期不同，不能仅用一个固定 token 大小描述所有对象。

本文的“原生 BF16”指 FULL 层的 K/V 表示；GDN 的 Conv 和循环状态分别保持模型规定的原生精度。

本文将“原生 BF16 HBM／二级缓存”作为现状对照。其中 HBM 部分采用本项目原生参考实现的真实视图，二级部分采用常规的 BF16 有效数据块转存示意。**当前 OSCAR 工程尚未实现二级缓存，不能将图中的二级端当作已有部署事实。**

### 1.2 真实地址组织带来的要求

本项目的原生混合缓存并不是每个物理页内依次摆放 `Conv｜K｜V`。它在一个共享 raw tensor 中，先排全部 Conv 区，再排全部 SSM/K 区，最后排全部 padding/V 区。一个物理页号对应的数据分布在不同区段。

因此，分层存储必须区分三件事：逻辑 token 范围、物理块身份，以及该块在各区段中的真实地址。若把页号 b 简单解释为一段从 `b×P` 开始的连续数据，就不能正确表达这套布局。

采用 INT2 后，这个问题进一步变成多格式布局问题：压缩 K/V 带有解码参数，原精度窗口有独立 K/V 和位置标签，GDN 状态保留原生精度。需要一种布局及控制方式，使这些数据既能按 HBM 内核要求放置，又能在二级内存中按有效载荷组织，而无需先将压缩历史展开成整段 BF16。

## 二、本次采用的源码与配置依据

本轮核查的本地及远端分支均为 `codex/oscar-bf16-attention`，提交为 `5613f748e432104e3038743744dc95ccef495b64`。

当前 candidate 的实际路径是：

- `experimental_aligned_math=True`、`experimental_bf16_attention=True`。
- 存储格式为 **`striped_fp32_v2`**，D＝256 时每 token、每 KV head 为 **144 字节**。
- 原精度窗口采用**物理页 padding 中的快照视图**，不是独立的 request-window sidecar。
- 当前默认 APC 关闭，但这不会把布局自动切换成 request-window 模式。
- GDN 继续使用原生 Conv 和 SSM 视图，工作区另行分配。

上述选择分别见 [候选开关](../../tools/serving_variants.py#L93)、[运行时格式](../../oscar_ascend/runtime.py#L348)、[布局转换](../../oscar_ascend/integration/specs.py#L115)及[实际窗口绑定](../../oscar_ascend/runtime.py#L829)。

### 2.1 当前配置可推导出的几何

依据工作区保存的官方 Qwen3.8-27B 配置快照、当前 TP＝4、MTP＝3 和原生形状计算代码，可以得到下表。**这是配置与源码推导，不是目标机器的内存分配采样。** 实际运行仍以模型配置、native spec 和 `cache_layout` 输出为准。

| 项目 | 当前配置推导值 |
|---|---:|
| 每卡 FULL KV head 数 H | 1 |
| FULL K/V head dimension D | 256 |
| GDN Conv | 默认 SD 视图 `(6,2560)`，BF16 |
| 每块 Conv 字节数 C | 30,720 B |
| GDN 循环状态 | `(12,128,128)`，FP32 |
| 每块 SSM 字节数 M | 786,432 B |
| 原生统一计费页 P | 1,603,584 B，即 C＋2M |
| 原生 FULL 物理块容量 | 1,536 token |
| OSCAR candidate FULL 物理块容量 | 5,376 token |
| candidate 原精度窗口预留行数 | 64 Sink＋256 Recent＋3 MTP＝323 行 |

因此，旧启动文档中的 `15,360 / 393,216 / 801,792` 不能继续用作当前默认配置的容量。尤其不能把 GDN 循环状态也画成 BF16：当前官方配置快照指定它为 FP32。共享 raw tensor 的数量与物理块数由实际分组和内存预算决定，不在图中固定为 16 个 tensor。

## 三、现状 HBM 布局：原生 BF16

设一个共享 raw tensor 的基址为 R，实际物理块数为 N。C 表示每块 Conv 字节数，M 表示每块 SSM 字节数。在本次原生对齐几何下，单个 K 页与 SSM 页同为 M 字节，P＝C＋2M。

```text
同一个 raw tensor，长度 N×P，以下是整池连续区段

A区：N×C                 B区：N×M                  C区：N×M
[ Conv / FULL留空 ]      [ GDN SSM / FULL K ]      [ GDN留空 / FULL V ]
0                       N×C                      N×(C+M)
```

对于物理块 b，真实起点为：

| 所有者与数据 | 字节地址 |
|---|---|
| GDN Conv | `R + b×C` |
| GDN SSM | `R + N×C + b×M` |
| FULL K | `R + N×C + b×M` |
| FULL V | `R + N×(C+M) + b×M` |

GDN 和 FULL 通过不同的活跃物理块身份使用共享池。**同一个活跃块号不能同时归两者所有。** 表中的 SSM 与 K 地址相同，表示按所有者解释的复用关系，不表示它们在该地址同时存放。

原生 FULL 页内，K 和 V 各自按 token、head、维度连续排列。对页内 token t、KV head h、维度 d：

```text
K(b,t,h,d) = R + N×C + b×M       + ((t×H+h)×D+d)×2
V(b,t,h,d) = R + N×(C+M) + b×M   + ((t×H+h)×D+d)×2
```

D＝256、H＝1 时，每 token 的有效载荷是 K 512B＋V 512B＝1024B；**这两个 512B 位于不同分区，不是 HBM 中相邻的一个 1024B 槽。** 二级缓存可以将选定范围的 K/V 归集成一个对象，但必须先按上述真实地址读取。

## 四、本发明的 HBM 基础：OSCAR 混合精度布局

本方案保留当前 candidate 的真实存储组织，不重新发明其量化算法。设 padding 字节数 F＝P−C−M，一个 raw tensor 的区段为：

```text
A区：N×C                 B区：N×M                     C区：N×F
[ GDN Conv / FULL留空 ]  [ GDN SSM / FULL INT2 ]      [ GDN留空 / FULL精确窗口 ]
0                       N×C                         N×(C+M)
```

原生 FULL 使用 B/C 两区保存全历史 BF16 K/V。OSCAR FULL 则将压缩 K/V 一起放入 B 区，在 C 区保存有限的原精度窗口和标签；GDN 的 A/B 视图保持原生语义。

### 4.1 B 区中的真实 INT2 槽

当前格式下，每个 token、每个 KV head 的 GM 槽是连续的 144B：

| 槽内偏移 | 长度 | 内容 |
|---:|---:|---|
| 0 | 64B | K INT2 codes |
| 64 | 64B | V INT2 codes |
| 128 | 4B | K scale，FP32 |
| 132 | 4B | K normalized zero，FP32 |
| 136 | 4B | V scale，FP32 |
| 140 | 4B | V normalized zero，FP32 |

该排列直接取自 [实际写入内核](../../csrc/kernels/store_aligned_int2.cpp#L134)及[实际解包读取](../../csrc/kernels/attention_aligned_unpack.h#L43)，不能用旧 canonical 格式中的字段偏移代替。`striped` 指 codes 的位排列，不表示将全池所有 token 的 codes 和 metadata 再拆成大数组。

页 b、页内 token t、KV head h 的槽起点为：

```text
INT2(b,t,h) = R + N×C + b×M + (t×H+h)×144
```

当前 APC-off 模式下，一个 FULL 物理块的 token 容量为：

```text
B_int2 = 128 × floor(M / (128×144×H))
```

按本次配置推导，B_int2＝5376，有效压缩载荷为 774,144B，B 区该页末尾剩余 12,288B。这里的 128 是内核及 metadata 的对齐单位，不是实际物理块容量。内核搬入 UB 后可能使用 160B 对齐步长，它也不是 GM 持久缓存槽大小。

### 4.2 C 区中的 BF16 窗口与标签

定义窗口行数 W＝Sink＋Recent＋MTP 预留，本次为 323。对物理页 b，窗口区起点：

```text
E_b = R + N×(C+M) + b×F
```

该页的布局为：

```text
[ BF16 K：W×H×D×2 ][ BF16 V：W×H×D×2 ][ int64 tags：W×8 ][ 剩余padding ]
```

按本次配置，K、V 各 165,376B，tags 为 2,584B，总计 333,336B。它们都是同一 raw tensor 上的视图，不是另外分配的全历史 BF16 缓存。

窗口行索引按 Sink 与环形区解释：全局位置小于 Sink 长度时使用对应 Sink 行，其余原精度行使用 `Sink + inPage % (Recent+MTP)`。tag 保存页内 token 偏移，用于判断该行当前对应哪个位置。它不是请求 epoch，也不是一个固定的全局 token 地址。

每页预留窗口容量不表示每页都写满同一份 Sink。实际 writer 只在全局位置属于 Sink 时写入相应行。有效窗口的 K/V 来自未经过 INT2 量化的原始 BF16 数据。

### 4.3 KV 池与临时计算空间分开

当前步的 Q/K/V、旋转缓冲、softmax/LSE、输出和 CV scratch 由独立工作区提供，不属于上述 raw KV 布局。图中的 BF16 窗口不能扩大解释为当前 chunk 的全部临时数据，更不能据此认定存在全历史 BF16 展开。

## 五、拟议二级布局：把有效区段组织成带类型的数据槽

HBM 按计算内核需要使用整池分区和物理页视图；二级内存按迁移与恢复需要组织对象槽。**布局切换是数据区段及地址组织的转换，不是将 INT2 反量化后冒充原精度。**

### 5.1 现状对照：BF16 数据槽

常规 BF16 分层基线可以按选定逻辑范围归集 K/V：

```text
FULL对象：[描述头][BF16 K载荷][BF16 V载荷]
GDN对象 ：[描述头][原生Conv载荷][原生SSM载荷]
```

这是二级存储的对照示意，不表示所有已有系统都采用完全相同的头部格式。其 FULL 历史仍是 BF16，单位 token 的 K/V 载荷仍为 1024B（D＝256、H＝1）。

### 5.2 本发明：混合精度数据槽

针对当前 OSCAR 物理布局，二级内存定义两种按所有者区分的数据槽：

```text
FULL混合精度槽：
[描述头][INT2有效槽序列][BF16窗口K][BF16窗口V][窗口逻辑位置]

GDN原生状态槽：
[描述头][Conv有效载荷][SSM有效载荷]
```

FULL 的 INT2 有效槽仍保留 `K codes｜V codes｜FP32参数` 的 144B 格式，原精度窗口单独保存。GDN 的 Conv 和 SSM 继续采用原生 dtype，本次配置分别为 BF16 与 FP32。FULL 槽和 GDN 槽属于不同对象，不能把两者当作同一物理块内并存的数据。

描述头只需表达恢复布局必需的信息：对象所属层或组、格式标识、逻辑 token 范围、有效数量、各载荷偏移与长度、窗口位置及版本。具体字段编码属于实现选择，不作为本稿叙述重点。

这套布局支持只迁移当前需要保存的载荷。热的 Sink 或 INT2 页可以继续驻留 HBM；需要下沉的压缩页按 INT2 载荷迁移，选定原精度窗口和 GDN 状态也可独立保存。**支持二级 INT2 数据槽，不等于每次都把整条 INT2 历史再复制一份。**

单独归档原精度窗口，并不意味着其所在物理块可以立即回收；仍被保留载荷使用的块继续驻留。

### 5.3 槽内偏移与恢复地址

二级对象基址为 L，各字段偏移分别记为 `o_Q、o_K、o_V、o_pos`，则：

```text
INT2槽(j,h) = L + o_Q + (j×H+h)×144
窗口K(u)   = L + o_K + u×H×D×2
窗口V(u)   = L + o_V + u×H×D×2
位置(u)    = L + o_pos + u×8
```

字段起点根据前一字段的有效长度和介质对齐要求确定，空段不分配实际载荷，原 HBM 中的无效 padding 不作为有效内容搬运。对原生 BF16 FULL 对象，则直接使用 BF16 K/V 段及其偏移，不套用 144B 解码格式。

源 HBM 的 ring tag 只有页内偏移。控制装置结合逻辑块映射形成窗口位置，不能用物理页号 b 直接代替逻辑 token 序号。恢复时若分配到新的池或物理页，按目标的 N、b、C、M、F 和格式重新计算地址，并重建目标窗口标签。二级槽不保存可直接回写的旧 HBM 指针。

## 六、控制装置的作用

控制装置保持三个明确功能即可：

1. **格式识别**：识别原生 BF16 FULL、OSCAR 混合精度 FULL 或 GDN 状态，读取其有效范围及布局参数。
2. **槽位转换**：按源区段取出有效载荷，组织为对应二级对象槽；反向恢复时保持各载荷精度及解码格式。
3. **地址映射**：建立逻辑对象与二级槽的关系，并根据新分配的 HBM 物理页计算恢复地址。

这些功能共同保证：一种逻辑缓存对象可以在不同层级采用适合该层级的布局，又不因页号变化、K/V 分区变化或精度不同而读错数据。必要的完成确认、版本校验和并发管理属于实现所需条件，图中不展开为复杂控制流程。

## 七、拟保护内容与技术手段的区分

本稿拟保护的重点为：**针对具有整池分区、压缩 K/V 槽和原精度窗口的 HBM 缓存，设置与之对应的带类型二级数据槽，并由控制装置完成有效区段归集、槽内偏移组织及恢复地址重定位，使 BF16 与 INT2 载荷在分层迁移过程中保持明确的格式和逻辑位置关系。**

其中，OSCAR 的量化方法、当前 144B 编码、原生共享页布局本身是已有实现基础，不应重新写成本发明从无到有的创造。拟申请时应围绕上述跨层级布局组合及其控制关系形成限定，并与已有多精度 KV 管理、卸载和地址转换技术进一步比对。

周期保存 SSM 与原精度窗口、命中后续算、选取历史原精度窗口参与后续注意力，均保留为可选使用方式。第一页的 8K→12K 场景用于说明这些数据如何被使用，不再将“精度补偿”或“周期检查点”单独标成受保护的创新。

现有公开资料中，DeepSeek-V4 §3.5.2 已讨论磁盘上压缩 KV 与 SWA 状态的不同存储策略；Intel US20250061316A1 已涉及多精度 KV 页保存与换入。因此，不能仅凭“混合精度＋二级内存”几个概念就认定新颖性。本稿给出的是需要继续审查的具体布局与装置组合。[DeepSeek-V4](https://arxiv.org/html/2606.19348v1#S3.SS5.SSS2)、[Intel 公开申请](https://patents.google.com/patent/US20250061316A1/en)

## 八、收益与证据范围

本方案的直接目标是：在维持 HBM 计算布局的同时，使二级缓存能够保留压缩历史、必要的原精度窗口和原生循环状态，避免为统一存储格式而先将全部历史展开成 BF16，也避免把无效 padding 当作有效载荷搬运。

144B 与 1024B 是 D＝256、单 KV head 下的历史存储载荷对照，不包含原精度窗口、GDN 状态、空闲区域、控制信息或临时工作区，不能写成整机显存或端到端性能倍率。

本项目的 raw 池按预算预分配。卸载并回收某个 block ID，首先得到的是可供其他请求复用的槽位，不会自动使已分配 raw tensor 的 HBM 字节数变小。实际可用容量、写入量、恢复时延和任务质量均需实现后验证。

本轮完成的是当前源码、配置推导和文档图示核对。当前工程仍没有二级缓存布局转换与迁移控制实现，且默认 APC 关闭；不能把本稿当成这些功能已经在服务中验收。未连接真机，未开展性能测试。

## 附：关键源码证据

| 内容 | 当前代码或本地参考位置 |
|---|---|
| candidate 与 request-window 的选择 | [serving_variants.py](../../tools/serving_variants.py#L93)、[runtime.py](../../oscar_ascend/runtime.py#L348) |
| C/M/P 从 native spec 推导 | [specs.py](../../oscar_ascend/integration/specs.py#L115) |
| 整池分区和 INT2 页容量 | [layout.py](../../oscar_ascend/layout.py#L89) |
| BF16 窗口 K/V/tag 的偏移与视图 | [lifecycle.py](../../oscar_ascend/lifecycle.py#L37) |
| 144B 实际字段与写地址 | [store_aligned_int2.cpp](../../csrc/kernels/store_aligned_int2.cpp#L134)、[attention_aligned_unpack.h](../../csrc/kernels/attention_aligned_unpack.h#L43) |
| 物理页绑定和实际 raw 块数 | [runtime.py](../../oscar_ascend/runtime.py#L790) |
| 原生 FULL K/V 与 GDN 切片 | 本地参考快照 `references/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:4569`、`:4681` |
| 原生页大小对齐 | 本地参考快照 `references/vllm-ascend/vllm_ascend/patch/platform/patch_mamba_config.py:64` |
| GDN 形状与 dtype | 本地参考快照 `references/vllm/vllm/model_executor/layers/mamba/mamba_utils.py:213`、`references/vllm/vllm/model_executor/models/config.py:536` |
| 官方模型配置快照 | `reports/official_alignment_20261018/official_qwen38_config.json`，SHA256 `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |

档案 #17–22 约束格式与解码参数一致，#37–49 约束物理身份与生命周期，#127/#128 要求区分状态跨度、真实物理页与 metadata128 视图。本稿的地址公式及图示据此避免将旧容量、请求跨度或虚拟块长当成真实页布局。

[配套两页可编辑 PPT](presentation/本发明技术方案_HBF混合注意力缓存管理_最终评审稿.pptx)
