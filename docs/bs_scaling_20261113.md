# BS 增长但吞吐不增长：当前源码与本轮改动

基底 `de0c85b`，分支 `oscar-new-7`。本轮沿用已有用户日志、BS8 算子表及源码，未连接真机、未运行性能实验。已经阅读启动文档 D.4、档案 #126/#145/#150/#151/#154/#164/#172 和优化台账；历史整段反量化约6.5秒是禁止重现的失败结构，不是当前耗时。生产只修改重复元数据读取和检查，完整母源、数值合同、原生调度、PT/clip、MTP/GDN、PC、服务参数与 timer 保持。

## 什么情况下容量优势不能兑现

同一阶段、相近历史长度下，每步输出率近似为：

```text
输出 tokens/s ≈ 实际参与执行的请求数 B × 每请求每步平均有效输出 a / step 时间 T
BS16 → BS28 的吞吐比 ≈ 1.75 × (a28/a16) / (T28/T16)
```

当接受量不变、step 时间也增长到1.75倍时，吞吐就不增长。Running 峰值只证明同时驻留的请求数量；不能代替每步的有效 B、MTP 接受量或时间。两侧配置 `num_speculative_tokens=3` 相同，也不等于每步接受 token 数必然相同。

| 情况 | 当前实现/已有证据 | 对 BS 扩展的影响与判断边界 |
| --- | --- | --- |
| 每请求历史工作占主导 | q4/q1 按请求、KV head、split 遍历历史。PC 命中后仍需用该历史做 decode attention | 相同历史长度下，B16→28 的 packed 读取和有效解包总量约×1.75。PC 命中率相同不消除这些工作 |
| 权重读取已被充分分摊 | 模型线性层可在 batch 内复用权重；每请求的 QK/PV、Vector 解包、softmax、精确窗口仍独立 | 每卡约14GB权重不能确定其时间占比，也不能证明 HBM 已饱和。共享部分越小，继续增 B 的收益越小 |
| 算子内部并行已满，后续任务排队 | 现有 main q4 条件模型中，B16/28 的 split=5/4，最忙核32K KV256循环104/192，64K为208/384 | 循环比约1.85，可能抵销1.75的请求增长；这是工作量，不是实测延迟。改 split 会改变归约顺序，未盲调 |
| 中间流量/Vector/同步限制 | INT2展开为FP32、BF16 tile发布、Cube读取、score/P交接、FP32 state仍存在 | 压缩槽144B不等于全部数据流只有144B/token。当前L1/score复用已优化部分；不能仅用压缩比或Cube峰值判 bound |
| 混合/续块 prefill 花费大 | 用户BS8同批11条、32K/50输出、MTP3：C4+C16累计13.782876秒，占OSCAR CV累计90.613% | `pure_decode_steps=916/1200` 的76.33%是次数比例。少量昂贵 prefill/mixed 步可以决定整轮；op累计不是 wall 时间 |
| 外部恢复/抢占等待或阶段不重叠 | 异步恢复已开启；私有抢占仍需保存、全TP确认和安全释放 | 更多驻留请求不保证全部在执行。相同PC命中率也不保证同样的DRAM/SSD层级和恢复时间；现有数据不足以认定它是主因 |
| MTP有效输出、爬坡/排空不同 | 同MTP配置不保证实际accepted相同；有限请求数据集存在尾批 | 对吞吐必须计有效输出，不能用verify输入行数、峰值BS或kernel调用数替代 |

例如仅作解释的模型 `T(B)=A+B*c`：若B16时共享成本A占10%，其余均随请求数增长，则B增加1.75倍只得到约1.045倍吞吐；A占30%也只有约1.148倍。这里的比例是条件示例，不是对14GB权重的测量或现场归因。

[现有任务计数](../reports/batch_scaling_20261112/workload.json)中N64/N112均已在graph捕获尺寸中，不能把本例笼统归为BS28额外padding。C4/C16共享的是同一请求内的query tile，不能让不同请求的KV因batch增大而自动共用。当前已有K/V旋转结果复用、主forward prepare复用、prefix单writer和批量counter读取，本轮不重复记账。

## 本轮已接入的三处改进

### 1. q4/q1/C4/C16 的单条物理页映射复用

`Physical(position)`原来每次均提交一次4B页表DMA，再读UB标量。同一AIV的连续run可能反复查询同一个虚拟128列。现在只缓存 `(request, column, virtualBlock)` 三个整数；每次 kernel Init 清空，绝不跨launch、layer或graph replay保留。请求切换、列切换都会重新读；bounds和缓存值合法性每次仍检查，非法缓存值不能被当作成功。

缓存值脱离`addressBuf`保存，tag/status复用这个UB不会污染它。**包括命中在内，每次原有MTE2_S仍执行**：原屏障也可能完成前一笔packed DMA，不能为了削减等待改变其完成边界。无新增UB/GM；新增key比较、分支和三个标量字段，寄存器/编译器spill成本尚未设备测量。

[静态复现脚本及报告](../reports/bs_scaling_20261113/physical_page_cache_workload.json)按真实loader遍历和既有split公式计数，单层、单KV头、两AIV合计，示例中每task开始按冷缓存计：

| history 场景 | 页表 DMA 原次数 | 新次数 | 同步屏障次数 |
| --- | ---: | ---: | --- |
| 32K，B16/S5 | 11,504 | 8,240 | 保持 |
| 32K，B28/S4 | 19,684 | 14,392 | 保持 |
| 64K，B16/S5 | 22,944 | 16,448 | 保持 |
| 64K，B28/S4 | 39,396 | 28,728 | 保持 |
| C4/C16 单个对齐KV256历史区间 | 16 | 4 | 保持 |

decode对齐区间可能零命中，此时有比较成本却没有DMA节省。每命中只省4B载荷和一次UB标量读；上表约27–28%是**这类页表DMA提交**的减量，不是CV时间、HBM流量或吞吐改善。原INT2解包和KV暂存工作量未减少。

### 2. C4/C16 精确窗口的连续有效行检查

`LoadWindowRuns`原来对每个有效tag行分别执行finite归约和V_S等待。现在仅在**连续有效tag run**内使用已有`FiniteRows256`充分界证书；健康完整16行可由16次scalar等待减为1次。单行/短run仍用原行谓词，极值、NaN/Inf和无法认证的数值继续完整逐行检查，不降低精度门。

不能跨无效tag合并：原顺序中的finite错误2后遇坏tag必须变4，坏tag之后再遇finite错误必须变2。新路径保持这个顺序、物理页错误、K后V、原Cast、全部DMA及其复用屏障；只在Cast完成后复用已闲置的naturalBuf作Abs工作区。FP32内容和最终BF16发布不改变，无新增UB/GM。新增Abs/证书归约成本，实际净收益待设备验证。

### 3. 多请求私有抢占共用counter回读

`handle_preemptions`多请求分支接入已有惰性`resolve_many`。同一CounterSnapshot只等待一次event并读取有效计数列；不同step/stream仍各自等待。逐条resolve→PUT→ACK→forget、每次保存的stream屏障及失败rollback保持，后项错误不能阻止前项已经成功的确认。

实际主机对照中，同快照28条MTP3抢占由28次event等待、84次item变为1次等待和none/align下2/3列读取；载荷、receipts逐字节相同。若只抢占快照中的少数请求，批量读取可能比逐标量多传少量元数据；这是减少调用的取舍。无抢占时本改动没有收益，不能从现有日志假定该路径很热。[详细证据](../reports/bs_scaling_20261113/offload_preemption_review.md)。

## 有依据保留的必要工作

没有跳过完全masked的历史末tile。原路径还会检查Physical/量化metadata，而且合法finite metadata可能展开为Inf；`P=0`时`0*Inf`的PV行为不能直接当成不执行。简单跳过会改变错误域或结果。没有删除identity MTP矩阵运算、修改split/softmax归约、修改capture尺寸/BS/预算，也没有引入整历史BF16缓存或跨请求误共享。

本轮改动属于减少确定的重复工作，不能消除每请求必要的attention、GDN、MTP和共享成本饱和。单凭当前聚合统计，无法保证它们足以让B28超过B16；更不能把已有BS8热点的13.78秒全算作可消除开销。

## 验证和启动接线

同ABI派生源由`generate_active_kernels`统一生成；母源SHA及完整逆变换不放宽。生成工具加入构建签名和probe源码绑定，旧构建无法复用。candidate直启及`start_node93_oscar.sh`实际编译、选择同一新实现，不新增用户参数。

主机验证包括原/新实际提取C++方法、延迟DMA完成模型、错误优先级、碎片/非法页表、同地址多代Init、缓冲污染、PC/MTP/私有抢占字节等价、组合源逆变换、缓存失效和candidate路由。功能probe增加固定地址改变物理页映射的graph replay，所有页载荷/窗口/tag/slot随映射同步迁移，使用已有独立oracle及eager/replay位比较。原本只有一个物理页的测试fixture在NPU分配前预留一个poison备用页，使测试能真正改变映射，生产KV容量不改。它是待运行的真NPU检查，不能把脚本存在写成设备通过。

最终 **361项主机测试通过，0失败/跳过**：108项专项、192项组合生成/构建/路由及原数值保护、61项功能probe/生命周期检查。确切命令、结果、源SHA、测试构造失败说明和验收边界见[validation.json](../reports/bs_scaling_20261113/validation.json)。本机没有CANN SDK；CANN编译、官方CPU-debug、NPU精度/真实图、模型质量及端到端吞吐均未获得本轮通过证据。
