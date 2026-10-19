# 当前 OSCAR 大 batch：本地 profiling、容量和吞吐模拟

母源：`73280bdfdbbd22a3fe2eb06b25c7ab95a9660d79`，分支 `oscar-new-3`。档案依据：#127/#128/#140（物理页与原生 planner）、#148/#154（数学边界与真实工作量）、#155/#156（不能由局部或两个总耗时拟合整轮）、#163/#167（当前 BF16 与最新内存日志）。亲读启动文档 D.4 原始打点；不读取失败 OSCAR 项目源码。

**结论：容量收益明确存在；当前实现尚不能凭容量收益认定吞吐超过原生。** 本次固定假设的稳态 decode 模拟，扩大 OSCAR batch 带来 **17%–21%** 的自身吞吐提升，相对原生为 **1.14–1.19 倍**。这是下文可复算的条件估计，不是目标 NPU 实测预测，更不是整轮 AISBench 倍率。仅将额外 GM 等效 HBM 比例从 25% 改为 50%，结果就变成原生的 **0.84–0.86 倍**；缓存和解包/同步成本足以抵消收益。

## 1. 实验与证据

本轮不改生产算子、默认开关、调度器、目标配置、原生参考树、精度门或 oracle；两个 candidate 入口仍实际使用既有 BF16 组合。本次新增的是离线分析工具、结果和回归。

| 实验 | 实际执行 | 能证明什么 |
| --- | --- | --- |
| 原生容量 planner | 从本项目只读原生源码抽取并执行 grouping/allocation；核对物理页、MTP/GDN 状态 | 给定模型形状、净 KV 预算下的驻留容量 |
| 真实算子控制流计数 | 主机 Clang 编译实际 6 路 Cube 函数体及共享 Unpack helper；78 个 decode 场景、30 个 Cube 形状检查、H16/32/64 三种 helper | API 调用、请求的矩阵形状/FLOPs、最少 GM 操作数字节和分片；不是 SDK 内部执行或 NPU 时间 |
| Python profiling | 实际 `from_common`、current/mixed planner、selector；36 组形状，10 次预热、每组 5×100 步；另用 cProfile | 本机 CPU 路由耗时及 B32/B33 临界分流 |
| 吞吐敏感性模拟 | 运行前固定 JSON 参数，1920 个参数组合 | 给定成本模型的吞吐、延时和盈亏平衡点 |
| 新增回归 | **37 passed，3.92 s**；包含错误 tile 步长负控、first-MTP 原生 current 排除、draft eager 形状、容量与公式边界 | 离线工具的逻辑一致性，不能替代设备验收 |

本机为 macOS/ARM、Python 3.14.6、CPU PyTorch；无 `torch_npu`、CANN `bisheng` 或可运行的 CAModel。未把 Mac 秒数换算成 NPU 秒数，未运行 AISBench，未启动真实模型服务。

数据入口：[容量](../reports/batch_capacity_20261019.json)、[算子计数](../reports/batch_kernel_cost_20261019.json)、[CPU profile](../reports/batch_host_profile_20261019.json)、[历史校准审计](../reports/batch_calibration_20261019.json)、[预声明参数](../reports/batch_simulation_assumptions_20261019.json)、[1920 点结果](../reports/batch_throughput_simulation_20261019.json)、[最终测试](../reports/batch_profiling_tests_20261019.xml)。报告记录对应源码 SHA256。

## 2. 先算真实容量，不用 INT2 编码倍率代替

采用当前配置：TP4、MTP3、APC 关闭、16384 token 调度预算、max-num-seqs=128、sink64/recent256、物理卡4–7/端口7878。日志 #167 四 rank 的净 KV 池均为 14.53 GiB，使用两位小数的保守下界 14.525 GiB。**该日志没有同轮原生净 KV 预算；表格是相同净预算比较。**

本地现有原生形状夹具为 target FULL16/GDN48、draft FULL1，每卡 Hq6/Hkv1/D256。当前目标模型的完整 config 不在本机；16 个 target FULL 另有旋转文件回执，其他维度仍为明确条件。没有把类默认配置当作目标模型实参。

原生 shape helper 实际给出 conv `(6,2560)`、SSM `(12,128,128)`，BF16 字节为 30,720 与 393,216。原生共享 pool 为 `[17,16,16,16]` 四组，17 个槽/池块，有 3 个 GDN padding 槽；每层物理页 817,152 B。默认 none 下原生 FULL 每页768 token，OSCAR 每页2816 token；每页另有 **333,336 B 精确快照**。因此物理 FULL 页 token 容量比是 **3.667 倍**，不能用 `1024/136=7.529` 的纯编码比代表实际空间收益。

预留每请求 512 个输出 token、3 个 FULL lookahead token、每 GDN 组4个状态，加全池1个 null 块：

| 输入长度 | 原生每请求全部 KV/卡 | OSCAR 每请求全部 KV/卡 | 原生驻留 BS | OSCAR 驻留 BS | BS 倍率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32K | 0.7245 GiB | 0.3105 GiB | 20 | 46 | 2.30 |
| 64K | 1.2808 GiB | 0.4658 GiB | 11 | 31 | 2.82 |
| 128K | 2.3805 GiB | 0.7633 GiB | 6 | 19 | 3.17 |

这些是驻留上限，调度准入、混合 prefill、输出增长、并发到达和排空会降低实际 decode BS。align 布局按峰值状态另算，在相同输出预留下为原生19/10/5、OSCAR37/25/15；不能混用 none 的表。

当前共享 workspace 为 **610,272,512 B/卡（582.001 MiB）**，旋转矩阵为 12,845,056 B/卡。均不能乘以 FULL 层数；净 KV 日志来自模型 profiling 后，不能再重复从14.53 GiB扣除这些空间。原生自身 workspace 不同，实测净 KV 预算可能不同，模型保留这个缺口。

## 3. 源码成本：省了常驻 KV，仍有大量临时操作

从 `attention_bf16_history_mdl.cpp` 的 `HistoryCubeTask`、共享 `attention_striped_unpack_bf16.h` 实际控制流计数，一个完整历史 KV256 tile、单本地 KV head：

| 项目 | q4 验证 | 后续 q1 MTP |
| --- | ---: | ---: |
| 原生 BF16 KV 一次逻辑读取 | 262,144 B | 262,144 B |
| OSCAR INT2 + FP16 参数 + 页表 | 34,832 B | 34,832 B |
| OSCAR 额外临时 GM 最少读写 | 667,648 B | 568,320 B |
| BF16 QK/PV 请求 FLOPs | 6,291,456 | 1,572,864 |
| Unpack helper 调用数，合计 K/V | 8 | 8 |

临时字节含 BF16 KV 发布、Cube 最少 operand 读取、score/P 与输出处理。额外的 Q 发布、任务末 FP32 inverse rotation 和 partial 写另计在完整任务中。**GM 请求量不等于 HBM 流量**：缓存可能服务这些请求，SDK 内部也可能再次读取；FLOPs 是 `SetSingleShape` 请求量，不是物理 Cube 指令计数。

仅比较历史流量，q4 要保留带宽优势，额外临时 GM 的等效 HBM 比例需低于 **34.05%**，q1 低于 **40.00%**，而且这尚未扣解包/同步时间。若假设有效 HBM 1 TB/s、20 Cube 完全活跃，压缩读取节省的全部时间只给每 Cube 每 KV256 tile 约 **4.546 μs** 的额外工作余量；存在临时 HBM 访问后余量继续下降。1 TB/s 是扫参值，不是本机测量或硬件规格声明。

在64K、BS31的模型步内，17次 q4 + 2次 q1 的 history/precise 源合计：必须读取的压缩/精确 KV及页表/tag **5.0505 GiB/卡**，其他 GM 最少请求 **93.4610 GiB/卡**。原生19次 BF16 KV 一次读取为36.8125 GiB/卡。93.4610 GiB 包含可复用的缓冲反复读写，不是分配了这么大内存，也不是断言全部经过 HBM。

扩大 BS 不摊薄这些逐请求历史扫描。C4/C16 只能在**同一个请求**中复用符合条件的 query group，不能把不同请求的 KV 合并。历史 split 已允许小 BS 并行利用多个 Cube，不能假设原生小 BS 完全闲置而 OSCAR 大 BS 获得线性加速。

主模型走捕获尺寸；本次 draft enforce_eager=true，first MTP 实际 N=4B，later MTP 继续保留 N=4B（仅 B 行有效）。例如 BS46，target N196/S2，first/later N184/S3。后两次 attention 是 q1，不表示 dense projection/MLP 只执行 B 行；原生 proposer 的三个 draft dense 调用可能合计12B行。first-MTP current 源使用原生 FIA，计数已从 custom CV 中移除，未捏造 FIA 内部字节。

## 4. 明确的吞吐预估与适用条件

为避免把未知参数伪装成校准，使用可直接修改的预声明实验情景：

1. 原生驻留 BS 下，long-history KV 服务时间归一化为1；公共固定成本F=0.5、随请求线性成本L=0.5。即原生这一步的模型份额为历史50%、可摊薄固定25%、线性25%。这不是实测阶段占比。
2. OSCAR 压缩/精确 KV 按源码计数；其余 GM 请求按25%等效 HBM计费（γ=0.25）。这不是标准 L2 hit-rate 指标。
3. 未被字节服务覆盖的额外解包、计算、同步、旋转/store/merge等开销，假设净串行部分为每请求原生历史服务的25%（U=0.25）；不是假设其免费，也不声称这能包住实际开销。
4. 两侧每请求每次验证输出 token 相同（接受长度比a=1），稳态纯decode、相同上下文长度，不包含混合prefill、爬坡和排空。

令 s=B/Bnative，A 为原生小批次历史字节，K/G 为 OSCAR 必须读 KV 与其他 GM 请求：

`Tnative = 1 + F + L`

`Toscar(B) = F + s × (L + U) + (K(B) + γ × G(B)) / A`

`吞吐比 = s × a × Tnative / Toscar(B)`

这是一个串行成本敏感性模型，不是完整设备仿真；未测流水重叠不从任意父子相位重复相减。不输出绝对 TPS 或服务秒数。

| 输入 | OSCAR BS 扩大 | 相对 OSCAR 同原生 BS | 扩大后相对原生 | OSCAR 单步延时变化 |
| --- | --- | ---: | ---: | ---: |
| 32K | 20 → 46 | **1.174×（+17.4%）** | **1.142×（+14.2%）** | 1.960× |
| 64K | 11 → 31 | **1.201×（+20.1%）** | **1.177×（+17.7%）** | 2.347× |
| 128K | 6 → 19 | **1.211×（+21.1%）** | **1.190×（+19.0%）** | 2.615× |

同一 OSCAR 版本先用原生容量 BS 时，在此情景中只有原生吞吐的0.973/0.980/0.983倍；扩大 BS 摊薄固定成本后才超过原生。吞吐提升伴随更长 step，不能把它同时写成单请求 token 延时降低。

仅改变额外 GM 等效 HBM 比例，其他参数不动：

| γ | 32K 相对原生 | 64K 相对原生 | 128K 相对原生 |
| --- | ---: | ---: | ---: |
| 0%，理想缓存情景 | 1.804× | 1.879× | 1.918× |
| 25%，上述预估情景 | 1.142× | 1.177× | 1.190× |
| 50%，较多临时数据出缓存 | **0.835×** | **0.857×** | **0.863×** |
| 100%，全部按 HBM 计费 | 0.544× | 0.555× | 0.557× |

在F/L/U/a保持上述取值时，超过原生要求 γ 分别低于 **34.66%/36.84%/37.54%**。若 γ=25%，但 OSCAR 每步有效输出只有原生80%，三种长度变成 **0.913/0.942/0.952倍**。因此接受长度、访存和附加时间必须一起成立；不能用 speculative_tokens=3 假定固定输出4个token。

1920点网格只是系统扫描假设，不是置信区间，不从中挑最快点作为当前设备结果。旧日志中的2.914/3.789仅是旧版本统计窗口，未用作本次a。历史 native FIA 1.4509ms对应32请求、20/23/27/30K各8个，单卡q4 Event；564.705 GB/s仅为该形状一次KV读取的逻辑归一化，也未拿来给当前 OSCAR 标定秒数。

## 5. PD 融合下的明确风险与判断

**当前设计的容量方向合理，性能路径仍不充分。** MDL缩小短query工作区、INT2小表直出BF16、共享workspace都减少了不必要成本；仍需支付临时BF16 KV发布/再次读取、softmax/P与partial处理、FP32逆旋转、split合并和MTP额外调用。

实际调用 `plan_mixed_decode` 验证：有历史的 continued prefill 与 q4短请求混合时，短前缀最多128token。B32还能拆分，B33即132token不拆分；长suffix≥8192时整批选择C16，512token suffix时整批C4。B32拆分也保留原整批S，常见为S1，不能套纯decode B32/S3的耗时。新的零历史prefill suffix有独立current-only路径，不受此边界同样影响。

本机最终 profiling中，continued混合 B32/B33/B46 的一次group构建+16层路由决策约 **184.6/187.2/201.9 μs**；fresh B46约81.4μs。说明边界是真实路径差别，**不能从相近的Python耗时推断两个设备算子同速**。纯target的capture决策约53μs也不是图回放时每步Python成本。

因此，当前32K容量虽允许46，不宜直接把46作为PD性能收益点；优先关注B32→33的分流边界。64K默认容量31、128K容量19，本身就不支持对全历史驻留直接设置BS64/128。应优先降低上述临时访存与同步，验证更大短前缀分区的完整forward成本，确认later-MTP实际padding成本；不能仅继续增加max-num-seqs。

本轮没有为了漂亮的模拟数字改生产默认。已有BF16筛查仍7/11失败（其中2例最终输出失败），目标机Trap也未归因解决；没有把此次成本计数提升为精度、NPU、图、模型质量或整轮性能验收。

## 6. 本地复现

以下仅用于当前已配置 `.venv` 的Mac开发机，真机启动不依赖这些命令。结果写新临时目录，不覆盖历史报告：

```bash
export SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
export OSCAR_TEST_CLANGXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
profile_out=$(mktemp -d /tmp/oscar-batch-profile.XXXXXX)
.venv/bin/python -m tools.experiments.batch_capacity_20261019 --output "$profile_out/capacity.json"
.venv/bin/python -m tools.experiments.batch_kernel_cost_20261019 --output "$profile_out/kernel.json"
.venv/bin/python -m tools.experiments.batch_host_profile_20261019 --output "$profile_out/host.json"
.venv/bin/python -m tools.experiments.batch_throughput_simulation_20261019 \
  --capacity-report "$profile_out/capacity.json" --kernel-report "$profile_out/kernel.json" \
  --output "$profile_out/throughput.json"
.venv/bin/python -m pytest -q tests/test_batch_capacity_20261019.py \
  tests/test_batch_kernel_cost_20261019.py tests/test_batch_host_profile_20261019.py \
  tests/test_batch_throughput_simulation_20261019.py
```

实验纠偏保留：初次模型测试暴露零临时字节情况下盈亏平衡式除零，已返回不适用而不伪造阈值；最终37项通过。初版host实验重复16次metadata builder，复查原生runner发现同group共享一次metadata，已修正并增加调用数测试，旧报告单独保存为`batch_host_profile_20261019_initial_repeated_builder.json`且其耗时不用于最终结论。target graph padding与eager draft padding、first-MTP native current成本边界也已交叉修正。
