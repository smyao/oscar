# 32K 已达标、64K 变慢：执行路径与通用优化

2026-10-27，限定分支 `codex/oscar-bf16-attention`。

**已找到会使 64K 的额外成本增长快于输入长度的执行机制：16K token budget 下，后续 prefill 反复访问越来越长的历史；C16 虽然共享了 INT2 解包，仍为每个 query group 执行 QK、softmax、PV，并在每个 KV256 上读写 FP32 在线状态。当前 v4 的 warm prefill 还必须在同一自定义 CV 路径处理 current 部分。这些工作没有随 KV 压缩消失。** 本轮针对其中重复的转换指令发射、完全可见块的逐行检查和 Scalar/Vector 等待做三项通用优化。

这解释了“64K 不一定比 32K 获得更多压缩收益”的机制，但**现有证据不足以把用户此次慢 10% 精确拆成各阶段秒数，也不足以承诺改后达到 30% 或证明 32K 的端到端时间不退化**。用户当前结果是实际性能事实；本文工作量是源码计数，两者不混为同一证据。

## 1. 本次事实与证据范围

| 项目 | 当前已知 |
|---|---|
| 用户负载 | 200 请求，客户端 batch size 32，prefix cache 开启 |
| 固定预算 | `max_num_batched_tokens=16384`，本轮不调整 |
| 用户 32K 结果 | 相比禁用 OSCAR 的基线，端到端总时间快约 30% |
| 用户 64K 结果 | 同条件下，比基线慢约 10% |
| 本次现场未知 | 两侧实际 prefix 命中 token、逐阶段设备时间、每步实际 decode batch、MTP 接受长度、抢占、输出 token 数 |
| 本次本地工作 | 源码复核、原生合同主机重放、公式计数、功能验证；没有连接真机或运行性能实验 |

200×BS32 不等于每个设备 step 都有 32 个 decode 请求。原生调度、prefill 混合、MTP、爬坡和排空仍会改变每步组成。没有现场轨迹时，不用客户端 BS32 代替设备实际 batch，也不把历史请求的输出长度套入这次测试。

档案依据：[#129–131](/Users/sunao2000/gpt_new_oscar_kimi/issue1_full_record.md:27954) 要求区分任务归属、有效工作与主机背压；[#148–151](/Users/sunao2000/gpt_new_oscar_kimi/issue1_full_record.md:28498) 要求区分共享解包、局部时间和整轮收益；[D.4 完整原始日志](/Users/sunao2000/gpt_new_oscar_kimi/oscar_ascend_agent_start.md:1118) 禁止全历史恢复，其旧 6.5 秒 dequant 不是当前占比。

## 2. 不是在累计 64K 处换了一个坏算子

当前 host 分流的长 prefill 阈值是 **本次 query 长度 `max_query_len >= 8192` 选 C16**，其余长 query 选 C4；不是按历史 context 到 64K 才切换。固定 16K 预算使 32K 和 64K 的主要 warm chunk 进入同一家族。q4/q1 decode 另有分流，仍完整读取压缩历史。[分流源码](/Users/sunao2000/gpt_new_oscar_kimi/oscar_ascend/ops/cv_dispatch.py:167)

任务准备为每个 query group 生成三源：历史 `[min(sink,context),context)`、精确 sink/recent 候选，以及本 chunk 的因果 current `[context,first_query_position+count)`。历史和 current 都不能因为输入长而跳过；历史 precision cut 的遮罩在 reader 内执行。[任务区间](/Users/sunao2000/gpt_new_oscar_kimi/csrc/kernels/attention_official_tasks.cpp:144)

Hq6/Hkv1 的一个 query group 为 `floor(128/6)=21` 个 query。C16 只把同一 request/head/split/context/range 的 16 个完整 group 聚在一起，共享一次 KV256 解包和 K/V 发布；**这 16 个 group 的 score、softmax、P、PV、max/sum、输出累积都独立**。[C16 聚簇条件与调度](/Users/sunao2000/gpt_new_oscar_kimi/csrc/kernels/attention_cv_official_cluster16.cpp:338)

### v4 的 current 路径为何重要

无历史的首块可以在一致旋转域中执行既有 native current-only attention，再量化写入缓存。已有历史的 warm prefill 需要把历史、精确窗口、current 的 **FP32 partial 合并后再一次逆旋转**。直接使用 native FIA 的 BF16 current partial 会提前舍入，所以 v4 显式关闭这条 mixed-current partial 路径，current 同样由 C16/C4 CV 计算。这是现有精度合同的选择，本轮保留。[首块路径](/Users/sunao2000/gpt_new_oscar_kimi/oscar_ascend/integration/current_only_dispatch.py:96)，[warm 路径与原因](/Users/sunao2000/gpt_new_oscar_kimi/oscar_ascend/integration/impl.py:107)

因此，多出来的后续 chunk 并非“仅把压缩历史读长一点”：它同时增加完整的历史计算、current 三角计算、精确窗口计算和在线状态交接。

## 3. 固定 16K 预算下的工作量

以下单位均为 **一个请求、一个 FULL 层、一个 TP rank，Hq6/Hkv1/D256，S1，无 padding/hole**。一项 QK/PV tile pair 指一个 query group 对一个 KV256 的一次 QK 和一次 PV，不是设备耗时。脚本没有给这些计数拟合秒数或加速比。[计数实现](/Users/sunao2000/gpt_new_oscar_kimi/tools/analyze_long_context_work.py:34)，[完整结果及源码 SHA](/Users/sunao2000/gpt_new_oscar_kimi/reports/long_context_20261027/work_model.json)

### 理想 16K 整块：帮助理解增长来源

这里明确假设该请求没有 prefix 命中，只是示例，**不是对用户现场命中率的判断**。

| 指标 | 32K：16K+16K | 64K：四个16K | 比例 |
|---|---:|---:|---:|
| 新输入 token / 量化写入 token | 32,768 | 65,536 | 2.00 |
| warm prefill token | 16,384 | 49,152 | 3.00 |
| 历史 QK/PV tile pairs | 49,984 | 299,904 | 6.00 |
| 共享后的历史解包 tiles | 3,904 | 23,424 | 6.00 |
| 自定义 current QK/PV tile pairs | 25,438 | 76,314 | 3.00 |
| cluster 循环状态读写地址字节 | 12.09 GiB | 72.56 GiB | 6.00 |

原因很直接：32K 的 warm chunk 面对一次 16K 历史；64K 的三个 warm chunk 分别面对约 16K、32K、48K 历史，累计为 `1+2+3=6`。首块仍只执行一次，已达标 32K 所受的历史成本远小于 64K。原生 attention 也有随长度增长的数学工作，不能把这六倍当成“比原生慢六倍”；真正差异是这部分数学之外额外的 INT2 展开、GM 交接、转换与同步。

### APC 当前实际 splitter 的主机重放

使用原生 `_mamba_block_aligned_split`、真实 production checkpoint policy、原生 manager/hash 合同，固定每次剩余预算 16384。没有执行 NPU，也没有重放用户的 BS32 到达/抢占轨迹。[主机重放实现](/Users/sunao2000/gpt_new_oscar_kimi/tools/experiments/prefix_checkpoint_audit.py:72)，[生产 splitter](/Users/sunao2000/gpt_new_oscar_kimi/oscar_ascend/integration/prefix_checkpoint.py:131)

| prompt | Native APC chunk | OSCAR APC chunk |
|---|---|---|
| 32,768 | 16,128 / 15,360 / 1,280 | 16,128 / 13,824 / 1,536 / 1,280 |
| 65,536 | 16,128 / 16,128 / 16,128 / 16,128 / 1,024 | 16,128 / 16,128 / 16,128 / 13,824 / 2,304 / 1,024 |

原因是 native GDN 768-token 状态块、OSCAR FULL 2304-token 物理页的共同 checkpoint，以及原生 `prompt-1`/MTP 尾页规则。两个长度都比对应 native 多一个 tail step；不是 64K 独有的新增分支。本轮不改这些已修复的状态/hash 生命周期。

对上面的 OSCAR **显式冷请求样本**，得到：

| 指标 | 32K 主机 chunk | 64K 主机 chunk | 比例 |
|---|---:|---:|---:|
| 历史 QK/PV tile pairs | 57,678 | 308,781 | 5.35 |
| 共享后的历史解包 tiles | 7,080 | 27,711 | 3.91 |
| 自定义 current QK/PV tile pairs | 18,616 | 68,058 | 3.66 |
| cluster 循环状态读写地址字节 | 14.06 GiB | 75.65 GiB | 5.38 |

解包增长与计算增长不同，是因为短 tail 选 C4，其共享粒度为 4；主干选 C16。这个区别是当前实际分流的结果，不能统一按“所有 chunk 都 C16”估算。

**APC 开启也不等于这些冷样本每条都发生。** 在本地主机的完全相同重复 prompt、checkpoint 尚存在的条件下，native/OSCAR 的共同命中结果分别为：32K `31488/29952`，64K `64512/62208`。OSCAR 对应剩余 tail 为 2816 和 3328 tokens，走 C4；现场共享前缀长度和缓存压力未知，因此不能将这些命中上限当成 200 请求的实际命中。JSON 同时给出明确标注的冷样本和“完全相同 prefix 被保留”的条件 tail，未选择其中任意一个冒充现场。

## 4. 各环节的结论

| 环节 | 已证实的增长或成本 | 仍未确定 |
|---|---|---|
| prefill 历史 | 多个 warm chunk 重复处理更长历史；QK/PV、softmax 与在线状态的累计增长大于新 token 数 | 本次现场占比、APC 实际减少了多少工作 |
| prefill current | 首块可用 native current-only；warm v4 current 走自定义 CV，并有大量全可见的三角内部 tile | 当前设备耗时及与历史路径的关键依赖占比 |
| decode / MTP | 每步要扫描更长历史；q4/q1 的 K/V/P BF16 发布仍有可消除的逐 repeat 标量发射 | 实际 batch、有效输出/步、MTP 接受率和每步延迟 |
| 量化写入 | 只写当步新 K/V；没有发现因 64K 重写全部旧历史的行为。无命中、相同输出时 prompt 写入 token 数翻倍 | 写入在此次端到端中的占比 |
| 反量化 | KV256 有界融合，无全历史 BF16 物化；C16/C4 减少了解包重复，未消除跨 query/跨 chunk 的重复 | 实际解包/访存是否成为关键限制 |
| 搬运 | 量化 payload 之外仍有 score/P/PV 和 FP32 acc/max/sum 交接 | 地址字节有多少命中 L2、多少到 HBM、是否带宽受限 |
| metadata / 调度 / GDN | 保留现有 native 调度与 GDN；APC 对齐影响 chunk/tail 个数 | 本次抢占、host 排队及各原生算子时间；不能由总时长反推 |

旧 C16 timer 的 `softmax_math` 和 BF16 转换结果只用于选择方向；它来自更早的指定32K诊断，**不套用为当前 v4、BS32、64K 的时间占比**。[原计时范围与限制](/Users/sunao2000/gpt_new_oscar_kimi/docs/c16_breakdown_analysis_20261020.md:23)

## 5. 本轮三项通用优化

### A. q4/q1 发布：相同转换，批量发射 repeat

decode 的三个 K/V/P 发布点从逐64元素发一条 normal-mask RNE 转换，改用已经存在的批量 repeat helper。每条最多255 repeats，16384元素明确拆成255+1；既有 stride、RNE、目标地址、DMA fence 与量化 metadata 解码保持。零 padding 转换不在这次替换范围。[严格变换](/Users/sunao2000/gpt_new_oscar_kimi/tools/official_decode_cast_transform.py:1)，[helper](/Users/sunao2000/gpt_new_oscar_kimi/csrc/kernels/attention_c16_cast.h:22)

| 一次发布元素数 | 原指令发射数 | 新指令发射数 | 转换元素数 |
|---:|---:|---:|---:|
| 1,536 | 24 | 1 | 不变 |
| 4,096 | 64 | 1 | 不变 |
| 16,384 | 256 | 2 | 不变 |

这是指令发射数量，不是64倍/128倍算子提速。64K 每步历史 tile 更多，重复成本被进一步放大；相同优化也覆盖32K及其他长度，不按输入长度硬编码启用。

### B. C16 完全可见 current/window：复用现有 finite 路径

旧代码只有完整历史 tile 使用批量 finite helper，current/window 即使无任何遮罩工作，也逐行构造可见范围。v4 固定 cut 保证 window candidate 连续；current tile 则必须在当前 score block 最早 query 的因果边界之前。满足严格证明才使用全可见路径，其余 diagonal、尾块和 padding 仍走原 mask。[可见性证明条件](/Users/sunao2000/gpt_new_oscar_kimi/csrc/kernels/attention_cv_official_cluster16.cpp:853)，[变换](/Users/sunao2000/gpt_new_oscar_kimi/tools/official_softmax_visibility_transform.py:1)

独立主机测试编译实际发出的 C++ predicate，并逐行逐列枚举原逻辑位置，覆盖 GQA1/6/16、S1/S3/S32、sink/recent 边界、请求偏移和 causal diagonal。该证明不替代设备数值验证。

### C. C16 finite 检查：充分界证书减少逐行同步

完整可见 score block 先在已结束 KV staging 的 UB 内计算绝对值非负和。若结果有限且不超过 `2^110`，则每个原输入绝对值均有界，原256项行归约的任意中间和远离 FP32 溢出。此时不必再逐行等待 Scalar 读取 ACC_VAL。证书不成立时，仍执行原逐行 count-mode vcadd/ACC_VAL 条件；NaN/Inf、巨大有限值和抵消场景不由证书代替裁决。[证书实现](/Users/sunao2000/gpt_new_oscar_kimi/csrc/kernels/attention_official_finite.h:14)

该证书不改变 score、不近似 softmax、不改变错误码。最多复用8192个 float 的死 dequant UB，无新增分配；小于8行保留原检查。它增加一次 Abs 和块归约，换取常见有界块中多次逐行归约/同步的移除，**净设备收益仍需实际验证，尤其不能仅凭少了同步次数就承诺32K不退化**。

三项改动经 [official generator](/Users/sunao2000/gpt_new_oscar_kimi/tools/generate_official_attention.py:123) 生成实际 q4/q1/C16 reader，候选继续通过统一 `tools/serving_variants.py` 的 official v4 路由选择；不是只修改 probe。原 C4、aligned 母源、timer、冻结 status guard、INT2 codec、FP32 metadata 策略、recipe3/PT、clip、旋转、窗口和启动参数保持。

## 6. 剩余的大成本与性能边界

**FP32 在线状态搬运仍存在，本轮没有消除。** 每个聚簇 query group、每个 KV256 的 Vector restore+spill 地址字节为：

`2 × 128 × (256 + 2) × 4 = 264192 bytes`

这里的256是输出acc列，2是max/sum；128是两个AIV共同负责的完整行容量。上表只累加 cluster 主循环的这部分，未包含初始化spill、最终restore、Cube atomic读写、score/P/PV/KV通信；也没有把这些地址访问等同于HBM流量。[实际状态读写](/Users/sunao2000/gpt_new_oscar_kimi/csrc/kernels/attention_cv_official_cluster16.cpp:1025)

要进一步明显削减这部分，需要保持数值顺序和有界工作区的状态驻留/多KV复用设计，并重新证明所有权和资源容量。不能用全历史展开、降低FP32精度、改16K预算、降低真实工作量或关闭prefix cache换取表面成绩。当前三项优化针对已有精确实现中的重复工作，未把未实现的状态重构计为收益。

32K 与64K的数学、模型参数和调度配置保持同一套；这构成非特例优化的证据。**32K端到端不退化和64K约30%收益仍是设备验收要求，不能由主机计数或局部CPU通过自动推出。** 本轮不将性能标为 passed，也不把目标不足包装为已经完成。

若按“总时间减少30%”解释目标，当前64K约为`1.10×native`，目标为`0.70×native`，还需消除当前OSCAR时间的`1−0.70/1.10≈36.36%`。若某待优化路径只占当前时间的比例`p`，其加速比为`s`，即使其他路径完全不变，也需满足`(1−p)+p/s≤0.70/1.10`。因此没有实际占比和完整新耗时，就不能用某个helper的发射次数下降代替这个总账。

## 7. 本地复现与验证记录

```bash
.venv/bin/python -m tools.analyze_long_context_work --output reports/long_context_20261027/work_model.json
.venv/bin/python -m pytest -q tests/test_long_context_work.py tests/test_official_softmax_visibility_transform.py
```

计数与可见性主机回归：**21 passed**。计数测试独立枚举 query leaders/cluster owners、当前因果 tile、cast repeat 覆盖，验证六倍理想历史增长和显式 prefix 条件。JSON 中四个 native/OSCAR、32K/64K 合同重放均 completed，symbolic state identity violation 为零；这些不是GDN真实状态payload、NPU、图、模型质量或性能通过。

最终本地验证如下，原始回执及源码 SHA 见[总验证记录](../reports/long_context_20261027/validation.json)：

| 验证层次 | 真实结果 |
|---|---|
| 主机/生成/路由/冻结身份 | 101 passed；生产源与 generator 逐字一致；25个配置、timer、母源等文件与基底逐字一致 |
| finite helper 官方 CPU-debug | 1507/1507 passed；原/新判定一致，score/input不变、UB/GM guards通过；不是1507个模型用例 |
| 完整 C16 官方 CPU-debug | 6组 baseline/candidate 逐位一致：成熟16成员簇+current、独立多KV、完整window/双head、mode1、metadata错误、全mask历史；健康用例独立v4 oracle通过 |
| 完整 q4/q1 官方 CPU-debug | 10组配对、42份输出共4,149,936字节逐位一致；6健康oracle/4错误域，含context850跨KV预取 |
| CANN ascend910b4 | 完整official C16/C4/decode设备库和独立finite helper编译通过；没有改ABI或SDK |
| NPU / graph / 模型质量 / 性能 | 本轮未运行；不将以上结果升级为这些验收 |

失败尝试没有覆盖：首次C16测试加载缺`libunified_dlog.so`返回127，修正测试环境；随后Q337/context577控制臂的SDK逐操作日志写满VM，未完成数值验收，原日志完整归档且调用退出120。改为覆盖相同风险的6个互补形状，并只在成功phase结束后清其可再生SDK trace，仍保留所有SDK检查、原180秒deadline和冻结容差。decode同一磁盘事故在未改动query-copy处出现SIGBUS/ENOSPC，首错另存后完成剩余配对；不能写成新内核的数值失败或通过。独立finite探针的构建接缝首错及修复也保留在其回执。

仍沿用固定 candidate 直启入口：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

本轮没有修改 prefix 开关或其他 argv；用户既有 APC 开启方式保持。新源码进入既有构建指纹，实际服务选择优化后的official q4/q1/C16；PT recipe与格式没有改变。
