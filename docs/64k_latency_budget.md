# 64K、120请求、K32：相对原生2–3倍的当前时延预算

**2026-10-10 用户最新修订**：整轮相对原生2–3倍可接受，保持普适优化与原生token budget。native实测1342.4437259秒，对应目标 **671.22186295 / 447.48124197秒**；相对当前OSCAR2467.1275737秒仍需 **3.676 / 5.513倍** 改善。最新可复算账本为`reports/latency_budget_64k_20261010.json`。这是目标预算，尚非改后预测。

下方保留2026-10-09的4–5倍预算历史与数据归因证据。其中whole-prefill消除历史工作的方案已因用户要求省略token budget而退休，不再计入任何当前速度评估；最新工具将其显式隔离为历史。当前P表示一次请求所有原生prefill chunks的服务成本总和。

本轮目标已由用户明确：相对原生22.4分钟快4–5倍。按附件精确的native Benchmark Duration **1342.4437259秒**，目标是 **335.610931475秒 / 268.488745180秒**。当前OSCAR为2467.1275737秒，因此实际还需7.351 / 9.189倍改进。此前相对当前OSCAR计算出的493–617秒目标已失效。

证据严格限于 `reports/evidence_64k_120_20261009.json`、其原文 `reports/target_64k_120_20261009.txt`、`docs/64k_performance_source_audit.md`及当前源码/指定原生参考。原文SHA256为`03eb53109a97a3e82b02c91550b5661dc8cf0c3360585693f567a3217c7311bb`。没有读旧LongBenchv2性能记录，没有性能拟合、VM、CPU-debug、CAModel或设备执行。档案#137/#140/#152/#155用于区分日志、算子、整轮证据，不移用其耗时。

**当前数据不支持宣称268–336秒可达。** whole64K有明确的大幅减少OSCAR历史prefill工作量的结构依据；剩余整段current attention、GDN、MLP/线性层、通信和长上下文decode没有本轮设备相位计时，无法算出可靠秒数下界或4–5倍预测。FULL KV 6–7倍的字节目标与速度目标分开验收。

## 1. 从日志进一步能识别什么

### 请求级验证事件可以反解，完整模型迭代数仍未知

原生 `SpecDecodingStats.observe_draft` 每调用一次只增加一个**请求级**事件Q；logger输出 `mean=1+A/Q`，三位per-position rate为各位置接受计数除Q。参见 `references/vllm/vllm/v1/spec_decode/metrics.py:39–45,75–115`。本轮分析同时使用A、D、两位mean、三位position rates和接受计数的前缀单调性求整数Q，未直接把D/3当无条件事实。

| 本轮已打印窗口 | 反解结果 | 边界 |
|---|---:|---|
| OSCAR 226个非零accepted完整窗口 | **唯一Q合计21125** | 包含warmup；不是全局模型轮数 |
| OSCAR原文line51，A0/D6、mean1.00 | **Q至少2，无无条件有限上界** | 原生invalid-draft修正可把某事件的有效D减到0；这些字段无法排除额外0-accepted事件 |
| OSCAR line565 | 整数A/D缺失 | 25.00/39.90 token/s不能恢复精确计数，保留unknown |
| native 135个完整窗口 | **唯一Q合计16316** | 包含warmup；line917另有不可恢复片段，不能宣称全程计数完整 |

源代码约束：`scheduler.py:2061–2076`先检查scheduled draft存在，再扣invalid数量并observe。因此即便D都是3的倍数，也不能无条件假设每个事件都留下3个有效draft。对于226/135个窗口，加入打印的接受均值与position率后整数解恰好唯一，这是比只除3更强的结果。

这些计数仍不能给出全局验证step：一个step可以包含1–32个此负载请求，且prefill、MTP与token上限会改变参与行数。例：OSCAR02:15:20（line146）A2777/D3969唯一对应Q1323；在客户端K32假设下至少需要42个含验证的全局step。若额外假设整个区间始终27个请求共同参与，则1323=27×49，但日志的两个Running端点不能证明此假设，不能把10/49秒称为已测设备cycle。

完整已打印窗口的平均接受长度OSCAR约2.914，native约3.789。它说明同样输出量下OSCAR通常需要更多请求验证事件，但由于缺失窗口、warmup、停止裁剪和批次变化，不能直接当成整轮准确1.30倍模型调用。即使接受长度提高到理论上限4，单看该因素也只是约4/2.914=1.373倍潜在decode事件减少，不能提供整轮4–5倍收益。

### 输入首输出 credit 的节奏，不是输入计算计时

原生 `metrics/stats.py:353–371`在请求仍被标记prefilling的首个输出更新其完整prefill统计，logger再累加computed prompt token（`metrics/loggers.py:142–146`）。因此打印prompt吞吐0只能说明该窗口没有对应credit，不能说明设备没做prefill。

两边Engine日志都连续每10秒一条。以65100-token整请求、无prefix命中及1个warmup为明确假设，将每个非零prompt窗口量化为1或2个credit，两边恰好各121个；去掉最初warmup为120个。真实logger毫秒级抖动未知，整数秒bin不是精确首token时间。

| cohort首输出credit | OSCAR | native |
|---|---|---|
| 首个bin | 02:06:50，line49 | 03:05:12，line749 |
| 最后bin | 02:47:10，line603 | 03:27:12，line1102 |
| 首末bin跨度 | 2420秒 | 1320秒 |
| 相邻credit bin间隔中位数 | 20秒 | 10秒 |
| 最大bin间隔 | 50秒 | 50秒 |

同bin的两个credit按0秒间隔记录，只说明打印分辨率不足。上述节奏混有排队、模型计算、decode及客户端补充请求，不是120次prefill耗时之和；不能把2420/120当作纯prefill20秒。

### 只能给出“全部首输出之后”的排空候选段

OSCAR在最后credit bin结束后，到Running=0的观测段为 **02:47:10→02:47:30，20秒**，后两个spec窗口A3149/D5028，对应Q1676；Engine generation credit近似4806个。native为 **03:27:12→03:27:22，10秒**，A1577/D1614，对应Q538、generation近似2101个。

它们比“prompt=0”提供更多约束：所有首次输出已被计入、Waiting已清空、Running下降。但没有scheduler抢占/重算事件及设备完成时间，仍不能无条件认证这就是纯decode。若明确额外假设无重算/外来请求，才可作为排空decode候选；对应240.3与210.1 token/s是**变动B的日志credit平均率**，不是K32稳态NPU指标。

A+Q分别为4825与2115，略多于generation积分，这也不是计数错误证明：原生在 `scheduler.py:1433–1455`先记录spec接受，再处理停止与输出裁剪。不能强制让这些数字等于最终输出token数。

## 2. 新目标要求的必要条件与一个可检验预算

| 指标 | native快4倍 | native快5倍 |
|---|---:|---:|
| 整轮目标T | 335.611秒 | 268.489秒 |
| 相对当前OSCAR需要 | 7.351倍 | 9.189倍 |
| 完成批次平均输入吞吐 | 23276.95 token/s | 29096.19 token/s |
| 完成批次平均输出吞吐 | 183.07 token/s | 228.84 token/s |
| K32下允许的平均客户端E2E上界 | 89.496秒 | 71.597秒 |

最后一行来自相同batch wall口径的占用积分：120×mean(E2E)≤32×T。当前OSCAR mean(E2E)=649.324秒，native=335.920秒。这个必要条件包含服务端排队，但没有把它分摊给某个kernel。

为后续短probe定义一个明确、保守、**没有从两个总wall拟合**的预算模型：P为一个whole65100完整target+首次MTP预填充的增量服务时间，D为context65100/K32下完整target q4＋三轮MTP的有效输出吞吐，U为额外无法重叠的开销。若按预填充服务与decode服务分时计算：

```text
120*P + 61440/D + U <= T
```

这是待验证的服务预算，不是当前实际混合时间线的分解。若混合计算使两项重叠或互相干扰，须用同shape短服务证据替换该模型，不能把重叠收益预先填进去。

| 假设可持续的纯decode D | 输出预算61440/D | 4倍目标允许P（U=0） | 5倍目标允许P（U=0） |
|---:|---:|---:|---:|
| 300 token/s | 204.80秒 | 1.090秒 | 0.531秒 |
| 400 token/s | 153.60秒 | **1.517秒** | **0.957秒** |
| 500 token/s | 122.88秒 | 1.773秒 | 1.213秒 |
| 1000 token/s | 61.44秒 | 2.285秒 | 1.725秒 |
| decode免费 | 0秒 | **2.797秒** | **2.237秒** |

如果实际whole P=2秒，目标分别要求D≥642.60 / 2156.64 token/s；P=3秒时仅120次whole就超过两项目标。日志有短时generation≈400 token/s的窗口，但不能拿它当已认证的D，更不能假定整个批次一直维持该吞吐。

若未来实测接受长度仍约2.914，K32下D400/500/1000分别要求完整验证＋三MTP cycle约≤233/186/93ms，且未计爬坡与排空损失。这是下一次短probe可直接否证的阈值。

## 3. whole64K 到底移除哪些工作

当前源码审计基线4defe20的candidate whole_limit=32256，65100理想分为16384/16384/16384/15948。前三段之后带历史的CV成本来自真正的INT2 history attention；新bounded whole资格覆盖65100后，fresh请求context=0，已有current-only证明会令这一fresh请求的history/window CV工作为零，随后仍执行INT2写入。**数据含历史的续算、decode和MTP历史不会因此变成BF16历史。**

理想单请求、S1、GQA6、D256、sink64/recent256，四块中的跨块query-key对数（暂不扣窗口修正）为1,589,182,464，占完整causal对2,119,037,550的 **74.995%**。whole将这部分attention交给合法的原生current FIA，**没有删除这些attention数学**。

由当前C16源码及 `docs/64k_performance_source_audit.md` 第4节的有限计数，每FULL层、每rank可消除的理想旧history CV包括：

- 28023个KV256恢复单位；相应FP32 K/V GM发布13.683GiB。
- 17856个簇-KVtile的在线状态restore/spill，GM读写70.295GiB。
- 对这些history单位的OSCAR QK/PV、softmax、flags和输出逆旋转路径；相同causal配对转由current FIA完成。

这些是源码copy量与数学路径变化，不是实测HBM事务或节省秒数。实际120请求的chunk/Q/C/S轨迹未提供，不能把理想字节直接乘120、套某个带宽，再宣称整轮完成时间。

仍存在的共同工作，在模型几何假设16 target FULL/48 GDN/1 draft FULL下：

| 不因whole消失的工作 | 有限计数/边界 |
|---|---|
| target FULL current attention | 每prompt每rank，QK+PV数学约`4*causal_pairs*D*Hq*16 = 2.0831e14` FLOP；是数学计数，硬件实际tile运算可更多 |
| 首次draft FULL | 若与Q65100同形，另约`1.3019e13` FLOP/rank；真实MTP移位/裁剪shape须由短probe确认 |
| GDN预填充 | 120×65100×48 = **374,976,000个layer-token位置**；不能取消状态计算或改变GDN数学 |
| 主模型与首draft的线性/MLP路径 | 相同Q假设下120×65100×65 = **507,780,000个layer-token位置**；矩阵维度/实际TFLOP率未随本轮日志提供，不能从“27B”直接伪造算力下界 |
| FULL持久化写入 | 120×65100×17 = **132,804,000个token-head/层位置/rank**；仍要精确窗口、旋转/量化/INT2编码 |
| 长上下文decode | 完整target q4及三轮MTP依然读取65100级历史；当前的request-window布局主要省物理字节，不删除这些读/解包/QK/PV |

因此whole可以去掉明确的OSCAR重复工作，但没有证明P会从当前首输出节奏降到0.96–1.52秒。输入TTFT包含网络和排队；不能用native最小TTFT9.777秒作为不可突破的设备下界，也不能反过来把它全部算作可消除OSCAR开销。

## 4. 共享计算的限制与仍在 OSCAR 范围内的重点

当前证据无法给出GDN/MLP/通信的**无条件秒数下界**，只能给出不可取消的正工作量。若只优化FULL/OSCAR，最终未改变或无法重叠的共同计算时间必须≤T。固定工作/固定调度的极端Amdahl条件是：若其它被优化部分全部免费，共同部分在native wall的占比仍必须≤25% / 20%。实际whole/batching会改变利用率、接受长度会改变decode工作量，所以该比例只是条件约束，不能拿两条wall解出实际共同占比。

仅达到native本身的1342秒也远未达到335/268秒。更大的驻留并发能减少等待、改变batch效率，但K32客户端最多32个活跃请求；native峰值12到32在“cycle耗时完全不变”的乐观条件下也只给2.667倍并发乘数。这个条件尚未证明，且OSCAR接受长度更低，不足以独立解释4–5倍目标。

| OSCAR可控方向 | 当前源码证据与可检验问题 | 不能先宣称的收益 |
|---|---|---|
| 完整whole65100准入＋bounded arena | 合法fresh域移除上述C16恢复与state traffic；短probe同时测完整P、首MTP、真实峰值HBM与decode邻居等待 | 不将13.683/70.295GiB换成假秒数；不把CPU调度契约当NPU已通过 |
| MTP显式identity旋转专化 | `runtime._prepare_rotations`明确draft identity但hadamard=False；`rotate_clip_store.cpp:103–113`通用分支对每列做Mul+ReduceSum。D256时每K/V token有2×256个向量dot reductions，完整首draft65100约33331200次此类reduction调用（不含其它相位） | 可研究精确identity专化，仍须保持有限值检查、负零/字节、clip与quant舍入；不能改MTP算法或旋转语义。此为源码机会，未引用旧实验耗时 |
| q4/q1历史解包与GM交换 | `attention_cv_decode_bundle.cpp:214–237,423`仍按KV256产生FP32 K/V再QK/PV；每token packed约136B，而发布FP32 K/V约2048B，另有score/P/PV。需新短profile给出unpack、发布、等待、Cube与GDN/MLP/通信时间 | 不能把INT2 7.53倍编码比当全历史device流量比；片上/L2命中未知，源码GM字节不等于DRAM字节 |
| MTP三轮与验证batch效率 | 本轮请求级事件可识别，完整K32 cycle未知；q4/q1/C16等路由已存在，N128/B32并非旧16K padding场景 | 不重复宣称已消除的padding收益，不关闭MTP，不降低精度或隐藏拒绝 |
| wrapper/prepare/status/量化写入融合 | 可以在外部边界减少自身重复metadata与launch；native GDN仍执行原有运算 | 尚无本轮host/device归因，不能把小相位优化算成整轮大幅收益 |

最快的反证顺序：先测一个无观测开销的whole65100完整服务P，再测context65100/K32的稳定q4＋三MTP cycle与实际接受长度；若P已超过2.797/2.237秒，上述“一输入一whole pass”的分时方案即使decode免费也不够。只有P与D进入表中范围，才值得继续用短mixed probe检查额外U、内存和队尾延迟。**不要求重跑已完成AISBench；没有这些短证据也不承诺4–5倍。**

## 5. 可复算产物

### 已排除：原生每步把整池非连续KV复制成连续布局

本轮复核`attention_v1.py:1175–1232`与`device/device_op.py:55–101`，A2普通FIA包装确实调用`key.contiguous()/value.contiguous()`；但当前目标对应的hybrid FULL reshape在`model_runner_v1.py:4489–4498,4567–4577,4634–4640`将conv前缀之后的K/V分别切成连续SoA张量，再用`view`成kernel128视图。因此不能仅凭`contiguous()`字样，就把它当成每层整池复制或原生22.4分钟的可消除成本。padded `as_strided`分支对应cache-only/extract-hidden-state spec，不是本目标FULL层。没有因此修改原生源码或新增优化。

`tools/analyze_64k_latency_budget.py`仅读取本轮evidence JSON与原文hash，输出 `reports/latency_budget_64k_20261009.json`，包含每个counter窗口整数反解、首输出credit bins、排空边界、目标预算和静态工作量。执行：

```bash
python3 -m tools.analyze_64k_latency_budget
```

这次命令成功完成离线数据核算；没有生成或执行性能模拟，没有修改生产源码、CANN、GDN、默认配置或验收门槛。GPU/图/精度/完整服务/实际268–336秒目标均未由本报告验证。
