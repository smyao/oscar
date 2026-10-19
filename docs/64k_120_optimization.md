# 64K / 120 请求 / K32：性能与显存优化记录

**2026-10-19 离线补证**：当前配置APC关闭、16384预算，以现有形状和相同净KV预算计算，64K+512输出的驻留BS为native11/OSCAR31。实际源码计数与1920点条件模型见[本轮实验](batch_profiling_20261019.md)：所选参数下扩大OSCAR batch约+20.1%，相对原生1.177倍；临时GM等效HBM比例改为50%即0.857倍。没有当前设备时间/接受长度校准，不能据此宣称120请求整轮达到22.4分钟的某个倍率；未重跑AISBench，未改变候选或提升验收状态。本文下方旧开启APC的目标描述不覆盖当前分支配置。

> 本文是[优化总 MD](../skills/oscar-quant-optimization/SKILL.md)下的本任务进展记录；总 MD 负责统一流程，[roofline 子 MD](quant_dequant_roofline_20261011.md)展开组合收益模型，[台账](optimization_ledger.md)汇集实际证据。

档案依据：#26–49/#69/#127–129/#137/#140/#142/#145/#148/#151–155、启动文档 D.4。工作从当前 `4defe20` 开始。本文随实现更新，不能将计划、字节推导或本机模拟写成真机通过。

**2026-10-12 当前目标**：单项完整相关路径相对当前基底至少1.15倍，精度/协议合格后可入组合；组合相对原生1342.4437259秒达到2–4倍，即335.61093148–671.22186295秒。默认prefix caching开启。旧2–3倍与none布局估算保留历史身份；不能拼乘单项或把helper收益当整轮通过。

**2026-10-10 历史验收目标（已更新）**：用户接受64K整轮相对原生提速2–3倍，并要求不同长度、并发和阶段通用。以已提供native1342.4437259秒为分母，目标为671.22186295–447.48124197秒（约11.19–7.46分钟）；此前4–5倍为历史目标。省略token-budget参数、FULL KV 6–7倍、原生GDN与冻结数值要求均保持。局部writer1.796×和unpack2.346×仍不能直接当作整轮预测。

## 2026-10-11 明确的组合方式与提速原理

**采用组合方式提高速度**：量化写入、反量化/数据布局/历史复用、紧凑FULL KV与任务执行效率共同推进，以合并版本的实际净收益裁决。过小的独立增量保留历史，不装进当前组合；同一成本只减少一次，不拼乘局部倍率。

64K decode的主机制是：INT2降低实际KV访存量和占用 → 可容纳更多请求 → 在调度和prefill供给允许时形成更大的实际decode batch → 在原生与OSCAR均访存受限、且OSCAR反量化与MTP未成为计算瓶颈的条件下，减少每个有效输出token的时间。**压缩带来的容量不自动等于实际batch，更不自动等于吞吐倍率**。必须包含weights、精确窗口、临时FP32数据、量化/反量化、MTP各轮、通信/同步与有效输出数量，并区分HBM、L2和不同计算单元的限制。

完整量化账及证据边界见[组合优化与roofline](quant_dequant_roofline_20261011.md)。这是本轮设计和验证的主要机制；不能靠增加无用访存把算子人为变成memory-bound，也不能把稳态decode倍率直接当成含prefill和排空的整轮倍率。

## 最新约束：全部服务使用原生 token budget（2026-10-09）

用户明确要求删去 `--max-num-batched-tokens`，此前的显存、速度目标不变。两个固定 candidate 入口及默认配置现在均省略该参数；不指定2048，不通过OSCAR调度器、history容量或隐藏配置提高原生预算。原生引擎解析的有效budget/capture容量仅用于预先分配工作区。

此前whole32768/whole66048方案已退出所有服务配置。下文保留的整段64K预填充收益、881MiB工作区及其额外激活预测属于**被本约束取代的历史方案**，不再用于当前速度预测。需要在原生chunked prefill下优化C4/C16历史计算与q4/q1 decode；不能继续计入“移除全部历史chunk”的收益。

请求独立精确窗口/双段INT2页的格式与6–7倍FULL KV目标继续保留，但生命周期已改用原生Scheduler/AsyncScheduler通知接缝，预算和队列决策不变。此研究优化尚未在默认candidate启用。FULL KV字节比与全部KV字节比不依赖whole准入；整卡容量必须以新的实际工作区和峰值重新核算。

关键主机证据：两个入口实际effective-config/exec与observer服务边界验证，原生预算2048/8192/16384下调度输出对照、forced reset/lease/epoch、实际引擎容量绑定，以及省略budget后独立算子和服务probe兼容。NPU、graph、整轮速度/显存仍未验收。

### 原生小分块对优化方向的影响

以下是条件源码账，不是把2048写回引擎：假设单个65100-token prompt每步独占全部有效预算、无prefix复用，current因果对数为`sum(chunk*(chunk+1)/2)`，其余为跨chunk历史对。混合decode会占用预算，实际chunk数可能更多。按Hq6/Hkv1/D256、20 Cube、C16 arena与capture≤budget，用当前`WorkspaceGeometry`复算：

| 假设引擎最终预算 | 理想prefill chunk数 | 跨chunk历史对占比 | OSCAR固定workspace/卡 |
| ---: | ---: | ---: | ---: |
| 2048 | 32 | 96.869% | 164.431 MiB |
| 4096 | 16 | 93.744% | 233.548 MiB |
| 8192 | 8 | 87.495% | 371.782 MiB |
| 16384 | 4 | 74.995% | 648.251 MiB |

因此本轮重心包含**小query chunk对长历史的C4/C16计算**，不能只优化纯decode或沿用whole64K移除历史工作的预测。以上占比是attention pair数量，不是wall占比；固定workspace下降也不是等比例提速。旧条件pool推导尚需按实际native预算重新计算整卡峰值，不把这张表当NPU显存验收。

## 本轮范围与目标

2026-10-09 用户新增优先级：重点压缩 OSCAR 量化/反量化及必要搬运成本。当前真实路径、待替换的标量打包、C4 16→32行方案，以及已完成的 Hadamard 索引向量化局部证据，集中记录于 `docs/quantization_hotpath_20261009.md`。元数据原型7.11×不作为完整写入、注意力或整轮收益。

- 只使用本轮 65,100 输入、512 输出、120 请求、客户端并发上限32的证据。LongBenchv2 后续仅作精度测试，不再用于性能研究。
- 原文：`reports/target_64k_120_20261009.txt`；SHA256 `03eb53109a97a3e82b02c91550b5661dc8cf0c3360585693f567a3217c7311bb`。用户提供数据，未生成或索取“真实数据集”。
- 保持 INT2 语义、冻结精度、原生 GDN 计算、MTP状态和外部插件边界；最终仍通过两个 candidate 一键入口交付。
- **用户已明确显存6–7倍以FULL注意力KV为目标，同时报告全部KV。** FULL KV、FULL+GDN/MTP KV、固定workspace与整卡HBM分开列账；并发峰值不能替代字节比，也不能推出等比例提速。
- **用户已明确速度分母为原生22.4分钟，最新接受2–4倍**：整轮目标335.61–671.22秒，相对历史当前OSCAR约需3.68–7.35倍改善。旧268.49–335.61秒属于已替换的4–5倍目标。

## TODO（按依赖推进）

- [x] T01：原样保存全部新日志，解析客户端表格与服务端样本，保留缺失和暖机边界。
- [x] T02：从指定原生源码独立推导物理页/共享池/GDN状态/workspace字节账，并验证公式。
- [x] T03：确认65100请求的当前准入、current-only、mixed、MTP、C16真实路径和工作放大。
- [ ] T04：建立可复现的64K调度/内存/工作量模拟；报告它能复现与不能复现实测的项目，固定参数后再评价改动。
- [x] T05：工作区按原生最终token/graph容量分配；此前大current/小history方案保留历史记录，不用于当前交付。
- [x] T06：删除OSCAR自设32K/64K预算与整段准入策略，使用原生调度；主机验证保持chunk、异步、抢占/恢复结果，真机接缝另验。
- [ ] T07：实现紧凑FULL页池及稳定请求身份的精确S/R/MTP窗口，消除按所有历史页重复付费。
- [ ] T08：验证FULL/GDN分配接缝与真实字节预算；必要时去除分配padding，保持GDN原始状态及计算。
- [ ] T09：校验MTP接受率下降的工作量影响；精度与性能分开，不能拿接受率改动充当无损优化。
- [ ] T10：冻结数值/非法输入/生命周期/跨页/释放复用/图检查；Lima CPU-debug与CAModel必须排他串行。
- [ ] T11：复算改后容量、整轮时间及长尾，标记是否达到相应目标；不足则继续结构优化。
- [ ] T12：把合格实现接入两个一键入口，保留精简关键日志与失败清理，完成本MD后交付真机认证。

## 1. 已固定的实测基线

| 指标 | 当前OSCAR | 原生 |
| --- | ---: | ---: |
| Benchmark Duration | 2467.1275737 s | 1342.4437259 s |
| 成功 / 总请求 | 120 / 120 | 120 / 120 |
| 总输入 / 总输出token | 7,812,000 / 61,440 | 7,812,000 / 61,440 |
| 平均TTFT | 208.9315 s | 230.3687 s |
| 平均E2E | 649.3244 s | 335.9201 s |
| 平均TPOT | 861.8 ms | 206.6 ms |
| P99 ITL | 13.4451 s | 2.8731 s |
| 服务端Running采样峰值 | 27 | 12 |

两侧工作量相同；OSCAR墙钟为原生1.838倍，平均首token后的时间为440.3929/105.5514=4.172倍。服务端可见峰值实际为27，不能用用户粗估26替代解析值。平均TTFT较好与整体吞吐较差同时成立。

服务端完整已打印窗口（包含暖机，缺失不补）的MTP接受率为63.7857%/92.9660%，加权平均接受长度2.91357/3.78898；drafted/3为请求验证轮数，不是引擎step。OSCAR 02:44:10 Spec后半缺失、原生03:15:32附近有断片，均已保留为未知。

必须复现的现象：首Engine样本至Running峰值490/110秒；OSCAR180/247活跃采样generation≤10tok/s，原生7/135；OSCAR19个双吞吐为零且Running>0的样本；短时高吞吐峰值409.9/355.8tok/s；等待归零后的60/20秒排空段。它们是logger采样现象，不是设备相位占比。prompt统计在首输出报送，prompt=0不证明pure decode；HTTP200不证明请求完成。

**复算与本机证据**：`python3 -m tools.analyze_64k_evidence`，8项解析测试通过；`reports/evidence_64k_120_20261009.json`。这里只完成实测解析，尚未建立改后全服务时间模拟。

## 2. 显存问题：字节去向已确认

详见 `reports/memory_64k_analysis_20261009.md` 和对应JSON。具体字节使用报告中明确标注的模型几何假设，不冒充本轮设备分配快照。

当前D256/Hkv1每token编码136B，裸格式比为1024/136=7.529倍；实际FULL页却仍是原生817152B，仅其中382976B用于2816个INT2 token。每页另预留333336B精确BF16窗口/tags，占整页40.79%，而非每个请求仅保留一份窗口。

输出末每请求/每rank：原生FULL1139.332MiB、GDN统一池158.977MiB；OSCAR FULL317.953MiB、GDN统一池158.977MiB。全部KV条件字节比2.722倍，与“不是6–7倍容量”方向相符；尚未用未知KV预算强行拟合27/12峰值。

候选N32768/C16固定workspace为1201.189MiB/rank。原生KV profiling已计入它，不能使用profiled KV预算时再重复扣除。

**当前采用结构**：保留原生单一共享页池和全部GDN状态；FULL自己的物理ID在SSM、padding两个SoA区域各存2816个INT2 token，总物理容量5632。精确窗口移至稳定请求身份的sidecar，每FULL层按配置128槽预留323行。双段地址、native spec/metadata128复制和stable lease已有主机实现；新读写核正做独立数值资格，生产候选尚未启用。完整协议见`docs/compact_pool_64k_design.md` §11–12。

按128槽在K32下摊销，FULL对原生FULL物理字节比约6.309倍（对逻辑BF16 payload约6.032倍）；全部KV含原生GDN状态为3.823倍。保留全部GDN/MTP状态时，进一步去全部分配padding的理想全部KV结构比约5.70倍；不能将FULL比值写成全部KV或整卡收益。

**条件容量预测**：`python3 -m tools.forecast_64k_memory`与11项边界回归。仅用暖机单请求3.7%反推的旧usable pool区间960–986块，每块17×817152B；sidecar增加691.737MiB，workspace回收320.221MiB，另计17层status及共用device map共10752B，对已profile预算只扣净增371.527MiB。额外native activation/graph峰值分别增加0/512MiB/1GiB/2GiB时，完整请求条件容量为38–39/37–38/35–36/32–33（客户端上限仍32）。K32允许未知额外峰值约2172–2530MiB。模型几何、暖机无其他占用、GDN四状态均为显式条件；这不是实测pool或实际峰值，不据此推断速度。

## 3. 性能问题：64K没有走整段路径

源码证据：`integration/whole_prefill_scheduler.py`、`tools/serving_variants.py`；细节见 `docs/64k_performance_source_audit.md`。

当前整段上限32768−128×4=32256；65100请求始终不合格，仍按16K分块，但32K workspace已分配。仅将预算改成65536仍不足（上限65024，比65100少76），并且直接扩大到65664将workspace推到2311.384MiB/rank，不能忽略对容量的反作用。

理想65100=3×16384+15948、无holes/S1时，后三个有历史的chunk：每FULL层/rank共28023个KV256恢复单位，FP32 K/V发布约13.683GiB，C16簇内状态读写约70.295GiB。C16确实共享解包，但每个成员仍做QK/PV/softmax和状态搬运。这是源码工作量；不等于真实HBM流量或实测耗时。

**当前改进方向**：联合设计whole-64K准入与有界history arena，避免用历史CV处理本可一次完成的初始prefill；保留长请求/历史续写的真实INT2路径。N128常规MTP不属于大padding裁剪条件；不重复计入已不存在的16K padding收益。

**已实现、未启用**：input66048/history16896的GraphWorkspace把总固定工作区1201.189MiB降到880.968MiB。BoundedWholePromptAsyncScheduler由原生AsyncScheduler执行队列、分配、抢占；含历史prefill的整步预算受16896约束，fresh65100可与31个q4同批。主机回归直接执行指定native完整schedule与AsyncScheduler，覆盖抢占/恢复/forced reset/旧异步结果；43项与10个subtests通过。EC/KV connector、动态chunk/batch覆盖scheduler和带历史重回WAITING的未资格streaming场景显式拒绝。正常大profile的metadata=None路径与捕图≤512已源审；不为超容量forced dummy伪造成功。

## 4. 改动与仿真结果表（持续填入真实结果）

| 问题 | 当前模拟/源码是否匹配实测 | 实际改动 | 改后证据 | 达标状态 |
| --- | --- | --- | --- | --- |
| 每页精确窗口、统一池浪费 | 当前2.722×全部KV字节账；暖机区间对驻留峰值方向相符，实际预算仍未知 | 双段INT2页＋128 stable request windows | 条件FULL6.309×、全部KV3.823×；新读写数值资格中 | FULL字节目标条件满足，尚待完整资格与真机字节核实；全部KV仅报告 |
| 64K整段路径未触发 | 源码明确阈值不足，与分块执行相符；未宣称唯一根因 | input66048/history16896，独立容量校验 | 固定workspace1201→881MiB，主机容量测试通过 | NPU/端到端速度未验 |
| 历史chunk拖慢decode/ITL | 源码确认重复历史和状态流量；缺少本轮device相位时间 | whole64K准入与历史总预算策略已实现、未启用 | native调度/async/抢占等43项主机回归＋10subtests | 未量化整轮速度 |
| MTP接受率差异 | 完整打印窗口的差异已复算，不能单独解释4.17×TPOT | 仅分析，未改变量化/采样语义 | 未完成 | 未达标 |

本轮尚无新真实NPU、图、整轮速度或显存验收结果。所有“不足”“未完成”和失败数据保留；不以编译通过、模拟局部倍数或拟合总时长冒充目标达成。

## 5. 当前本机资格进展（不是交付完成声明）

- 新双段写入：CANN编译通过；官方CPU-debug 19/19例通过，包含MTP拒绝0–3、lease/epoch复用、真实32 live/128 padding、非法qstarts和全padding未发布map。报告`reports/request_windows_store_20261009/cpu-suite/cpu_cases_report.json`。
- q4/q1新读路径：CANN编译通过；官方CPU-debug 11/11例通过。正常输入逐位对照＋独立oracle；新地址/epoch非法域分别验证明确错误状态，不把错误域称为正常数值oracle通过。报告`reports/request_windows_attention_20261009/cpu-suite/cpu_cases_report.json`。
- C16：Q336/C65/R1的一簇双臂用例曾在120秒期限处终止；原失败保留。随后复用完整old输出、再次验证任务/oracle/状态/guards，候选在原120秒限制内完成；segment0/1两例均通过所有partial/LSE/status/counters逐位及source0+1独立oracle。第二例先完成独立基线再验证候选。此处不测试source2数学，不作速度证据。
- 外部集成：稳定身份map在原生input-prep event范围内、condense/reorder之后发布，保留deferred corrections；sidecar在model-load profiler范围内分配，保持原生GDN spec/页池/状态。native真实manager的扩容、释放及5632→metadata128复制主机回归通过。
- 真NPU writer/六种reader/固定地址改输入graph门已实现并连接，当前未运行；直接serve仍保留无probe语义，最终observe必须在这些门通过后才起服务。
- 完整构建：新七份独立kernel与两个Torch binding、36项capability登记已完成；Lima CANN9.1/ascend910b4全库与扩展成功链接，signature=`d83607ded865b9a7e2afbf2d325565efd847c0c6822e616c40b9a2aba1903e8e`。此前8路C++编译进程被杀的两份日志保留，改用显式`--jobs 2`完成；默认真机编译设置未改。证据`reports/request_window_fullbuild_20261009/`。
- 关键集成回归：234项＋16个subtests通过；官方CPU数值合计32例（19写＋11decode读＋2 C16读）。源身份打包检查、真实Torch schema解析、native scheduler/manager及一键显式研究配置均检查；这些不冒充真实NPU或整轮性能。
- 当前candidate默认preset尚未切换，未push。汇总`reports/restructure_64k_progress_20261009.json`，后续完成时更新。

## 6. 为速度目标淘汰的错误方向

详见`docs/64k_latency_budget.md`、`docs/64k_activation_peak_audit.md`、`docs/64k_decode_transfer_audit.md`。

- q4/q1 MMAD实际M为24/6，没有M64/M128逻辑计算浪费；旧固定shape实验只有1.01586倍，不能再次包装成大优化。
- packed DMA是二维DMA，不是64次独立提交；旧SIMD标记里的该阶段仅占history约3%，且不是当前bundle，不据此新增burst/Gather改动。
- 指定native树中直接UB→L1的实例属于arch35；A2实例仍走GM桥接。480KiB L1生存账不足以证明A2传递及冻结FP32计算可用，先做最小资格验证的计划，未重写attention。
- 当前bundle已合入批量窗口读取、精确range mask、history流水和544B natural stride。旧大头占比不能移用；同源normal/profile诊断原型已编译并完成首个上下文的数值和扰动对照，完整相位采集仍未完成，不预填残余占比。
- whole66048新增native激活峰值不能取零。源码的条件范围已接近K32可承受的额外峰值余量，完整runtime峰值仍需短probe核实；不同阶段内存不能相加当并发峰值。

## 7. 当前bundle残余定位与接缝修复

- 首个C1025、R32/N128/S3、20核完整q4 CAModel正常/插桩对照完成：normal=1229822、profile=1242723 task ticks，插桩扰动+1.049%，低于预先5%研究门；partial/LSE/status逐位一致，两臂独立oracle通过。完整输出与哈希见`reports/request_window_decode_profile_20261009/`。
- 三类actor分别核算的q4 history/window/current KV单元均为96/96/64，q1为96/96/32，与全部任务账本一致，没有过滤任务。SYS_CNT为零，不能给相位毫秒或占比；完整marker采集继续，尚不评价新优化，更不把插桩比当加速。
- q1同形CAModel完整配对完成：normal=1033238、profile=1065947 task ticks，插桩扰动+3.166%，通过原先5%门；partial/LSE/status逐位一致，两臂独立oracle通过。output/LSE最大绝对误差分别约4.82e-7/5.92e-7。q1/q4的有效行数、当前源任务数与owner调度不同，不能用两个tick数反解固定成本占比。四臂均在原600秒期限内完成，汇总`reports/request_window_decode_profile_20261009/first_context_summary.json`。
- C1025的每个历史split仅1个KV256块，未进入`hasNext`连续预取；它只完成当前数值/ABI/插桩扰动资格。下一点必须覆盖首/中/末迭代（C2049每split3块，C4097每split5块），仍保持R32/N128/S3/20核，按实际工作量与资源预算选择，不能把C1025占比外推64K。
- 下一点已选最小C2049，不运行C4097。先运行q4/q1 normal，每臂运行前固定900秒上限：总KV单元从256/224增至448/416，首点实际host时间351/291秒；新预算用于新工作量，不修改首点的600秒门。尚无下一点结果，不将静态工作量比当作耗时比。
- C2049首个q4 normal在对话中断后仍有VM进程，已接管原进程观察至退出，没有重复启动。它最终缺少completion、数值输出和完整task profile；局部subtask记录不能当作成功或性能证据，原目录与中断记录已归档。后续将监管与stdout/stderr文件放在VM本地，保持预先900秒期限，防止SSH/对话生命周期切断结果；本次退出的具体原因未确认。
- 发现并修复入口问题：省略CLI variant而显式配置经规范化改变budget/scheduler时，直启曾仍读旧文件；现在实际服务使用同一effective配置，避免“计划显示66048、服务仍读16384”。默认候选未改变。
- 发现并修复报告门缺口：六reader报告现在必须匹配case_plan声明的shape、实际core、输入SHA；observe还核对本次冻结acceptance SHA及物理设备选择。主probe只查询一次实际core并写入报告。59项相关主机回归通过，不能据此补写NPU通过。
- 生命周期复核未发现当前支持域内可指认的跨请求污染、MTP回滚或GDN页重叠新问题；此结论不替代目标图/整服务验证。

## 8. 原生预算下的下一步定位（持续更新）

- 当前dispatch明确在小于8192的prefill query上选择C4，2048预算不会自动进入C16。C4的正常矩阵/softmax/状态工作必须单独测量，不以decode比例替代。
- C1025/q4/R32/N128/S3/20核已取得完整60 actor的MOV相位记录：task=1,243,728 ticks，相对同profile核先前1,242,723仅增加0.0809%；partial/LSE/status逐位一致，独立oracle、所有actor闭合/调用计数和5%日志扰动门通过。来源`reports/request_window_decode_completion_20261009/`。这一点每个history split只有一块，不能推出长历史流水、C4或整轮加速。
- 此短点core0/AIV0的history span=528,798 cycles（五个任务），unpack=46,527、softmax=55,391；其余包括实际等待、query阶段和收尾。它已否定“这个短形的unpack占大多数”的判断，但不能证明长形也一样。全20核分布见`phase_summary.json`，未把actor时长相加成wall。
- FIFO日志方式在SDK初始化时等待，尚未进入kernel，已停止并完整保留失败、确认清理。改用本次显式completion-only普通文件采样：每actor最多64MiB，默认旧cap不改、popped丢弃、完整读取后回收raw；成功记录不覆盖FIFO失败。
- C4诊断使用当前生产源的可逆normal/profile克隆，独立schema32补充spill/restore；Q1344/R1/C769/20核代表mixed步骤中的prefill分区，source2按生产抑制，仍保留完整task表。源码身份检查通过，正在串行CANN/metadata工具编译；数值、插桩扰动和相位结果尚未获得。
- 后续C4正常/插桩库与metadata导出器已编译通过；真实prepare导出4032个任务，确认13成熟簇、history共享72个KV256、window67个、source2健康任务全部end=begin。加入独立field31定位原有限性归约，正常源码不变。Q1344 normal在预先900秒期限处终止，无completion/数值输出；清理完成，子进程在清理中返回-11，不能据此认定正常计算发生数值错误。证据`reports/c4_phase_profile_c769_20261009/summary.json`。不加时、不采用局部tick；下个有限诊断点拟为Q336，保留真实C4和全部任务，不作为饱和K32或整轮性能结果。
- Q336正常臂随后完成：2,543,951 ticks、host538.09秒，全部1008任务/输入guards/status/真实cluster统计及所有query的source0+1独立oracle通过，最大output/LSE误差约1.065e-6/7.738e-7。它是低占用阶段定位点，不代表Q2048/K32饱和性能。配对profile保持同库/输入/配置及900秒期限；首次在kernel启动前因tmpfs不足4.5GiB而拒绝，校验本地完整副本后仅回收四份旧workspace共91,750,400B，原资源门未降低、失败保留。新profile正在运行，阶段占比和扰动尚未给结论。
- 单纯把当前FP32 Q/K/P/V改BF16已被有限值宿主反例否定：输出误差0.09267，超过冻结容差0.00796；见`reports/single_bf16_cast_counterexample_20261009.json`。原PR的decode和prefill舍入位置不同，不能以“累加仍FP32”替代数值证明，当前内核数学未改。
- 原ReduceSum的逐行内部同步不能靠延后外层Fence消除，直接换二维归约也未证明错误契约等价。一个待测候选是整数high16充分条件：证明所有原score求和有限时才省掉原检查，阈值外仍执行原路径；完整证明、位级API证据及收益淘汰门见`docs/cv_safe_finite_range.md`。未写入生产核、未声称提速。
- Q336 profile已完成数值和SDK计时：2,660,850 ticks（相对normal +4.595%），所有output/LSE/status/clusterStats逐位及独立oracle通过。只有core3两个AIV的日志在最后写回阶段达到64MiB上限，完整marker门仍为失败；没有放宽门或补写丢失区间。全部finite_reduce叶子已闭合、调用数对真实tasks/profile逐actor核实，core3两AIV的history分别140,861/133,275 cycles，window9,545/9,362；这不足以支持1.5倍整核，故淘汰safe-range单点，不实现。
- C4完整循环子范围已独立提取，所有开放父区间及截断尾部保留未知。第二个full256回合观测390,694 cycles；跨member流水条件预算270,443（1.445倍），再加固定有界next-KV预取预算216,547（1.804倍稳态）。新增状态流量、实际数据ready/ack延迟和12.5MiB/20core临时区已计入。短Q336整核仅约1.226倍，不是达标结论。详见`docs/c4_cross_member_pipeline_budget.md`；下一步评估原生Q2048/Q1924实际两遍分配的关键核，不能将簇内倍数当完整核或模型倍数。
- 另一个待资格路线是Cube PV以FP32 atomic-add直接更新已写好的alpha-state，省掉额外状态往返。当前A2源码支持完整K256后才写C，已有原生F32 Fixpipe先例，但与Vector Add在tie/subnormal/±0等上的逐位语义还没有实证；先做独立最小算术门，未改生产路径、未给它虚构零成本。
- 该最小算术门随后已完成：anti-K128、random、ties、subnormal/±0、nonfinite共五组，官方CPU-debug和CAModel均逐位通过，active/padding/NaN payload位差均0，输入/guards/三actor状态完整。每次paired CAModel总74,698 ticks只是两路径合并资格测试，不能当成atomic单路径提速。证据`reports/c4_atomic_state_qualification_20261009.json`。原型还未验证AIV收到AtomicDone后立即MTE2读取更新状态，正在独立v2补此接缝；真实NPU、图与完整C4多tile语义仍未验。
- 默认小预算下的完整history分配已进一步核算，比较分母固定为当前实际未均衡代码：Q2048的联合流水/均衡条件预算约2.107倍，Q1924约1.878倍；新GPU planner和尾/固定成本仍未知。详见`docs/c4_complete_partition_budget.md`，没有把簇内预算换写成模型4–5倍。
