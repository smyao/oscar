# 完成状态与证据

本表保留启动文档稳定ID。`passed`仅限该项声明范围；本项目整体未完成。CPU测试、Python wheel、CANN编译、设备完成、图捕获、图回放、性能分别列出。所有源码当前在本地分支 `codex/oscar-ascend`，远端为用户指定GitCode仓库；推送状态以最终Git提交/远端校验为准。

| ID | 状态 | 要求 | 本轮证据/缺口 |
|---|---|---|---|
| [x] A01 | passed | 建立 reference 清单，锁定 OSCAR PR 与相关 vLLM/Ascend commit，确认未参考失败仓库。 | docs/reference_manifest.md；reports/local_environment.json；两棵native树HEAD/status和全参考文件指纹。PR/paper为提供的快照pin，无独立git历史。 |
| [ ] A02 | in_progress | 记录实际软件、NPU、CANN/编译环境、模型配置和原生源码状态。 | macOS与Lima VM环境已记录；VM CANN9.1/Torch2.12 CPU，非目标Torch2.10/NPU。用户要求移除真机启动环境审计，实际目标调用结果由探针记录；该项不作为部署前置闸。 |
| [ ] A03 | in_progress | 跑通原生目标配置的基线，保存命令、输入、版本、MTP/图模式与资源记录。 | 用户提供了2026-09-07原生同负载服务片段（档案#137），但本轮版本、完整命令、逐请求输入和计时证据未配对核实；不可作为当前正式基线。 |
| [x] A04 | passed | 读懂第 3.3 节调用链，记录 FULL/GDN、TP、KV 分组、MTP 和图模式的真实入口。 | docs/runtime_implementation.md；tests/test_runtime_native.py执行真实native分组、manager、reshape、MTP/图builder方法；目标worker执行另列D08。 |
| [ ] A05 | in_progress | 核实第 4 节全部歧义，给出层/组/Tensor/地址映射与字节来源。 | 动态C/M/P与B2304公式、真实native config/形状函数已验证；MTP3示例C30720/P817152；#127补齐prefix关闭的none模式，Bfull2816、GDN请求块262144，真实页初始化复验待回传。 |
| [x] A06 | passed | 读取 PR 的配置、旋转、量化和精度验证方法，完善本 Checklist，列出仍缺失的输入。 | docs/semantics.md；ops/reference.py、rotations.py与独立PR语义测试。 |
| [ ] A07 | in_progress | 任务开始时已把 `issue1_full_record.md`（含第三部分 [74]~[86]）完整读一遍并掌握其查表用法（症状速查表 → 条目跳转），并将其中与当前任务相关的故障模式登记进第 8.2 节回归检查；确认不会参考任何失败 OSCAR 实现的代码（第 0.2 节红线）。 | 已阅读启动文档全约束、D.4全文、档案速查/总索引与相关条目；子代理分别查读各子系统。未把26,251行档案逐字通读签成完成。 |
| [ ] B01 | in_progress | 产出 `docs/design.md`，列出外部新增文件/函数/类及其包装的原生入口。**结构必须符合第 5.0 节**：全局端到端架构（生命周期图/联动矩阵/时间线/两本账）先行，逐点展开固定六问，自审清单逐项打勾；碎片化设计视为 B 阶段未完成。 | 全局设计/矩阵/时间线/两本账/差异与D.4四问齐全；四条性质中的真实性能与H18仍未过。 |
| [ ] B02 | in_progress | 完成请求窗口、物理页、GDN 隔离、TP 分片、页生命周期和 MTP 提交/回滚设计。 | canonical物理页snapshot、MTP slot-derived位置、prefix/回滚方案已实现；真实native方法与CANN CPU-debug通过，NPU生命周期未运行。 |
| [ ] B03 | in_progress | 给出所有 payload、元数据、padding、workspace 和总 HBM 公式，证明压缩收益可落地。 | 真实native allocator/grouping/LCM/hash/manager与动态字节账已验证；实际NPU容量收益仍未测。 |
| [x] B04 | passed | 完成算子清单、AscendC 融合划分、Decode Stage1 伪代码和逐阶段读写流程。 | 七个AscendC符号完整源码、ABI、Cube/Vector/同步/UB与GM预算；docs/operator_inventory.md、cv_implementation.md、rotation_pipeline.md。 |
| [ ] B05 | blocked | 给出 decode 计算量、HBM 流量、MTP 跨 query 复用和 Host 调度预算。 | H18总历史读取亚线性与精确dense attention存在数学冲突，已提出澄清。 |
| [ ] B06 | in_progress | 固定功能、精度、性能测试方法和验收门槛，设计一键脚本及错误清理流程。 | PR容差未改，部署/故障处理/测量与profiler工具已实现；模型级质量门槛仍待实测前定义。 |
| [ ] C01 | in_progress | 独立插件/算子工程可安装、可编译、可加载，原生源码未修改，安装脚本不使用 `cp`。 | Python包及CANN9.1/910B4交叉编译/链接通过；独立direct-launch库采用源签名SONAME，非custom OPP vendor。node93后续构建及75项store/merge原语NPU执行通过，确认已越过#125加载故障；后续已完成TP4/MTP编译及KV预算，后续已通过短请求，长请求无进展（#129）。 |
| [ ] C02 | in_progress | 版本能力 probe、延迟注册和幂等 Hook 通过 CLI、API server、各 worker 初始化验证。 | 轻量可撤销hook、factory、binding、真实native类/方法契约通过；实际TP4 worker已走到缓存分配后的MTP metadata初始化；#128修复显式虚拟块副本，后续启动待真机确认。 |
| [ ] C03 | in_progress | 实现并验证 INT2 编码、旋转/裁剪/量化、KV store 与 Recent→History 增量迁移。 | AscendC rotate/clip/quant/pack/raw snapshot已实现，官方CPU-debug通过；node93的store原语及非法值/边界NPU用例通过，融合旋转/裁剪路径仍未验收。 |
| [ ] C04 | in_progress | 实现真实 fused INT2 CV Decode Stage1（CV=Cube/Vector，见 H07 注），并完成必要的分块/分段输出合并。 | 真正Cube QK/PV/逆旋转与Vector SIMD解包/softmax已编译、CPU-debug通过；node93首个CV用例D64/Q1/context17输出超差（#126）。已定位并修订query有效行→padding的MTE3→V同步缺口，目标复验/性能未完成，H18冲突保留。 |
| [ ] C05 | in_progress | Sink/Recent Attention、History Attention 和参考数学语义一致，数值稳定。 | 三source causal/GQA与稳定LSE真实CPU-debug通过；node93 merge原语NPU用例通过，但CV首用例输出超差，其LSE断言尚未到达（#126）；完整数值验收未通过。 |
| [ ] C06 | in_progress | 实现原生页表兼容的物理分配与视图，GDN 不受影响，HBM 中无冗余完整历史副本。 | 真实native grouping、BlockPool、Full/Mamba managers与GDN reshape执行通过；production raw分配与view已接通；#127修正none模式容量，保留GDN原对象、分组、原生状态数；最新真机已通过KV容量计算并进入分配后MTP初始化；#128元数据副本修复后需继续复验。 |
| [ ] C07 | in_progress | 在确有需要的路径提供有界恢复能力，证明 decode/verify 不走全历史恢复。 | prefix直接保留有界canonical BF16 snapshot，无全历史恢复函数；设备tag不匹配显式错误，CPU-debug故障注入通过。 |
| [ ] D01 | in_progress | 首次 prefill 正确，未引入不必要的重复压缩和恢复。 | current chunk直接精确attention后仅压缩新KV，AscendC CPU-debug通过；真实模型首次prefill未测。 |
| [ ] D02 | in_progress | chunked prefill 正确，已处理历史不随每个新 chunk 重搬运或重压缩。 | 旧history在CV tile内消费，无旧chunk重复quant或完整恢复；真实模型chunked prefill未测。 |
| [ ] D03 | in_progress | 普通 decode 持续进入压缩历史路径，无全历史 BF16 HBM 恢复。 | INT2 fused路径与静态shape自适应split已实现并CPU-debug验证；真实模型decode未测。 |
| [ ] D04 | in_progress | MTP draft/verify 与原生流程一致，完整接受、部分接受、完全拒绝后的缓存和 GDN 状态正确。 | 实际native MTP拒绝语义已复现并修正：后续draft按slot/无序BT恢复逻辑位置；CPU回归与CANN CPU-debug通过，NPU未测。 |
| [ ] D05 | in_progress | MTP 多 query 复用历史 tile，读取量测量支持“不随 `q_len` 线性倍增”。 | Mq64容纳GQA6×MTP4；split历史分区互斥、真实多split CPU对拍通过。NPU读取流量待profiler。 |
| [ ] D06 | in_progress | 混合长度和 prefill/decode/verify 混合调度正确，不依赖逐请求 Python 热循环。 | 设备批量metadata、mixed Q1/Q4、多KVhead、padding孔洞与17query跨tile CPU-debug通过；真实异步混合调度未跑。 |
| [ ] D07 | in_progress | 16K、32K、50K 长输入实际使用 OSCAR，跨页与跨窗口边界均正确。 | node93串行16K/32K/50K请求已HTTP完成（档案#130）；混合并发曾撞300s死线（#131），完整跨页/窗口数值对拍未验收。 |
| [ ] D08 | in_progress | TP=4、异步调度、目标图模式和 W8A8 权重量化共存，所有 rank 路由正确。 | TP4/MTP服务和图回放日志已出现（档案#129/#135/#138）；需保留图捕获返回、逐rank路由、真实回放及精度各自证据，不能以HTTP 200代替整项通过。 |
| [ ] D09 | in_progress | 前缀缓存、页共享/复用、请求取消/结束、抢占恢复等目标环境可达生命周期通过验证。 | 真实native block manager与canonical snapshot/回收代数已验证；全服务prefix/取消/抢占待NPU。 |
| [ ] E01 | in_progress | 核心算子对齐参考；覆盖 pack/unpack、量化边界、尾块和 attention 数值误差。 | CANN交叉编译、官方CPU-debug和CPU oracle通过；node93逻辑npu:0通过75项原语NPU探针（54store/21merge），随后首个CV数值用例失败；probe-cv的3项通过仅为Python/源码契约测试，其余CV/旋转用例因maxfail=1未由该日志证明完成。 |
| [ ] E02 | not_run | GDN 隔离检查通过；FULL 量化后的层输出、模型输出与约定质量指标达标。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] E03 | not_run | MTP 接受率、接受长度、输出正确性和有效生成吞吐完成对照。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] E04 | not_run | 固定 token 数和固定 HBM 预算两种口径下证明真实缓存收益，计入全部新增开销。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] E05 | not_run | profiler 证明无禁止的 CPU/AiCPU 数据处理、同步、冗余 KV 双写及全历史恢复。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] E06 | failed | 按第 10 节逐工况比较原生性能；无未解释退化，不以平均值掩盖慢项；**任何算子/相位耗时不得差于附录 D.1 原生基线**（上一版 dequant 6.5s/卡的崩盘即判失败的标准，见附录 D.3）。 | node93的16K阶梯与用户32并发日志已暴露严重性能问题（档案#134/#135/#137-#139）；本轮没有严格配对的当前原生报告，也没有修复后性能通过证据。K1→K4约4倍墙钟受16K每步token预算混淆，不能单独归因算子串行。 |
| [ ] E07 | in_progress | 对照第 8.2 节验证历史故障防护，保存实际覆盖结果而非仅声明"已避免"。 | 历史故障约束已落实；本轮VM发现Log alias与Matmul Init重载问题并修复；CANN plog本任务PID诊断与失败保码通过本地测试。node93回传已登记#123–#126；后续75项原语NPU执行通过确认加载错误已越过，首个CV数值用例暴露query缓冲同步缺口；源码修订后的CV真机复验待回传。 |
| [ ] E08 | in_progress | **无回退逻辑验证**：静态审查 + 全量工况扫描证明代码中不存在任何回退/降级路径；16K/32K/50K 与各种 batch size 下 OSCAR 压缩路径全部实际命中（H13/H17），不存在静默走原生 BF16 全历史的分支。 | 静态/路由/错误注入通过；长序列真实压缩路径命中仍需目标probe。 |
| [ ] F01 | in_progress | 一条 Bash 命令完成安装、编译、所需校准、probe、资源清理和正式启动。 | 一键安装/编译/NPU算子与CV/旋转对拍/自动旋转文件/TP4服务probe/清理/正式服务；默认0–3卡和8989端口，实时输出并落盘。#138确认探针→清理→正式服务首度走通；后续性能实测严重退化，不能据流程完成宣称性能或精度通过。 |
| [ ] F02 | in_progress | 验证正常退出、失败退出和中断后的清理，重复执行不会遗留 worker 或加载旧产物。 | 真实CPU子进程/HTTP/超时/信号/释放失败与诊断故障注入通过；NPU资源回收尚无实证。 |
| [ ] F03 | in_progress | 正式服务按目标配置启动并完成真实请求，记录 OSCAR compressed decode/MTP 路由证据。 | #138一键流程已进入正式serve；用户随后通过service_direct.sh对OSCAR服务发出请求并获HTTP 200（#137），但正式服务的完整路由、质量与负载性能证据尚未验收。 |
| [ ] F04 | in_progress | 交付代码、设计、完整 Checklist、测试/性能/显存报告、日志及 README 中的一键命令。 | 源码、设计、VM日志/报告、README和测量工具已交付；真实模型精度/性能/容量报告尚缺。 |
| [x] F05 | passed | 检查所有硬约束和未完成项，最终结论准确列出已验证范围、失败项及阻塞项。 | 本表和reports/local_validation.json严格区分源码、编译、CPU-debug、NPU/图/性能；未测项保留未通过。 |

## 当前必须正面解决的事项

1. H18复杂度口径；精确attention需要读取全部压缩历史。
2. 已实现的CV/rotation/clip在目标NPU上的精度、图、性能验证。
3. 已实现的prefix、GDN隔离、MTP和固定buffer在真实完整模型上验收。
4. 本次配置已按附录 A/F 固定0–3卡、8989端口和ascend910b4；最新node93复跑进入service-probe并加载TP4/MTP模型；#127已被后续运行越过；#128的MTP虚拟块视图修复后需重新验证服务启动。
5. 模型质量门槛在实测前冻结；paired基线、NPU精度/图/资源/性能逐项过门。

本地构建/测试结果与失败细节见 `reports/local_validation.json`、`reports/pytest.xml`、`reports/build.json`、`reports/readiness.json`。档案#123–#126记录用户回传的node93真实故障与源码修订；#125加载故障已由后续75项原语NPU成功执行越过。最新原文保存在`reports/target_cv_failure_input.txt`，其中首个CV数值用例失败，完整模型、图和性能仍未验收。

本轮 #127：原文 `reports/target_uncached_layout_failure.txt`，关闭prefix时的FULL页布局修复保留全部原生GDN状态。新日志到达服务探针但没有HTTP健康/图捕获/回放成功证据。

本轮 #128 原文为 `reports/target_metadata_block_failure.txt`：编译及初次profiling已返回，容量值仅为原生估算；MTP虚拟块副本修复后仍无正式请求/图回放成功证据。

#129：短请求的实际HTTP完成证据已出现；长请求无完成记录。已修复源码任务分布和未来tile冗余，并新增硬截止及显式debug同步；完整16K/32K/50K与性能仍需真机重跑。同步调试结果不能用于原生性能比较。

#130：用户新日志确认128/16K/32K/50K串行HTTP请求均完成，各生成16token；混合并发仍在进行，不能宣称全流程/质量/性能通过。原文见reports/target_serial_50k_progress.txt。本轮分块DMA优化经过438项本地测试和67项官方CPU-debug，目标性能复验待回传。

#131：混合并发四请求中mixed-1/2/3全部撞300s请求死线，service-probe相位失败（mixed-0-128完成，清理与资源回收正常）。attention_progress心跳证明全程推进、非死锁：49K KV每FULL层组4–15s，混合约99K prompt token共享16384-token引擎步，失败性质为当前CV内核速度下的时间预算。300s死线与PROBE_LENGTHS未动；内核级优化待OSCAR_TIMING=1真机打点定位。原文为用户回传摘录及真机20260922T081333.912496Z日志。

#132：OSCAR_TIMING=1打点（20260922T084539.494672Z）确认host launch全部亚毫秒，墙钟为设备执行以prepare背压形式呈现：串行49K KV mtp prepare host_s=7.38s×4rank、layers.63窗口约22s；混合16K约1.5s/层组、32K约4.4s/层组。设备成本随KV近似线性，CV历史扫描为主项；GDN与FULL的逐kernel拆分需设备trace（原生profiler窗口或ProfileSession+tools/summarize_profile）。验收边界未动。

#133：原生TorchNPUProfilerWrapper的/start_profile与/stop_profile窗口在真机20260922T093958Z杀死全部四个worker（start报线程亲和错误，stop在RECORD状态触发四份segfault→EngineDeadError），探针失败。该HTTP profiler集成已整体移除；逐相位设备时间改由OSCAR_DEBUG_SYNC检查点+tools/summarize_timing提供（一键入口scripts/debug_service.sh），同步值仅用于归因。

#134：debug-sync归因（20260922T104823.837447Z）给出设备时间单瓶颈：fia（CV注意力内核）device_s=2463.859/868次/p95=7.166s，约占99.3%；stores/rotate/merge/guard合计≈18s，prepare≈0.7s（原生GDN开销可忽略）。混合超时依旧（归因运行不验收）。CV内核历史扫描是唯一优化目标。

#135：#134内核微调（逐行区间masking替代逐元素标量读写）真机复测：fia均值2.839→2.054s/call（-28%）、p95 7.166→5.043s（-30%），有效但未达量级；debug-sync下混合并发首次300s内完成、whole-service passed。deploy的full-service门却与探针自报passed矛盾（门已改为打印实读status/resource_release），待service-probe.json实读值裁决。CPU-debug 24项与CANN VM编译、20000例等价模拟、450项本地测试通过。

#136：full-service门失败根因为文件名冲突——探针报告`service-probe.json`与run_phase相位账本`<name>.json`同路径，探针退出后账本覆盖报告。探针全门本就通过（probe_result.log：passed/passed）。报告改名`service-probe-report.json`，测试含账本覆盖回归；门保持严格。正式服务仍待下一次真机全绿。

#137：用户提供OSCAR与原生同配置同数据集32并发对照：原生Running 1→26、聚合gen 156 tok/s；OSCAR Running停滞2–4、gen 0–0.7 tok/s。差异链闭合于CV内核（#134/#135），MTP/调度/GDN/量化排除。功能探针保留为验收证据；新增真实负载性能测量相位（32并发20–30K、计数器吞吐、爬坡采样），只测量不验收，写入report["performance"]。

#138：一键流程首次走完探针→清理→正式服务（#136修复生效）；真实负载首测completed=3/32、prompt_tps=499、gen_tps=0.2。修正收集循环误捕单请求超时的记账bug。按用户指令改为并发阶梯probe（臂1/4×16384，调度/算子/访存三层分离，收尾自打CONCURRENCY_ARM/VERDICT）；impl相位补requests字段，TIMING_BUCKET加reqs维度。

#139：并发阶梯诊断快路径`scripts/probe_concurrency.sh`（--ladder模式，跳过安装/编译/功能相位）；心跳1秒间隔下用`summarize_progress`从心跳差重建每层驻留wall（臂窗口×reqs×KV分桶，跨臂空档单列）。一条命令出齐调度/算子/访存三层证据，诊断不冒充验收。

2026-09-23复核（档案#130–#139；分析见`docs/performance_findings_2026-09-23.md`）：#137中“差异链闭合于CV、排除调度/MTP/GDN”和#139中“心跳差为每层算子/访存成本”的表述超出所贴生产日志。#134/#135的独立debug-sync运行确实把主要同步时间归于FIA，但与本轮生产负载不是同一计时；本轮16K阶梯K1=19.0秒、K4=76.5秒，每条请求恰好占满16384-token调度预算，约4倍墙钟不足以单独证明算子串行。`PROGRESS_BUCKET wall_s`是四rank宿主心跳间隔之和，含其他工作，不是设备算子时间或HBM访存量。用户32并发服务片段缺逐请求usage/TTFT、当前版本配对原生证据和模型级精度门；E06仍为failed，E02/E03未验收。此处为旧条目解释边界的更正，不追加新的真机错误编号。

本轮探针改版（防档案#94/#95/#129/#131/#136/#138/#139）：`tools/service_probe.py`在正式一键功能门后发4条20/23/27/30K不等长的synthetic streaming请求，严格记录TTFT/TPOT、终态usage、metrics及失败；任一诊断请求或指标缺失即阻断正式serve。正式serve中的`benchmarks/passive.py`只观察用户原客户端负载；`tools/summarize_progress.py`明确区分四rank心跳累计与设备算子时间。独立`tools/probe_native_current_fia.py`仅为真机可行性实验，不进入生产路由。此轮本地全仓pytest 466 passed、124 skipped、6 subtests passed；未运行新一轮目标NPU，未修改AscendC、`references/`或冻结精度阈值，E06仍不通过。

一键配对改版（用户要求一次启动、少量可复制日志；档案#70–73/#94/#95/#125/#131–139）：`scripts/install_probe_serve.sh`自动在同一次调用中先跑原生4路20/23/27/30K synthetic、确认进程和NPU释放，再跑OSCAR完整服务探针及相同4路synthetic；性能诊断每路输出64token，功能探针仍16token，300秒死线不变。`tools/paired_concurrency_probe.py`按冻结延迟/吞吐比值逐请求比较，任一失败或证据不足阻断正式serve；单批次、预热不完全配对，因此结果不解锁E06。终端默认compact，仅少量阶段/资源状态、失败和至多8条配对`PERF_*`摘要，完整日志/真实退出码落盘。`scripts/probe_concurrency.sh`作为省去完整安装/功能门的单命令快路径，自动核实签名复用或重建AscendC，新构建须先过真实NPU CV/旋转数值门和资源释放，之后顺序跑原生与OSCAR、每轮确认资源释放。正式一键流程强制本轮新鲜NPU数值探针一次，避免与快路径缓存证据混同。旧16K/16-token性能阶梯已移除；原有16-token功能门仍保留。此轮仍未在目标NPU运行，速度是否追平原生保持未验收。

#140 真机配对结果：同一脚本对20/23/27/30K、K4、64输出token分别跑原生/OSCAR，两边4/4完成且清理及NPU释放通过；原生14.7s、OSCAR182.5s，TTFT p50慢11.53倍，TPOT p50慢14.68倍，客户端性能门按冻结1.0阈值`rc=2`失败。E06从“未验收”进一步有了当前同轮**失败实证**，不能因服务和精度探针完成而启动性能通过结论。HCCL INFO、捕图期间STARTUP_WAIT及TBE关停EOF都不是这次真正失败点。原文及归因边界见档案#140和`docs/performance_findings_2026-09-23.md`；后续内核改动前仍需本轮设备相位/来源归因。

#140后的局部性能修订：CV在短query时只处理有效softmax行，并把四个32-token KV子块合成一次128-token有界工作单元，减少重复Cube调用与跨核握手；没有全历史物化、原生路径回退或精度阈值变更。`reports/kv128_local_validation.json`记录CANN ascend910b4编译及官方CPU-debug 27/27通过、源码指纹；**目标NPU精度、FULL_DECODE_ONLY图捕获/回放和20–30K/K4性能尚未运行**。快路径现先核实当前签名，必要时重建并跑真NPU CV/旋转数值门，避免用旧`.so`误报新内核速度；正式一键流程强制每轮新鲜数值门。E06继续保持失败/待复验，不能从静态约4倍tile轮次减少推导已经追平原生。

#141：用户新实测OSCAR K4墙钟约131.7s（前轮182.5s），有收益但仍比原生慢约9倍，E06未通过。本轮进一步改FP32 Cube basic64×64×128、解析mask、Brcb scale/zero、整块FP32 SoftmaxFlashV2及批量alpha；主模型eager prefill current改用原生causal FIA，与实际CV history/window合并，draft/graph decode仍CV。最终同一内核通过CANN ascend910b4编译和27/27官方CPU-debug；主机517 passed、127 skipped、6 subtests。源码哈希与证据在`reports/deep_attention_local_validation.json`和`reports/deep_attention_cv_cpu_debug.json`。一键脚本自动新增实际current/CV/merge的真NPU门，失败不拉模型。**本轮目标NPU精度、图、服务和追平原生的结果仍待实测**，不以SDK指令数量或本地对拍解锁性能。

#142：用户回传OSCAR捕图0/34时507035/vector trap。原生非capture的MTP dummy warmup标为ChunkedPrefill且slot全-1，新增current路由误启用真实slot guard；用外部`_dummy_run` ContextVar显式区分dummy，完整CV padding仍执行。49项相关主机回归通过（含原生预热编排与真实prefill恢复），AscendC及冻结配置未变。**修订后目标捕图/服务/性能仍待复验**；故障原文见`reports/target_graph_warmup_failure.txt`。
