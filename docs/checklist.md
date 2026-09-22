# 完成状态与证据

本表保留启动文档稳定ID。`passed`仅限该项声明范围；本项目整体未完成。CPU测试、Python wheel、CANN编译、设备完成、图捕获、图回放、性能分别列出。所有源码当前在本地分支 `codex/oscar-ascend`，远端为用户指定GitCode仓库；推送状态以最终Git提交/远端校验为准。

| ID | 状态 | 要求 | 本轮证据/缺口 |
|---|---|---|---|
| [x] A01 | passed | 建立 reference 清单，锁定 OSCAR PR 与相关 vLLM/Ascend commit，确认未参考失败仓库。 | docs/reference_manifest.md；reports/local_environment.json；两棵native树HEAD/status和全参考文件指纹。PR/paper为提供的快照pin，无独立git历史。 |
| [ ] A02 | in_progress | 记录实际软件、NPU、CANN/编译环境、模型配置和原生源码状态。 | macOS与Lima VM环境已记录；VM CANN9.1/Torch2.12 CPU，非目标Torch2.10/NPU。用户要求移除真机启动环境审计，实际目标调用结果由探针记录；该项不作为部署前置闸。 |
| [ ] A03 | not_run | 跑通原生目标配置的基线，保存命令、输入、版本、MTP/图模式与资源记录。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [x] A04 | passed | 读懂第 3.3 节调用链，记录 FULL/GDN、TP、KV 分组、MTP 和图模式的真实入口。 | docs/runtime_implementation.md；tests/test_runtime_native.py执行真实native分组、manager、reshape、MTP/图builder方法；目标worker执行另列D08。 |
| [ ] A05 | in_progress | 核实第 4 节全部歧义，给出层/组/Tensor/地址映射与字节来源。 | 动态C/M/P与B2304公式、真实native config/形状函数已验证；MTP3示例C30720/P817152，最终目标config仍待现场确认。 |
| [x] A06 | passed | 读取 PR 的配置、旋转、量化和精度验证方法，完善本 Checklist，列出仍缺失的输入。 | docs/semantics.md；ops/reference.py、rotations.py与独立PR语义测试。 |
| [ ] A07 | in_progress | 任务开始时已把 `issue1_full_record.md`（含第三部分 [74]~[86]）完整读一遍并掌握其查表用法（症状速查表 → 条目跳转），并将其中与当前任务相关的故障模式登记进第 8.2 节回归检查；确认不会参考任何失败 OSCAR 实现的代码（第 0.2 节红线）。 | 已阅读启动文档全约束、D.4全文、档案速查/总索引与相关条目；子代理分别查读各子系统。未把26,251行档案逐字通读签成完成。 |
| [ ] B01 | in_progress | 产出 `docs/design.md`，列出外部新增文件/函数/类及其包装的原生入口。**结构必须符合第 5.0 节**：全局端到端架构（生命周期图/联动矩阵/时间线/两本账）先行，逐点展开固定六问，自审清单逐项打勾；碎片化设计视为 B 阶段未完成。 | 全局设计/矩阵/时间线/两本账/差异与D.4四问齐全；四条性质中的真实性能与H18仍未过。 |
| [ ] B02 | in_progress | 完成请求窗口、物理页、GDN 隔离、TP 分片、页生命周期和 MTP 提交/回滚设计。 | canonical物理页snapshot、MTP slot-derived位置、prefix/回滚方案已实现；真实native方法与CANN CPU-debug通过，NPU生命周期未运行。 |
| [ ] B03 | in_progress | 给出所有 payload、元数据、padding、workspace 和总 HBM 公式，证明压缩收益可落地。 | 真实native allocator/grouping/LCM/hash/manager与动态字节账已验证；实际NPU容量收益仍未测。 |
| [x] B04 | passed | 完成算子清单、AscendC 融合划分、Decode Stage1 伪代码和逐阶段读写流程。 | 七个AscendC符号完整源码、ABI、Cube/Vector/同步/UB与GM预算；docs/operator_inventory.md、cv_implementation.md、rotation_pipeline.md。 |
| [ ] B05 | blocked | 给出 decode 计算量、HBM 流量、MTP 跨 query 复用和 Host 调度预算。 | H18总历史读取亚线性与精确dense attention存在数学冲突，已提出澄清。 |
| [ ] B06 | in_progress | 固定功能、精度、性能测试方法和验收门槛，设计一键脚本及错误清理流程。 | PR容差未改，部署/故障处理/测量与profiler工具已实现；模型级质量门槛仍待实测前定义。 |
| [ ] C01 | in_progress | 独立插件/算子工程可安装、可编译、可加载，原生源码未修改，安装脚本不使用 `cp`。 | Python包及CANN9.1/910B4交叉编译/链接通过；独立direct-launch库采用源签名SONAME，非custom OPP vendor。目标加载仍待NPU环境。 |
| [ ] C02 | in_progress | 版本能力 probe、延迟注册和幂等 Hook 通过 CLI、API server、各 worker 初始化验证。 | 轻量可撤销hook、factory、binding、真实native类/方法契约通过；实际API/server/所有worker未运行。 |
| [ ] C03 | in_progress | 实现并验证 INT2 编码、旋转/裁剪/量化、KV store 与 Recent→History 增量迁移。 | AscendC rotate/clip/quant/pack/raw snapshot已实现，官方CPU-debug通过；目标NPU数值未测。 |
| [ ] C04 | in_progress | 实现真实 fused INT2 CV Decode Stage1（CV=Cube/Vector，见 H07 注），并完成必要的分块/分段输出合并。 | 真正Cube QK/PV/逆旋转与Vector SIMD解包/softmax已编译、CPU-debug通过，S2/S20对拍及MTP共tile复用；NPU/性能未测，H18冲突保留。 |
| [ ] C05 | in_progress | Sink/Recent Attention、History Attention 和参考数学语义一致，数值稳定。 | 三source causal/GQA与稳定LSE真实CPU-debug通过，含mixed batch/empty/非法数据；NPU未测。 |
| [ ] C06 | in_progress | 实现原生页表兼容的物理分配与视图，GDN 不受影响，HBM 中无冗余完整历史副本。 | 真实native grouping、BlockPool、Full/Mamba managers与GDN reshape执行通过；production raw分配与view已接通，NPU分配未跑。 |
| [ ] C07 | in_progress | 在确有需要的路径提供有界恢复能力，证明 decode/verify 不走全历史恢复。 | prefix直接保留有界canonical BF16 snapshot，无全历史恢复函数；设备tag不匹配显式错误，CPU-debug故障注入通过。 |
| [ ] D01 | in_progress | 首次 prefill 正确，未引入不必要的重复压缩和恢复。 | current chunk直接精确attention后仅压缩新KV，AscendC CPU-debug通过；真实模型首次prefill未测。 |
| [ ] D02 | in_progress | chunked prefill 正确，已处理历史不随每个新 chunk 重搬运或重压缩。 | 旧history在CV tile内消费，无旧chunk重复quant或完整恢复；真实模型chunked prefill未测。 |
| [ ] D03 | in_progress | 普通 decode 持续进入压缩历史路径，无全历史 BF16 HBM 恢复。 | INT2 fused路径与静态shape自适应split已实现并CPU-debug验证；真实模型decode未测。 |
| [ ] D04 | in_progress | MTP draft/verify 与原生流程一致，完整接受、部分接受、完全拒绝后的缓存和 GDN 状态正确。 | 实际native MTP拒绝语义已复现并修正：后续draft按slot/无序BT恢复逻辑位置；CPU回归与CANN CPU-debug通过，NPU未测。 |
| [ ] D05 | in_progress | MTP 多 query 复用历史 tile，读取量测量支持“不随 `q_len` 线性倍增”。 | Mq64容纳GQA6×MTP4；split历史分区互斥、真实多split CPU对拍通过。NPU读取流量待profiler。 |
| [ ] D06 | in_progress | 混合长度和 prefill/decode/verify 混合调度正确，不依赖逐请求 Python 热循环。 | 设备批量metadata、mixed Q1/Q4、多KVhead、padding孔洞与17query跨tile CPU-debug通过；真实异步混合调度未跑。 |
| [ ] D07 | not_run | 16K、32K、50K 长输入实际使用 OSCAR，跨页与跨窗口边界均正确。 | 16K/32K/50K真实请求probe已实现，本机VM无模型/NPU，未实际运行。 |
| [ ] D08 | not_run | TP=4、异步调度、目标图模式和 W8A8 权重量化共存，所有 rank 路由正确。 | 固定地址/实际输入视图契约、dtype与core属性处理已实现；TP4/异步/W8A8/图实机未跑。 |
| [ ] D09 | in_progress | 前缀缓存、页共享/复用、请求取消/结束、抢占恢复等目标环境可达生命周期通过验证。 | 真实native block manager与canonical snapshot/回收代数已验证；全服务prefix/取消/抢占待NPU。 |
| [ ] E01 | in_progress | 核心算子对齐参考；覆盖 pack/unpack、量化边界、尾块和 attention 数值误差。 | CANN交叉编译、官方CPU-debug和CPU oracle通过；110个显式NPU测试在本机not_run，不计通过。 |
| [ ] E02 | not_run | GDN 隔离检查通过；FULL 量化后的层输出、模型输出与约定质量指标达标。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] E03 | not_run | MTP 接受率、接受长度、输出正确性和有效生成吞吐完成对照。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] E04 | not_run | 固定 token 数和固定 HBM 预算两种口径下证明真实缓存收益，计入全部新增开销。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] E05 | not_run | profiler 证明无禁止的 CPU/AiCPU 数据处理、同步、冗余 KV 双写及全历史恢复。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] E06 | not_run | 按第 10 节逐工况比较原生性能；无未解释退化，不以平均值掩盖慢项；**任何算子/相位耗时不得差于附录 D.1 原生基线**（上一版 dequant 6.5s/卡的崩盘即判失败的标准，见附录 D.3）。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] E07 | in_progress | 对照第 8.2 节验证历史故障防护，保存实际覆盖结果而非仅声明"已避免"。 | 历史故障约束已落实；本轮VM发现Log alias与Matmul Init重载问题并修复；CANN plog本任务PID诊断与失败保码通过本地测试。node93 回传的 devices:null 与开发命令误用已登记 #123/#124，修订默认配置及部署文档，真机复跑待回传。 |
| [ ] E08 | in_progress | **无回退逻辑验证**：静态审查 + 全量工况扫描证明代码中不存在任何回退/降级路径；16K/32K/50K 与各种 batch size 下 OSCAR 压缩路径全部实际命中（H13/H17），不存在静默走原生 BF16 全历史的分支。 | 静态/路由/错误注入通过；长序列真实压缩路径命中仍需目标probe。 |
| [ ] F01 | in_progress | 一条 Bash 命令完成安装、编译、所需校准、probe、资源清理和正式启动。 | 一键安装/编译/NPU算子与CV/旋转对拍/自动旋转文件/TP4服务probe/清理/正式服务；按最新要求保留探针并移除环境/原生源码/readiness审计，默认0–3卡和8989端口；全目标流程未运行。 |
| [ ] F02 | in_progress | 验证正常退出、失败退出和中断后的清理，重复执行不会遗留 worker 或加载旧产物。 | 真实CPU子进程/HTTP/超时/信号/释放失败与诊断故障注入通过；NPU资源回收尚无实证。 |
| [ ] F03 | not_run | 正式服务按目标配置启动并完成真实请求，记录 OSCAR compressed decode/MTP 路由证据。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] F04 | in_progress | 交付代码、设计、完整 Checklist、测试/性能/显存报告、日志及 README 中的一键命令。 | 源码、设计、VM日志/报告、README和测量工具已交付；真实模型精度/性能/容量报告尚缺。 |
| [x] F05 | passed | 检查所有硬约束和未完成项，最终结论准确列出已验证范围、失败项及阻塞项。 | 本表和reports/local_validation.json严格区分源码、编译、CPU-debug、NPU/图/性能；未测项保留未通过。 |

## 当前必须正面解决的事项

1. H18复杂度口径；精确attention需要读取全部压缩历史。
2. 已实现的CV/rotation/clip在目标NPU上的精度、图、性能验证。
3. 已实现的prefix、GDN隔离、MTP和固定buffer在真实完整模型上验收。
4. 本次配置已按附录 A/F 固定0–3卡、8989端口和ascend910b4；目标真机复跑结果仍待回传。
5. 模型质量门槛在实测前冻结；paired基线、NPU精度/图/资源/性能逐项过门。

本地构建/测试结果与失败细节见 `reports/local_validation.json`、`reports/pytest.xml`、`reports/build.json`、`reports/readiness.json`。档案 #123/#124 仅记录用户回传的 node93 真实启动错误与源码修订，未声称真机修复已通过。
