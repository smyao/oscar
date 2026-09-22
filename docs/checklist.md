# 完成状态与证据

本表保留启动文档稳定ID。`passed`仅限该项声明范围；本项目整体未完成。CPU测试、Python wheel、CANN编译、设备完成、图捕获、图回放、性能分别列出。所有源码当前在本地分支 `codex/oscar-ascend`，无远端。

| ID | 状态 | 要求 | 本轮证据/缺口 |
|---|---|---|---|
| [x] A01 | passed | 建立 reference 清单，锁定 OSCAR PR 与相关 vLLM/Ascend commit，确认未参考失败仓库。 | docs/reference_manifest.md；reports/local_environment.json；两棵native树HEAD/status和全参考文件指纹。PR/paper为提供的快照pin，无独立git历史。 |
| [ ] A02 | in_progress | 记录实际软件、NPU、CANN/编译环境、模型配置和原生源码状态。 | 已记录macOS/Python/工具缺失；目标模型config、NPU/CANN/驱动实际情况尚未获取。 |
| [ ] A03 | not_run | 跑通原生目标配置的基线，保存命令、输入、版本、MTP/图模式与资源记录。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] A04 | in_progress | 读懂第 3.3 节调用链，记录 FULL/GDN、TP、KV 分组、MTP 和图模式的真实入口。 | docs/native_integration.md 已核实原生接缝；生产生命周期仍待实现和验证。 |
| [ ] A05 | in_progress | 核实第 4 节全部歧义，给出层/组/Tensor/地址映射与字节来源。 | layout.py/specs.py与native padding/虚拟页证据及CPU隔离测试；真实allocator未运行。 |
| [x] A06 | passed | 读取 PR 的配置、旋转、量化和精度验证方法，完善本 Checklist，列出仍缺失的输入。 | docs/semantics.md；ops/reference.py、rotations.py与独立PR语义测试。 |
| [ ] A07 | in_progress | 任务开始时已把 `issue1_full_record.md`（含第三部分 [74]~[86]）完整读一遍并掌握其查表用法（症状速查表 → 条目跳转），并将其中与当前任务相关的故障模式登记进第 8.2 节回归检查；确认不会参考任何失败 OSCAR 实现的代码（第 0.2 节红线）。 | 已阅读启动文档全约束、D.4全文、档案速查/总索引与相关条目；子代理分别查读各子系统。未把26,251行档案逐字通读签成完成。 |
| [ ] B01 | in_progress | 产出 `docs/design.md`，列出外部新增文件/函数/类及其包装的原生入口。**结构必须符合第 5.0 节**：全局端到端架构（生命周期图/联动矩阵/时间线/两本账）先行，逐点展开固定六问，自审清单逐项打勾；碎片化设计视为 B 阶段未完成。 | docs/design.md已给架构/矩阵/时间线/预算/差异；四条性质未全绿。 |
| [ ] B02 | in_progress | 完成请求窗口、物理页、GDN 隔离、TP 分片、页生命周期和 MTP 提交/回滚设计。 | 窗口oracle、物理布局和metadata基础已实现；production prefix/owner/MTP事务尚缺。 |
| [ ] B03 | in_progress | 给出所有 payload、元数据、padding、workspace 和总 HBM 公式，证明压缩收益可落地。 | layout.py给真实payload、SoA、padding、LCM公式；固定HBM实际容量收益未测。 |
| [ ] B04 | in_progress | 完成算子清单、AscendC 融合划分、Decode Stage1 伪代码和逐阶段读写流程。 | docs/operator_inventory.md/ascendc_design.md；store/merge有源码，CV等明确未实现。 |
| [ ] B05 | blocked | 给出 decode 计算量、HBM 流量、MTP 跨 query 复用和 Host 调度预算。 | H18总历史读取亚线性与精确dense attention存在数学冲突，已提出澄清。 |
| [ ] B06 | in_progress | 固定功能、精度、性能测试方法和验收门槛，设计一键脚本及错误清理流程。 | configs/acceptance.json已冻结PR阈值；模型质量门槛仍为null，性能/部署工具已建立。 |
| [ ] C01 | in_progress | 独立插件/算子工程可安装、可编译、可加载，原生源码未修改，安装脚本不使用 `cp`。 | Python包wheel/安装/entrypoint检查可执行；缺CANN而未编译AscendC；custom OPP未交付，当前是direct-launch候选路线。 |
| [ ] C02 | in_progress | 版本能力 probe、延迟注册和幂等 Hook 通过 CLI、API server、各 worker 初始化验证。 | plugin.py的轻量/可撤销/幂等、绑定和缓存测试通过；实际CLI/API/所有worker待真机。 |
| [ ] C03 | in_progress | 实现并验证 INT2 编码、旋转/裁剪/量化、KV store 与 Recent→History 增量迁移。 | 精确oracle和quant/pack/scatter AscendC候选源码；融合rotation/clip及迁移尚缺。 |
| [ ] C04 | blocked | 实现真实 fused INT2 CV Decode Stage1（CV=Cube/Vector，见 H07 注），并完成必要的分块/分段输出合并。 | 真实fused CV未实现；A2的UB→L1软件通路经GM，不冒充片上融合。 |
| [ ] C05 | in_progress | Sink/Recent Attention、History Attention 和参考数学语义一致，数值稳定。 | CPU causal/GQA/LSE/window oracle通过；NPU attention主体尚缺。 |
| [ ] C06 | in_progress | 实现原生页表兼容的物理分配与视图，GDN 不受影响，HBM 中无冗余完整历史副本。 | SSM SoA、native padding、虚拟页映射实现与CPU测试；生产allocate/reshape provider尚缺。 |
| [ ] C07 | not_run | 在确有需要的路径提供有界恢复能力，证明 decode/verify 不走全历史恢复。 | 仅测试oracle允许有界/完整恢复；没有生产全历史dequant路线。 |
| [ ] D01 | not_run | 首次 prefill 正确，未引入不必要的重复压缩和恢复。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] D02 | not_run | chunked prefill 正确，已处理历史不随每个新 chunk 重搬运或重压缩。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] D03 | not_run | 普通 decode 持续进入压缩历史路径，无全历史 BF16 HBM 恢复。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] D04 | not_run | MTP draft/verify 与原生流程一致，完整接受、部分接受、完全拒绝后的缓存和 GDN 状态正确。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] D05 | not_run | MTP 多 query 复用历史 tile，读取量测量支持“不随 `q_len` 线性倍增”。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] D06 | not_run | 混合长度和 prefill/decode/verify 混合调度正确，不依赖逐请求 Python 热循环。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] D07 | not_run | 16K、32K、50K 长输入实际使用 OSCAR，跨页与跨窗口边界均正确。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] D08 | not_run | TP=4、异步调度、目标图模式和 W8A8 权重量化共存，所有 rank 路由正确。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] D09 | not_run | 前缀缓存、页共享/复用、请求取消/结束、抢占恢复等目标环境可达生命周期通过验证。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] E01 | in_progress | 核心算子对齐参考；覆盖 pack/unpack、量化边界、尾块和 attention 数值误差。 | CPU语义与ABI契约通过；NPU probe已写但本机无NPU，未运行。 |
| [ ] E02 | not_run | GDN 隔离检查通过；FULL 量化后的层输出、模型输出与约定质量指标达标。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] E03 | not_run | MTP 接受率、接受长度、输出正确性和有效生成吞吐完成对照。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] E04 | not_run | 固定 token 数和固定 HBM 预算两种口径下证明真实缓存收益，计入全部新增开销。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] E05 | not_run | profiler 证明无禁止的 CPU/AiCPU 数据处理、同步、冗余 KV 双写及全历史恢复。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] E06 | not_run | 按第 10 节逐工况比较原生性能；无未解释退化，不以平均值掩盖慢项；**任何算子/相位耗时不得差于附录 D.1 原生基线**（上一版 dequant 6.5s/卡的崩盘即判失败的标准，见附录 D.3）。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] E07 | in_progress | 对照第 8.2 节验证历史故障防护，保存实际覆盖结果而非仅声明"已避免"。 | 文件头引用、静态禁止模式、真实子进程失败/超时与native完整性检查通过；真机回归未跑。 |
| [ ] E08 | in_progress | **无回退逻辑验证**：静态审查 + 全量工况扫描证明代码中不存在任何回退/降级路径；16K/32K/50K 与各种 batch size 下 OSCAR 压缩路径全部实际命中（H13/H17），不存在静默走原生 BF16 全历史的分支。 | 静态/路由拒绝测试已建立；不等于生产压缩路径全工况命中。 |
| [ ] F01 | in_progress | 一条 Bash 命令完成安装、编译、所需校准、probe、资源清理和正式启动。 | scripts/install_probe_serve.sh具备本地可审查流程；完整服务probe/runtime未实现，必然中止。 |
| [ ] F02 | in_progress | 验证正常退出、失败退出和中断后的清理，重复执行不会遗留 worker 或加载旧产物。 | 本地进程组超时/清理与错误码测试；NPU worker/HBM释放未验证。 |
| [ ] F03 | not_run | 正式服务按目标配置启动并完成真实请求，记录 OSCAR compressed decode/MTP 路由证据。 | 真实目标环境尚未运行；所需生产核心或验证仍未完成。 |
| [ ] F04 | in_progress | 交付代码、设计、完整 Checklist、测试/性能/显存报告、日志及 README 中的一键命令。 | 本地源代码/设计/报告/README已保存；完整实现和目标验收尚未达成。 |
| [ ] F05 | in_progress | 检查所有硬约束和未完成项，最终结论准确列出已验证范围、失败项及阻塞项。 | reports/readiness.json及本表区分源码、CPU、设备、图和性能；没有将基础测试算总验收。 |

## 当前必须正面解决的事项

1. H18复杂度口径；精确attention需要读取全部压缩历史。
2. 真正fused Cube/Vector attention及融合rotation/clip的数据通路。
3. prefix共享、GDN物理页、MTP接受/拒绝和图固定buffer的完整运行时。
4. 本次设备号/端口、真机连接及目标编译环境。
5. 模型质量门槛在实测前冻结；paired基线、NPU精度/图/资源/性能逐项过门。

本地构建/测试结果与失败细节见 `reports/local_validation.json`、`reports/pytest.xml`、`reports/build.json`、`reports/readiness.json`。原始档案不追加虚构的真机修复条目。
