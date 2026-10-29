# OSCAR 优化与失败证据索引

> 2026-10-29 OptR精度续修：修正拟合窗口依赖轨迹最终长度、FP32代理与BF16写读失配、请求均值掩盖单次decode退化，以及全头一起接受/拒绝丢失可保留收益的问题；增加真实长上下文训练，v3身份自动隔离旧PT。证据与启用方式见[本轮记录](optr_accuracy_fix_20261029.md)。固定candidate启用；线上计算不增，模型分数/设备性能尚无本轮实测。

> 2026-10-28 OptR官方分阶段目标/昇腾准备加速：NPU FP32 proxy、CPU正交参数化与最终验证；固定candidate启用，原在线路径不变。官方数学对照/CPU功能/产物恢复见 `reports/optr_sequential_ascend_20261028/`；未连接NPU，设备拟合、模型质量、性能未验收。详见[实现记录](optr_sequential_ascend_20261028.md)。

2026-10-28用户批准实验PT：保留validation和数值门，独立test退化记录为failed并允许实验发布/后续服务，不按test重选，不恢复recipe3，不加命令开关。schema2明确发布政策和test状态，旧schema1继续严格检查；兼容审核过的旧拟合断点以避免重做已完成头/层。训练数值与线上核/runtime/timer/argv不改，不申领精度或速度收益。证据见`reports/optr_experimental_publication_20261028/verification.json`。

2026-10-28用户要求先测OptR：恢复当前原生BF16固定candidate的recipe5默认，覆盖下方#173静态撤回处置，不增加开关或静默改走recipe3。训练/trace/断点身份、独立质量门、线上核/runtime/argv/timer保持；本次不计精度或速度收益。固定入口、双缓存实际选择及失败阻断的本地证据见`reports/optr_candidate_enabled_20261028/verification.json`。

2026-10-28启动显存不足（#174）：四worker初始化时各仅1.19GiB空闲，尚未加载新权重，未确认占用者。修复直启exec后无人监督清理的缺口，新增本checkout逐卡锁、正式服务完整退出/信号清理与有界只读失败诊断。在线核/数学/argv/预算/timer保持，不领取性能或模型收益；真实主机进程及源码基底配对见`reports/direct_service_lifecycle_20261028/verification.json`。

2026-10-28 OptR独立test失败（#173）：第7层通过联合validation但独立test退化，禁止发布的质量门正确触发。固定candidate撤下过早默认化的recipe5，恢复recipe3，BF16/FP32 metadata及其他性能组合保持；显式研究继续严格失败并保存完整指标与身份。不把恢复启动路径算作OptR收益，不将失败测试改成选择集。见[原因、处置与验证](optr_independent_test_failure_20261028.md)，覆盖下方历史默认描述。

2026-10-27 OptR准备超时续修（#172）：目标构建/原生capture已完成，CPU全层拟合错误套用1800秒总时限。改为有限规模预算+真实进展监督，新增认证头/完成层原子断点；验证固定侧历史与teacher复用，配对结果逐位一致，不降低拟合与验收配置。362项本地主机/CPU检查和Python包验证通过；无在线源码或工作量变更，不申领目标性能/质量收益。见[修复证据](optr_preparation_timeout_20261027.md)。

2026-10-27 OptR输出感知旋转：固定BF16 candidate准备阶段启用recipe5，在已认证recipe3上离线精修Rk/Rv。联合KV、BF16 gate、GQA/全部TP交叉项、真实frame分片纳入输出误差；候选含原矩阵，独立test只判通过/失败，不参与选择。线上核/runtime/布局/argv/timer不改；首次采样和CPU拟合有成本，匹配PT复用。CPU/主机验证不代表真机模型精度或性能不退化。见[源码复核与实现](optr_implementation_20261027.md)。

2026-10-23 PC联合检查点保留：主机真实native链复现20请求decode周转后GDN checkpoint被回收，联合命中29952→0。外部插件持有每活跃请求最后可读prompt已有GDN页，完成/取消/抢占/卸载释放；配对修复后保持29952，全体释放无泄漏。最多3页/请求的真实池占用代价，PT/数学/argv/timer/CANN不变，现场命中与模型质量未验收。见[报告](prefix_checkpoint_retention_20261023.md)。

2026-10-23 PC低命中修复：外部插件按实际FULL/GDN共同边界选择checkpoint，并计入prompt−1/原生MTP尾页；低预算保留native状态块推进，共同checkpoint之后仍调用原生尾部切分，避免partial状态误hash。15000-token无MTP本地重复前缀命中0→13824；32768/MTP/剩余15360为0→29952。227项相关主机测试通过；保持GDN/spec/hash/argv、timer及CANN，非对齐状态与encoder输入保留原生路径。实际流量命中率、NPU/图/模型质量及性能未验收，不能消除稀疏状态和完整页drop的固有限制。见[排查、修复及证据](prefix_checkpoint_alignment_20261023.md)。

2026-10-23 FP32量化等级读端修复：实际BF16行1/1.0078125经过真实144B codec在mode0逐位恢复，而旧decode mode1变1/1.015625。candidate所有阶段和capture统一mode0，保留INT2、PTrecipe3、CANN及原生argv；新增运行策略overlay，不放宽v4容差。数学/路由检查不代表LongBench目标已达成，见[实现与证据](fp32_dequant_precision_20261023.md)。

2026-10-23 recipe4质量回归：用户报告约10点损失。实际clip selector的D256反例：代理995.372→421.137，选择.96→.94，attention正确1.0却变0.761594，两种dequant模式均成立。撤下candidate默认recipe4并恢复recipe3，双缓存回归验证不误用recipe4；这不能证明10点全部归因已闭合，长历史跨tile CPU检查继续。见docs/oscar_logic_review_20261023.md。

2026-10-23 剩余精度差对齐：用户确认center后约3点差，目标1点以内。确认唯一新增明确配方缺口为逐头K clip；补齐PT/TP/writer接入及本地预先固定的四指标非劣校准选择，自动升级recipe4。官方未公开实际Qwen表/selector，不声称逐值复刻；反量化与S_v复核一致，窗口/tile差异保留声明，不盲改CANN。见[细粒度证据与修复](head_clip_alignment_20261023.md)，真机质量仍未验收。

2026-10-21 用户要求直接修复center配方：candidate升级recipe3/schema2逐KV头K/V旋转与key mean，默认TP4按rank选择；全部K来源统一BF16中心化，MTP独立零均值。旧PT自动失效，验证后复用已有原始校准样本重拟合，缺失时自动采样。只交付一键启动，不要求诊断实验；数学/迁移/运行时功能验证与LongBench未验收分列，见[实现与边界](official_center_recipe_20261021.md)。

2026-10-21 v4质量反馈续查：用户报告精度仍低，实际v4启用标识待核。定向复核未发现重复RoPE/gate/GQA或Q scratch覆盖K/V的新错误；确认官方结果表center配方与本地共享层V1 PT仍不同。新增只读运行配置/PT识别和已保存校准样本的perhead/center量化误差诊断，不改生产、不重新加载模型、不连接NPU，不将score代理冒充LongBench。见[证据与下一步](precision_followup_20261021.md)。

2026-10-21 官方V1算术链修复：新增v4 reader与prepare，历史/HP/current统一旋转域，FP32 partial合并后一次BF16逆旋转；current-only不再省略旋转，混合current避免提前BF16舍入。单forward精度边界固定，native MTP/APC/物理快照保持；不声称完全复刻SGLang分配器。新合同/独立oracle及一键gate同步，旧母源/guard/timer不改。CANN、CPU与主机证据及未验收的NPU/图/质量/性能见[交付说明](official_math_v4_20261021.md)。

2026-10-20 timer驱动C16定向优化：只针对实测softmax链和BF16转换，保留同一256列归约/ACC_VAL，省略无用标量结果UB发布；C16专用normal-mask转换批量发射repeat，RNE/stride不变。candidate统一启用`c16_tuned`，原核保留显式control，现有timer逐字不变。12例完整C16与355例helper CPU-debug、55能力CANN fresh构建通过；主机148通过/3项已有静态失败/1项跳过。仅交付待真机认证，不计提速或>30%收益；后续依据实际收益决定保留/继续，不能将全部23.8ms视作可消除。见[实现与证据边界](c16_targeted_optimization_20261020.md)。

2026-10-20 精度排查后的模型/校准身份修复：原生load_model前绑定实际runner模型/量化模式，修复手动CLI与OSCAR配置漂移仍可选择同形状旧PT的缺口。保持当前910B4原生GDN的BF16状态，撤回仅凭HF声明FP32判定原生配置错误的推断。服务/采样/PT profile统一有效精度；recipe v2绑定采样manifest与K/V产物，拒绝无来源或不匹配的导入；独立prepare入口先解析auto量化。旧自动PT换路径生成，新产物可复用。主机回归和原生算子能力源码核验见[排查与修复边界](qwen38_precision_audit_20261020.md)，没有真机、质量或速度验收，不宣称用户约5点差已修复。

2026-10-20 C16内部实测收口：用户run `2ae27c282db34edfbb383587023dcb63`完整通过；64/64个目标调用主AIV的具体非等待首位为softmax_math。代表AIV23.4–23.8ms，BF16转换7.5–7.6ms，INT2解包约5.4ms，历史读取2.0–2.6ms。C16优化顺序调整为softmax链→转换→状态/交接，不能套用到q4 decode；诊断扰动约11%，不折算为原1199ms或整轮收益。本轮只交付分析。[证据与定向方案](c16_breakdown_analysis_20261020.md)。

2026-10-20 C16实验启动故障：用户日志确认cache load内层kernel输入解包expected20后持续health空等。新增输入契约预检、外层/内层缓存身份隔离及quiet启动明确worker异常快速清理；不关graph/MTP、不改数学或模型参数，不计性能收益。[修复与边界](c16_startup_fix_20261020.md)。

2026-10-19 C16内部诊断实验：新增单次candidate 32K请求的内部计时，拆分历史读取/解包/发布、QK/softmax/PV、状态读写与同步等待。使用独立计时克隆和真实输入逐位校验，原生产内核保持；局部完成屏障扰动另报，不按计时占比重分旧1199 ms，不计任何性能收益。[脚本及解释](c16_internal_breakdown_20261019.md)。

2026-10-19实测分析报告：依据用户恢复的32K BS4/4/8摘要，先验证OSCAR的AIV向量重建/转换成本；Vector/MTE2活动比约3.79/3.87，不能冒称完整计算/搬运独占时间占比或HBM饱和。报告逐项列证据、C16/q4热点和一对一优化/否决条件，本轮未改算法或测速。[分析报告](oscar_32k_bottleneck_analysis_20261019.md)。

2026-10-19摘要恢复：用户三组32K请求已推进至服务清理/离线解析，最后报str+float。按官方ts字符串/dur数值格式统一解析，并接官方CPU→NPU flow归因；新增仅恢复现有导出的--summarize-only，不重跑模型/SDK，保护原记录。113项主机测试含48窗口真实CLI通过，实际瓶颈仍待用户摘要。[恢复命令与证据](timer_summary_recovery_20261019.md)。

2026-10-19 timer首轮修复：用户native BS4 control26.2秒/profile29.83秒均完成，但单tokenflush空TPOT median中断，daemon worker解析被SDK拒绝导致设备任务缺失。修为空统计null+样本数；worker只存raw，服务释放后由普通子进程离线解析并验证实际任务表。105项主机回归通过；OSCAR对照及bound仍待用户新日志。[修复说明](timer_capture_parse_fix_20261019.md)。

2026-10-19计时专项：用户反馈OSCAR可容纳约31条/原生15条但仍慢，要求暂停大规模优化。按其后续要求交付32K native BS4 / OSCAR BS4 / OSCAR BS8的control与有界PMU诊断，默认256输出；少量终端摘要、完整数据落盘，等待用户运行后归因，不新增任何算法优化或性能达标结论。[设计与一键命令](timer_32k_diagnosis_20261019.md)。

2026-10-19 后续：离线数据通过后，校准 LLM 两层 fork 触发 `Invalid thread pool!`（档案 #166）。校准子进程导入前强制使用原生支持的 spawn，覆盖 EngineCore/TP worker，正式服务参数不改；验证与边界见[线程池启动修复](rotation_process_startup_fix_20261019.md)。

2026-10-19：修复自动 PT 准备下载 GPQA 时的 SSL 证书错误（档案 #165）：随包发布官方原字节压缩数据、校验 SHA 并纳入 PT 缓存身份，默认准备不再联网。55 项主机回归与隔离 wheel 离线数据验证通过；没有内核改动、真机连接或测速。[修复及复用/生成条件](rotation_data_startup_fix_20261019.md)。

2026-10-18：默认原版Qwen3.8-27B（无强制ascend量化、按原生RoPE），保留Qwen3.5/W8A8选择；默认官方qqt/sst校准和新BF16/144B数学路径，性能测试停止。main/stable未动，真实NPU/图/模型质量未据本地证据升级。见[本轮实现及边界](qwen38_official_alignment_20261018.md)。

## 2026-10-16 独立BF16计算分支

主分支7047ce36保持，新增 `codex/oscar-bf16-attention`。QK/PV/Rv均改BF16乘数和FP32累加，保留FP32 online softmax/LSE/merge；独立BF16v2 oracle容差预冻结，旧FP32 oracle不改。4种reader接入共享candidate，相关生命周期门支持不测速的正确性验证。48cap新鲜CANN/Torch构建、126源三方hash、q4/q1及C4/C16官方CPU-debug正常/故障通过。用户取消测速后不再执行模拟/真机性能验证；早先q4候选VM OOM中止原样保留。不得推导30%收益、访存bound或真实NPU通过。见[完整交付说明](bf16_attention_20261016.md)。

## 2026-10-15 stable 同步与 B1 迭代基底

- 用户确认stable的APC开/关均可工作；`f96820f1`已快进同步主分支。固定stable全部vLLM参数、显式16384、原生调度器，默认APC关闭。新目标为32K/64K各自相对同参数原生端到端提速超过30%，覆盖下方历史2–4倍目标。禁止连接真机，不再请求AISBench。
- 用户明确接受B1作为主分支新基底并取消单项必须1.15倍的限制。按完整相关路径净收益逐项叠加，精度/协议不放宽；新设备门为不回退，最终30%不是局部倍率相加。仅用户验证后自行更新stable。
- 同库C65完整三臂：stable3888161、pipeline3302518、B1 3172246 ticks；B1/stable 1.22568倍，B1/pipeline 1.04107倍。1008任务/60actor/逐位及oracle通过。默认C16选择B1，q4选择已完整配对的hybrid L1，q1保留MDL、C4保留已有pipeline。B1增加47.5 MiB/rank固定workspace；此结果不含设备merge/copy，也不代表32K/64K整轮。
- 全库44能力CANN编译通过；CPU、CAModel、主机接缝分别记录。先前超时/ENOSPC/旧门裁决保留为历史；没有新增真机档案。详见[本轮范围与入口](stable_optimization_20261015.md)、[B1证据](../reports/c16_l1_reuse_20261015/summary.json)。

### B1 第一轮增量：最终 Rv 驻留

新同库B1→Rv完整C65配对3168844→3045844 ticks，1.0403829倍，新增UB/L1/GM为0；C337多KV与末成员QR错误隔离CPU逐位通过。按用户最新规则纳入主分支candidate，同一B1算子/ABI/两入口自动启用。旧母源与此前测量保留，不拼乘为稳定版总倍率。源码已打包，新完整扩展构建、NPU/graph与整轮目标另行验证。[配对](../reports/c16_rv_reuse_20261015/completed/rv-camodel-c65/comparison.json)。

### B1 第二轮增量：有限值结果批读＋成熟簇跳步

新同库C320 Rv基底2884171→组合2640555 ticks，1.0922594倍、UB/L1/GM均0增量；完整CPU两臂、实际Softmax12特殊值病例、CAModel全1008任务/60actor/逐位与独立oracle通过。原FP32归约及掩码/尾块/空行分支保持；跳步只在全16成员已验证执行后，第一遍错误/status不变。主分支C16 B1同算子/ABI自动使用新源，设备与端到端仍待验。[报告](../reports/c16_b1_followup_20261015/summary.json)。

### 小增量重开与metadata去重

- query Batch8已按用户新门重新纳入：当前Init68082→Batch864537，同库完整1.05493倍；每行FP32顺序不变，+7KiB/core Hadamard UB、GM0增量，writer Batch8仍不选。现有rotation NPU/graph门实际选择同一新源，未以本地证据标设备通过。[审查](../reports/query_batch8_reopen_20261015/independent_review.json)。
- mixed prepare跨FULL层复用已接candidate，并以现有arena两段分区保留short/long两张表。180项窄主机检查通过，实际任务/错误字段与层操作等价；observe新增真实NPU metadata on/off非回退门。32→2调用和96→6MiB写量只计结构工作，不领CV或模型倍率。[边界](../reports/mixed_prepare_reuse_20261015/summary.json)。

### 本轮最终 decode 选择与完整构建

同库多KV q4有限值批读828355→820349 ticks（1.0097593倍）纳入；q1 794580→793232（1.0016994倍）保留研究、生产MDL不变。两族CPU/全task-source/逐位与oracle通过，原错误预期2失败和真实LoadQueries2→metadata3诊断完整保留；不推断NPU NaN行为。[报告](../reports/decode_finite_20261015/summary.json)。

最终440主机检查通过，8重复大形状跳过；全新完整CANN/Torch44cap、153.02秒、rc0、无旧构建复用，121实际VM源码与冻结host一致。[构建](../reports/stable_loop_fullbuild_20261015/summary.json)。E06整轮目标及NPU/graph/模型质量不因这些本地结果变更为通过。

## 2026-10-14 stable 故障修复

- 目标为用户回传的APC开启时MTP code5、关闭时后期异步Trap。原文与证据边界见档案#160。
- 否决把native位置直接当完整物化context的方案：正位置偏移反例存在未写KV间隙，metadata通过不等于attention合法；生产改动已撤回。
- 外部采样边界适配：修复generic greedy未保护masked行的写入，并为原生cu[-1]读取提供32字节零前缀。固定设备形状、无主机数值回读；随机算法/RNG/GDN/冻结核保持原逻辑。合法mixed源码反例沿原生方法导致code5，适配后回到合法slot；125主机测试通过。只推stable，两固定入口自动启用，默认APC关闭/预算16384。
- [报告](../reports/sampling_boundary_fix_20261014/summary.json)明确：源码执行模型选择未定义值不代表目标Ascend编译事实；第二段没有精确状态码，真实NPU及两故障消失未验收。不计性能倍率，不要求用户补跑AISBench。

## 2026-10-13 当前并行裁决（后续于下方历史记录）

| 路线 | 当前完整证据 | 本轮选择 |
| --- | --- | --- |
| 077 discard/APC 兼容 | 270b46b4已推送原生discard桥、maxlen-wrap合法行修复和prior_work/精确code诊断；28项原生方法、CPU D256链和增量APC writer→q1跨页/rollback通过；#158的64K Trap根因仍未确认 | 保留严格活行检查、只隔离原生明确discard；不领性能倍率，不称真机100%验收 |
| direct-NZ 对当前MDL q4 | 同库完整934323→1143373 ticks，0.81716倍；两侧完整输出/LSE/status逐位、独立oracle、全task/source和六份完整trace通过 | **退出组合**；真实publisher已计费，正在按trace定位回退，不重复helper或旧实验。[结果](../reports/decode_direct_nz_attention_20261013/completed/comparison.json) |
| MDL q1 对当前Normal q1 | 6个CPU病例通过；同库完整948223→797943 ticks，**1.18833倍**；逐位/oracle/全task/source通过。基线后置padding检查错把-1要求为0，已验证完整数值/trace后单独追加校正链，原rc1保留、未重跑内核 | 达单项1.15门，两个candidate入口启用；完整41项CANN/Torch构建通过，NPU/graph仍待验；属于MTP later-q1局部，不能外推16个target FULL层或整轮。[结果](../reports/q1_mdl_qualification_20261013/completed/comparison.json) |
| C4 当前H16→H32成员流水 | Q336/C1341完整配对 **3540306→2671072 ticks，1.32543倍**；全1008任务、336query独立oracle、60actor通过；大形CPU超时保留，小形CPU及大形prepare完整 | 达单项1.15门，两个candidate入口启用现有独立pipeline；默认C16共享arena已覆盖，未增加保留空间。NPU/graph仍待验。[结果](../reports/c4_composed_long_20261013/completed/comparison.json) |

三个性能候选分别有负责人，主代理统筹真实VM独占队列、兼容修复和两个一键入口；状态以完成回执为准。总目标仍为native2–4倍，当前尚未建立整轮达标证据。

> 2026-10-12组合门更新：单项完整路径 `t_base/t_new>=1.15`、精度/协议合格即可入组合；组合相对原生2–4倍。C4 H32→member pipeline已测1.148704倍，严格尚未达到新15%门，不向上取整；保留研究证据与必要依赖身份，待完整合并基底证据再裁决。兼容/正确性修复不要求独立速度倍率。

更新：2026-10-11。按当前工作区源码和已保存报告核查；**不代表所有改动已经推送，也不代表真 NPU、graph 或整轮性能通过**。档案依据：`issue1_full_record.md` #126/#140/#145/#148/#154–156、启动文档 §0.7 与 D.4 完整日志。

当前目标：65100 输入 / 512 输出 / 120 请求 / K32，相对已有 native 1342.4437259 秒实现 2–4 倍，即 335.61–671.22 秒；FULL KV 6–7 倍，同时报告全部 KV。原生 token 预算、GDN、冻结精度门不变。不同长度与低并发也须保留正确性和可用性。

**本轮后续进展**：已将H32＋C4 member pipeline打包为独立可测算子，并同步request-window reader、两入口共享flag、实际选核/ABI/workspace2494464B和专用成熟簇多KV NPU/图门；完整生产CANN/Torch编译通过，39项能力齐全。见[构建摘要](../reports/c4_member_pipeline_fullbuild_20261011/summary.json)与[Python接缝验证](../reports/c4_member_pipeline_python_20261011.json)。2026-10-12按新组合门收口：**已退为显式research开关，默认candidate不启用、未发布；同形H16→组合完整数据尚缺，真实NPU/图/整轮仍待认证**。专用gate固定要求完整路径至少1.15倍，既有1.148704倍不能四舍五入过门。跨请求decode流水长形仅1.137倍，暂不扩写triad；K/V连续NZ已完整测得回退并停止；当前根据多KV完整交付链研究AIC官方MDL大块装载，尚无性能通过声明。

本轮新增完成：独立 `rotate_vector_init_out`＋writer pack/Init 的完整 CANN 库与 Torch 扩展 fresh build 已通过（signature `79550b3a08cdb12739b9ceeb60fc521f0d1569238d6fc59f45929bdd5a1125b5`），manifest 37项能力含新算子；实际候选选核/缺失NPU证据拒绝等25项主机检查通过。详见[本地构建与接缝摘要](../reports/quant_bundle_fullbuild_20261011/summary.json)。这是后续于表内旧“fresh build待验”的结果；NPU、图和整轮性能仍未验收。K-major三例周期配对虽数值通过，但PV回退，已结束该版本，见[完整裁决](../reports/kmajor_20261011/decision.json)。完整C4 B-NZ与Vᵀ-NZ均已测回退、退出组合，未启用生产；仅K/V的L1复用另行审查，尚未验证或否决；按[总 MD](../skills/oscar-quant-optimization/SKILL.md)的组合闭环，用[roofline 子 MD](quant_dequant_roofline_20261011.md)核算容量、实际batch与完整成本。

### 完整 B-NZ / Vᵀ-NZ 研究的完成与退出

- `csrc/experiments/c4_b_nz/` 已完成独立 CPU/CANN 编译；在 H32＋member-pipeline 基底只改变 B 格式与真实有界publisher，永久16KiB索引使显式UB为174144B，原数学/错误域/任务与归约顺序不变。未进生产。
- 同源完整 CPU baseline 23.45秒完成、oracle/guards/status通过。第一次运行在SDK硬编码 `/tmp` 的内存映射分配遭根磁盘耗尽，SIGBUS；后续baseline通过后两臂累计npuchk日志填满tmpfs，候选与监管报告写出失败。失败日志已完整归档，未改成通过。
- 隔离并归档日志后，candidate单臂仍在原120秒期限超时，cleanup完成、无完整数值输出。该失败原样保存在[完整证据归档](../reports/c4_b_nz_20261011/timeout-after-archive/evidence.tar.gz)，不据有限actor成功行声称完整CPU通过。
- 四个最终publisher方法逐字提取的独立CPU门已通过：history旧lane16与precise16各131072个K/V原始word、±0/NaN/Inf/subnormal/空间标签、输入不变和双guards。它只证明新增布局接缝，不证明完整C4、NPU或速度。
- 原`cv_replay.py`完整CPU前置门保持不变。新的**研究入口**`cv_replay_publisher.py`明确要求已成功的baseline完整CPU/真实prepare加独立publisher CPU门；完整candidate CPU仍标未完成。该入口在原600秒期限内完成两臂，完整输出/独立oracle/全部1008任务与状态/60actor门通过：**1608607→1743703 ticks，0.92252倍（慢约8.4%）**。已明确否决这个B-only完整版本，不进组合；[配对结果](../reports/c4_b_nz_20261011/completed/reports/camodel-valid/comparison.json)。没有更改冻结精度、生产探针或将CPU超时重标为成功。
- **Vᵀ-NZ也已完成并退出组合**：同库Q336/C337/S1/20core，**1610898→1734415 ticks，0.92878倍（慢约7.67%）**。两臂在600秒期限内rc0、cleanup完成；完整CAModel逐位/独立oracle/1008任务/60actor通过，publisher CPU原始位与guards通过；**完整candidate CPU未运行**。更少V发布burst没有转成完整核提速，未测分相位归因，不把回退武断归到单一Gather/DMA成本。[Vᵀ裁决](../reports/c4_b_nz_vt_20261011/decision.json)、[配对](../reports/c4_b_nz_vt_20261011/reports/vt-camodel-valid/comparison.json)。B-only的[裁决](../reports/c4_b_nz_20261011/decision.json)、CPU超时与全部原失败继续保留。
- 当前A2/CANN公开`DataCopyUB2L1Impl`及`DataCopyUB2L1ND2NZImpl`在AIV侧通过GM workspace软件模拟：先`copy_ubuf_to_gm`，再发送GM→L1消息，不能据API名字申领无GM直通收益。[SDK源码123–213行](../artifacts/cann_sdk_cube_public_20261010/impl/basic_api/dav_c220/kernel_operator_data_copy_impl.h)。**仅K/V的L1复用仍单独审查**，其可行性、所有权和净收益未定；两次布局回退不构成对该方向的否定。以上都是本机研究证据，NPU、graph与整轮目标未验收。

## 阅读规则

- **已接入**表示当前生产源码或 candidate 选择器存在该逻辑；设备验收另看报告，不能等同已发布。
- **保留组合**仅指当前选中的候选或必要依赖。用户最新限定过小单项不入当前组合：writer Batch8 3.8%、query Batch8在Init上5.5%、static Matmul 1.6%改列历史备用；它们不是数值失败。保留明显的writer pack+Init和query Init；完整C4流水1.148704倍低于1.15门，移出默认组合、保留显式研究，H32仅作研究依赖。旧临时接入状态由此选择覆盖，不能混称全部已启用。
- **数值否定**、**已测回退**、**约束失效**分别记录。超时、日志截断、空间耗尽不自动等于算子数学错误。
- 倍数分母严格保留。仅同输入、同库、同配置、完整完成的配对可得该算子倍率；不相乘不相干局部倍率，不把容量比换算吞吐，不把当前路线套入 D.4 旧版 95.4% 占比。
- 此页覆盖能追溯到源码、档案或报告的主要路线；**没有落盘的中间想法未记录，不能声称 100% 穷举历史**。更早故障以档案症状索引为入口。部分旧 README/汇总仍写“待测”，以本页链接的后续具体比较报告为准，原失败证据不覆盖。

## 一、当前 candidate 已存在的路径

| 方案、当前代码 | 结果与精度范围 / 收益分母 | 启用与组合边界 | 证据 |
| --- | --- | --- | --- |
| SIMD INT2 pack + Hadamard Init；`csrc/kernels/rotate_clip_store_striped.cpp`、`striped_store_pack.h`、`hadamard_indices.h` | 完整 writer 同库 old **173419** → pack **136213** → pack+Init **96566** ticks；old→组合 **1.79586×**。CPU 完整字节、oracle、guard；CANN/扩展完整编译，request-packed 接缝 CPU 通过 | **本地生产源已接入** striped writer；`rotate_clip_store_request_packed.cpp` 同步包含两项。Init 是 pack 后额外收益，不再另乘 Init helper 的 7.112×。Q优化改走独立可选算子，canonical FE0源码保持冻结；见下项当前选择 | [三臂完整 writer](../reports/quant_store_simd_pack_20261010/comparison.json)、[生产接缝](../reports/quant_production_integration_20261010/integration-summary.json)、[Init helper](../reports/quant_hadamard_indices_20261009/summary.json) |
| q4/q1 decode bundle；`attention_cv_decode_bundle.cpp`、`attention_striped_unpack_padded.h` | 包含 HalfKv64、两块有界 KV 流水、精确窗口批量读、区间 mask 等。完整合成 K32 / 20–30K / q4 对旧 striped SIMD：**16290667→9589788，1.69875×**，全部 tasks/source、partial/LSE/status 逐位、两臂独立 oracle | **candidate 已选择**；该组合内各单项不能再次计入新增收益。此长 q4 结果不是完整 K32 q1 或整轮目标证据 | [完整 q4 配对](../reports/decode_integrated_batch32_retry_after_oom_20261007/q4_r32_n128_20to30k_s3-comparison.json)、[交付索引](../reports/decode_candidate_delivery_20261007.json) |
| history C4/C16 复用、natural V 发布；`attention_cv_window_range_cluster4.cpp` 等、`ops/cv_dispatch.py` | 当前 C4 仍 HalfKv16；自然 `[KV,D]` V 配合非转置 PV，已消除旧 ND V 转置 Gather/散写。C16 仅 max_query_len≥8192 路由；不能把 native 小 chunk 当作 C16 | **candidate 已存在**。旧 natural V 收益不是新的 NZ 发布收益。C4/C16 与新 H32/pipeline 改同一 reader，须形成单一派生链 | [历史复用](history_reuse_experiment.md)、[mixed 路由](mixed_cv_optimization.md)、档案 #144/#150–154 |
| current-only / 首轮 MTP current FIA / later MTP compact / mixed 分区；`integration/impl.py`、`compact_later_mtp.py`、`mixed_decode_split.py` | 当前未压缩部分已走符合原生调用条件的 FIA；later MTP 已去除已审计的大 padding；短 decode 前缀与长后缀已分开。没有可再申领的单项完整分钟收益 | **candidate 已存在**。历史仍 INT2，不动 GDN；后续 actual-N split 是另一项尚未实现的策略，不能重复计 compact 收益 | [现有组合](decode_candidate_20261007.md)、[剩余重复审计](quant_dequant_remaining_redundancy_20261010.md) |

## 二、正收益路线的当前选择与历史记录

| 方案、当前代码 | 已有结果与精度范围 / 分母 | 生产与组合依赖、剩余门 | 证据 |
| --- | --- | --- | --- |
| canonical Q Hadamard Init + Batch8；`experiments/hadamard_batch8/rotate_init.cpp`、`rotate_batch.cpp` | 同库完整 N128/H6/D256 Q rotate：old **110343** → Init **68082** → Batch8 **64537**；old→最终 **1.70976×**，Init 后 Batch8 仍 **1.05493×**。输出/status 位一致；7 个 CPU 边界覆盖 D64/128/256、1/7/8/9 行、padding、±0/nonfinite | **当前只选Init，完整1.62074×；Batch8额外5.5%不列本次组合**。**本地已完成**独立`rotate_vector_init.cpp` / `rotate_vector_init_out`映射及源反向恢复；共享candidate开关已选此算子，canonical FE0与原ABI保持。未发布，fresh build/CPU seam/NPU/graph待验：[当前集成报告](../reports/quantization_bundle_integration_20261011.json) | [三臂完整 rotate](../reports/hadamard_batch8_20261010/reports/batch8-rotate-camodel128-v2/comparison.json)、[范围汇总](../reports/hadamard_batch8_20261010/summary.json) |
| writer Batch8；同目录 `writer_batch.cpp` | **已含 pack+Init** 的 current **95930** → Batch8 **92403**，完整 writer 净 **1.03817×**；CPU 两例完整字节/oracle/guards 通过 | **历史正收益，本次不交付**。3.8%增量过小；临时本地Batch8映射**已撤回**到已测pack+Init原快照，request-packed已同步重生；未发布/fresh build待验。保留原完整数值证据；禁止用另一库的旧96566与新92403拼倍率 | [完整 writer 配对](../reports/hadamard_batch8_20261010/reports/batch8-writer-camodel128/comparison.json) |
| C4 H32 padded unpack；`experiments/c4_unpack32/` | 193 helper CPU 用例逐位/oracle/guards；helper **98478→41974，2.34617×**。完整 Q336/C113 C4 **1355015→1346984，仅1.00596×**；CPU valid/bad80/bad96 全位一致 | **仅作下项pipeline必要依赖，未启用**。不单独交付/计入2.346× helper收益；保留旧 alternating16-row lane错误归属与Softmax scratch可见容量 | [helper](../reports/quant_unpack32_20261010/comparison.json)、[完整 C4](../reports/quant_unpack32_20261010/full_c4_camodel_comparison.json) |
| C4 member pipeline + atomic accState；`experiments/c4_member_pipeline/` | **两臂均已 H32**；完整 Q336/C337/R1/S1/20core、全部1008任务/source0+1：**1849738→1610282，1.14870×**。CPU C113/C337/bad336 全位；atomic v3 即时回读及 atomic→normal 也已通过 CPU/CAModel | **显式research，默认candidate不启用、未发布**。1.148704严格低于1.15门，不能向上取整；增量在H32之上，不能乘helper2.346，也不能跨C113/C337乘1.00596。新增640KiB/core；原C1/precise路径不变，NPU/graph/长形仍待组合门 | [完整核配对](../reports/c4_member_pipeline_20261010/reports/c4-pipeline-camodel337/comparison.json)、[bad336](../reports/c4_member_pipeline_20261010/reports/c4-pipeline-cpu-bad336/comparison.json)、[atomic](../reports/c4_atomic_state_recovery_20261010/summary.json) |
| static Matmul 配置；`experiments/decode_static_matmul/` | 完整 q4/R1/C20000/S3 在 history-pipeline 基底上 **2464632→2426161，1.01586×**；partial/LSE/status 逐位、两臂 oracle | **历史正收益，本次不交付**。1.6%增量过小；不是精度失败。若改用低层K-major将被替代，不能同时记两项收益 | [配对](../reports/decode_static_long_camodel_20261007/comparison.json) |
| active/identity writer；`experiments/active_identity_store/` | q1 identity 完整 writer 局部 **8.56288×**；已有835fc22片段中两次 q1 store 只占步 **0.266%**，不是当前整轮占比 | **历史备用，未列当前交付**。Hadamard和identity分支互补，但现有整步贡献证据小；identity的finite/±0/status不能未经资格直接用拷贝替换，暂不重跑同一micro | [完整 q1](../reports/active_identity_store_camodel_20261006/n128_h1_d256_bf16_b128_decode_q1_identity-g16-comparison.json)、[旧片段范围](../reports/active_identity_store_decode_cost_20261006.json) |
| 按实际短分区选 split；尚为方案，当前仍保留原 `cv_shape_tokens` | 源码确认小 R 可偏载。naive actual-N 的 q4 R5/N20 会 S20，不是 S4；同一条件预算含16层长 C4+短 q4+2次q1，长后缀 context63K 仅约1.038×（q1系数是假设） | **保留结构分析，未列当前交付/未实现/未测速度**。必须权衡新增task/merge固定成本、短context、split归约误差；不能用“5核变20核”冒充整步4×。预算不是实测正收益 | [全 mixed 条件预算](../reports/mixed_actual_split_budget_20261010.json) |

## 三、布局、容量与下一结构原型

| 方案、当前代码 | 已证范围 / 收益分母 | 组合冲突与下一必要门 | 证据 |
| --- | --- | --- | --- |
| 双段 packed FULL + request-stable window；`request_window_storage.py`、`request_windows.py`、`lifecycle.py`、`attention_request_window*.cpp`、`rotate_clip_store_request_{packed,window}.cpp` | 源、绑定、完整 CANN/扩展、store19 CPU + reader等局部数值已完成；C16 oracle 仅source0+1。字节账条件值 FULL **6.3088×**、全部KV **3.8234×**；不是HBM峰值或速度倍率 | **已注册、research显式开关，candidate默认未开**。已解耦 whole-prefill，保持native调度；lease/preempt/reset/reorder/epoch/capture门不可省。writer新算法与六条reader必须在同一生成链同步；须把 pipeline/NZ 变更迁移到新地址协议，不能拼两个不一致副本 | [当前进度](../reports/restructure_64k_progress_20261009.json)、[19例writer](../reports/request_windows_store_20261009/cpu-suite/cpu_cases_report.json)、[字节/ROI账](../reports/roi_decode_overlap_20261010.json)、[设计](compact_pool_64k_design.md) |
| B-NZ / AB-NZ Matmul；`experiments/fp32_nz_b_matmul/`、`fp32_nz_ab_matmul/`、`fp32_nz_matched_shapes/` | 已测免费host预pack上限、完整MMAD序列/整数oracle/输出位相同。matched QK126：ND28760→B22118→AB19436；PV24：ND25957→B13777→AB13817，**AB在PV24反而比B多40ticks** | **免费预pack微核只保留为历史上限**。已补完整B-NZ与Vᵀ-NZ真实publisher，均回退并退出组合；AB完整发布尚未实现。B与AB不相乘，QK/PV须按M分域，不能跳过真实publisher成本重新宣称净利 | [B-only](../reports/fp32_nz_b_matmul_20261010/reports/nz-b-camodel/comparison.json)、[AB](../reports/fp32_nz_ab_matmul_20261010/reports/nz-ab-camodel/comparison.json)、[matched三臂](../reports/fp32_nz_matched_shapes_20261010/reports/nz-matched-camodel/comparison.json) |
| AB-NZ的Q/P与最终归一化A布局；仍为资源/协议方案 | SDK已核FP32 C0=8与NZ精确地址；完整B-only/Vᵀ方案已回退，不能沿用免费预pack倍率作为整核收益 | **AB完整方案未实现/未启用**。P不能与ND score原地alias；独立P、物理OrgM32/128、tail零化、MTE3_V均必要。C4 pipeline两P槽使每core需求2756608B，当前4997120B C16 arena可容纳；修改stride必须Cube/Vector/allocator一致。C16自身再扩P另算 | [KV布局](nz_history_layout_20261010.md)、[Q/P布局](nz_query_probability_layout_20261010.md) |
| 按D列分工NZ；只读设计 | 可减少C4 publish burst，但packed/metadata双读；decode H64本已较少burst，不能套C4收益 | **未实现/未测，未入组合**。B-NZ回退尚无相位归因，不能据此认定列分工会获益；旧虚拟lane错误归属必须保留 | [列分工](nz_column_owner_20261010.md) |
| 完整C4 B-NZ / Vᵀ-NZ；`experiments/c4_b_nz/`、`c4_b_nz_vt/` | 同库各自配对：B **1608607→1743703**（慢8.40%）；Vᵀ **1610898→1734415**（慢7.67%）。两例CAModel完整逐位/独立oracle/1008tasks/60actors通过，真实K/V publisher已计时，Rv一次256KiB位pack未计时 | **两版本均退出组合、未启用生产**。B完整candidate CPU仍120秒超时；Vᵀ完整candidate CPU未运行；两者publisher CPU通过。不能写成全CPU门完成，也不能跨库把两候选互相拼倍率 | [B裁决](../reports/c4_b_nz_20261011/decision.json)、[Vᵀ裁决](../reports/c4_b_nz_vt_20261011/decision.json) |
| 仅K/V的L1复用；另行只读审查 | 当前公开UB→L1接口经GM软件模拟，不能宣称无GM直通；Cube内已有K/V的L1复用是另一项资源/生命周期问题 | **未验证、未否决、未启用**。待确认实际可复用数据、L1容量、跨成员所有权和完整净收益；不重跑已回退的两个布局版本 | [SDK接口](../artifacts/cann_sdk_cube_public_20261010/impl/basic_api/dav_c220/kernel_operator_data_copy_impl.h)、[当前决策](quant_dequant_next_decision_20261011.md) |
| FP32 K-major低层Matmul；`experiments/fp32_kmajor_matmul/` | 当前独立源码与有限fixture；初始事件冲突、v2 PV的FMATRIX协议错误已修。v3 CPU/CAModel三例有限输入配对位、oracle与每输出K128顺序通过；qk24/cancel **13943→12557，1.11038×**，PV126 **18975→25853，0.73396×**。**本版本PV明显回退，不入组合、不扩形状**；根因仍查完整trace，不归咎精度。[完整配对](../reports/kmajor_20261011/v3/reports/kmajor-v3-camodel-v2/comparison.json)。针对真实重复A的L1→L0加载；全NZ L1、K外/N内/独立M64和M62，保持每输出K128顺序/HF32off/small-M屏障 | **未启用**。在NZ基底上评估增量，不再重复记ND2NZ收益。使用独立L1/L0资源、不与MatmulImpl共占；whole128 ZA子视图未证，首版改两次独立Load3D。预pack仍免费上限，之后须接完整publisher与attention | [独立原型](../csrc/experiments/fp32_kmajor_matmul/)、[API可行性](fp32_manual_matmul_feasibility_20261010.md) |
| C4尾波按代价拆回原C1；只读候选 | 当前R1/Hkv1/GQA6/S1条件下Q2048有21成熟簇，个别core两簇；Q1924无同样尾波，Q8192全拆会变慢。对当前pipeline用A/C1、B/C4和固定成本参数式，理想1.5×需`B≥A+F+3P` | **未实现/未测**；增加KV展开与总工作，必须以完整关键路径净收益选择，不能拿旧条件1.62×当新组合性能 | [尾波预算](c4_tail_wave_budget_20261011.md)、[大矩阵合并暂缓](c4_wide_matmul_feasibility_20261011.md) |
| 同core跨独立request流水；`experiments/decode_task_overlap/` | v1两UB状态/两GM上下文、相同轮数不同request配对已实现；CPU/CANN编译及q4/q1两请求短形逐位/oracle通过。完整短q4 **810475→775979，1.04445×**；完整31KV长q4 **3351197→2948283，1.13666×**，两臂完整位/oracle/task/actor通过；长CPU120秒未完成单列。K32实际prepare表配对80/96 history tasks | **独立研究，暂不入组合或扩写triad**。长形完整净收益不足以支撑旧条件1.56倍；不同轮数和奇数triad只有协议。需新的明显净贡献证据再增加复杂调度；不把局部13.7%变成整轮目标 | [实际裁决](../reports/decode_task_overlap_20261011/decision.json)、[执行记录](../csrc/experiments/decode_task_overlap/README.md)、[旧条件预算](../reports/roi_decode_overlap_20261010.json) |

Kt/Vt共同连续NZ发布的新实验：`experiments/c4_kvt_nz/`将K也改为物理`[D,KV]`/NZ，QK用非转置B、PV用转置B，K128/FP32归约维度保持。每完整KV发布528块→32连续块，新增64次index Cast仍在核内计费；UB182336→165952B。CPU/CANN编译、history/precise各131072原始word及guards通过；**完整C337配对逐位/oracle/1008任务/60actor通过，但1612253→1723131，慢6.88%，v1不入组合**。基线仍ND H32+pipeline；完整候选CPU未跑，旧B-only CPU超时保留。[实际裁决](../reports/c4_kvt_nz_20261011/decision.json)。后续uint32索引常驻UB版本也已完成：1611590→1709129，慢6.05%；精度/全部任务/actor通过，仍退出组合。移除转换没有扭转完整核退化，因此暂不继续扩展NZ布局。[原型与预声明门](../csrc/experiments/c4_kvt_nz/README.md)

## 四、明确失败、受限或曾被误判的路线

| 项目 | 当前结论与代码状态 | 原因 / 证据，后续避免重复 |
| --- | --- | --- |
| A2 L2 `NORMAL` / `PERSISTENT` 参数微调 | **源码筛选排除当前调参方案，未改生产** | `GlobalTensor`默认NORMAL，当前work无显式bypass；所读A2设备分支仅DISABLE改变地址，PERSISTENT/LAST不提供新驻留行为。约8GiB是累计GM访问、非实测HBM；现CAModel L2关闭，真机命中率/回写未知，未猜L2容量。仅K/V的L1复用独立待审查；附录同时记录Python链路无大整区复制及merge FP32中转字节上限：[缓存与复制审计](../reports/quant_cache_policy_audit_20261011.json) |
| 双BF16残差＋三项补偿 | **只读筛选后暂缓，未实现/未数值否定** | 三次完整BF16 Matmul的条件成本已高于FP32；融合版改变累加链且需增加分解/重组/同步，尚无完整精度与明显净收益证据。只有实际融合成本和冻结数值资格同时成立才重开：[具体边界](dual_bf16_residual_feasibility_20261011.md) |
| 无条件BF16输入代替FP32历史；FP16 PR prefill dtype链 | **数值否定，生产未改** | FP32累加不能补回输入cast丢失；single-BF16有效值反例误差0.09267>冻结门；PR dtype链7个store可达stress失败。不得换oracle或放宽门：[BF16反例](../reports/single_bf16_cast_counterexample_20261009.json)、[prefill反例](../reports/prefill_tile_stress_20260930.json)。FP32/BF16两个可精确表示微形状的1.213/1.326×只是诊断：[dtype实测](../reports/fp32_bf16_mmad_ratio_20261010/decision.json) |
| M32/N128/K64 tiling消小M barrier | **当前具体方案已测回退，未启用** | 同形完整marked QK24 20724→27856，慢34.41%，虽8个BAR归零。不是禁止所有未来tiling：[原始裁决](../reports/decode_mmad_barrier_20261010/decision.json)。不要直接删SDK规定的小M屏障 |
| weighted decode owner | **q4小正/q1回退，保留分域研究，未启用** | 两核短形q4 1.07753×，q1 0.97935×，两臂位/oracle过；不能全域启用：[q4](../reports/decode_weighted_smoke_20261007/q4_multi_request_s3-comparison.json)、[q1](../reports/decode_weighted_smoke_20261007/q1_multi_request_s3-comparison.json) |
| safe-range消逐行finite ReduceSum | **未实现；单点不足目标，非数值失败** | C4部分trace仅final截断，finite叶子60actor闭合计数齐；该部分成本不足单独1.5×。现列历史分析、不入当前交付；如未来组合条件改变须重核，不能跳过NaN/Inf/error语义：[叶子范围](../reports/c4_phase_profile_q336_20261009/finite_leaf_evidence.json)、[闭合子范围](../reports/c4_phase_profile_q336_20261009/core3_closed_subscopes.json) |
| whole32K/64K/bounded-whole scheduler提预算 | **用户约束失效，当前服务强制关闭** | `tools/serving_variants.py`清除owned scheduler、max_num_batched_tokens/history override；入口保持native默认。旧whole预算收益全部退出当前预测。历史`decode_candidate_20261007.md`整段准入描述已过期，不能重新启用来凑目标 |
| 旧batched4、旧同步/尾部错误 | **真NPU数值失败或已修bug，不是小收益候选** | 档案#126零填覆盖、#145 Softmax第二块覆盖未完成写回、#148质量回退、#154 batched4 CPU过而NPU oracle失败。新的H32/pipeline必须保留逐位/错误域/设备/图门，不能以旧CPU通过代替 |
| 全历史BF16恢复、缓存target/draft的“同位置KV”共用 | **结构/语义不可直接采用，无生产改动** | D.4完整恢复约6500ms是失败边界；target/draft有不同投影和raw LayerState，后续draft还依赖上一步结果，同位置不等于同值。只允许固定tile工作区：[剩余重复审计](quant_dequant_remaining_redundancy_20261010.md) |

## 五、实验设施失败也要保留

| 失败 | 可采用 / 不可采用的证据 |
| --- | --- |
| C4 Q1344/C769 normal 900s超时 | cleanup完成但无完成输出；不得取局部ticks或说数学已错。后续Q336是明确缩小的诊断点，不代表K32/2K饱和：[失败索引](../reports/c4_phase_profile_c769_20261009/summary.json) |
| Q336 C4 profile两个AIV末尾64MiB截断 | 整trace门仍失败；数值完成与有限闭合循环单列，不补finalize： [范围说明](../reports/c4_phase_profile_q336_20261009/partial_diagnostic_summary.json) |
| Batch8主query CPU日志填满tmpfs | 失败完整保留；不能伪称CPU主形通过。7个小边界CPU与后来完整query CAModel是独立证据：[汇总](../reports/hadamard_batch8_20261010/summary.json) |
| C4 CPU比较器误判bash-wrapper fixture路径 | 不重跑算子；严格解析受控exec、核prepared metadata/tasks/positions修复后另存verified报告，旧失败comparison保留：[修复后资格](../reports/c4_member_pipeline_20261010/comparison-cpu113-verified.json) |
| K-major 首例 CPU 初始 `M_MTE1(0)` 报 `set twice consecutively` | **本机CPU协议错误，数学尚未开始，不是真机条目或数值回归。** 真SDK `TPipe::Init` 在 arch2201 AIC 已 Alloc+Set 0/1/2（97–111行），析构会 Wait+Release（533–539）；不是前一 ICPU_RUN 污染，TBuf Init 本身未分配事件。自有3个free及2个ready必须逐个 `AllocEventID`，完成配对drain后 `ReleaseEventID`，不触碰TPipe保留0/1/2；`FetchEventID`不置占用位，不能连续Fetch充当并发槽。原CPU失败证据保留，修复后的完成另验；[SDK定义](../artifacts/cann_sdk_cube_public_20261010/impl/basic_api/kernel_tpipe_impl.h) |
| K-major v2 PV126 CPU `Load3D` 断言 `l1W > 0` | **本机API配置错误，非数值回归/非真机故障**。同版本QK24已完整逐位通过。公共`LoadData3DParamsV2`自动reset在`kernel_operator_mm_base_impl.h:173`调用LEFT FMatrix设置，并不随`fMatrixCtrl=true`改RIGHT；SDK原生B Prepare单独设置RIGHT。已修为显式`SetFmatrix(...FMATRIX_RIGHT)`加`reset={false,true}`；v3 CPU三例随后通过，旧失败不覆盖：[旧失败及QK完成](../reports/kmajor_20261011/load3d-failure/)、[SDK封装](../artifacts/cann_sdk_cube_public_20261010/impl/basic_api/kernel_operator_mm_base_impl.h) |

## 六、组合顺序与最少关键验证

1. **先交付明显增益的量化基底**：writer/request-packed只保留已测pack+Init；Q使用独立Init-only算子，canonical FE0保持原样。Batch8与static Matmul不入本次组合。核源恢复、ABI、UB、padding/nonfinite/±0/status与fresh build/CPU seam；沿用既有配对，不重做旧micro。
2. **C4沿一条继承链合并**：生产H16→H32（旧lane16错误域）→已测member pipeline/atomic；保留C1/precise、全部任务和native current suppression。以合并版本做完整核/错误预取/跨轮与图接缝门，不能拿不同context的单项比相乘。
3. **request windows并行完成容量协议，最终合流**：先保证native scheduler通知与lease/reset/reorder/epoch正确、实际token/capture容量分配；上述writer和六reader均由统一源生成。它不是量化优化的数学前置，也不能以6.3088×容量代替速度。最终一键bundle只使用一套明确格式/ABI。
4. **完成布局裁决，审查新的L1复用条件**：B-NZ、Vᵀ-NZ及当前PV K-major均已完整测回退，本版本退出组合，不重复micro或原样重跑。仅K/V的L1复用仍独立核实；先有SDK能力、容量/生命周期与可省实际工作证据，再决定新实验。AB尚无完整资格，不能用其免费预pack上限补足已失败的整核账。
5. **用剩余瓶颈决定actual-N split/owner与跨task流水**：重新核完整native-budget mixed关键路径，只将有明显收益依据的结构方案列入当前组合；短context、R1/小R、q1/q4/长C4不能凭一条长形外推。只运行新组合必要门，已有成功/失败micro不重复。
6. **交付门**：`tools/serving_variants.py`统一直启和observe的实际开关，验证q4/q1/C4与request-window格式选择；原生budget与GDN不变。真实NPU、同地址改输入graph、短合成服务完成/清理和性能分别记录。组合总收益只来自完整新旧配对或标明假设的阶段加权账，尚无证据宣称native2–3倍已达到。

当前选择以用户最新“过小单项不入组合”为准：保留明显增量与其必要依赖，过小正收益仍存档且不伪装成数值失败；失败根因和原始证据保持可追溯，避免重复实验。

## 2026-10-12 继续执行与 prefix caching 接缝

- **Prefix caching 必需兼容**：修复显式配置未传到原生服务、开启时无必要地禁止 mixed decode split 两处源码缺口；用户追加后所有服务入口默认显式prefix=true，主动false诊断保留；native/baseline/candidate一致。188项主机生命周期/入口/分流检查通过，包括实际原生hash/共享/refcount/驱逐和正context的main/first-MTP；NPU、图与真实服务命中仍未验收。用户未提供错误栈，不能把两处源码问题认作其未知错误的根因。请求sidecar研究路径仍未具备跨请求prefix恢复，不默认启用。[实现和边界](prefix_caching_compatibility.md)、[host报告](../reports/prefix_caching_host_20261012.json)。
- **完整decode交付时序**：现有源d817的QK/PV方法提交标记后仍有MMAD/FIX执行。小型R2/q4/C1341、4完整KV+尾行的6文件完整completion/popped已完成：normal1072361/profile1093991 ticks，扰动2.017%，全部数学逐位/oracle与3actor通过。中间完整轮PV周期约44K，QK/PV结果交付约86.5%，softmax与P交接约13.5%，KV供给暴露等待为0；仅本小形，不是NPU或整轮占比。下一主线审查AIC B块ND2NZ装载的串行暴露，不重复AIV NZ发布失败路径。[当前审计](../reports/decode_cost_audit_20261012.md)。
- **一侧BF16残差＋仿射码值**：2/3分量在完整原空间identity抵消用例均超冻结门；三分量输入重构残差0仍失败，停止该具体算法，不泛化所有BF16路径。新增3倍MAC、仅减少22.37%逻辑GM工作，普通用例通过不能覆盖抵消失败。逆旋转后移只有约0.585%矩阵工作上界，GQA6跨head共享已存在，均不作为主线收益。[结构筛选](../reports/decode_structure_screen_20261012.md)、[CPU证据](../reports/affine_bf16_residual_screen_20261012.json)。

- **实际dequant FP32值的BF16平面积**：与仿射码值方案分开筛查。2×2/3×2在4个CPU完整oracle用例通过（含identity抵消），不代表普适精度。4/6倍MAC、理想GM增加23.7%/43.4%；已有同形完整Matmul FP32/BF16仅1.213/1.326倍，暂无明显净收益支持，未入组合。合法FP16 metadata不保证两BF16平面精确的表示反例已记录。[筛选](../reports/dequant_bf16_product_screen_20261012.json)。

- **默认APC align容量重新核算**：当前条件参考几何、65615 tokens、无共享时FULL2.96552×，计原生5页GDN峰值预留后全部KV2.29545×。工具已支持新默认并执行7项主机规则测试；未测HBM，6–7×仍待实现兼容APC的新布局，不能沿用none投影。[报告](../reports/memory_64k_prefix_align_20261012.json)。

> 2026-10-12 #157 更新：MDL 与 rotate_vector_init 两算子按用户指示保持默认 candidate 开启；077548fa 真机首崩为冻结 status_guard Trap（详见档案 #157），归因此前不可读，现已加 `OSCAR_GUARD_HOST_SYNC=1` 按族归因开关（默认关闭、不改变默认路径）。两算子的 observe NPU/图门仍欠账，未把默认开启写成设备验收通过。

- **AIC官方MDL大块装载研究**：根据完整多KV交付链提出，保持现d817 AIV publisher、FP32/K128计算及错误域，仅切官方Normal/MDL模板；CANN编译与L1/L0资源静态断言已通过。官方CPU编译首次因独立暂存漏传oscar_launch.h失败，补公共头后编译与完整数值通过，原失败保留；这不是算子数值或真机错误。完整配对1079788→931042，1.159762932倍，全部输出/LSE/status逐位与oracle/3actor/六份完整日志通过，已达到新1.15倍单项门。实际B装载232→34条，history QK/PV由8×32KiB变1×256KiB，稳态PV周期约1.322倍；未少算GM字节，也不是整轮倍率。已打包独立MDL q4算子并接两固定candidate入口，q1/C4不变，observe新增NPU/graph/1.15倍完整CV+merge门；完整40项CANN/Torch编译通过，NPU/整轮未验收。[源与协议](../csrc/experiments/decode_mdl_load/README.md)。

- **MDL部署证据**：[完整配对及装载机制](../reports/decode_mdl_load_20261012/summary.json)、[真实装载对比](../reports/decode_mdl_load_20261012/load-analysis.json)、[生产包装与host接缝](../reports/mdl_decode_packaging_20261012.json)、[40项完整构建](../reports/mdl_decode_fullbuild_20261012/summary.json)、[真实NPU门实现](mdl_decode_gate.md)。C4的相同MDL机制已作独立H16基底研究准备，尚未执行，不外推q4收益或默认启用。

- **MDL后N128/K128尾段合并**：官方资源合法但当前q4增量筛选未达15%：QK总LOAD_2D仍32条，B失去双缓冲；即使乐观将QK/PV装载后尾段减半，稳态条件倍率约1.123。未生成/未运行，保留C4/q1有新完整证据后重开条件。[审查](../reports/mdl_n128_k128_feasibility_20261012.md)。

- **C4 MDL资格仍未建立**：真H16基底CPU20全1008任务28.28秒通过；候选CPU20在120秒未完成。独立CPU4保留所有任务且按源码重建owner后，baseline仍120秒超时，已停止换拓扑或加时。原始日志、44份失败PID专属GM临时数据均已归档校验，资源已回收；未证实OOM/SDK死锁，不判数学错误，更不判通过。C4生产未启用MDL。[状态](../reports/c4_mdl_load_20261012/summary.json)、[超时审查](../reports/c4_mdl_cpu4_timeout_review_20261012.md)。
- **压缩码字直接NZ研究**：在UINT16码字上新增4352B Gather，再按原Shift/And和FP32 Mul/Add形成NZ；history免去后置64KiB FP32重排。显式UB171680B，Cast/Mul/Add各4→8、source1/2精确域转换及Rv常量均须计费；host全码型/地址筛选通过，真实CPU/CANN helper已通过：K/V各65536码型与17个旧新FP32bits/status/非法值/尾部/guards用例，12.374秒；完整消费者和速度未验收。[设计](../csrc/experiments/decode_direct_nz/README.md)。
