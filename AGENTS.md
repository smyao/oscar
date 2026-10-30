# 当前工程约束

> 2026-10-30全面自查续修：修复seed原顺序采样混入held-out来源，candidate升级OptR v7，source seed仅用同一固定train分区，身份oscar_base_shared_train_v2；旧seed/PT/fit隔离，历史v1–v6原生trace严格认证复用。达到512行的完整prompt即停止；共享OSCAR统计/官方OptR目标/整层接受保持。离线固定缓存≤8GiB沿用原驻留，超限按query加权分块（完整K/V不截断），明确额外离线搬运及总峰值未测；默认无额外NPU重算。正交门显式拒绝非有限Gram误差。630项本地回归通过，线上CANN/runtime/timer/argv不改，不升级NPU/质量/性能验收；仅本地提交不推送。见docs/optr_oscar_base_alignment_20261029.md自查节。

> 2026-10-30用户要求尽量保持原OSCAR逻辑：固定candidate使用OptR v6，在指定oscar_base的共享、零均值seed上每层只优化一对Rk/Rv，保留官方FP32独立GQA目标与K→V；撤掉当前逐头独立搜索/头子集混选，最终整层接受。v6发布/加载严格共享，旧PT/fit隔离，seed及认证v1–v5 raw trace复用；不改变采样/clip/窗口/线上CANN/runtime/timer/argv。四卡层并行和已授权实验test政策保持，仅本地提交不推送。CPU对照不代表模型精度恢复；离线全头固定缓存有额外驻留量，线上操作无新增。详见docs/optr_oscar_base_alignment_20261029.md的10-30续修。

> 2026-10-29用户指定本机OSCAR拟合：明确授权只读 `/Users/loki.yao/Documents/workgit/oscar_base`（9cd2429），不扩展到其他禁止参考树。固定candidate继续OptR，改用该源码的FP32共享qqt/sst、CPU FP64 eigh+epsilon/UHP、零K均值作独立recipe6 seed，再运行官方OptR FP32独立GQA K→V目标；撤掉v4自定义训练目标，保留部署验证和已授权实验test政策。policy v5隔离旧PT/fit；同采样覆盖的历史raw trace精确认证复用，旧FP64 Gram不能冒充新seed统计。前4卡、四worker、线上CANN/runtime/timer/argv不改；本轮仅本地commit不推送，不连接真机。详见docs/optr_oscar_base_alignment_20261029.md；数学/CPU通过不能宣称模型精度恢复。本条覆盖下方旧拟合目标及旧拟合断点兼容描述。

> 2026-10-29 OptR离线并行：用户要求缩短2～3小时拟合，固定candidate在采样结束后自动按完整层调度至本次配置的四张NPU，每卡一个spawn进程；先绑定逻辑设备再初始化精度选项。CPU线程有界，单worker无进展独立超时，失败/信号/启动中断清理owned进程，全部退出后才发布。每层K→V、样本/80步/精度/联合选择/test政策和线上代码保持；仅精确认证兼容4204e22的v3串行断点，旧v1/v2不沿用。完整PT及原生trace身份不变，无需新开关。按用户要求本轮仅本地commit、不推送；见docs/optr_parallel_fit_20261029.md，未连接真机或验收设备速度。

> 2026-10-29 OptR精度续修：用户反馈OptR比原OSCAR掉点更严重，要求直接修复且继续启用。candidate仍为recipe5，但拟合policy升级 `optr_frame_bf16_ascend_v3`：按真实forward固定cut及BF16写读数值链训练，train覆盖4K/32K/64K；validation逐请求/逐forward/双分片非劣，缓存各头输出后穷举最多16种组合保留有效头。旧完整PT/拟合断点不得沿用，新旧采样计划不一致时自动重采，不能把旧validation/test转为训练。独立test仍按已授权实验政策记录，不参与重选；线上CANN/runtime/timer/argv不改。详见docs/optr_accuracy_fix_20261029.md；本地反例与测试不是模型质量/NPU验收。

> 2026-10-29部署更新：用户要求当前代码改回默认前四张物理卡 `0,1,2,3`，端口保持 `7878`，覆盖下方2026-10-28后四卡默认值。安装/打包编译/采样/OptR拟合/探针/正式服务仍统一读取 `configs/target.json`；固定candidate命令不附加设备参数，单卡拟合逻辑 `npu:0` 对应物理卡0。`--rear-cards`保留为显式后四卡选项，显式自定义配置仍有效；不连接真机或测速。

> 2026-10-28本地部署更新：用户要求启动、打包等默认适用后四张物理卡 `4,5,6,7`，端口保持 `7878`。以 `configs/target.json` 为唯一默认来源，安装/编译/采样/OptR拟合/探针/正式服务共用本次选择，覆盖下方历史前四卡默认值。固定candidate命令无需追加参数，`--rear-cards`保留兼容；单卡拟合的逻辑 `npu:0` 映射物理卡4。显式自定义配置仍有效，不继承其他任务卡号，不连接真机或测速。

> 2026-10-28用户要求参考官方OptR拟合并加快昇腾准备：candidate采用sequential K=KL+post-W_O误差、V=固定P_K的value误差，官方80步/lr0.02/零基每20步验证；NPU执行FP32大矩阵、CPU执行matrix_exp/Adam及最终部署验证。保留Qwen gate/D256、原采样、整层validation逐请求非劣和实验test记录规则。版本化新目标，不复用旧拟合PT/断点，只精确认证复用原生trace。线上runtime/CANN/timer/argv不改，不连接真机。见docs/optr_sequential_ascend_20261028.md；设备速度/质量未实测。本条覆盖旧CPU拟合及拟合身份不变要求。

> 2026-10-28用户明确授权“允许生成实验PT，先测试OptR”：独立test仍执行并完整记录，退化不阻断实验PT/后续服务；validation逐请求非劣、数值/身份/正交检查继续保持，test不参与重选矩阵。固定candidate无需新增参数。新证书schema2明确experimental、publication_policy及独立test通过/失败/退化层，服务加载实时显示，禁止把实验放行写成质量通过。本条覆盖下方独立test退化禁止发布的旧要求。保留历史拟合/trace身份，审核通过的旧头/完整层断点可自动复用，旧失败报告不充当已完成层；线上核/runtime/argv/timer不改。

> 2026-10-28用户明确恢复OptR测试：固定 `candidate` 在当前原生BF16、qqt_sst/perhead_center、显式预算配置直接启用OptR recipe5，无需新增开关，不允许通过撤下OptR默认来绕过测试。本条覆盖下方#173的默认撤回处置；独立test失败报告和禁止发布仍保持，不按test重选或放宽阈值。保留已有显式Hadamard/W8A8/native-budget兼容入口；显式请求不兼容OptR仍报错。线上核/runtime/timer/argv不改，不宣称原第7层质量失败已修复或模型/性能已验收。

> 2026-10-28直启资源生命周期（#174）：用户回传4个worker在新权重加载前各仅余1.19/29.49GiB，低于0.9所需26.54GiB；占用者身份未获证实。固定直启以外层监督替代exec，按本checkout物理卡持锁覆盖build/prepare/serve，退出及INT/TERM/HUP有界清理本次新进程组，日志/原始子进程状态及shell退出码落盘。正常服务不加时限、探针或设备轮询；仅显存错误后一次有界只读占用诊断。不自动杀历史/未知进程、不降预算、不改卡号或在线核/argv/timer；无真机复验。

> 2026-10-28 OptR独立test失败处置（#173）：用户回传第7层联合validation选中候选后，独立test明确退化，准备rc1而非超时。撤下未验收OptR的candidate强制默认，恢复perhead_center recipe3及原FP32 metadata/BF16性能组合；固定直启/探针共用，残留optr配置清除，显式不匹配PT仍报错。显式variant=None研究保留recipe5与全部原质量门，失败保存逐请求/分片原始指标，不按test重选、不放宽门、不自动重试。线上核/runtime/timer/argv不改；未实现OptR精度收益，未连接真机或验收性能。见docs/optr_independent_test_failure_20261028.md。本条覆盖下方10月27日的candidate默认OptR描述。

> 2026-10-27 OptR准备超时续修：用户回传CANN构建和7条原生采样均已成功，CPU拟合被通用1800秒总时限终止（#172）。candidate离线准备单独按任务规模分配有限总预算，当前上限24h；保留默认1800秒真实无进展终止，心跳不续期。认证原生trace不失效，新增原子头/完成层断点并保留全部联合validation/独立test门。只减少离线重复计算，不减样本/steps，不改线上核/runtime/timer/argv，不宣称目标整服务已复验。见docs/optr_preparation_timeout_20261027.md。

> 2026-10-27 OptR离线精修：用户要求review后直接改造且不增加在线开销。原生BF16 candidate在recipe3基础上启用独立recipe5输出感知旋转；gate后联合KV、GQA/全部TP输出、动态分片及独立test必须非劣，变更层验证总误差严格改善。仅准备/加载最终矩阵，不改线上CANN/runtime/timer/argv/MTP/GDN/graph。首次需要新增原生采样和CPU拟合，完整PT匹配后复用；不能冒称零启动成本或已达到模型/性能验收。显式Hadamard/W8A8保留旧路径；见docs/optr_implementation_20261027.md。

> 2026-10-23部署配置更新：用户要求当前服务默认使用前四张物理卡 `0,1,2,3`、端口 `7878`，以 `configs/target.json` 为准，覆盖下方历史设备/端口默认值。固定candidate直启命令无需附加参数；`--rear-cards`仍为显式后四卡选项。档案 #123：所有子进程共用本次配置，不继承其他任务卡号。

> 2026-10-23精度续修：candidate默认fp32_metadata读端策略，现有mode0用于prefill/decode/MTP/capture；已存FP32量化等级不再附加BF16 metadata/减法舍入。独立overlay继承v4门，旧合同/核不改；recipe3/PT/clip/旋转/窗口/argv保持，recipe4实验仍关闭。此为精度优先政策，区别官方CUDA的decode速度取舍；不能宣称模型已达1点以内。见docs/fp32_dequant_precision_20261023.md。

> 2026-10-23紧急回归：用户反馈recipe4默认后差距扩大到约10点。已用实际selector+D256反例证明四项score代理改善仍可令attention输出从1变0.7616。candidate撤下recipe4默认，恢复用户实测约3点的perhead_center recipe3、K=.96/V=.92，清除残留clip开关并绑定recipe3 PT；研究能力保留但不自动启用。继续检查跨KV256历史累积，未完成前不得声称内核或模型质量已验收。不要求用户诊断实验。

> 2026-10-23：用户实测center后差距从5–6点降到约3点，目标仍需1点以内。candidate进一步补齐逐头K clip，recipe4绑定本地预先固定的四项score代理非劣选择，不能冒称官方未公开clip表/选择器或LongBench达标。自动认证复用recipe3/2原始校准样本生成新PT，缺失才自动校准；只交付固定candidate命令，不要求用户诊断实验。CANN/timer/argv/窗口/MTP保持，详见docs/head_clip_alignment_20261023.md。

> 2026-10-21追加：用户拒绝诊断实验，只运行固定candidate一键命令。candidate升级官方perhead_center配方，recipe3/schema2；优先验证并复用旧校准原始样本重拟合，缺样本自动校准，新PT匹配后复用。Qwen27B TP4按全局KV头切矩阵与均值，全部K来源一致BF16中心化，MTP为identity+零均值。仅必要本地功能验证，不连接真机、不测速；CANN/timer/argv不改，不宣称质量已恢复。见docs/official_center_recipe_20261021.md。

> 2026-10-21最新：用户要求修复与官方OSCAR数值链不对齐并推送。candidate采用独立v4合同：历史/精确窗口/current统一BF16旋转域，CV partial保持FP32至LSE合并后再一次BF16逆旋转；current-only同样旋转，混合current不提前舍入。窗口在单次forward固定cut，保留native物理快照/MTP3/APC；不声称复刻SGLang不支持MTP的8-token分配器或未知V2 PT。旧合同/母源/冻结guard和timer保持不变；只做本地源码/CANN/CPU功能验证，不连接真机、不测速，只推BF16分支。详见docs/official_math_v4_20261021.md，模型质量未验收不得称5点已恢复。

> 2026-10-20最新：用户恢复额度并明确授权仅根据两天timer结果做定向优化，随后要求“不要改timer”。本轮限定C16 softmax有限性检查与BF16转换发射开销；现有timer脚本、hook、schema、汇总及生成器保持不变。保留FP32 online softmax/max/sum/alpha、原错误域、全部跨核/跨DMA所有权、MTP/graph/argv。允许源码/CANN/CPU功能验证，仍不由agent连接真机或启动性能实验；速度收益未测不得宣称。只推BF16分支，main/stable不动。

> 2026-10-19最新：用户要求停止大规模优化，仅交付当前分支32K计时诊断，由用户真机运行反馈。默认三臂改为native BS4、OSCAR BS4、OSCAR BS8（替代最初15/31），输出256；关闭采样的对照和有限profile重复分开。允许本轮显式诊断入口采集timer/设备PMU，不恢复日常入口计时或性能优化。终端摘要≤200行且≤50KiB，细节落盘；保留原生graph/MTP/APC/预算/精度，禁止把host时间、四rank累加或小batch推断冒充设备/整轮结论。仍不由agent连接真机，仅推BF16分支。

> 2026-10-18最新：默认所有入口使用原版Qwen3.8-27B，移除默认ascend权重量化及旧YaRN×4覆盖，按模型自身RoPE；显式W8A8仍支持，MTP/APC-off/16384/设备端口不变。官方OSCAR339368e的qqt/sst→U·H·Pbr替代默认data-free H；模型采样在NPU，离线FP64统计/eigh采用CPU同精度实现，不是推理回退。candidate默认使用版本化144B FP32 normalized-zero metadata/BF16旋转与attention，原136B与旧oracle留作历史；新v3容差预先固定。用户禁止性能测试/真机连接，仅源码/编译/功能/图契约验证。只推codex/oscar-bf16-attention，main/stable不动。

## 2026-10-16 BF16分支任务（优先于旧FP32-only门）

- 用户报告当前main的32K约34分钟并同意保留；只在`codex/oscar-bf16-attention`开发和推送，基底为最新main `7047ce36`，main/stable不移动。默认vLLM argv、设备、端口、原生调度/GDN/prefix/MTP保持。
- 用户明确撤销attention只能FP32计算的旧约束，要求以原生BF16计算方式提速并自行评估必要精度。QK/PV/最终Rv矩阵乘法统一BF16操作数、硬件FP32累加；online softmax/max/sum/alpha、LSE与merge保持FP32。INT2 codec、量化及Q旋转规则不变。
- 新独立合同`configs/acceptance_bf16.json`与`tools/bf16_reference.py`在设备测量前定义；旧FP32 oracle保留差异诊断，不能用旧FP32逐位门否决已授权的BF16舍入。新oracle容差不事后放宽，metadata/status/错误域和图地址仍严格。
- 用户随后明确本轮不用测速，只需把系统调通后由用户真机验证。停止CAModel/performance实验和新BF16入口速度门，只验证编译、正确性、错误域、图/启动接缝；performance标为用户要求未测，不能伪造passed。继续禁止连接真机，不能把BF16标签当作已转为访存受限或端到端>30%。


## 2026-10-15 当前交付约束（优先于下方历史分支策略）

- 用户确认 `codex/aisbench-stable` 的 APC 开/关均可工作；以 `f96820f1` 整体同步主分支作为优化基底，保留所有已工作机制。默认启动参数冻结：APC 关闭、显式 token budget 16384、原生调度器，以及 `configs/target.json` 的其余设置。完整 argv 冻结证据在 `reports/stable_32k_64k_20261015/frozen_launch.json`，不能通过修改启动参数换性能。
- 本轮目标改为 32K 和 64K、300 请求、512 输出，两场景端到端性能均比同参数且完全禁用 OSCAR 的原生基线提升超过30%。完整客户端证据及两种30%口径见 `reports/stable_32k_64k_20261015/analysis.md`；旧120请求、2–4倍等目标不用于本轮验收。
- 本轮禁止以任何方式连接或使用真机；本地源码、CPU-debug、CANN编译与CAModel独立完成，最后交付一键命令由用户验证。不要索要已完成AISBench重跑，也不能把本地算子倍率写成整轮达标。
- 用户随后明确接纳完整stable→B1（3888161→3172246 ticks）进入主分支作为继续迭代基底；不因相对中间pipeline仅4.1%而剔除。每点不再机械要求1.15倍，保留真实净收益并按新基底去重计账，精度和协议门不变。继续叠加到两场景相对同参native端到端超过30%；完成后推主分支给用户验证，stable仅由用户验证后维护。

先读 `oscar_ascend_agent_start.md`，按其档案引用制度执行。`issue1_full_record.md` 实际为173条（G1–G34、#1–#139），按标题定位；旧标题条数与旧行号存在漂移。

- 只允许参考本项目 `references/` 内指定PR和原生实现；禁止读任何失败OSCAR项目代码。所有参考树只读，不运行格式化或修改。
- 适配代码为外部插件与独立AscendC工程，不能覆盖原生vLLM/Ascend源码，不使用 `cp` 部署。
- 配置以本次 `configs/target.json` 为准：按本次启动文档附录 A/F 默认物理设备 `0,1,2,3`、SOC `ascend910b4`、端口 `8989`。不能从旧项目记忆或已有环境继承其他任务卡号；显式配置非法时仍报错。
- 用户另明确授权直启入口`install_serve.sh --rear-cards`本次使用物理设备`4,5,6,7`和端口`7878`；这是显式可选模式，默认配置和其他入口不变。
- 用户最新要求：保留真实 NPU 算子、CV/旋转和完整服务探针；移除默认部署中的环境清单、原生源码扫描、readiness 前置审计。正常流程为安装/编译/探针/自动旋转文件/服务探针/资源清理/正式服务，任何探针失败不得继续。该要求优先于启动文档旧的环境审计步骤。
- 部署阶段及服务子进程的输出和错误必须实时打印到当前终端，同时保留相位日志、状态和真实退出码；不能只把 traceback 写入文件后让用户手工查找（档案 #94/#95/#125）。
- `docs/checklist.md` 是42个稳定checkpoint；源码、CPU、CANN、设备完成、图捕获、图回放、质量、性能分别记录。
- CV/history/window、production RuntimeProvider、物理页精确快照、MTP位置修正和全服务probe已有实现。CANN VM编译与官方CPU-debug通过；node93已通过75项store/merge原语真NPU探针，后续日志已进入TP4/MTP模型加载，最新已完成编译/初次profiling与KV预算，后续已通过health与128→16短请求；后续128/16K/32K/50K串行请求均已完成；混合并发已现300s请求死线超时失败（心跳证明非死锁、为时间预算问题），性能未验收（#130/#131）；已支持none/align布局并区分物理存储与metadata128虚拟块，完整NPU/图/性能未验收，readiness或编译成功不能冒充这些证据。
- `ops/reference.py` 是独立测试oracle，绝不加入生产路由。精度冻结于 `configs/acceptance.json`，不能事后放宽。
- 凡修改CANN算子，先亲读档案同类条目与D.4完整原始打点，文件头加入档案引用及D.4四问。
- 新真机错误才追加档案；本机CPU测试或理论判断不能伪写为真机记录。
- 用户已授权完成后推送至 https://gitcode.com/jpl123/gpt_new_oscar_kimi.git；只推送本工程，保留真实验收边界。
- 每轮交付必须同步到固定直启命令 `git pull --ff-only && bash scripts/install_serve.sh --variant candidate`：不仅编译最新产物，还要实际启用本轮候选优化；不得仅更新probe。候选开关统一由`tools/serving_variants.py`维护，直启与probe共用；回归须验证配置传到服务且q4/q1/C4选择了对应算子。后四卡模式仍叠加`--rear-cards`，默认设备与端口不变。
- #157：status_guard 属 fe0 字节冻结集（`FE0_KERNEL_SHA256`），不得改动；生产 trap 归因统一用 `OSCAR_GUARD_HOST_SYNC=1`（默认关闭、图捕获路径不受影响），异常链给出 site/family，不得为归因修改冻结算子或新增生产回读。2026-10-13复审：later-draft 只允许依据原生 `discard_request_mask` 将明确丢弃行置为 padding，由外部 proposer 作用域与 metadata builder 在 prepare 前完成；禁止从 slot 解析失败反推 discard，禁止每层全表扫描清空活行。该主机路径不经 pin/生成链，冻结算子不变。
- 用户已在835fc22完整运行过AISBench；后续不得再要求重跑AISBench。分析已有复制日志和片段，必要验证只用短时合成算子/图/服务probe；片段缺失时不得伪造全程占比，也不能把短probe当成完整模型质量或8分钟性能验收。

真机部署只需 `git pull --ff-only && bash scripts/install_probe_serve.sh`，使用当前 Python 环境。`.venv/bin/python -m pytest -q` 仅用于已配置虚拟环境的本地开发机；`bash scripts/validate_vm.sh` 仅用于带 Lima 的 Mac 宿主机，均不是真机前置步骤。单独 NPU probe 是 `python3 -m tools.deploy --only probe-ops`，无 NPU 时必须失败，不能用 CPU 替代。已在真机跑通过完整探针后，可用 `bash scripts/install_serve.sh` 仅编译安装并直接启动正式服务（无任何测试/probe）；它不提供验收证据。

## aisbench-stable 分支启动策略（用户 2026-10-13 最新指令）

本分支同步 codex/oscar-ascend 当前逻辑，但默认关闭 prefix caching，显式传入 `--max-num-batched-tokens 16384`，使用原生调度器。此要求优先于历史文档中的默认开启 APC/省略预算约束，仅适用于本分支；不表示真机 Trap 已修复或性能已验收。
