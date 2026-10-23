# 当前工程约束

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
