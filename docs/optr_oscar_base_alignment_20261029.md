# 按用户指定 oscar_base 恢复 OptR 拟合来源

## 2026-10-30：相对原 OSCAR 的再次精度排查与 v8 修复

本轮以 `360a682` 为基底，再次只读对照用户指定的 `/Users/loki.yao/Documents/workgit/oscar_base`（`9cd2429fb4fb934889fa6f51da5d54396012b5df`）、本工程原生参考源码和已固定的 OptR 官方目标。该本机仓库不是官方 OSCAR 仓库，二者不能混称。沿用档案 #170/#173/#175/#176 的教训：局部输出误差改善不能直接等同于模型任务精度改善，现场 40.94 的 SHA/PT 未知，不能用本地反例宣称已解释全部掉点。

**结论：发现并修复了两项确定的候选误接受问题。** 默认升级 `optr_oscar_shared_projection_ascend_v8`，仍执行共享 OptR。共享零均值 seed、原官方 FP32 独立 GQA 的 K→V 目标、80 步/lr0.02/验证频率、采样、clip、精确窗口、四卡层并行和线上数值链均保持。本次改动集中在候选验证、证书和版本隔离；不是替换成另一套自定义优化损失。

### 1. 已复现：帧均值与末 token 都改善，仍可能损伤其他 prefill query

v7 已要求每个请求、每个 forward 和每帧末 query 非劣，但未检查帧内其他已采样 query。prefill 中间位置的 hidden state 会继续影响后续层的 K/V，因此不能因其不是末 token 就忽略误差增加。

本次用真实 `fit_shared` 跑完默认 80 步，不替换优化器或 attention 重放：H=2、D=8、GQA=2、hidden=12、完整 K/V 长度81；两个连续 forward 共采12个 query，train/validation/test 使用独立随机样本，W_O 是 BF16 可表示值。选中 K 第80步、V 第61步。

| validation 指标（单个分片） | seed | OptR 候选 | 旧门结果 |
| --- | ---: | ---: | --- |
| 全部 query SSE | 3.000481882 | 1.090141121 | 改善63.67% |
| 第一个 forward SSE | 0.009283797 | 0.004670466 | 改善 |
| 第二个 forward SSE | 2.991198085 | 1.085470656 | 改善 |
| query 43 SSE | 0.000091371 | 0.000356456 | **增加290.12%，旧门未检查** |
| query 77 SSE | 0.184739374 | 0.232181066 | **增加25.68%，旧门未检查** |

两个 forward 的末 query 也均改善，故 v7 确实接受该候选。v8 保留每个已捕获 query 的 SSE/teacher energy，并逐位置要求非劣；这个候选会被拒绝。最终仍按整层选择一对共享矩阵，不重新引入逐头混选。新测试独立证明逐 query 检查本身足以拦截，不借用新增的投影检查获得通过。

证据：`tests/test_optr_query_gate.py`、`reports/optr_precision_review_20261030/query_counterexample.json`。这些是可复现的合成 CPU 数值，290.12% 是某一小误差位置的相对增长，不是 LongBench 掉点比例。

### 2. 已复现：FP32 W_O 误差更小，不保证 BF16 输出误差更小

当前回放已包含 BF16 attention 和 gate，但旧选择器在 W_O 的 FP32 投影结果上结束。原生参考 `vllm_ascend/ops/linear.py` 使用 `F.linear`，`RowParallelLinear` 再执行 TP all-reduce；BF16 模型存在可见的低精度输出边界。距离 BF16 舍入中点很近时，连续 FP32 误差的大小顺序可以反转。

精确反例全部由 BF16 可表示操作数构成：gate=[1,1]，W_O=[1,1/256]，teacher attention=[1,1]，seed=[1,1/2]，candidate=[1,9/8]。

| 结果 | teacher | seed | candidate |
| --- | ---: | ---: | ---: |
| FP32 投影 | 1.00390625 | 1.001953125 | 1.00439453125 |
| BF16 输出 | 1 | 1 | 1.0078125 |
| FP32 SSE | — | 2^-18 | 2^-22（改善16倍） |
| BF16 SSE | — | 0 | 2^-14（退化） |

另有局部 rank 输出本身已精确可表示、仅 TP 最终输出舍入就产生同类反转的独立反例。测试将两者嵌入 D256/GQA6/TP4，并单独隔离 attention 结果以定位投影边界。这里没有借助不确定的 HCCL 归约树解释错误。

v8 复用原来的每 rank FP32 W_O 乘积，同时检查三种输出视图：

1. `fp32`：原有 FP32 投影、CPU FP64 TP 求和指标，仍作为总量严格改善的必要条件。
2. `bf16_output`：同一求和结果按 BF16 输出舍入。
3. `bf16_rank_output`：每 rank 的投影先按 BF16 舍入，再求和并按 BF16 输出舍入。当前生产采样强制一个 KV 头对应一个 TP rank，因而此处 head/rank 映射一致。

三种视图均检查请求、forward、末 query 和全部已采样 query 非劣；不同视图的 teacher energy 分别计算。CPU FP64 求和是明确的参考约定，**不代表逐位复刻实际 NPU GEMM、融合通信或 HCCL 中间累加**。新增检查针对已证明遗漏的输出舍入风险，不把参考视图宣称为真机数值验收。

证据：`tests/test_optr_projection_gate.py`、`reports/optr_precision_review_20261030/projection_counterexample.json`。调用计数验证每 rank 仍只有原来的一次 attention replay、actual/teacher 各一次 W_O 投影，没有因三种视图重算 attention 或增加矩阵乘法。

### 3. 与指定 oscar_base 的差异及剩余掉点可能性

| 环节 | 当前对照结论 | 对精度的影响与处置 |
| --- | --- | --- |
| seed 统计与旋转公式 | FP32 共享 qqt/sst、按实际行数合并 rank、CPU FP64 eigh+epsilon/UHP、零 K 均值已对齐 | 实际参考函数对照、分块统计、非连续 Q/有效行截取通过；本轮未发现新的确定性偏差。重复特征值可有不同合法特征基，属于固有数值敏感性，不擅加 ridge 或改变参考公式 |
| 最终旋转参数结构 | v6 起每层一对共享 Rk/Rv，v8保持 | 已撤掉逐头独立拟合和子集混选；仍可能因局部非凸优化选到泛化较差的旋转，不能保证严格优于 seed |
| 核心优化目标 | 保持固定官方 FP32 独立 GQA KL/output，先 K 后固定 K 优化 V | 官方代理与联合多头、Qwen gate、BF16 部署存在差别；最终部署验证缓解该差别，但不能保证未见数据质量。按用户要求不再自创训练目标 |
| 数据与覆盖 | 当前 GPQA 记录三分区；每分区打包4K/32K/64K各一条请求，并采decode8；每forward最多8个query。seed只取同一train分区前缀、最多512行 | 198条源记录不等于198个独立长上下文请求。指定 oscar_base 使用其短中文校准输入，两者语料并不相同；少量打包请求和稀疏位置未必覆盖 LongBench 的语言、检索和长距离分布。此轮未增加采样量或改变数据流程 |
| 验证接受 | 新增所有已采样 query 与两种 BF16 投影视图 | 已修复上述两个确定反例；覆盖范围仍是已捕获的位置，不是整段全部token。单层输出 SSE 也不等同答案正确率 |
| 在线旋转精度 | 指定 oscar_base 的 `backend.py:569–580` 使用 FP32 旋转/逆旋转；本工程 `oscar_ascend/runtime.py:813` 采用已授权的 BF16 操作数路径 | 这是仍存在的实际数值差别，不能把“seed公式相同”说成“线上完全相同”。本轮不更改线上核和既有性能合同，不能据 CPU 验证承诺消除该差别 |
| 层间分布 | 原生 BF16 teacher 捕获，离线逐层校准 | 推理时前层 INT2 误差会改变后层输入；单层固定 trace 不包含完整的跨层误差传播。增加接受粒度只能降低已见风险，不能证明整模型不退化 |
| 独立 test | 按用户授权执行、记录，但退化仍可发布实验 PT | v8新增视图/query退化同样显示 failed、保留具体位置，不用test重选或改标passed。实验放行本身不构成质量保证；没有擅自撤销该授权 |
| 旧产物/实际部署 | 当前 policy/profile及拟合实现哈希严格隔离，固定candidate选择v8 | 旧v7 PT/fit不能冒充新验证结果。现场40.94使用的commit/PT仍未知，无法断言当时已加载本次或前轮修复 |

上述区别中，采样分布、FP32/BF16线上差别、官方损失与部署差别、跨层误差传播都可能使 OptR 不如对照 OSCAR；现有证据不足以给它们排序或分配掉点占比。不能为了宣布精度改善而任意加步数、换loss、改clip、放宽阈值或绕过OptR。

### 4. 兼容、开销与验证

新策略身份为 `optr_oscar_shared_projection_ascend_v8`；最终仍 recipe5，seed仍 `oscar_base_shared_train_v2`/recipe6。旧最终PT和fit断点隔离，当前train-only seed可复用；v1–v7原生trace按历史原身份和实际采样覆盖精确认证，不重新贴标签。七版冻结policy/profile SHA回归通过。缺失query/投影视图、混用样本、teacher身份不一致、非法数值或不完整证书均拒绝。

线上 attention、codec、runtime、CANN、timer、MTP/GDN/graph、设备0–3、7878及完整服务argv均未改。OptR保持启用，没有新增关闭或绕过开关。新增成本是**离线CPU转换/求和/逐query指标和证书的保存、启动校验**；无新增attention回放或W_O矩阵乘，但不能宣称首次准备与加载总开销为零，也没有设备测速证据。新PT首次需按新身份拟合，旧认证采样可复用。

统一相关回归 **694 passed、0 failed、0 skipped**，包括参考OSCAR公式、官方OptR目标/梯度、真实80步反例、D256/GQA6投影、证书/发布/断点/四进程清理、旧模式和固定入口。首轮针对性检查有3个旧测试夹具缺少v8证据而失败，补全夹具后74项通过，随后统一694项通过；生产门未放宽。禁网wheel构建、包内字节匹配、隔离导入和基底启动配置/argv对照均通过，详见 `reports/optr_precision_review_20261030/verification.json` 及同目录原始日志。

本轮仅本地提交、不推送，不连接真机、不重跑AISBench，也不升级既有CANN/NPU/图/模型质量/性能验收状态。固定命令仍为 `git pull --ff-only && bash scripts/install_serve.sh --variant candidate`；因为本轮未推送，远端pull尚不能取得本地修复。不能将本地694项通过写成 LongBench 40.94 已恢复或性能实测无回退。

## 2026-10-30：修改后的全面自查与修正

本轮以 `0e3cd5f` 为基底，按“检查 → 可复现问题修正 → 完整复查”执行。固定 candidate 升级为 `optr_oscar_shared_train_seed_ascend_v7`，继续实际执行共享 OptR。下节 v6 的共享数学仍适用；其 seed 直接复用和无界固定缓存描述由本节覆盖。

### 发现并修正

1. **seed 与 held-out 来源交叉。** 原 seed 按 GPQA 原始顺序取前 512 行，而 OptR 按记录 SHA 排序分 train/validation/test。真实 bundled 198 条记录中，原始前三条分别属于 test、test、validation；因此初始化也用到了 held-out 来源。现抽出同一分区函数，seed 只取 66 条 train 记录的有序前缀。统计仍取最多 512 个对齐 Q/K/V 行，FP32 qqt/sst、rank 合并、FP64 eigh+epsilon/UHP、共享和零均值均不变。该遗漏损害验证独立性，不能直接归为 LongBench 40.94 的根因。
2. **采够 seed 后仍校验无用 prompt。** 原流程先渲染完整 token budget，再截取覆盖 512 行的完整 prompt；后面的超长 prompt 可能阻断本无需使用它的准备。现完整 prompt 达到采样上限就停止，实际参与的 prompt 仍严格执行预算、长度和完整性检查。recipe2/3 行为保持。
3. **大采样配置的离线固定缓存无界。** 共享搜索同时驻留全头 train/validation，默认形状固定缓存约 3.60 GiB，但合法的四个约 64K 长度、每 forward 64 个 query、decode 64 配置约为 50.47 GiB。现先按 tensor 形状核算：不超过 8 GiB 保持原驻留方式；超过时按 query 分片，一次固定缓存不超过 512 MiB，保留完整 K/V、所有 query、原顺序和步数。chunk 按 query 数加权，保持原均值目标；单个完整历史加一条 query 仍超限时，在设备分配前明确失败。额外搬运和固定概率重算只发生于超预算离线配置；这两个阈值不是包含 autograd、工作张量和 allocator 的实际总峰值保证。
4. **正交检查对非有限结果显式拒绝。** 三个检查入口补上 Gram 误差的 finite 检查，复用已有标量，无新增矩阵乘法或放宽容差。负例模拟 backend NaN；本地有限极大值反例产生 Inf 已被旧门拒绝，不能把防御性修正写成已复现 NPU 故障。

新 seed 身份为 `oscar_base_shared_train_v2`（仍 recipe6）；最终 PT 仍 recipe5。旧 seed、最终 PT 和拟合断点不能充作当前结果，固定命令自动准备新身份；原生 trace 在 OSCAR 关闭的隔离进程采样，与 seed 数值无关，覆盖一致时仍按 v1–v6 原身份精确认证复用，不将旧 seed 换标签。历史六版 policy/profile 冻结 SHA 均匹配。

### 修正后的复查

| 检查面 | 结果与边界 |
| --- | --- |
| 本机 OSCAR 数学 | 只读指定源码 AST 对照、FP32 统计与 rank 合并、FP64 UHP、共享/零均值通过；本轮未更改这些公式 |
| OptR 核心 | 固定官方独立 GQA K/V loss、共享参数梯度、逐步 Adam、固定选中 K 后的 V 通过，含 D256/GQA6 |
| 分块等价与生命周期 | loss/gradient/Adam、完整 K→V 搜索和选择与驻留路径匹配预定 FP32 容差；weakref 检查捕获并修正评分 `enumerate` 暂存前一行导致重叠驻留的问题 |
| 默认准备开销 | 原逐头一次 stage、共享 W_O、teacher 和 V 固定概率缓存保持；选中 K 仍只搬运一次，默认不增加 NPU 重算 |
| 数据隔离与兼容 | 真实 GPQA seed train-only；trace 全计划逐 token SHA 与修改前一致；recipe2/3 保留；旧 seed/PT 拒绝、历史 raw trace 精确认证 |
| 接受/发布/加载 | 整层共享候选，validation 每请求/forward/terminal 非劣和整层严格改善保持；test 仍按已授权实验政策记录，不参与矩阵重选；原子发布/中断恢复/四进程清理通过 |
| 服务与安装包 | candidate 继续 OptR，摘要显示 train seed；前四卡、7878、完整 argv、线上 attention/codec/runtime/timer 保持；wheel 构建、字节核对及隔离导入通过 |

统一回归 **630 passed、0 failed、0 skipped**。证据：`reports/optr_self_review_20261030/verification.json`、`pytest.xml`、`source_package_checks.json`；seed 两项真实红例日志保留。分块开发期间的一项单 query 测试夹具错误和评分引用滞留在统一回归前修正，没有放宽产品质量门。

仍有明确适配差异：参考 OSCAR 用短中文 prompt，这里用固定 GPQA train 前缀；官方 OptR 独立优化各头，这里按用户要求约束为每层共享矩阵；官方 FP32 优化目标与当前 Ascend BF16/FP32 metadata 部署由独立输出验证连接，采样包含稀疏 query、真实 forward 窗口及 Qwen gate。它们没有被宣称为源码逐位等同，也没有证明对任意数据集都改善。独立 test 仍可产生带退化记录的实验 PT。

本轮未连接真机、未做 NPU/图/LongBench/速度测试；不能宣称准确率恢复或实测性能无退化。线上每 token 计算和 KV 格式未变，产物加载仅增加已有标量的有限性判断。只本地提交、暂不推送。原固定命令不变；远端未获得本地提交前，真机 `git pull` 不会取得本轮修复。

## 2026-10-30：保留 OSCAR 共享结构，只替换寻优过程

本轮基底为 `d08c3d7`，对应用户“尽量保持原有 OSCAR 逻辑，只更换核心 OptR 的拟合寻优”。固定 candidate 现在使用 `optr_oscar_shared_upstream_ascend_v6`，仍实际执行 OptR，不增加开关。下面的 10-29 记录保留为历史；其中逐头独立优化、头子集选择和 v5 身份已被本节覆盖。

已确认的差异：指定参考 `oscar_base@9cd2429` 的 `tools/gen_rotations.py:207–231` 始终为每层输出一对共享矩阵；d08c3d7 只让初始 seed 共享，随后独立优化每个 KV 头，并允许头子集混选，最终矩阵不再共享。这是源码上的偏离，尚不能归因为用户全部精度损失。

本轮改动限于离线搜索和相应产物认证：

- 每层只学习一个 `A_K` 和一个 `A_V`，`R = R_base @ exp(A−Aᵀ)`，所有 KV 头使用同一 R。仍使用固定官方 FP32 独立 GQA 损失、原 clip/窗口、80 步、lr 0.02 和原验证频率。
- loss 按 `mean_KV_heads(mean_requests(phase_loss))` 归约。每一步固定同一个 R，逐请求累计 dL/dR，再一次反传到 A、一次 Adam 更新；不逐头轮流更新，不平均已独立拟合的矩阵，不二次除以头数。
- K 在全头 held-out 均值上选出唯一结果后，所有头固定该 K，再拟合共享 V。最终只允许整个共享候选与整个 OSCAR seed 二选一，保持每请求/forward/terminal query 非劣；不再混选部分头。独立 test 仍只报告实验结果，不重选矩阵。
- 保持现有 perhead_center 存储与 TP ABI，离线复制同一矩阵；发布、加载和完成层断点要求所有头的 Rk/Rv 严格相等、K mean 为零。共享拟合断点单独保存，不能用旧 head 断点代替。
- 已认证 recipe6 seed 直接复用；采样覆盖相同的 v1–v5 raw trace 按原身份认证复用。旧逐头 PT/fit 不再加载，固定命令自动准备新 v6 PT。

这称为“在 OSCAR 共享矩阵约束下使用 OptR 目标”，不是宣称官方 OptR 本身使用共享参数。D256、4 个 KV 头时，每层 K/V 的有效正交自由度从 261120 降至 65280；可减少针对少量校准请求的自由度，但不构成泛化或 LongBench 分数改善保证。新可行域仍包含原 OSCAR seed，验证未改善时保留该层 seed 是优化器的接受规则，不是关闭 OptR。

本轮保留的边界：采样仍为当前 GPQA/长序列 trace，没有改成参考短中文 prompt；seed 仍为参考数学的 FP32 共享统计/FP64 eigh+epsilon/UHP、零均值。INT2、K=.96/V=.92、sink/recent、BF16/FP32 metadata 数值链、MTP/GDN、CANN/runtime、timer、argv、前四卡和四 worker 均未变。不声称整套数据采集或线上实现与该参考逐位相同。

成本：线上每 token 没有新增操作、buffer 或 KV 字节。离线仍计算全部头的损失；CPU 正交矩阵/Adam 更新从每头一次变为每层一次。流式反向只保留一个请求的大图，V 阶段缓存选定 K 的概率后释放 Q/K/teacher/mask。固定 train/held 张量会同时驻留所有头，固定缓存约为旧逐头拟合的 H 倍；不能声称离线峰值显存不增或速度实测提升。

验证入口：`reports/optr_shared_oscar_20261030/verification.json`。覆盖固定官方函数在共享参数约束下的 loss/梯度、D256/GQA6、逐步 Adam、条件 V、严格共享、独立 test 报告、旧 PT/trace 身份、断点恢复、真实本机四进程和固定启动接缝。首次回归有一个旧负例同时破坏共享和正交性，因此先触发新共享门；已改为同时破坏各头相同元素以单独验证正交门，原失败记录保留。

无真机连接、NPU/图、模型准确率或性能测试；CPU 通过不能标为质量恢复。仅本地提交，暂不推送。固定命令仍为 `git pull --ff-only && bash scripts/install_serve.sh --variant candidate`，远端尚未包含本地提交时不能据此声称真机已生效。

## 以下为 2026-10-29 历史交付记录

本轮将固定 candidate 改为：**用户指定 OSCAR 的共享、零均值基底 + 官方 OptR 的 FP32 分阶段优化 + 现有部署输出验证**。它继续实际运行 OptR，不增加关闭或绕过开关。修改只涉及离线准备、身份认证和启动摘要，线上 attention、codec、缓存布局、MTP/GDN、timer、服务 argv 均保持。

本轮基底为 `478fe18c850edfdba4505c37c38da44fd25afa58`。用户指定参考 `/Users/loki.yao/Documents/workgit/oscar_base`，明确授权只读该目录；未读取其他被禁止的实现，未修改参考目录。按用户已有指令只做本地提交，不推送，不连接真机、不跑性能实验。

## 源码核对结论

参考仓库为 `smyao/oscar`，commit `9cd2429fb4fb934889fa6f51da5d54396012b5df`。它是用户指定的本机实现，不能称为官方 FutureMLS-Lab OSCAR 仓库。

| 环节 | 用户指定 OSCAR | 修改前 OptR v4 | 本轮 |
| --- | --- | --- | --- |
| K 统计 | FP32 `Q_g.T @ Q_g / (N·G)`，同层跨头、跨 rank 合并 | recipe3 逐 KV 头 FP64 统计、trace 归一 | 按指定源码共享统计，不做 trace 归一 |
| V 统计 | FP32 `w=kᵀΣqk`，归一后平方根加权 V 的协方差 | recipe3 逐头 FP64 energy moment | 按指定源码 FP32 sqrt-weighted 顺序 |
| 特征分解 | 合并后 CPU FP64 `eigh(sym(cov)+1e-6 I)`，`U·H·Pbr` | 逐头 eigh，无这项 epsilon | 按指定源码；epsilon 先 FP32 创建再转 FP64 |
| 基底矩阵/均值 | 每层共享一对 R，零 K mean | 每头不同 R，非零 K mean | 共享 R 展开到现有逐头存储，均值严格为零 |
| OptR 优化目标 | 该仓库没有 OptR 优化器 | BF16 STE、native teacher、gate、joint GQA 自定义目标 | 固定官方 OptR 源码的 FP32 原空间 K/V 损失 |
| 梯度归约 | OptR 官方对所有 KV 头再取平均 | 分别训练各头时漏掉该系数 | backward 除以全局 KV 头数，lr 保持 0.02 |

源码哈希写入 seed profile：`calib.py` 为 `a4db413470e123102b1423bfb933e83a789b417590ebde2ddfd9eb2fc1b5b9b7`，`gen_rotations.py` 为 `0dfbd105e9ff63018bbaacd93a84e6a8dea1efe3c99a07fa7e2d410e38ab405e`。

以上是确定的实现差异。现有 40.94 分数没有现场 commit/PT 身份与同条件配对结果，不能把某一个差异认定为全部掉点原因，也不能把数学对齐写成质量已恢复。

## 实际拟合路径

1. `tools.prepare_optr_seed` 在独立原生采样进程中收集 post-RoPE Q/K/V。按参考默认上限，只保留前 512 个对齐 token；不足 512 时使用本次实际完整样本数。取得上限后立即计算协方差并释放收集缓冲。
2. 继续使用当前固定 GPQA/chat-template 数据来源，保留完整 prompt，取得 512 行的那个完整 prompt 后停止后续前向。没有复制参考的短中文 synthetic prompt；这一采样差异已写入 provenance。原配置 30000 是选取完整 prompt 的预算上限，不表示此 seed 拟合用了 30000 行。
3. CPU 合并 rank 的 FP32 协方差，按指定源码进行 FP64 eigensolver 和 U·H·Pbr。生成独立 recipe6 seed，矩阵展开成现有 `[global_kv_heads,D,D]`，K mean 填零。`perhead_center` 字段在此仅是既有存储 ABI；`fitted_grouping=layer`、`key_centering=none` 与启动摘要说明真实拟合含义。
4. 在基底上实际运行 OptR K→V 优化，来源固定为 `daniel-eai/Output-Aware-INT2-KV-Cache-Quantization@66d46561b9bdf5d715f591e0f5845a4cbd927dee`。对给定 μ，K 使用 `Keff=quant((K−μ)Rk)Rkᵀ`；本轮 μ 按所选 OSCAR 基底为零。K loss 为每个 GQA 头的 `KL(Pt||Pq) + mean((((Pq−Pt)V)WOjᵀ)²)`；V 固定选中的 K，使用 `mean(((Pq(Veff−V))WOjᵀ)²)`。逐 GQA 头和序列分别平均。
5. 保持 80 步、lr=0.02、上游零基 0/20/40/60/79 验证，以及 `R_base @ exp(A−Aᵀ)`。各头独立 Adam 的反传 loss 除以全局 KV 头数，复现上游 all-head mean；不通过缩小学习率替代。
6. 现有 BF16/native teacher/gate/全 TP 联合输出只用于最终部署验证，不再定义训练损失。每请求、每 forward、terminal query 和双分片的非劣验证不放宽。独立 test 仍完整执行并按已授权实验政策报告；不用于重选矩阵，退化 PT 明确标为实验结果。

保留两项必要部署适配：INT2 分组仍为现有 head_dim=256 和 K=.96/V=.92；一个采样轨迹包含多个真实 forward，精确窗口按各 forward 的 context/cut 计算，不能直接使用轨迹最终长度的 recent 窗口，否则重引 #175。稀疏 query、GPQA 拼接与 gate-aware 部署验证也是现有适配。因此只声称指定纯数学函数与官方优化公式对照成立，不声称复刻官方完整模型、数据、CUDA 执行或论文结果。

## 缓存、启动和成本

- 新拟合 policy 为 `optr_oscar_base_upstream_ascend_v5`，最终仍为 recipe5 OptR PT；中间 source seed 独立为 recipe6。旧 PT/拟合断点自动隔离，不能混入新均值/矩阵。
- 原 recipe2/3 拟合实现及其 standalone prepare 保持。旧 raw payload 只有 FP64 QGram，不能拿它冒充参考 FP32 `Q.T@Q`，所以首次新 seed 会自动短采样。
- 相同采样覆盖的 v1–v4 原始 OptR 长序列 trace 按历史完整 profile、rank、请求、张量哈希认证复用。历史 profile 仍记录原 recipe3；不会重新贴成 recipe6。默认仍为 train/validation/test 各 4K、32K、64K，共 9 条独立请求，每条 8 个 decode token。
- 完整层继续由四个独立 spawn worker 调度，使用本次配置的前四张物理卡 `0,1,2,3`。seed 文件和完成标记原子发布，损坏或不完整产物不进入拟合，CPU 测试产物不能进入生产。
- 推理每 token 的算子、矩阵尺寸、KV 字节、设备 buffer 与服务参数无新增；同形状源码工作量保持。首次新 seed 采样和重新拟合有离线成本，设备拟合耗时及模型端到端性能未测，不承诺具体分钟数或速度倍率。

固定入口无需增加参数：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

本轮按用户要求未推送；远端取得本地提交后，这条命令会自动启用新拟合路径，无需手工删 PT。启动摘要可见 `optr_fitting_policy=optr_oscar_base_upstream_ascend_v5`、`optr_seed=oscar_base_shared_zero_mean`、`key_centering=false`。显式指定的旧 PT/旧 pair 仍因身份不符报错，不能静默替换用户明确指定的文件。

## 验证与未完成边界

测试包含指定参考函数的 portable AST 快照，并与实际只读参考的 SHA/AST 核对；包括 FP32 协方差、跨 rank 合并、FP64 eigensolver/U·H·Pbr 的 CPU 逐位配对，以及 D256/GQA6、官方 OptR loss/梯度、多头 Adam 每一步参数和最终选矩阵对照。另覆盖 seed 上限/释放、旧 Gram 拒绝、原子发布/恢复、旧 trace 身份、CPU 产物隔离、真实主机进程并行一致性、默认入口和原 recipe2/3 回归。

最终 **543 项通过，0 失败、0 跳过**；Python wheel 构建、隔离导入及 12 个生产变更模块的包内字节校验通过。机器可读结果与 JUnit 位于 `reports/optr_oscar_base_alignment_20261029/`，早期测试 fixture 的失败及修正保留记录。源码核对确认 `rotations.py` 仅改离线身份函数，候选配置与 argv 对比基底完全相同。未修改 CANN，因此本轮未重新编译 CANN；未运行目标 NPU、图、LongBenchv2 或吞吐验证。已解决本轮确定的拟合来源偏离，**模型精度是否改善仍未验收**。
