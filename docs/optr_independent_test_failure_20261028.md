# OptR 第7层独立测试失败：原因复核与处置

> 最新明确授权：用户选择“允许生成实验 PT，先测试 OptR”。现采用schema2实验发布证书：独立test退化保留失败指标与状态，继续后续层、发布和服务；validation、数值合法性、矩阵身份及正交性检查保持。下文禁止发布和需要确认的描述属于授权前历史，不再代表当前行为。本变更不意味着退化已修复，模型质量仍未测；固定candidate无新增参数。证据见`reports/optr_experimental_publication_20261028/verification.json`。

> 2026-10-28后续用户指令覆盖默认处置：用户要求先测试OptR，禁止通过开关绕过或关闭。当前固定candidate已恢复recipe5，准备/加载均直接走OptR，无需新增参数；下文撤回默认及recipe3选择的描述是当时的历史处置，已被本指令替代。独立test失败报告、禁止发布和全部质量阈值未改，本次未修复已观察到的泛化失败，也未重新取得真机质量或性能证据。已有匹配trace/拟合断点身份不因本次预设修改而变化。

这次失败来自质量验收：第7层候选在联合 validation 被接受，在独立 test 上退化，程序因此禁止发布。没有证据支持将它修成“忽略异常即可成功”。此前把未完成完整验收的 OptR 强制作为 candidate 默认过早；本次先撤下该默认，恢复原 recipe3，同时保留显式研究能力与原质量门。**OptR 获得更高模型精度的目标尚未完成。**

## 现场能证明什么

2026-10-28后续运行 `20261028T031557.011852Z` 的[新增原始日志](../reports/optr_test_metrics_20261028/target_failure.txt)已给出具体指标；[机器可读分析](../reports/optr_test_metrics_20261028/analysis.json)保留原值及计算结果。prepare耗时3629.800522秒（60.50分钟），rc1、未超时、cleanup_complete=true。第7层独立test总normalized error由0.003534305215807693降至0.0035226326595371314，相对改善0.3303%；但同一条`test_65536_2`在single_split和captured_runtime_frames下分别相对退化0.08509%和0.08620%。两条失败记录具有同一个sample hash，不能说成两条独立请求退化。

本次可以确认触发的是严格逐请求门，而非总量门：`regressed_total_metrics=[]`，两行64K输出误差上升。源码`nonworse`显式要求总量和每行同时非劣，日志所示比较与实现一致；不是编译错误、超时或已证明的数值计算bug。该幅度是CPU重放输出误差的相对变化，不是LongBench分数，也没有证据将它当作浮点噪声。按相邻心跳定位，第7层联合validation开始到prepare失败约1047.7–1107.7秒（17.46–18.46分钟），这是阶段时间窗，不是设备内核计时。

本次源码复核未改训练、选择、test或发布标准。若要在独立test退化后仍生成实验PT并继续模型测试，必须明确改变原来的发布判定并标注未通过独立test，不能把它称作质量修复或通过。本条补充失败证据，不改变固定candidate继续启用OptR的用户要求。

用户本轮粘贴的 [原始失败日志](../reports/optr_independent_test_failure_20261028/target_failure.txt) 对应目标 run `20261027T133233.150213Z`。`prepare-rotations` 运行 3498.906897 秒后返回 1，`timed_out=false`、`cleanup_complete=true`。baseline 和 selected 均完成 test 的 `single_split`、`captured_runtime_frames` 检查，随后报 `independent test regressed; artifact not published`。这次已经越过旧1800秒时限；不是编译报错或超时复发。

前一份 [65536字节截断附件](../reports/optr_independent_test_failure_20261028/head_progress_truncated.txt) 显示第7层 head0–2 保留基底，head3 的验证误差从 0.003319737907123906 降到 0.0030706464108648376，约改善7.50%。这是逐头 validation 指标，不是独立 test、全层模型输出或任务得分。

当前 `refine_layer` 只有在联合 validation 非劣且总误差严格下降时才另算 selected 的 test，未接受时直接复用 baseline 的 test。因此这条调用链支持“通过联合 validation 的候选未能通过独立 test”。日志没有保留具体失败请求、分片或数值，无法判定退化发生在4K、32K还是64K，也不能判断是总量退化还是某一请求退化。没有现场启动SHA，不把本地基底提交当作现场身份。

## 源码复核与可能原因

| 因素 | 代码证据与含义 | 当前结论 |
| --- | --- | --- |
| 训练长度不足 | `optr_policy.py` 默认 train 仅4096，validation/test为4096、32768、65536；长历史的量化误差和softmax竞争未直接进入默认训练 | 明确的覆盖差异，可能影响泛化，未证明是本次全部原因 |
| 监督稀疏、自由度高 | 默认每frame最多8个query、8个decode token；当前4K请求可在16384预算内一次prefill，context=0走未量化current分支（CPU近似），直接含历史INT2误差的训练主要来自少量decode。D256的两个正交矩阵合计65280个自由度 | 过拟合风险较高；query个数不是独立标量约束个数，不能仅凭计数断言欠定 |
| 验证多长度不等于多份独立内容 | `calibrate_optr_traces.py` 对每个split拼接同一token流，再取各长度前缀，长于源流则重复；train/validation/test来源记录分开，但同split长度高度相关 | 现有验证长度覆盖不能替代内容多样性 |
| 选择集收益不保证新请求收益 | 各头K/V训练及step选择看validation，联合检查仍看validation；独立test只在最终检查使用 | 质量门发挥了作用，不能把它改为按test重选矩阵 |
| 严格逐请求门 | `nonworse` 同时要求总量和每个请求非劣，不允许平均改善掩盖局部变差 | 可能总量改善仍失败；现有日志不足以确认此分支，不能归咎于浮点噪声 |
| 数值或实现错误 | 复核固定侧K/V缓存、阶段切换、best更新、联合TP投影及发布路径，暂未发现能确定解释这次退化的错误 | 未找到确定性bug不等于已证明与NPU逐位一致；仍保留CPU数学参考边界 |

即使未来每层离线检查都通过，也不直接保证全模型质量：当前按原生teacher激活独立拟合各层，未闭合量化误差跨层传播及自回归分布变化。这一限制此前已记录在 [实现说明](optr_implementation_20261027.md)。

## 已修改的行为

1. `tools/serving_variants.py` 在准备前静态撤下 OptR 默认。固定 `candidate` 恢复 perhead_center recipe3，清除残留 `rotation_refinement`、`optr_calibration`，继续保留 BF16、FP32 metadata、K=.96/V=.92及全部已有优化组合。没有切换到会撤掉这些功能的 `baseline` variant。直启和probe使用同一选择器。
2. recipe3 与 recipe5 使用不同认证缓存身份。两种PT同时存在时，固定 candidate 准备和加载实际选中 recipe3；显式指定不匹配的PT仍失败。没有删除用户产物或重标配方。匹配 recipe3 可直接复用，缺失时仍按原来的认证/自动校准链处理。
3. 显式研究配置经 `variant=None` 保留 OptR，原训练样本、步数、损失、选择规则和独立 test 门均不变。未加入“失败后换回基底继续发布”、自动重试或降低门槛。
4. OptR 独立test失败时，先尝试在本次身份目录原子保存 `layer-N-independent-test-failed.json`，包含完整validation/test、逐请求/分片原始值、矩阵哈希、head decisions、policy和来源身份；再打印总指标及退化项。文件不符合完成层断点命名，不能作为通过凭证。写文件或输出异常仍抛出携带报告的 `IndependentTestRegression`（ValueError子类），保留原错误链；没有最终PT发布。

这是发布预设的回归控制，不是本次 OptR 研究获得非劣结果。已暴露的test失败保持失败，不能通过改默认设置把研究结果记为通过。

## 开销与缓存边界

[逐项比较](../reports/optr_independent_test_failure_20261028/source_and_preset.json) 记录当前目标配置在修改前后的唯一有效配置差异为 `rotation_refinement: optr → absent`；完整服务argv和所有在线功能开关一致。此前记录的690个受保护文件逐字节保持，涵盖线上核/runtime/集成、目标配置、验收合同及timer等。训练策略、优化器、trace数学和验收函数也未修改。没有新增在线矩阵乘、KV存储、设备同步或回读；失败报告只在离线失败分支产生。

准备工具属于既有拟合实现哈希的输入，因此诊断修改会建立新的fit checkpoint身份；不能声称旧head/layer断点直接恢复。原生trace的身份、路径不变，匹配的采样仍可复用，旧文件保留。固定 candidate 已不执行 OptR 拟合，直接走recipe3。

没有测目标机性能，不能把“在线源码和工作量不变”说成端到端性能已经验收。现有图缓存对有效配置/插件身份的隔离规则保持，切换配方可能发生一次图缓存重建；不把首次启动成本说成零。

## 验证与交付

本地372项主机/CPU测试通过，失败和跳过均为0；纯Python wheel禁网构建、隔离导入及变更模块字节校验通过。完整命令、JUnit和包验证记录在 [verification.json](../reports/optr_independent_test_failure_20261028/verification.json)。覆盖固定入口及旧模式、recipe3/5双缓存、完整OptR准备成功/失败、总量改善但单请求变差仍拒绝、报告文件/输出双失败、断点身份和原超时监督。

本轮没有修改CANN源，未连接真机、运行模型质量或性能测试；目标整服务、NPU和graph状态未升级。改动保存在当前工作区，未推送。固定入口语法仍是 `git pull --ff-only && bash scripts/install_serve.sh --variant candidate`，代码合入部署分支后才会通过git pull取得；无需新增手工配置、删除PT或调超时。

## 后续精度方向

重新启用默认OptR前，应先固定新实验方案：让训练覆盖长历史与chunked prefill，增加不同来源的训练/validation请求和代表性query；评估限制旋转扰动幅度、降低可训练自由度等泛化约束。所有选择必须限于训练/validation，并计入离线采样、内存和拟合成本。最后用未参与这些调整的新holdout确认收益；本次已失败的test保留为历史证据，不能循环调到它通过便宣称独立验证成功。

这些是待验证方向，本轮没有为制造“成功”而实施未经验证的新训练配方。原先约3点差距以及小于1点目标均未被本次修复证明改善。
