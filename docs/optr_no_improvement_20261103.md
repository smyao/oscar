# OPTR v10 零候选改善：现场失败与诊断修复

用户于2026-11-03回传2026-11-02的 `prepare-rotations` 日志。当前本地基底 `oscar-new-6@8c414aa` 已包含上一轮 v10；日志没有现场 Git SHA 或最终 PT 回显，不将本地提交写成已确认的部署身份。依据档案 #170/#172/#174/#175，保留验收门、线上算子及性能来源，不能用“让服务启动”代替有效精度改善。

## 日志能够确定什么

- 整个准备阶段耗时 **5488.758832秒，约91.48分钟**，`timed_out=false`、`interrupted=false`、`cleanup_complete=true`。这不是单独的拟合耗时，也不是推理性能数据。
- 四个拟合worker均 `returncode=0`，并完成清理。最后的 `ERR99999` 跟在Python `ValueError` 之后，本片段不支持将其判为NPU算子异常。
- 可见的51/55/59/63层均选择 `candidate=seed`、`accepted=false`，并保存和收集 completed layer checkpoint。四层 validation/test 的 before 与 after 都相同。
- 最终 `require_publishable` 因整份证书 `accepted_layers` 为空而拒绝发布。因此这次没有生成新的合格 OPTR PT，也没有从该入口进入正式服务。仅凭末段不能列出所有更早层的候选值或具体拒绝条件。
- 这些层的 `independent_test=passed` 只说明未改动的recipe3与自身比较通过。被拒绝的候选没有进入独立test，不能解释成“候选独立test通过但发布失败”。

原始日志按字节保存于 [user_log.txt](../reports/optr_no_improvement_20261103/user_log.txt)，结构化事实在 [feedback.json](../reports/optr_no_improvement_20261103/feedback.json)。

## 为什么不能仅据此修改门或学习率

本地重新检查了上一轮成功合成例：它把首帧gate设为0，因此没有体现首帧BF16舍入对严格门的阻力。将首帧gate恢复为非零，原12个候选全部失败；其中K step1总SSE从0.317372降到0.169638，但首帧SSE由约0.00007263升到0.00016790，约增加131%。严格门拒绝它是正确行为。

实际代码的FP32代理在context=0时全部KV为exact，K/V损失和旋转梯度均为0；部署回放仍包含BF16旋转、中心化、逆旋转与gate/projection舍入。D256、8query的独立合成例再次确认这个盲点。默认lr=.02下，D256第一步矩阵相对Frobenius变化约0.581，第一次检查已不是基线附近的微调。

这证明现有目标和候选步幅存在局限；**尚未证明现场所有候选正是被首帧拒绝**。在较小学习率上找到一个合成成功例，也不能直接作为现场改默认lr的依据。本轮保留拟合策略和所有验证条件，不增加未经现场拒绝证据支持的新候选搜索，不发布全不变PT冒充OPTR提升。完整本地反例与复现测试见 [first_prefill_proxy_blindspot.json](../reports/optr_no_improvement_20261103/first_prefill_proxy_blindspot.json)。

## 本轮实际改动

新增 `tools.optr_rejection_report`，直接读取已完成的候选包及逐候选validation指标。工具认证原recipe3 pair、模型/profile、base/trace文件hash、当前拟合实现digest、完整FULL层覆盖、候选矩阵hash及保存的决策；报告不读取原始trace张量，不调用优化器、attention回放、模型或NPU。

每个候选区分：总误差退化、没有严格总改善、总误差改善但细粒度门失败、仅BF16投影视图退化，并提供具体request/partition/frame/query/terminal位置及差值。首帧context=0与已有context的帧分开统计；query计数包含分区和视图，不冒充唯一token数。报告中的“最低总误差候选”只作解释，不用于重新选择或发布。

固定入口 `tools.prepare_rotations` 捕获这一条特定的零改善错误后，自动生成 `no-improvement-report.json`，以现有 `OPTR_PROGRESS` 前缀实时显示摘要，兼容compact模式。原异常、退出码和服务阻断保持；诊断失败也不覆盖原异常。

本轮**未修改拟合digest清单里的任何文件**；digest仍为 `a30488344d57137d7bfc0623e074716744369eef999ac1827014fd907e02a917`。采样、矩阵、政策、候选数量、PT身份和checkpoint身份均不变。若同一现场目录的原pair、source/trace和completed checkpoint完整，固定入口续跑会复用全部完成层，worker计划为0，不会再采样、拟合或回放候选；仍有文件校验/I/O，并会保持原零改善失败。若清理了缓存或更改配置，这项复用前提不成立。

更新代码后，独立读取本次已有证据的命令是：

```bash
python3 -m tools.optr_rejection_report \
  --config logs/20261102T122655.396037Z/effective-target.json
```

该命令只生成诊断报告；成功退出表示报告生成成功，不表示精度通过或服务可以启动。它不要求新AISBench/NPU实验。固定candidate命令也已接入相同报告；没有自动修改当前服务参数。

## 验收边界

本轮修复的是“昂贵拟合完成后，只给零改善异常、没有可见拒绝原因”的诊断缺口，**没有解决真实数据上零候选改善，也没有恢复服务或证明精度超过OSCAR**。下一步算法决策应依据已保存的真实候选误差：首帧失败则优先研究包含BF16精确路径的训练约束；仅少数历史query或投影视图失败则检查对应联合目标和步幅；全部总误差变差则先审查训练轨迹本身。候选选择只能使用validation，独立test不能拿来反复调参。

本地验证及源码身份列于 [validation.json](../reports/optr_no_improvement_20261103/validation.json)。没有连接真机、测速、运行新的质量评测、修改CANN/运行时/PC/GDN/MTP/timer/argv，也没有删除已有PT或已完成工作。

既有准备/证书/入口/迁移保护联合回归 **256通过**；最终报告、非零首帧反例、真实认证续跑及异常保留专项 **39通过**（两组有重叠，不相加）。wheel内新报告模块和准备包装源码逐字一致，隔离导入不初始化NPU，`git diff --check`通过。本轮仍为本地修改，未提交或推送。
