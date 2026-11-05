# OPTR v11 现场失败与诊断摘要修正

> 后续状态：用户又提供了 layer59/63 的独立小步摘要；本轮后续已将固定 candidate 改为 v12 原生联合残差策略，见[续修说明](optr_native_residual_20261104.md)。下文保留 a599e8a 之前“仅修诊断”的历史事实，不代表当前默认策略。

遵循档案 #170/#172/#174/#175/#176/#177；本轮工作区基底为 `oscar-new-6@ccdd417`。用户提供的最新日志证明 v11 这次未找到任何通过严格 validation 的候选，上一轮小步搜索未解决现场问题。本文不把诊断工具修正写成算法或模型精度修复。

## 已有证据

原字节附件保存在 [user_log.txt](../reports/optr_v11_failure_20261104/user_log.txt)，SHA256 为 `ab2712381fc18819e995307db6630721bde9f1654743d82451c731fa8d00ec1c`。解析和与上一份日志的逐字段比较保存在 [analysis.json](../reports/optr_v11_failure_20261104/analysis.json)。日志没有部署 Git SHA，不将本地 HEAD 当作已证实的现场版本。

- 16 层各 76 个候选，其中原候选 12 个、小步候选 64 个；合计 192 + 1024 = 1216 个，全部未通过。
- 四个 worker 的退出码均为 0；准备阶段最终退出 1，无超时、无中断、清理完成。直接阻断点是 `require_publishable` 的“无严格改善，不发布 PT”。
- 整个准备阶段耗时 6806.486049 秒，即 113.441434 分钟。这不是推理耗时；片段无法拆分拟合、回放和报告各自占比。`recomputed_fit_or_replay=false` 仅描述末尾生成报告的操作，不能解释为整个准备阶段没有计算。
- layer63 的 validation/test before=after，`independent_test=passed` 是保留 recipe3 的自比，不能当作小步候选通过独立 test。
- 16 行 `best_total_candidate`、总误差、拒绝原因、query 检查计数和最差项，与上一份 v10 日志完全相同，全部来自原来的 12 个候选。当前片段没有给出新增 64 个候选各自的拒绝详情。95–236 个退化检查不能移作小步候选统计，也不是独立 token 数。

## 源码定位与处置

`tools/optr_rejection_report.py` 原来将原候选与新增候选合并后，只显示全体中总 SSE 最低者。因此即使新候选有不同的失败模式，也会被更低 SSE 的原候选遮住。完整 JSON 已保留每个小步的指标和拒绝分类，缺失的是终端摘要，不需要重新拟合来补齐。

本轮保留原摘要字段，增加单独的 `repair_search`：候选数、严格总误差改善数、仅 BF16 视图退化数、BF16 操作数完全相同数、拒绝原因分布、总误差已改善者的最少 query 退化检查数，以及新增候选自身的最佳候选及最差项。零候选或零改善不会写成通过；零基线的相对误差保持 null。

原有固定 candidate 入口发生同一错误时自动输出新摘要，仍保持原错误与退出码。策略、半径、拟合 digest、checkpoint identity、发布门、独立 test、CANN、runtime、PC/GDN/MTP、PT/clip/窗口、timer 和启动参数没有修改；不增加 v12 搜索或清理已有工作。

源码也解释了上轮方案为什么没有成功保证：每个头每侧只取一个既有方向，八个半径沿该方向移动；训练的总量代理没有直接约束每个部署 query。即使沿某方向总误差下降，个别位置的误差仍可能上升。对光滑局部模型，若某个位置的方向导数为正，单纯缩小正步长仍会退化；BF16/INT2 的离散变化又不能由这个光滑模型保证。该分析说明方案能力边界，**不能据此断言本次 1024 个小步实际被哪种条件拒绝**；这需要现成 JSON 中的具体指标。

## 不重算的读取方式

可以直接提供这次已生成的完整 `no-improvement-report.json`；无需重新启动服务、采样、拟合或运行 AISBench。文件在本次日志明确给出的目录：

```text
/workspace/oscar/oscar-new/artifacts/rotations/a7b5defbf8558637d2de5bb4a03c7c28d4ff73d1556cd4378f2fbfebebe5bfeb/optr_1f7c086deaddaed2efdd/no-improvement-report.json
```

新增的只读命令在含本轮代码的工作区使用：

```bash
python3 -m tools.optr_rejection_report --saved-report /完整路径/no-improvement-report.json
```

它只读取指定 JSON 并打印摘要，不加载 Torch、模型、PT 或 checkpoint，不写文件，不读取 test 来重选。输出带输入 SHA256，标明 `saved_json_only_checkpoints_not_reauthenticated`；只做 JSON 结构与摘要一致性检查，不冒充 checkpoint 认证。原 `--config` 模式仍认证实际 checkpoint。现有完整报告无须重新生成；只提供终端 `OPTR_PROGRESS` 片段不满足完整输入格式。

这次现场报告未在本地工作区或附件中提供，无法据片段还原 1024 个小步的真实指标。后续需要该已有文件来区分总误差回退、逐位置回退、BF16 视图回退和无效小步，而不能先选定一种归因再扩大搜索。

## 本地验证与验收边界

相关检查结果见 [validation.json](../reports/optr_v11_failure_20261104/validation.json)。测试覆盖：原候选掩盖新候选的数值反例、BF16 视图单独拒绝、操作数不变、空搜索、零基线、缺失/不一致报告、无第三方依赖 CLI、16 层摘要体积和原文件保持；真实 CPU v10/v11 完成层通过固定准备入口恢复时为 0 worker，禁止捕获/拟合/回放/子进程的哨兵没有触发。

相对 ccdd417，895 个既有生产文件逐字不变；唯一改动的既有生产文件是诊断模块，另新增标准库 JSON 摘要模块。完整 fitting digest 保持 `3ba079f899434f1f859806c799bb9cb9d72424d93ca5d07f738fc55b4ee0d251`，逐字清单见 [protected_source_sha256.json](../reports/optr_v11_failure_20261104/protected_source_sha256.json)。这些检查证明本轮诊断改动的范围和续跑行为，不是模型精度或性能验收。

本轮没有真机连接、NPU/图测试、性能实验或 AISBench；没有新的 OPTR PT 或模型分数。此前 42.52 与 recipe3 OSCAR 平均 45+ 的质量差距仍未解决，不能承诺已经超过 OSCAR。本轮改动仅在本地，未提交或推送。
