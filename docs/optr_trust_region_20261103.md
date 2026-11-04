# OPTR v11：依据真实逐位置退化报告修正离线候选搜索

> 2026-11-04现场反馈更新：本方案新增的1024个小步全部未通过严格validation，准备耗时113.44分钟，没有新PT。此前合成CPU通过不代表现场有效；下文“未得到现场结果”为当时状态。真实失败、摘要缺口与后续边界见[记录](optr_v11_failure_20261104.md)、档案#177。

本轮基底为 `oscar-new-6@e62d6d2`，遵循档案 #170/#172/#174/#175/#176：原 recipe3 是比较基线，代理改善不等于模型改善，不改在线算子和已确认的性能来源，不降低严格门，也不根据 test 调参。本轮没有连接真机或执行 AISBench。

## 现场证据与诊断修正

原始新附件保存于 [user_log.txt](../reports/optr_tail_regression_20261103/user_log.txt)，SHA256 为 `4bb47e214e72fee8b0aa22baeb78c477c348140d1dc48c14e4713bd942f5623d`，逐层解析见 [analysis.json](../reports/optr_tail_regression_20261103/analysis.json)。

16 层、每层 12 个候选，全部拒绝。各层最低总误差候选的总 SSE 相对 recipe3 下降 1.649%–57.054%，但各有 95–236 个 query 检查退化；最差项增加 21.66%–143.29%。这些计数包含两个 partition 和三个投影视图，不能当作独立 token 数或退化百分比。

16 个显示的最差项 context 都大于零：11 个是 current_end-context=16384 的有历史多 token 帧，5 个是差值为 1 的单 token 帧。最差视图分别为 bf16_rank_output 8 项、fp32 5 项、bf16_output 3 项。fp32 视图也位于 BF16 attention/gate 后；不能把这些退化都解释成最后一次 BF16 cast 噪声，更不能沿用“首帧导致现场全部拒绝”的推断。上一轮首帧反例作为局部目标盲点仍成立。

此次报告耗时 56.543872 秒，复用了既有结果，没有重拟合或重放；最终仍保持零改善异常。它证明诊断复用成功，不证明算法改善或恢复服务。

## 为什么调整候选生成

现有训练使用 FP32 代理，总目标没有对每个位置、每个投影视图的误差退化施加约束。完整候选同时调整多头 K/V，代理中的总收益可能伴随某些 query 退化。原严格部署门正确拦截了这些候选，但原 12 个整层检查点不足以寻找基线附近的可行改动。

本轮保留原拟合器、80+80 步、学习率、原 12 个候选及其优先选择规则；仅在原候选全部不能通过时增加固定搜索：

1. 从每个 KV 头每一侧最早的非零正 step 检查点取方向，按 step 固定排序；缺少可用方向时才考虑 primary。方向顺序不看 validation/test。
2. 以原 recipe3 的 R0 为原点。取 A=(R0ᵀRc−RcᵀR0)/2，并用 ||A||F/√D 归一化，生成 R(ρ)=R0·exp(ρA/normalizer)。这保持正交结构，不对输出做线性插值。
3. 固定半径为 .02、.01、.005、.0025、.00125、.000625、.0003125、.00015625。每个候选只改一个头的一侧旋转；其他头、另一侧和均值全部取原 recipe3。不会累计多个独立通过的头，避免破坏联合输出门。
4. TP4/H4 每层最多 64 个微步候选。缓存原始逐头 projected output，只重放被改的头，再按原顺序重建全层三种投影视图。两个 partition 不同时，每候选分别回放该头，最多 128 次 head-partition 回放，另有最多 8 次基线回放；不能称为零开销。只缓存投影输出/teacher，不持久保存多份旋转历史 KV。
5. 所有候选通过原 request/frame/query/terminal、双 partition、三视图非劣门后，按 validation 总 SSE 最小选唯一赢家，且必须严格改善。浮点容差不变。若转为 BF16 后矩阵逐位等于原矩阵，其全部部署操作数等价，可以直接复用基线指标，不将其算作改善。
6. 在读取独立 test 前，保存并认证所选矩阵、完整 validation 决策、原候选身份和固定搜索参数。test 失败拒绝发布；续跑锁定相同选择，不重选半径。全部层仍无严格改善时继续拒绝发布，不发布不变 PT 冒充 OPTR 提升。

## 缓存身份与断点恢复

固定 candidate 默认 `optr_recipe3_repair=single_head_trust_region_v1`，新 policy 为 `optr_recipe3_trust_region_ascend_v11`。v10 历史格式仍单独支持；旧 v9 的采样不能冒充 recipe3 独立样本。

现有 v10 capture 保留其真实 profile、source 排除证明和 tensor hash。只允许复用指定 v10 实现摘要 `a30488344d57137d7bfc0623e074716744369eef999ac1827014fd907e02a917` 的完整失败工作：认证原 recipe3 pair、base/trace/profile、全部 FULL 层、完成层和 proposal manifest/file SHA、每个候选矩阵及保存的严格门决策。旧目录只读；旧完成层不能直接成为 v11 完成层。

通过认证后，直接复用旧拟合方向和原 12 个候选的 validation 指标，不再调用模型采样或 NPU 拟合。新修复基线会重新回放，并要求与保存的基线指标完全相同。source 的完整快照摘要还进入新 checkpoint identity 和证书，旧 source manifest 发生变化时不能复用前一份 v11 完成层。新代码有新的 fitting digest，不伪装成旧代码身份。若没有旧工作，则正常采样/拟合后执行相同修复策略。

新矩阵、选择锁和完成层保存到 v11 独立目录。旧 PT、原始 capture、旧 proposal 均保留。固定错误报告支持 v10/v11，并解释已保存修复候选的拒绝条件，不重新搜索。

## 在线范围与代价

CANN/q4/q1/C4/C16、KV codec/布局、runtime、PC/GDN/MTP、clip、均值、窗口、timer 和 vLLM argv 均未改。启动卡号 0,1,2,3、端口 7878、TP4、16384 budget、PC 默认关闭和 MTP3 保持；独立后四卡选项保持。逐字保护清单在 [protected_source_sha256.json](../reports/optr_tail_regression_20261103/protected_source_sha256.json)。

推理只加载新常量，仍使用原操作和张量布局。新修复会增加离线 CPU 回放、证据文件与加载校验成本，实际准备耗时未测；不把代码不变写成新的端到端性能实测通过。最终证书由现有 runtime 在准备常量期间校验，线上缓存仍只保留原设备旋转和均值，不保留离线投影缓存。

## 本地验证与边界

真实 CPU 数值反例恢复非零首帧后，原 12 候选全部被拒；新固定搜索选择 head1/K/.02，validation SSE 从 0.3173715594 降至 0.1843001312，独立 test 从 1.9121566628 降至 1.8243649712，全部严格门通过。D256/GQA6 测试确认缓存路径与原 `_metrics` 的完整逐项证据相同。它们均是合成局部证据，不是现场层激活或 LongBench 分数。

完整 pipeline 测试包含真实 CPU recipe3 pair、source 排除计划、trace hash、旧 v10 失败检查点导入、原门实际回放、新 PT 发布/复用、独立 test 失败后的锁定恢复、source 篡改拒绝，以及没有旧 source 的新 v11 路径。结果、打包与测试命令见 [validation.json](../reports/optr_tail_regression_20261103/validation.json)。

联合主机回归 1453 通过、1 项可选本地源审计跳过。最终补齐 partial proposal 与原 source 矩阵/指标绑定后，126 项相关复验通过（与联合回归有重叠，不相加）。真实 spawn worker 验证旧 source 路径不初始化 NPU/不重新拟合，完成层恢复为 0 worker；214 个 wheel 源文件与工作区一致，隔离导入与 fitting digest 匹配。`git diff --check` 通过。本轮为本地工作区改动，未提交或推送。

当前只能确认源码/CPU功能范围通过；没有 v11 现场新候选结果，不能保证其在真实数据上一定有通过者，也不能宣布 OPTR accuracy 已超过原 recipe3 的 45+。BS1、PC-off、MTP-off、稀疏 query 与短生成的校准覆盖限制仍存在。

当前工作区直接启动命令保持：

```bash
bash scripts/install_serve.sh --variant candidate
```

代码同步到部署仓库后仍使用原固定命令：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

保留现有 recipe3 PT、native-traces 和 fit-checkpoints；删除它们会失去本轮复用条件。本轮不要求额外诊断实验或重跑 AISBench。
