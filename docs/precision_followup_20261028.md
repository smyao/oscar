# 2026-10-28 精度续修裁决

**当前没有同时满足质量与性能保护要求的新默认优化，目标未完成。** `bcbf4153` 已撤下上一轮 `native_exact` 默认改动，current-only及相关probe恢复用户确认32K/64K性能达标的614c实现，并清除旧生成配置残留。recipe3/PT、clip、16K预算、PC/MTP、q4/q1/C4/C16和timer保持；撤下不等于质量达标。[撤下证据](native_exact_current_precision_20261028.md)、[129项主机回归及源码身份](../reports/precision_followup_20261028/withdrawal_validation.json)。

用户最新单轮为127/127、FAIL=0，四项38.58/44.19/35.71/38.58，API1173.43秒。它不能代替两侧各10轮的top3统计。交付门仍是accuracy/easy/hard/short各项与基线差距小于1个百分点，并保护已达标的32K和64K性能，禁止按输入长度或数据集选择版本。[原文节选](../reports/precision_followup_20261028/user_result_20261028_080248.txt)。

## 已复现的问题

1. **MTP分组改变精度归属。** 同一逻辑query和已接受前缀，q4与逐token q1的fixed-per-forward cut不同，少数旧token分别使用BF16或INT2。实际codec与独立attention反例可改变同一固定head的argmax。
2. **APC联合状态可以混配。** 真实native主机生命周期复现：同token前缀经不同chunk轨迹重算，lookup选择A的FULL页与B的GDN页。实际数值反例也证明两条轨迹可能生成不同FULL/GDN内容。

两项都有[可复现脚本、源码SHA与范围](../reports/precision_followup_20261028/README.md)。这是本地合成输入与原生合同证据，不是完整Qwen激活重放；**没有证明用户全部LongBench分差或耗时差均来自这两处**。

## 未接入production的方案

| 方案 | 决定性反例或成本 | 裁决 |
|---|---|---|
| 将cut统一为query-relative | 当前q4与全BF16均为0.2001953125，修改后变0.333984375并翻转argmax；统一了分组，却降低了原本正确的一边。仅改decode还会漏掉mixed C4/C16路径。 | 不接默认 |
| 将259行全部作为精度窗口 | 额外3行为MTP拒绝回退预留。q4写768..771后拒绝draft，扩大窗口会读取已被770/771覆盖的511/512，违反tag有效性。 | 不改窗口或writer |
| APC canonical发布/驱逐门 | 推广新FULL会使有效长命中6912→4608；保护旧FULL则B/C重算后仍持续miss。 | 不通过删cache换一致性 |
| matched-cohort lookup overlay | 可以保持同2304命中并修正物理组合，但正确FULL_B可能已在free queue，而原FULL_A仍活跃。真实native admission所需页数1→2，紧容量时从可调度变不可调度。若真实依赖已回收，同长度组合也不存在。 | 不把同命中长度或零新增分配当成性能无损；不接默认 |

MTP依据见[反审与运行结果](../reports/precision_followup_20261028/MTP_REPAIR_REVIEW.md)。APC依据见[两种实验结果](../reports/precision_followup_20261028/APC_EXPERIMENT_RESULTS.md)及[独立容量反例](../reports/precision_followup_20261028/apc_capacity_review.json)。APC实验11项通过包含确认上述失败反例，不能解读为方案放行。所有实验保留在reports目录，没有接入candidate、修改原生参考树或部署到真机。

## 来源核对

先前把通用脚本默认group128当成目标模型依据的判断已撤回。已保存的官方339/386版本Qwen3.8-27B专用脚本均为group256、Lloyd-Max关闭；与本工程当前组宽一致。两份目标脚本SHA相同，这不是新的精度优化理由，不据此更改packed ABI/PT/参数。[目标脚本](../reports/official_alignment_20261018/official_qwen38_recipe.sh)、[386版本副本](../reports/official_semantics_audit_20261021/upstream/rotation/run/qwen3.8-27b.sh)、[纠正记录](../reports/precision_followup_20261028/group_size_source_correction.md)。

本轮交付内容是撤下无效默认改动和保存经过反例审查的结果。质量目标继续未完成；不把恢复614c、局部正确性修复设想或主机测试包装成满意的新优化。
