# 精确current政策的历史交付与撤下

2026-10-28，仅 `codex/oscar-bf16-attention`。用户已确认614c739的32K、64K性能数据集均达标；本轮保护这组性能实现，处理127条PC开启质量集。未使用评测题目拟合PT或修改采样/评分规则。

## 2026-10-28 后续反馈与处置

**fef8f5ba的native_exact默认改动已撤下。** 用户最新提供单轮127/127、FAIL=0，accuracy/easy/hard/short为38.582677165354326 / 44.18604651162791 / 35.714285714285715 / 38.582677165354326，进度19:04、API1173.43秒。原文节选及结构化记录见[反馈](../reports/precision_followup_20261028/user_result_20261028_080248.txt)与[JSON](../reports/precision_followup_20261028/user_result_20261028_080248.json)。客户端节选没有服务加载SHA，不伪填该身份，也不从单轮推导十轮top3统计。

按用户“如果上一轮改动无效，请弃用”的要求，current-only dispatcher、相关probe及生命周期检查恢复614c实现；所有variant清除旧生成配置中的`oscar_current_only_policy`，避免遗留配置重新启用已撤下政策。独立overlay及其生产模块退出活跃树，原文件保存在[退役文件目录](../reports/precision_followup_20261028/retired_native_exact/)。生产recipe3拟合的数学反例仍可独立复现，但它没有证明用户要求的模型质量收益，不能继续作为该默认改动的交付依据。

此次撤下不等于质量达标，也不把单轮成绩全部归因为某一算子。既有四项质量与32K/64K性能约束继续同时成立。下文保留上一轮的设计、局部结果及限制，均属**已撤下方案的历史记录**。

## 十轮日志与四项交付门

按文件内容，10月22日且首行`cat baseline.log`为native，10月28日为OSCAR。两侧各10轮，每轮127/127、FAIL=0。原始附件按字节保存，来源SHA见[input_manifest.json](../reports/precision_stability_20261028/input_manifest.json)。指标只从原始evaluation字典抽取一次，并交叉核验重复打印的两位小数表。

| 指标 | 基线各指标top3均值 | OSCAR各指标top3均值 | 基线减OSCAR，百分点 |
|---|---:|---:|---:|
| accuracy | 47.24409449 | 42.78215223 | 4.46194226 |
| accuracy_easy | 51.93798450 | 49.61240310 | 2.32558140 |
| accuracy_hard | 44.84126984 | 40.07936508 | 4.76190476 |
| accuracy_short | 47.24409449 | 42.78215223 | 4.46194226 |

四项都没有达到差距小于1个百分点的要求。若top3指按overall选择同三轮，easy差距为3.87596899个百分点，candidate第三名有3/5/9轮并列；完整两种口径和并列分支保存在[日志分析](../reports/precision_stability_20261028/log_analysis.md)，不擅自选择更有利口径。采样十轮平均与top3不是同一个验收统计；四指标仍分别检查。

全十轮overall标准差为baseline3.1496、OSCAR1.3486；附件证实OSCAR均值和最高水平下降，不能由这些汇总直接认定其随机波动更大或发生内存污染。请求提交节奏不作为本轮精度归因或修改理由。

## 确认并消除的误差

原v4对于已经证明无历史、只有精确current的请求，仍执行：Q旋转、K中心化及旋转、V旋转、native FIA、一次BF16输出舍入和V逆旋转。这条路径没有读取INT2历史。

理想正交且无限精度下，这组变换等价于原空间attention；共同K均值只产生每个query的共同logit平移。但实际矩阵与中间结果使用BF16。真实recipe3拟合出的FP32正交矩阵，转BF16后不再严格正交，旋转和逆旋转之间还有额外舍入。

使用**生产recipe3 fitter**、披露的4-rank×512条合成校准样本、D256/Hq6，得到了确定反例：

- Rk/Rv的FP32 Gram最大误差约`4.77e-7/6.56e-7`，BF16后约`9.96e-4/1.08e-3`。
- 单token时attention必须直接返回V；旧旋转往返改变774/1536个元素，最大绝对差`0.00830078125`，相对L2约`0.00316567`。
- n33/n257对独立原空间FP64 attention及BF16输出接口的对照，最大绝对差`0.015625`。这不是用户真实激活或模型分数。
- 对同一层相同incoming K/V，旧路径预旋转后writer与新路径store时旋转的四份writer输入逐位一致，三个形状共12份比较通过。

[可复现脚本及数据口径](../csrc/experiments/native_exact_current_20261028/README.md)、[数值与矩阵/输入SHA](../reports/precision_stability_20261028/native_exact_current/counterexample.json)。这证明可消除的额外误差，不把`0.0083`转换成某个LongBench分数收益。原v4是显式有限精度选择，本轮是新的精确单源政策，不伪称已发现官方CUDA或缓存生命周期bug。

## 新执行链与性能保护

历史fef8f5ba candidate默认`oscar_current_only_policy=native_exact`（现已撤下），只修改**已有current-only资格内部的算术**；原CPU序列上界、设备prepare零context证明、slot/padding检查、首MTP限制和capture/draft排除条件原样保留。未增加任何按输入、context区间、历史长度或数据集选择版本的判断。

1. 原Q由现有`copy_validate_bf16_query_out`检查并写入独立、预分配的`query_rot_bf16`，FIA消费其原空间副本。
2. 原K/V与该Q进入现有native因果FIA，shape、GQA、scale、mask及cumulative不变。
3. 保留原尺寸的连续输出规范化copy、LSE复制和`copy_validate_current_out`；不做最终逆旋转。每个被guard读取的status都有本次写入。
4. 缓存写入仍调用原`store_rotated`：相同BF16中心化、K/V旋转、K=.96/V=.92、INT2 codec、精确snapshot和页所有权。所有后续历史/混合读取仍使用原旋转域。

当时实现：[current_only_dispatch.py退役副本](../reports/precision_followup_20261028/retired_native_exact/oscar_ascend__integration__current_only_dispatch.py)、[native_exact_current.py退役副本](../reports/precision_followup_20261028/retired_native_exact/oscar_ascend__integration__native_exact_current.py)。没有错误后切换路径或历史BF16恢复。

| 工作 | 原路径 | 新路径 |
|---|---:|---:|
| Q旋转 + 最终逆旋转 GEMM | 2 | 0 |
| K中心化、K/V旋转 | 原实现 | 同一原实现 |
| 原生FIA、Q/output校验、copy、guard | 原数量/形状 | 数量/形状不增加 |
| 新增UB/GM缓冲、算子 | 0 | 0 |
| q4/q1/C4/C16历史核、prepare、writer、merge | 614c源码 | **逐字不变** |
| PT、clip、16K预算、PC/GDN/MTP、argv、timer | 614c配置 | **保持** |

每rank Hq6/D256、N个fresh query，删除的矩阵算术为`4×N×Hq×D² = 1,572,864×N FLOPs`。这只是实际删掉的计算量，不是墙钟倍率。K/V旋转从原FIA相位移入store相位，**总次数仍是2**；timer脚本/钩子/schema未改，不能把相位组成变化误读为新增writer成本。

已有的内核/分流源码身份门继续保留。由于本轮明确改变current-only算术，原整文件哈希门改为：其资格/混合prefix/padding证明与614c逐字一致；candidate配置除新policy外与614c快照完全一致；PT cache identity及artifact路径不变。没有把新代码哈希直接替换成“通过”的证明。

## 探针和独立参考

旧v4 oracle及容差不改。新增[精确current合同](../reports/precision_followup_20261028/retired_native_exact/configs__acceptance_native_exact_current.json)和[原空间独立参考](../reports/precision_followup_20261028/retired_native_exact/tools__native_exact_current_reference.py)。只对已资格、无历史的样本使用原Q/K/V的FP64期望；历史、未启用shortcut和后续q1仍调用原v4参考。

current-only/首MTP探针继续执行实际服务forward、相同store和后续decode检查；新policy、合同、overlay/oracle SHA及raw-Q检查调用证据均绑定，旧回执不能放行新policy。disabled对照显式使用旧rotated_v4，避免对照配置自相矛盾。

本轮主机、数学与调用链回归 **302 passed**，覆盖共享/strided QKV、混合prefix、padding/slot/context错误、首MTP、capture/later排除、LSE三dtype、stale status、full-input容量大于history容量、旧策略兼容、两个入口、PT复用及同源内核保护。[验证总记录](../reports/precision_stability_20261028/validation.json)。没有更改CANN核，不重复将其原有编译/CPU结果包装成本轮新设备完成。

## 仍存在的算法边界

另发现并保存了合法chunk/PC种子依赖反例：同一5000-token请求在两种合法剩余预算下，均得到2304-token共同checkpoint，但同一旧token可分别处于精确current或INT2 history。真实144B/mode0反例输出`0.19921875→0.33203125`；将三个source都换回原BF16数据时差值为0，明确定位到精度层级而非分源merge。[反例与限制](../reports/precision_stability_20261028/chunk_dependence/README.md)。

本轮没有以改变调度、扩大窗口、全历史恢复或改clip来掩盖这项依赖，也不把它当成已闭合的现场分数根因。新精确current政策消除该路径自身的多余旋转误差，不能消除所有INT2表示误差或warm路径的精度层级依赖。后续层hidden states会随正确的attention输出改变，因此“同incoming K/V的writer逐位相同”不等于“整个模型缓存内容不变”。

最终验收仍同时要求：四指标10轮top3差距各<1个百分点，以及已达标的32K/64K性能保持。本文没有为新策略填造模型分数。

## 一键入口

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

撤下后固定命令继续复用recipe3 PT，无需删PT、重新拟合或增加诊断参数。current-only恢复614c的v4旋转域；原历史/混合读核和FP32 metadata政策保持。当前入口不再生成或显示`native_exact`政策。
