# LongBenchv2 40.94 反馈：OptR 联合输出校准修复

基底 `645d2b0`。本轮继续默认启用 OptR，修改离线目标和候选校验；不增加每 token 的推理操作。档案依据为 #148/#170 的代理与模型质量边界、#172 的准备成本和断点身份、#173 的独立测试政策、#175 的校准修复。

## 现场证据与可确认范围

用户提供 LongBenchv2/version 324293 的 accuracy=40.94488188976378、easy=48.837209302325576、hard=36.904761904761905，反馈低于原 OSCAR。原文保存于 `reports/optr_joint_calibration_20261029/target_feedback.txt`。这是未达到用户要求的实际结果，不能用拟合 loss 改善代替它。

片段没有服务 SHA、PT 证书或同条件原 OSCAR 分数，因此不能量化本次相对掉点或确认加载了 v3 修复。日志时间为08:48，本地 v3 提交时间为11:49+08:00，也不能在未核对目标时区和部署身份前据此作确定版本归因。9.94秒属于评估阶段，不能当作生成耗时。该结果没有用于训练、挑矩阵、调阈值或挑学习率。

## 复现的校准缺口与改动

### 1. GQA 输出的交叉项被训练目标遗漏

原 `_projected_mse` 使用逐查询头的 Gram，计算各头投影误差平方的平均。模型和最终校验则先相加所有查询头的投影，再计算误差。两个头的误差 `[1,-1]` 可完全抵消；换成 `[0.5,0.5]` 后，旧指标从1降至0.25，保留相同归一化的联合误差却从0升至0.5。因此旧目标可能将优化引向联合输出更差的方向。最终校验能拒绝一部分坏候选，但不能让训练自动寻找正确方向。

新目标在每个 KV 头内保留全部 GQA 交叉项：令 `B` 表示 BF16 舍入，`g` 为捕获的 sigmoid gate，`a/t` 为候选/目标 attention，计算

```
z_h = B(B(a_h) * g_h) - B(B(t_h) * g_h)
loss_output = mean_query,hidden((sum_h z_h @ W_O,h.T)^2) / GQA
```

保留 `/GQA`，避免在 K 的 `mean(KL)+lambda_output*loss_output` 中无意把输出项权重扩大六倍。采用直接 FP32 投影，不构造大联合 Gram；后者在近抵消情况下可能因舍入产生负二次型。平方范数非负，且无额外 Gram 构造阶段。最终层选择仍汇总所有 TP/KV 头，不把头内目标当成完整层目标。

### 2. 使用真实捕获目标，并对齐 gate 的 BF16 边界

原 K 输出项仅使用 `(P_candidate-P_dense_teacher)@V`，没有使用已捕获的原生 attention teacher。新 K 保留原 dense teacher 的 KL 项，同时使用捕获的原生 BF16 attention 作为输出目标。修改捕获 teacher 会影响输出项，而不会改变 KL，已通过独立公式验证。

原目标用 `error*gate` 近似两个门控输出之差。BF16 下这不等价：例如 attention 为1.0078125与1、gate为0.10009765625时，前者约0.0007820，实际分别舍入后的差为0.0009765625。新目标分别舍入两个乘积。

V 保持官方顺序拟合：固定选中的 K 概率，条件 teacher 为其未量化 V 输出，经 BF16 attention/gate 边界处理后比较。没有改为同时训练 K/V 或用独立 test 重选。80步、Adam、lr=0.02、验证时机、窗口和 INT2 clip 均保持。

参考为[固定提交的官方拟合实现](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/blob/66d46561b9bdf5d715f591e0f5845a4cbd927dee/optr/calibrate_optr.py)。联合 GQA、原生 K 输出目标和 BF16 gate 是针对当前架构的明确适配，不能宣称与官方 CUDA 实验逐位一致。

### 3. 防止平均值掩盖最后一个 prefill token

原验证已逐请求、逐 forward 非劣，但一个 prefill forward 内仍汇总多个 query。末尾 query 决定首个生成 token，其误差可能被同一帧其它 query 的改善掩盖。

新证书对每帧保存 `terminal_query`，绑定 `position=current_end-1`、SSE 和 teacher 能量。validation 必须同时满足逐请求、逐帧、逐末尾 query 及两种分片非劣，并使总 SSE 严格改善。缓存投影反例中，帧 SSE 从9降至1，但最后 query 从0升至1，旧门接受、新门拒绝。解码单 query 帧的两种指标必须严格一致。独立 test 的末尾退化也完整记录，仍遵循用户已授权的实验发布政策，不用于重选或阻断实验 PT。

这些 query 本来就在采样中，无需扩大数据集、重新执行 attention 或增加候选重放次数。

## 数据、生效与准备成本

- 默认仍为9条轨迹：train/validation/test 各含4K、32K、64K。源记录按完整题目分离；三个长度仍由同一 split 的 GPQA 流拼接或重复，不能称为9条独立自然长文档。训练按请求平均，不按64K token数加权。数据域偏差、稀疏 query、短 decode 以及全模型跨层误差积累仍是剩余泛化风险，本轮没有用未经验证的大幅加样本掩盖它们。
- 固定 candidate 的 policy 更新为 `optr_joint_gqa_bf16_ascend_v4`。新目录身份自动隔离旧完整 PT；v1/v2/v3 拟合断点不能冒充新目标。相同目标内部的中断恢复保留。
- 完整且配置/来源/hash/采样覆盖一致的 v3 原生轨迹可以自动认证复用，只重拟合。旧默认7条轨迹不能冒充当前9条；缺少匹配数据时原有采样流程自动执行。没有从独立 test 借数据。
- 保留四卡独立进程、逐层动态分配、每卡一个任务和 CPU 线程上限。直接投影增加部分离线矩阵计算；以 G=6、D=256、hidden=5120 为例，投影权重30MiB。生产加载器的 train/held-out 共用同一固定权重，新暂存也只保留一份；相比旧两份分头 Gram 共3MiB，主要暂存增加27MiB/worker。不增加完整历史或模型副本，并省去 Gram 构造。这不能被说成离线零成本，也未测真机拟合时间。
- 线上仍只消费同形状的 Rk/Rv/k_mean。CANN、INT2格式、vLLM argv、runtime、timer、MTP/GDN、设备0/1/2/3和端口7878保持。

固定启动入口无需新参数，仍为 `git pull --ff-only && bash scripts/install_serve.sh --variant candidate`。当前遵照用户“保留本地提交、暂不推送”；因此远端 pull 暂时不会取得本轮代码。已有本轮代码的 checkout 可运行 `bash scripts/install_serve.sh --variant candidate`，自动完成所需准备。启动的 `OPTR_ARTIFACT` 回执会标明 v4 policy 和新 fitting_objective。

## 验证边界

独立公式与梯度、BF16 gate 反例、原生 teacher 敏感性、D256/GQA6、末尾 query 门、报告/证书、历史原始采样认证与旧拟合隔离、四进程恢复和串并行一致、固定入口及 wheel 结果保存于 `reports/optr_joint_calibration_20261029/verification.json`。

上述证据证明具体校准缺口已修正，不能证明 LongBenchv2 已提升或达到论文分数。本轮未连接真机、未重跑 AISBench、未修改在线内核，没有新的 NPU、图、模型质量或端到端性能验收结果。
