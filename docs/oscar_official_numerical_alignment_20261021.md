# Qwen3.8 OSCAR 官方数值链对齐审计

本页记录7786910f的修复前审计。后续用户授权的实现、数值合同和验证边界见[official_math_v4_20261021.md](official_math_v4_20261021.md)。

审计日期：2026-10-21。工程基线：7786910f。官方 sglang-rebase 固定到 38680852e52c10bd6b9e9a64ed9bc815e742859b；URL、下载状态和 SHA256 见 [upstream_manifest.json](../reports/official_semantics_audit_20261021/upstream_manifest.json)。用户本次反馈：更新身份校验并重新测试后，原版27B仍损失约5点，W8A8约0.5点。不得将上一轮358项主机检查视为该质量问题已解决。

档案依据：#13–20 的 LSE/输出偏差边界，#86/#97 的校准层数和来源，#126/#145 的设备内存证据，#148 的算子正确与模型质量分离，#164 的 BF16 舍入契约。D.4 原始日志已读；本轮没有真机、性能实验或全历史生产反量化改动。

## 结论

**没有发现 OSCAR 对 W8A8 实施专用反量化补偿；但是当前 FULL Attention 数值链没有完整对齐官方。** 已确认的关键差异是精确窗口/current 与历史使用不同计算坐标系，以及每个历史 partial 提前 BF16 舍入和逆旋转。它们可以在相同输入、甚至 INT2 编解码没有误差时产生不同输出，不能归为只能接受的底层库差异。

本轮只提交可复现的审计产物，没有修改生产数学。反例证明差异存在，不证明每项能解释多少 LongBench 分数，也未承诺修改后恢复5点。

## 已确认的不等价

| 环节 | 官方固定版本 | 当前工程 | 后果 |
|---|---|---|---|
| Q/K 的比较坐标系 | HP、INT2、current 都使用旋转后的 Q/K | history 用 qr，exact/current 用原 q/k | BF16 旋转不严格正交，跨来源的相对 logit 出现额外偏移 |
| 精确窗口/current 的 V | 保留旋转后的 BF16 V，并在最终结果统一逆旋转 | 保存和计算原空间 BF16 V | 不是相同的舍入链；即使不使用 INT2 也不等价 |
| 逆旋转的位置 | FP32 partial 合并，写出 BF16结果，再一次逆旋转 | 每个 history partial 先转 BF16并逆旋转，再与其它来源合并 | 多出分段舍入，输出可能依赖 split 划分 |
| current-only | 仍走当前 Q/K/V 旋转及最终逆旋转 | 直接原空间 native FIA | 没有完成与官方相同的数值操作 |
| prefill 精确窗口边界 | prefix slot 的 HP/INT2 类别在一次 forward 中固定，随后拼接整块 current | 随 query position 移动 recent cut，同一旧 token 可在块内从 exact 改走 INT2 | 仅64/256这两个窗口大小相同不能证明访问语义相同 |
| online softmax tile | GQA6 INT2 默认64，某些batch为32；HP为32 | 固定256 | P 在每个tile做BF16舍入，tile改变也会改变数值；这不是首要归因证据 |

官方源码：quantized_kv_prefill.py:123–201、503–529；triton_backend.py:1448–1575、3645–3658、3733–3797；decode_attention.py:1207–1241、2533–2590。当前源码：integration/aligned_math.py:38、51–54；attention_cv_aligned_decode.cpp:258、649–666；integration/impl.py:292–311；current_only_dispatch.py；attention_tasks.cpp。

prefill 的固定类别结论不假定某个现场 token 一定为 HP：上游可能在入场前已经 flush；本轮反例枚举固定 HP 和固定 INT2 两种情况，均与本地逐 query 切换区别。详见 [attention_audit.md](../reports/official_semantics_audit_20261021/attention_audit.md)。

## 最有区分力的数值反例

使用合法 D256 正交 FP32 矩阵，在两侧都按相同规则转 BF16。两枚原始 K 完全相同，Q 相同；历史 V 全为+1，精确/current V 全为−1。本例 INT2 K/V 量化后反量化与输入逐值相同。

| 计算量 | 当前工程公式 | 官方统一旋转域公式 |
|---|---:|---:|
| history logit | 16.1378173828125 | 16.1378173828125 |
| exact/current logit | 16.0 | 16.1378173828125 |
| 最终 BF16 输出 | 0.06884765625 | 0 |

理想实数下 R·R转置=I，但转成 BF16 并计算旋转后不再精确满足这个等式。因此 history 的 qR·(kR)转置 与 exact 的 q·k转置 不能直接当作同一数值坐标系。官方对两种来源使用相同旋转，避免此例的来源偏置。该反例在本地 prefill/decode 两种反量化模式下均成立。

另一个5-token反例采用单位旋转、无 INT2 值误差，仅把历史拆成3+2两个 partial：当前每段先BF16舍入后合并得到1.0078125；先合并再舍入为1.0，本地不拆分时也为1.0。说明提前舍入独立于旋转矩阵质量，也会引入额外差异。

这些是人工输入上的 CPU 数学执行，不是 CUDA/NPU kernel 执行或真实模型激活。复现脚本直接提取官方纯 Torch 旋转函数，再配合明确声明的本地 BF16 oracle；不拿它冒充硬件逐位对拍。主代理已独立运行两个脚本成功：

- [attention_repro.py](../reports/official_semantics_audit_20261021/attention_repro.py) → [attention_repro.json](../reports/official_semantics_audit_20261021/attention_repro.json)
- [codec_rounding_probe.py](../reports/official_semantics_audit_20261021/codec_rounding_probe.py) → [codec_counterexamples.json](../reports/official_semantics_audit_20261021/codec_counterexamples.json)

## 没有发现的 W8A8 补偿与已对齐范围

当前 FULL KV writer 的输入不包含 W8A8 权重 scale/offset。其 scale/zero 根据旋转后的 K/V 行自身的 min/max 计算。W8A8 的权重/激活反量化由原生线性层负责，OSCAR接收BF16 Q/K/V；这不等于 OSCAR 为 W8A8增加了补偿分支。W8A8会改变采样和推理的真实Q/K/V分布，但不能仅由两种模型的分数差推断存在补偿机制。

在**相同旋转后BF16输入、相同阈值、uniform单组D256**的前提下，官方与当前声明的 clip 下标、clip→min/max顺序、FP32 scale/normalized-zero、+0.5整数舍入、prefill/decode反量化步骤一致。独立提取官方 Torch quantizer的4组检查通过；这是限定条件下的公式对照，不是整个 Attention 已对齐。详见 [codec_audit.md](../reports/official_semantics_audit_20261021/codec_audit.md)。

V1 startup 的FP64 qqt/sst和U·H·Pbr，在两组相同TP/GQA样本上直接提取官方方法对拍，协方差与FP32矩阵差均为0。198条GPQA消息也完全相同。**官方 startup 同样采用短 GPQA、prefill-only，不能继续把这一点当成本地特有的实现偏差。** CPU/CUDA eigh仍可能给出不同的合法特征向量符号，但本轮没有证据将其归因为用户损失，也不自行新增官方没有的符号规则。

## 官方发布配方与本地V1默认不是同一个已验证对象

用户给的 qwen3.8-27b.sh 默认通过 need_rot 加载已有 RotationZoo PT，STARTUP_CALIBRATE默认关闭。本地则默认自行生成每层共享的V1矩阵。官方当前代码还支持逐KV头矩阵、k_mean、独立q_rotation/o_rotation及逐头裁剪；本地没有这些能力。

但不能把这些可选能力直接宣称为官方 Qwen3.8 默认必用。官方 README 说最新结果中多数dense/hybrid使用逐头旋转、key centering、逐头裁剪；逐头clip来自独立环境变量/JSON，并不会自动从PT推断。公开 RotationZoo 固定快照35c854c4cb6deffa62b93c1939053ae9343a7df6中没有Qwen3.8目录，不能证明镜像内实际PT是哪种格式。详情见 [calibration_audit.md](../reports/official_semantics_audit_20261021/calibration_audit.md)。

官方 README 的Qwen3.8结果为 GPQA-Diamond 88.4→86.9、KL@128K=0.029，并不是这次用户LongBench的同一结果。这个评测范围区别不撤销本轮已经复现的数值不对齐。

## 修复边界

优先修复已证明的共享V1路径：所有Q/K/V来源统一计算坐标系，partial保持FP32到合并后，再进行一次最终BF16输出/逆旋转；current-only、q1/q4/C4/C16、MTP、graph及cache窗口语义必须一起覆盖。不能只给writer换矩阵或只改probe。保持融合有界tile，不复制官方全历史BF16物化的工程做法。

如果进一步采用V2，则必须同时实现逐头TP切分、GQA广播、所有来源一致的key centering、独立Q/O变换、逐头clip表。只中心化历史而不处理exact/current会产生新的相对logit错误。原生GDN/SSM继续当前BF16路径，timer与冻结旧合同不改；应另立官方数学合同，而不是改宽现有误差阈值。

现有 acceptance_aligned_bf16.json 的 adaptation_boundary 明确允许原空间窗口和逐源逆旋转。此前本地oracle验证的是这个自定义合同，不能继续把它称作官方端到端数值对齐。当前证据确认了需要修改的数学链，但没有给出5点损失的因果占比或修复后的模型验收。
