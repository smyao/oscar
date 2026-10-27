# OptR 接入当前 OSCAR 的改造范围与性能分析

**建议接入 OptR 的离线旋转优化方法，以当前 recipe3 的逐头矩阵为起点，训练适配 Qwen3.8-27B 和当前 Ascend 数值链的新矩阵；第一阶段保持现有推理算子、INT2 格式和窗口协议。按这种边界实现，在线计算量、KV 容量和图节点数量可以保持不变，主要新增成本发生在首次采样与离线训练。**

直接运行上游脚本、把上游 PT 填入当前服务并不足以完成移植。必须适配完整 Q/K/V 与 gate 采样、混合架构的 FULL 层编号、GQA/TP 头映射、D256 量化分组、BF16 舍入、实际窗口，以及版本化产物认证。OptR 有希望改善当前精度，但没有证据保证把当前 32K 任务损失压到 1 点以内。

分析日期为 2026-10-27。本地基底为 `oscar-new-4-back` 的 `4f4ef7ef589741c3866ee3aac62448ac142fe813`；上游固定为 [`66d46561b9bdf5d715f591e0f5845a4cbd927dee`](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/tree/66d46561b9bdf5d715f591e0f5845a4cbd927dee)。本文中的时间和精度数据均标明来源；本次是静态分析，没有修改生产代码、连接真机或执行训练、性能测试。相关源码指纹和容量推算保存在 [证据记录](../reports/optr_analysis_20261027/evidence.json)。

**一、OptR 到底改变了什么**

OSCAR 的现有校准用 Q 的二阶矩及能量加权 V 协方差构造旋转。OptR 继续使用固定正交旋转，但把选择目标推进到 attention 输出经过输出投影 `W_O` 后的误差。它训练的是每层、每 KV 头的旋转参数，不是重新训练模型权重，也不是在每次请求中动态寻找矩阵。

用行向量记法，推理关系为：

\[
Q_r=QR_K,\quad K_r=(K-\mu)R_K,\quad V_r=VR_V,
\]
\[
O_r=\operatorname{Attention}(Q_r,\operatorname{DQ}(K_r),\operatorname{DQ}(V_r)),
\quad O=O_rR_V^T.
\]

`DQ` 表示量化再反量化；精确窗口使用未量化的旋转域 K/V。当前 candidate 已经具有这条执行结构，因此 OptR 可以只更换矩阵内容。

上游 [`calibrate_optr.py`](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/blob/66d46561b9bdf5d715f591e0f5845a4cbd927dee/optr/calibrate_optr.py#L25) 的关键过程如下：

1. **K 中心化。** 从每个 KV 头的 K 中减去固定通道均值。理想算术下，同一 query 对所有来源的 logits 都减去同一个常数，因此 softmax 不变。此条件要求历史、精确窗口和 current 的处理一致。
2. **正交校正。** 以基础矩阵 `R0` 为起点，令 `R = R0 @ matrix_exp(A - A.T)`。反对称矩阵的指数为正交矩阵。代码用 Adam 优化 A，最终导出已经合成的 R，而不是在服务中保留两级旋转。
3. **先优化 K。** 保持 V 为全精度，最小化 attention 概率的 KL 加上由概率变化产生的 post-`W_O` 输出均方误差。
4. **再优化 V。** 固定选出的 K 旋转，在量化 K 决定的概率下优化 V 引入的 post-`W_O` 误差。
5. **按头保留验证集上更好的矩阵。** 默认每阶段 80 步、学习率 0.02，每 20 步及最后一步检查验证损失；默认改进 margin 为 0。未改善的头保留基础矩阵。这是离线模型选择，不是运行时回退。

其 K/V 损失可以概括为：

\[
L_K=D_{KL}(P\Vert P_K)+\lambda\,\operatorname{MSE}((P_K-P)V W_O^T),
\]
\[
L_V=\operatorname{MSE}(P_K(\widehat V-V)W_O^T).
\]

实际代码按 query head 计算，再对共享同一 KV 头的 query heads 求平均。`quant_int2_ste` 在前向执行裁剪、分组 min/max、四级舍入和反量化，在反向用直通估计器传播梯度。默认配置为 group128、K clip0.96、V clip0.92、sink64、recent256，只对每段末尾 64 个 query 计算损失。[量化与窗口代码](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/blob/66d46561b9bdf5d715f591e0f5845a4cbd927dee/rotation/kv_calib_utils.py#L27)、[校准入口](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/blob/66d46561b9bdf5d715f591e0f5845a4cbd927dee/scripts/calibrate.sh#L32)。

这比只优化 K/V 重构、logits MSE 或统计代理更接近模型真正承受的误差，尤其适合继续处理本项目 recipe4 曾暴露的“代理改善但输出恶化”问题。不过，分阶段损失仍不等于最终整层输出误差：K/V 误差之间、不同 query head 输出之间都可能存在交叉项。因此还必须检查最终 K+V 组合的真实输出，不能只看 `Kbest` 和 `Vbest`。

**二、当前版本已经具备哪些基础**

| 能力 | 当前 candidate | 接入 OptR 后的处理 |
|---|---|---|
| 每层每 KV 头的 K/V 正交矩阵 | recipe3 已有 | 作为优化起点，替换为合成后的矩阵 |
| K 中心化 | 已有，BF16 减法后 BF16 旋转 | 第一阶段保持运算顺序 |
| K/V clip | 0.96 / 0.92 | 第一阶段固定，不重启 recipe4 代理选择 |
| Q、历史、精确窗口、current 的旋转域 | v4 已统一 | 保留 |
| attention 部分结果与合并 | FP32 partial / LSE / merge | 保留 |
| 输出逆旋转 | 合并后一次 BF16 `Rv.T` | 保留 |
| metadata | FP32 scale / normalized zero，mode0 | 校准与验证按此建模 |
| D256 INT2 存储 | 每 token、每 KV 头 K/V 共 144B | 保留 |
| MTP | 独立 identity + 零均值 | 首版保留，不能使用 target 某层矩阵代替 |
| GDN 状态 | 原生路径 | OptR 不直接改造其状态量化 |

依据为 [candidate 选择](../tools/serving_variants.py)、[实际旋转及逆旋转](../oscar_ascend/integration/official_math.py)、[runtime 载入及 MTP 隔离](../oscar_ascend/runtime.py)、[现有校准](../oscar_ascend/rotation_calibration.py)。

还有一个直接影响收益预期的对照差异：上游 `run.sh` 的 `oscar` 分支只加载基础矩阵，`oscar_optr` 分支才额外设置 `K_MEAN`；基础矩阵构造产物还是 layer grouping，而 OptR 输出为 KV-head grouping。当前 candidate 已经具有逐头中心化与逐头拟合，不能把上游的全部涨点归为“在当前 recipe3 上再训练旋转”的净收益。[上游模式选择](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/blob/66d46561b9bdf5d715f591e0f5845a4cbd927dee/run.sh#L89)、[基础矩阵产物](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/blob/66d46561b9bdf5d715f591e0f5845a4cbd927dee/rotation/compute_kv_rotation.py#L144)。

**三、必须改造的地方**

| 改造项 | 当前接缝 | 具体要求 | 工作量判断 |
|---|---|---|---|
| 原生轨迹采样 | `tools/calibrate_model_rotations.py`、`integration/rotation_capture.py` | 保存逐 query Q、gate、位置、请求边界，以及完整因果 K/V 历史；支持长 prefill 和 decode 观察 | 大 |
| 离线优化器 | 新增独立模块，复用 `rotation_calibration.py` 的基础矩阵与身份信息 | 实现正交参数化、STE、K 后 V 的损失、独立验证、D256 和 gate 适配 | 大 |
| 生产数值链参考 | `official_math.py`、`tools/bf16_reference.py` 对应的参考合同 | 匹配 BF16 旋转与舍入、FP32 metadata、窗口 cut、合并顺序 | 大，决定结果是否可信 |
| 模型结构和权重读取 | `tools/model_config.py`、`tools/prepare_rotations.py` | 解析 `text_config`，只遍历 FULL 层，正确定位 `o_proj` 与 TP shard | 中 |
| 产物与缓存认证 | `rotations.py`、`rotation_calibration.py`、`prepare_rotations.py` | 新 recipe / objective / trace schema，完整记录模型、基础 PT、数据及数值合同 | 中 |
| 启动配置传递 | `tools/serving_variants.py`、现有 prepare/serve 流程 | 准备、探针、服务读取同一产物；固定 candidate 命令最终能够启用 | 中 |
| 线上旋转、INT2 writer、CV reader | `runtime.py`、`official_math.py`、AscendC kernels | 推荐首版仅接收新矩阵，保持尺寸、dtype、ABI、算子选择和预分配 | 小或无需算法修改 |
| 功能与质量验证 | 既有 oracle / probe / artifact 检查 | 验证新矩阵、窗口、TP、MTP、图复用及组合输出；质量独立记账 | 必需 |

这是“较完整的离线校准改造，加上较薄的服务接入”，不是重写 Ascend attention，也不是只增加一个配置项。

**1. 当前保存的样本不足以训练 OptR**

`OfflineLayerAccumulator` 保存 `q_gram`、原始 K/V 和计数，没有保存逐 token Q。`Q^TQ` 丢失了 query 与位置、请求的对应关系，无法恢复每个 query 的 softmax，也无法计算 OptR 的输出损失。已有原始校准文件可以复用来构造基础矩阵、均值和检查身份，不能仅靠它们完成新的输出感知训练。[采样累加器](../oscar_ascend/rotation_calibration.py)。

现有采样器限定完整 prompt 单次 prefill，`max_tokens=1` 只从 prefill logits 采样，不产生 decode forward；同时不允许 prompt 超过 chunk budget。针对 32K/64K 的新采样必须显式支持跨 chunk 的逻辑请求，把绝对位置、累计历史、当前 query 范围和窗口边界一起保存，不能把每个物理 chunk 当成独立短句。[采样入口](../tools/calibrate_model_rotations.py)、[当前采样限制](../oscar_ascend/integration/rotation_capture.py)。

建议让 K/V 以请求为单位流式落盘，只为预先选定的位置保留 Q 和 gate，覆盖长上下文中的多个位置以及生成阶段。模型前向继续在原生 NPU 上执行，离线拟合进程在模型退出后工作。离线 CPU 数学不属于生产推理回退；若选择 NPU 优化，则需要单独核实 `matrix_exp`、反向和优化器支持，不能沿用上游 `cuda else cpu` 的设备选择而把生产流程静默放到 CPU。

**2. Qwen3.8-27B 的 gate 必须进入 loss**

本项目保存的模型配置是 64 层，其中 FULL 层为 `3,7,…,63` 共 16 层，其余 48 层为 linear attention；FULL 层 D=256、24 个 Q 头、4 个 KV 头，GQA=6，hidden size=5120。TP4 下每 rank 对应 1 个 KV 头与 6 个 Q 头。[模型配置快照](../reports/official_alignment_20261018/official_qwen38_config.json)。

原生 Ascend FULL 路径为：

\[
Y=[\operatorname{Attention}(Q,K,V)\odot\sigma(g)]W_O^T.
\]

gate 在 attention 后、`o_proj` 前逐元素相乘。上游 loss 直接计算 `d_out @ WO.T`，没有这个 gate。适配时应使用原生 BF16 轨迹中的 gate，按下面的形式衡量误差：

\[
e_{K,j}=\big([(P_{K,j}-P_j)V_h]\odot\sigma(g_j)\big)W_{O,j}^T,
\]
\[
e_{V,j}=\big([P_{K,j}(\widehat V_h-V_h)]\odot\sigma(g_j)\big)W_{O,j}^T.
\]

gate 值来自相同 teacher 输入，这能够衡量该层的局部量化影响；它并不覆盖前层量化造成的后续 hidden/gate 漂移，后者仍需多层 replay 或模型质量验证。实现依据见 [Ascend gate 顺序](../references/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py)。

上游 `WOCache` 假设顶层 config 有 hidden/head 参数，并硬编码 `model.layers.{i}.self_attn.o_proj.weight`；主循环是 `range(layers)`。这些假设不适用于本项目的 `text_config`、多模态包装及非连续 FULL 层。应从当前 native 模型或经过验证的 safetensors 索引定位权重，显式检查 `[5120, 24×256]` 全局形状及 TP 输入列分片，不能按 `hidden_size / num_heads` 反推 D。[上游权重读取](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/blob/66d46561b9bdf5d715f591e0f5845a4cbd927dee/optr/optr_common.py#L86)。

训练可按 KV 头分解，但最终验证应额外计算同一 query 的完整输出误差 `||sum_j e_j||²`。逐头 `mean(||e_j||²)` 会忽略跨头抵消或叠加；TP 每 rank 的误差平方和也不等于 all-reduce 后的整层误差平方。

**3. D256 应先沿用 group256，不能只复制上游默认 group128**

当前 writer 对 D256 的 K/V 各做一次裁剪、min/max 和量化，K/V 合计 128B INT2 payload 加 16B FP32 metadata，即 144B。[当前 writer](../csrc/kernels/store_aligned_int2.cpp)。

上游校准默认 group128。在 D256 下，这会产生每个 K 或 V 两组量化参数；如果校准使用 group128、服务使用 group256，优化的是另一种量化器。推荐首版设 group256，冻结当前 clip 和 codec，把 OptR 本身的收益隔离出来。

还应检查裁剪的分组范围：上游 `quant_int2_ste` 在每个 group 内求裁剪阈值，而本次源码的 grouped Triton writer 先对整个 head 求阈值，再分 group 求 min/max。D 等于 group size 时两者一致，扩展到 D256/group128 后就不能默认一致。这是实际读代码才能发现的移植接缝，不能靠同时填写 `GROUP_SIZE=128` 解决。[上游 grouped writer](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/blob/66d46561b9bdf5d715f591e0f5845a4cbd927dee/engine/sglang-research/python/sglang/QuantKernel/oscar_rotation_clip_int2_kv.py#L142)。

如果另外研究 group128，并保持当前每组 FP32 scale/zero，那么 K/V 的 metadata 由 16B 增至 32B，总大小为 **160B，较 144B 增加 11.11%**。它需要修改存储布局、budget/spec、writer、所有 reader、探针和 cache format 版本，还会增加分组参数读取与处理。单组排序变短可能抵消部分开销，但不能从字节数直接推断整轮速度；物理页和快照使实际容量变化具有台阶。此方向应作为独立实验，不应被包装成 OptR 的必要代价。

**4. 必须对齐真实 BF16 运算顺序**

上游校准工具把 trace 转为 FP32，在 FP32 中做旋转、反量化、逆旋转、softmax 和 `W_O` 误差；服务侧则存在 BF16 和 Triton 专用路径。当前 candidate 采用 BF16 Q/K/V 旋转、BF16 QK/PV 操作数、FP32 softmax/LSE/merge、最后一次 BF16 逆旋转。

尤其要区分：

- 本项目先做 `BF16(K - mu)`，再做 BF16 旋转；上游 pool 使用 `K @ R - precomputed(mu @ R)`。实数代数相等，BF16 舍入后不保证一致。
- 本项目精确窗口也处于 BF16 旋转域；上游离线工具把窗口直接恢复为原坐标 FP32 reference，不能代表本项目的“未量化但已旋转舍入”误差。
- 本项目在所有来源 FP32 合并后只逆旋转一次；离线工具逐 token 反量化后逆旋转再做 attention，理想算术等价，有限精度并不等价。
- FP32 metadata 策略不等于 attention 全 FP32；概率和矩阵输入的 BF16 舍入仍应计入最终候选检查。

上游顺序见 [`unified_kv_pool.py`](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/blob/66d46561b9bdf5d715f591e0f5845a4cbd927dee/engine/sglang-research/python/sglang/srt/mem_cache/unified_kv_pool.py#L759)，本项目顺序见 [`official_math.py`](../oscar_ascend/integration/official_math.py)。

建议采用两层校准：可微参考负责优化，独立的部署数值参考负责候选选择。关键 BF16 cast 可用前向舍入、反向 STE 的方式近似训练，但这不保证与 Ascend 硬件逐位相同；预先冻结的 oracle 与必要设备功能验证仍是独立关口。正交性检查应继续在导出的 FP32 矩阵上执行，BF16 转换产生的非正交残差则用实际 attention 输出验证，不能放宽旧门槛掩盖问题。

**5. 窗口必须按真实 forward 建模**

上游 `restore_exact_window` 对一段长度 L 的 trace 恢复 `[0,64)` 和 `[L-256,L)`，末尾 64 个 query 共用这套窗口，再用 causal mask 屏蔽未来。它没有完整重演本项目的分页、cached cut、current chunk 和 MTP 生命周期。

当前 v4 是每次 forward 固定 cached cut，current token 另受因果范围约束；因此长 prefill 的 current chunk 可能比 256 大得多。校准不能把这部分错误地当成历史 INT2，也不能把所有历史当成 BF16。应从采样 metadata 保存的请求上下文重建 quantized history / exact history / current 三个集合，用同一个规则训练和验证。[当前窗口合同](../docs/official_math_v4_20261021.md)。

上游 dump 记录的是每次调用的 Q/K/V 和 extend lengths，不能仅凭 chunk 文件名认定已有完整请求历史。默认优化数据来自短 GPQA prefill dump；K 均值则来自生成 trace 再 replay 的数据。当前中心化均值来自 prefill 样本，分布也不同。首轮可保持现有均值以单独判断旋转优化收益，随后再独立比较覆盖生成阶段的均值。[上游准备脚本](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/blob/66d46561b9bdf5d715f591e0f5845a4cbd927dee/scripts/prepare.sh#L135)、[dump 代码](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/blob/66d46561b9bdf5d715f591e0f5845a4cbd927dee/engine/sglang-research/python/sglang/srt/layers/attention/triton_backend.py#L1780)。

**6. 新产物必须有自己的身份**

上游虽然也使用 `format_version=2`，其字段是分开的 K/V 文件、`source_grouping=kv_head`、数字 layer id 和 `rotation`；本项目是统一的层名、`Rk/Rv/k_mean`、`layer_kv_head`、模型指纹及 recipe。当前 `require_calibrated` 还明确要求 `qqt_sst_r_h_pbr` 和 FP64 统计来源。因此上游 PT 不具备直接导入条件，不能只把 objective 改名骗过校验。[本项目产物校验](../oscar_ascend/rotations.py)。

新 recipe 应记录：基础 PT hash、模型/权重/配置/tokenizer 身份、FULL 层与全局头映射、trace schema 与样本 hash、train/validation/test 划分、group/clip/window、gate 与输出投影定义、全部关键 dtype、数值合同、训练步数、按头选择结果及联合验证指标。recipe3 保留为明确对照，新的优化产物不得冒称原始 qqt/sst 配方。

上游每 20 步使用同一 held 集选择 best，它属于验证集而非最终独立测试集。当前项目应按完整请求划分数据，避免同一长请求不同 chunk 分到两边；另留未用于选矩阵和调超参的测试请求。少量 improvement 或某一平均 loss 下降不应自动启用全部矩阵。

缓存复用必须认证身份，不能照搬上游“两个 PT 存在就复用”。启动前一次性载入并预分配矩阵，不能在请求中反复 load/convert/expand；更换旋转后必须销毁旧 KV/prefix 缓存并重建相应图，旧坐标系的缓存不能和新矩阵混读。

**四、性能会不会变差**

**推荐方案下，没有必需的在线增量计算，但端到端性能仍需要区分固定计算与生成行为。**

| 维度 | 推荐首版的影响 | 原因及边界 |
|---|---|---|
| 单步 Q/K/V 旋转 | 计算量不增加 | 用合成后的 R 替换旧 R，仍为相同尺寸 BF16 GEMM |
| KV 写入和读取 | 格式、字节和算子数不变 | 保留 group256、144B 和现有 CV 路径 |
| 输出逆旋转 | 次数不增加 | 仍在 FP32 合并后一次逆旋转 |
| KV 容量与 GDN 状态 | 不因 OptR 增加 | 无残差缓存、附加通道或高精度历史 |
| 图与调度 | 可以保持现有结构 | 常量须在 profiling/capture 前完成加载 |
| 首次部署 | 成本明显增加 | 新采样、离线优化与验证 |
| 后续启动 | 小幅认证/加载成本 | 复用经过认证的产物，不重新训练 |
| 真实生成吞吐 | 可能提高，也可能降低 | 输出长度、EOS、MTP 接受率和调度行为会随 logits 改变 |

以一个新 query token 同时写入一个 K/V token 为例，当前每 FULL 层、每 rank 的四类旋转总算量为：

\[
(6_Q+1_K+1_V+6_O)\times256^2=917,504\ \text{MAC}.
\]

OptR 合成矩阵后仍是同一个数。这是算术工作量，不是实测时延。若错误地在线执行 `x @ R0 @ deltaR`，则会增加相应的第二次 GEMM、中间缓冲与 launch；`matrix_exp`、loss、反向和优化器全部必须停留在离线阶段。

16 个 target FULL 层、每层 4 个 KV 头的 K/V BF16 矩阵合计为 16MiB，即 TP4 每 rank 4MiB；当前 recipe3 已经需要这些矩阵，所以推荐方案的这部分净新增是 **0**。若同时常驻 base、delta 和最终矩阵才会额外占用内存，那是可避免的实现开销。FP32 产物与其它已有副本另按实际加载器计账。32K/64K 的 INT2 payload 和窗口页不随 OptR 增长，此前按当前格式计算的 KV 节省量无需因“加入 OptR”重算；改 group 或窗口后则必须重算。

下面几种做法会产生实质风险：

| 做法 | 可能的负向影响 | 建议 |
|---|---|---|
| 在线串联 base 与 correction | 多一次矩阵乘法和搬运 | 离线预合成 |
| 在线逐 query head Python 循环 | 多 launch、小 GEMM 效率低、图捕获复杂 | 保留当前 TP4 的共享矩阵批量计算 |
| 将旋转或 attention 整体升为 FP32 | 带宽、工作区、计算吞吐变化 | 精度参考与生产 dtype 分开，首版保留 BF16 |
| 为复现训练公式恢复全历史 | 历史展开和搬运成本随长度增长 | 只在离线 replay 使用，生产沿用有界 KV tile |
| 同时切换 group128 或扩大窗口 | metadata、缓存页、算子工作增加 | 与 OptR 分开研究 |
| 在线采样、loss 或 CPU 同步 | 增加同步、传输、日志和生命周期负担 | 独立采样进程，正式服务不挂训练 hook |
| 未初始化新图或混用旧 prefix cache | 正确性故障，可能影响命中和吞吐 | 以产物身份建立完整缓存生命周期 |

MTP 尤其需要单独看待。即便 draft 矩阵保持 identity，target 模型 logits 改变后，draft 与 target 的一致性仍可能变化。可以用 `吞吐 ≈ 每步有效输出 token / 每步时间` 理解：如果单步耗时不变而接受率下降，端到端吞吐仍会下降。输出变长同样可能拉长整轮耗时，所以不能只根据相同算子图宣称整轮没有负向影响。

**五、上游的性能优化不能与 OptR 核心算法混为一谈**

论文在 Qwen3-4B、A100 40GB 的特定 decode、prefill 和 2K 输入/4K 输出负载上报告，相对其优化后的基础流水线变化在 2% 内。它不是 Ascend910B4、D256、TP4、GDN 和 MTP 的性能保证。[论文效率分析](https://arxiv.org/html/2608.02691v1#S6)。

仓库默认开启 `SGLANG_OSCAR_ABSORB_V_ROTATION=1`。当前提交的 `maybe_absorb_oscar_v_rotation_into_qkv` 确实把 V 旋转折叠入 V 投影权重，并标记 `oscar_v_rotation_absorbed`，使部分在线 V 旋转可以跳过。[默认开关](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/blob/66d46561b9bdf5d715f591e0f5845a4cbd927dee/run.sh#L197)、[折叠实现](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/blob/66d46561b9bdf5d715f591e0f5845a4cbd927dee/engine/sglang-research/python/sglang/srt/models/utils.py#L647)。

但这里需要保留一个源码与论文的区别：论文描述同时将逆旋转吸收入 `W_O`；本次固定提交里上述函数只处理 V 投影，`prepare_quantized_extend_qkv` 仍设置 `need_v_inverse=True`，Triton 路径仍调用 `apply_inverse_v_rotation`。不能仅按论文描述认定这个提交已经消除了所有 V 侧旋转开销。[显式逆旋转](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/blob/66d46561b9bdf5d715f591e0f5845a4cbd927dee/engine/sglang-research/python/sglang/srt/layers/attention/quantized_kv_prefill.py#L687)。

对当前工程，V 投影折叠可作为后续独立优化，但不是接入 OptR 的前提。它会改变 BF16 运算顺序：`BF16(BF16(XWv)Rv)` 与 `BF16(X·BF16(WvRv))` 并不逐位相等，还要适配 fused QKV、gate 所占的 Q 分片偏移、TP、MTP 和权重打包。因此首版不建议同时引入。

把输出逆旋转直接折入静态 `W_O` 则还有更强的障碍。当前输出为：

\[
(O_rR_V^T\odot g_t)W_O^T
=O_rR_V^T\operatorname{diag}(g_t)W_O^T.
\]

逐 token 的对角 gate 一般不与稠密旋转交换，不能改成先在旋转域逐元素乘同一个 gate、再使用固定折叠权重。除非另行改变 gate 的表示和计算，否则应保留逆旋转在 gate 之前。类似地，任意旋转也不能无条件跨过 Q/K 的 norm 和 RoPE 合并进投影。

**六、离线成本怎样控制**

上游矩阵优化使用固定轨迹，不保存模型全部层的反向图，成本远低于全模型训练，但 D256、长历史和 gate 会扩大成本。论文的单 A100 校准为 6.25–11.03 分钟，明确排除了 BF16 trace 收集；不能据此估计本模型的完整首次启动时间。[论文离线成本](https://arxiv.org/html/2608.02691v1#A1)。

仅按本模型 16 个 FULL 层的原始 BF16 张量计算，一条完整轨迹的数据量如下，未含权重、metadata、优化器和 autograd 中间量：

| 保存方式 | 32K | 64K |
|---|---:|---:|
| 全位置 Q/K/V 与逐 Q 头 gate | 14GiB | 28GiB |
| 全历史 K/V，Q 和 gate 各只存 64 个选定位置 | 约 2.023GiB | 约 4.023GiB |
| 单 query head 的 64×L FP32 logits 一份 | 8MiB | 16MiB |

前两行是所有 target FULL 层、全局所有头的总量；均匀 TP4 分片后每 rank 约四分之一。多条轨迹和转 FP32 会继续放大；不能把这张表理解成训练峰值显存。

可优先采用以下优化，它们只改变离线计算组织：

- 按层、KV 头、请求和 query microbatch 流式处理，累计归一化后的梯度后再更新，避免把所有 heads 的 autograd 图同时留在显存。
- K 阶段复用 teacher 概率与输出；V 阶段冻结 K 后，预计算相应 `P_K` 或按块重算，避免每轮重复做无关的 K 量化和 softmax。是否缓存由内存预算决定。
- 对逐头目标预计算 `G_j=W_{O,j}^TW_{O,j}`。令 `z=e_j⊙gate_j`，则 `||zW_{O,j}^T||²=zG_jz^T`，可减少大 hidden 维输出投影。该等价只针对这个逐头指标，不能代替有交叉项的完整多头输出验证。
- 只保存部署所需的最终 R 与均值；A、梯度、Adam 状态和训练中间结果不进入服务。

从 D128 增到 D256，单头矩阵参数量增至 4 倍，矩阵指数的典型稠密运算规模约增至 8 倍；但本模型 FULL 层数、KV 头数和轨迹长度也不同，因此不应直接用该倍率外推总校准时间。

**七、精度收益应怎样预期**

上游 README 的五任务平均分如下，分数单位为百分点。它说明该方法有实证价值，但不是当前 32K 长输入任务的预测。[上游结果表](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/blob/66d46561b9bdf5d715f591e0f5845a4cbd927dee/README.md)。

| 模型 | BF16 | OSCAR | OSCAR + OptR | 相对 OSCAR |
|---|---:|---:|---:|---:|
| Qwen3-4B Thinking | 67.58 | 62.71 | 65.38 | +2.67 |
| Qwen3-8B | 65.66 | 61.34 | 65.68 | +4.34 |
| Phi4-14B reasoning plus | 58.57 | 52.63 | 55.60 | +2.97 |

长上下文应看对应的检索结果，而不是把最大生成长度当输入长度。论文 Qwen3-8B 的 RULER-NIAH 为：

| 输入长度 | BF16 | OSCAR | OSCAR + OptR | OptR 与 BF16 差距 |
|---|---:|---:|---:|---:|
| 32K | 98.70 | 83.76 | 86.42 | -12.28 |
| 64K | 84.22 | 57.54 | 68.65 | -15.57 |

这组结果同时表明长上下文有所改善、仍可能明显有损；不能给出“采用 OptR 即可恢复到 1 点以内”的承诺。[论文长上下文结果](https://arxiv.org/html/2608.02691v1#S5)。

对本项目，OptR 最有可能改善的是旋转目标与最终输出不一致、少数敏感头误差被统计平均掩盖，以及校准数据补齐后对长历史和生成阶段的不适配。它不会自动修复错误页映射、MTP 位置、KV256 跨 tile 累积错误、源集合遗漏、部署加载旧 PT 等实现问题；这类问题必须由独立正确性证据定位。

**八、推荐实施与验收顺序**

1. **冻结当前对照。** 固定 recipe3、K0.96/V0.92、group256、v4、mode0、窗口、argv、MTP 和当前产物身份。先定义新 loss、采样 schema 与验证准则。
2. **完成采样与离线原型。** 通过外部 worker extension 获取 Q/K/V/gate 与请求位置；复用基础矩阵，在独立进程优化。旧统计样本不足时明确补采，不能合成逐 query Q 冒充原生轨迹。
3. **做部署数值验证。** 检查无 INT2 时的旋转/逆旋转残差、中心化一致性，覆盖 D256/GQA6/TP4、全部来源、长历史多 tile、K/V 同时量化、gate 以及多头最终输出。使用未参与选择的请求检查局部指标和尾部异常。
4. **接入版本化产物。** 复用原 runtime 的 matrix slots 和 dtype，检查全局头到 rank 的切分、MTP identity、图地址和 prefix 缓存代际；candidate 选择、准备、probe 与正式服务必须一致。
5. **分别记录功能、质量和性能状态。** 既有 CPU/算子门通过不等于模型质量通过；有生成质量结果也不等于速度达标。性能判断应拆分单步时间、有效输出、MTP 接受率、输出长度和容量，不能只看局部算子时间。

本轮没有授权实施或真机运行，因此以上是可交付的改造方案，不是已完成的接入。未来实现仍应遵守当前“不由 agent 连接真机、不修改 timer、不要求重跑已完成 AISBench”的约束；必要设备功能证据与完整模型质量证据分别保留真实状态。交付时沿用固定命令 `git pull --ff-only && bash scripts/install_serve.sh --variant candidate`，不得仅在离线 probe 中启用新矩阵。

**九、历史故障对本方案的约束**

| 档案或已有证据 | 对应决策 |
|---|---|
| 档案 #13–20 及 v4 数值修复 | 全部来源使用一致坐标系；只在 FP32 merge 后一次逆旋转 |
| 档案 #86 及 runtime 的 target/draft 隔离 | target FULL 与 MTP 分开标识，不能借用矩阵或均值 |
| 档案 #126/#145/#164 | CPU 数值、CANN 指令、设备异步与图验证分列，不因训练 loss 改善跳过 |
| 档案 #148 的局部通过但模型退化 | 按最终模型质量裁决，不把局部代理当质量验收 |
| 档案 #157 的冻结 guard | 首版不改冻结核、错误域和生产同步行为 |
| recipe4 attention 输出反例 | K/V 各自损失改善后仍验证最终组合输出，保留独立测试集 |
| 启动文档 D.4 的全历史恢复成本 | 上游离线反量化/逆旋转公式不进入生产全历史路径 |

综合判断：**OptR 值得作为当前精度修复的优先方向，适合采用“新离线校准器 + 现有线上执行链”的方式实现。最难的部分是让训练看到当前模型实际承受的误差；保持线上算子不变本身并不难。**
