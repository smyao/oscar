# OSCAR 数值语义与旋转契约

本文件仅记录本工作区从指定 PR/论文直接核实的语义，以及本机 oracle 的范围。
`oscar_ascend/ops/reference.py` 是离线测试 oracle；生产算子、后端不得导入或调用它。
它会完整重建历史以建立对拍答案，这个操作不允许出现在生产 decode/verify 路径。
本轮没有 NPU 编译、算子运行、图捕获/回放、模型质量或性能证据。

## 固定参考

- PR：`references/oscar-vllm-pr46774`，vLLM PR #46774，
  `57286d5d2cb08c3dcd8c17bb59e132d6985e6796`。
- 论文：`references/oscar-paper`，
  `41ebcdba3db5f0ce1339c3727caea80df575d437`。
- 来源登记：`references/PROVENANCE.md:17-20`。这些是固定本地快照，未宣称当前上游状态。
- 未访问任何旧失败实现。档案只用于错误回归：G19/G20/G23（采样身份）、
  G25-G34/#4-#22（量化/LSE）、#23-#25（协方差与特征分解）、#86（层覆盖）、
  #96/#97（工具环境与旋转来源）。

## 数值契约

| 运算 | 精确定义 | PR 证据 |
| --- | --- | --- |
| 旋转 | `x_rot = x.float() @ R.float()` | `oscar_attn.py:235-243` |
| clip | ratio 为零时不裁剪；否则 `t=quantile(abs(x_rot),ratio,dim=-1)`，截断到 `[-t,t]`，quantile 默认线性插值 | `oscar_attn.py:239-242` |
| 量化粒度 | 每 `(token, KV head)` 的完整 D 维向量一组，K/V 各自参数 | `triton_oscar_store.py:16-17,41-56` |
| scale/zero | `s=max((max(x)-min(x))/3,1e-8)`，`z=min(x)`；二者**先**舍入 fp16，再转回 fp32 参与编码 | `triton_oscar_store.py:44-53` |
| 编码 | `q=clamp(trunc((x-z)/s+0.5),0,3)`，不能用 ties-to-even `round` | `triton_oscar_store.py:55-56` |
| 打包 | `q[4b] | q[4b+1]<<2 | q[4b+2]<<4 | q[4b+3]<<6`，LSB-first | `triton_oscar_store.py:58-61` |
| meta | 每向量 `fp16 scale`、`fp16 zero`，各 low byte 在前 | `triton_oscar_store.py:70-77` |
| slot | `[K indices, K scale, K zero, V indices, V scale, V zero]` | `triton_oscar_store.py:118-145` |
| 反量化 | `x_deq=q.float()*s.float()+z.float()` | `triton_oscar_decode.py:104-114,133-141` |
| GQA | `kv_head = query_head // (num_query_heads/num_kv_heads)`，除法整除 | `triton_oscar_decode.py:58` |
| 历史 attention | Q 用 `Q@Rk`，量化 K/V 在旋转空间累积，历史输出用 `@Rv.T` 回到原空间 | `oscar_attn.py:599-616` |
| 分段合并 | natural-log `L=logsumexp(L_i)`，`O=sum(exp(L_i-L)*O_i)` | `oscar_attn.py:745-749` |

单 head 单 token 的原始 slot 字节为 `2*(ceil(D/4)+4)`：D=128 时 72 B；
D=256 时 136 B。这里没有隐含 32 B padding，生产物理对齐需另计，不得将 160 B
冒充 PR 固有字节数。`group_size < head_dim` 在 PR 配置中显式拒绝
（`config.py:155-162`）。oracle 支持尾部非四倍数维度，最后一个字节的无效高位为零。

空 segment 的输出为零、LSE 为 `-inf`；全空合并也保持零/`-inf`。NaN/+inf
不是“空 segment”，必须报错。这样直接防御档案 G30-G34/#6-#12 的未写 LSE 和无限值。
`attention()` 的 Q 形状为 `[Q,Hq,D]`、K/V 为 `[K,Hkv,D]`；默认 causal Q 位置是
`[K-Q,K)`，支持显式 `query_positions`/`key_positions` 对拍非连续分段。

## 必须登记的 PR 差异与未覆盖范围

1. **零 scale 数值域**：PR 的 `1e-8` 转 fp16 得到零。常量向量、极窄动态范围，
   或溢出的 fp16 meta 不具备可靠可移植的除法/整数转换结果。本实现显式 `ValueError`，
   不悄悄增大 epsilon，不伪造量化答案。生产必须把该域处理写入差异登记并关门；
   当前 oracle 不代表此域已被正式算子支持。
2. **固定请求窗口**：设计的逻辑窗口为 `s=min(S,L)`、`r=min(R,L-s)`，
   Sink `[0,s)`、History `[s,L-r)`、Recent `[L-r,L)`。
   PR `oscar_attn.py:259` 会将 Sink floor 到整页；`:644-683` 根据 staging owner
   决定 BF16 覆盖，失效时改用 INT2。任务要求固定窗口且禁止此类退化，因此生产必须
   使用无丢失 owner/生命周期方案，不能用 PR 驱逐测试充当成功标准。
3. **MTP 与图**：PR `oscar_attn.py:137-141` 为 `AttentionCGSupport.NEVER`、
   `supports_spec_as_decode=False`，没有提供本任务需要的 fused q_len=4 verify 协议。
   oracle 可验证其 causal/GQA 数学结果；并未验证原生提交/拒绝、页复用或图回放。
4. **历史读取复杂度**：精确全注意力必须消费全体压缩历史，读取量至少 Ω(L)。
   可消除的是全历史 BF16 HBM 副本/恢复写回，而非宣称精确注意力历史读取亚线性。
5. **论文模拟器不能替代 PR store oracle**：论文
   `rotation/compute_kv_rotation.py:139-147` 没有 fp16 meta 先舍入，精确编码以 PR 为准。

## 旋转 artifact

项目 schema 是对 PR 分开的 K/V checkpoint 的有指纹封装；不是声称上游可直接读取此文件：

```text
format_version: 1
source_grouping: layer
objective: hadamard | identity | qqt_sst_r_h_pbr | 明确的外部来源
head_dim: D
model_fingerprint: SHA256(模型配置 + 已核实的权重清单 SHA256)
pr_fingerprint: 固定 PR commit
test_only: bool
layers:
  model.layers.N.self_attn.attn:
    layer_id: N
    Rk: fp32[D,D]
    Rv: fp32[D,D]
```

`validate_artifact` 检查精确层集合、global layer id、模型/PR 指纹、维度、fp32、
有限性、正交性；不接受缺层。identity 必须在创建和加载时显式 `allow_identity=True`。
如果目标 draft 层采用允许的 identity 策略，调用方仍需记录授权、显式层名与告警；
不能将主模型缺层或名称解析失败处理成 identity。档案 #86 证明校准引擎不含 draft
时只有 16 个 FULL 层，故真实 target/draft 覆盖必须由运行时发现并验证，不能硬猜 17。

`build_hadamard_artifact` 是论文 `compute_kv_rotation.py:299-312` 明示的
data-free 正交旋转，来源为 `hadamard`，`calibrated=False`。它不建立端到端质量结论。
代码默认要求 NPU；CPU 只能显式 `device='cpu', testing=True`，产物带 `test_only=True`
且不得进入生产。`save_artifact` 仅把已经计算好的小型旋转常量迁移 CPU 序列化，
不将 Q/K/V 数据计算迁移 CPU。`load_artifact` 使用 `weights_only=True`。

## 真实样本校准接口

`sample_covariances` 同时接收同一份 Q/K/V 与非空 sample id：

- Q 按 KV head 的 GQA 组展开，`Cq_h=Q_h.T@Q_h/Nq`，`Ck=mean_h(Cq_h)`。
- `w_t=k_t.T@Cq_h@k_t`，归一化到 `sum(w)=T`；
  `Cv=mean_h((sqrt(w)*V_h).T@(sqrt(w)*V_h)/T)`。
- 非有限样本、负权重、零总权重显式失败；协方差对称化后再次检查有限性。
- 不重跑两遍引擎获取 Q 与 K/V，避免档案 G19/G20/G23 的 sample identity 漂移。

`build_sample_artifact` 必须传入 `eigensolver`，验证返回设备、shape、有限性、特征残差
与正交性。之后按论文 `:234-264` 的 **`R@H@P`** 顺序构建旋转；P 由降序特征值与
bit reversal 生成。论文 CLI 实際默认 `--composition plain`（`:343`），本接口明确
命名 `qqt_sst_r_h_pbr`，不混淆默认值。

该接口不包含真实模型采样 worker，也不提供已经证实 NPU 执行的 eigensolver。
NPU fp32 累积与论文 CPU fp64 有数值差异，需要单独对拍。正交 max-abs 门为 `5e-4`，
相对 Frobenius 特征残差门为 `5e-4`；这两项是新工程检查、尚无目标 NPU 通过证据，
不得称为论文精度验收。返回 NPU tensor 本身不证明算子内部无 CPU/AiCPU 路径，仍需 profiler。

## 本机验证证据

执行：`.venv/bin/python -m pytest -q tests/test_reference.py tests/test_rotations.py`。
环境：macOS CPU，Python 3.12，torch 2.14.0；与目标 torch 2.10/NPU 环境不同。
首次验证 **58 passed**，测试包括独立 `struct` fp16/byte oracle、half-bin ties、tail bits、
causal GQA 与 torch SDPA、分段 LSE、空 row、旋转等价链、window 边界、指纹/缺层/identity
失败、论文独立 fp64 协方差公式及坏 eigensolver 反例。

这些结果仅证明本机参考工具的行为。生产 kernel 编译、数值、图捕获、图回放、模型精度、
MTP 接受行为、HBM 节省、各长度性能均为 **NPU 未验证**。
