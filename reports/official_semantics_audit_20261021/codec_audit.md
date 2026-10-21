# Codec 与舍入顺序独立审计

官方固定版本：`38680852e52c10bd6b9e9a64ed9bc815e742859b`，文件来源与 SHA 见本目录 `upstream_manifest.json`。本报告未修改生产代码、未连接真机、未执行性能测试。

## 可确认的 codec 对齐范围

结论严格限定为：**给定相同的旋转后 BF16 向量、相同裁剪阈值、uniform INT2、D256 单组**。这不等于最新 Qwen3.8 RotationZoo 的实际 PT/recipe 已对齐；per-head 旋转、中心、裁剪需要另外核验实际 PT。

| 环节 | 当前生产源码 | 固定上游源码 | 结论 |
|---|---|---|---|
| 旋转操作数与输出 | `oscar_ascend/integration/aligned_math.py:37-38,50-52` | `quantized_kv_prefill.py:123-136`；`unified_kv_pool.py:530-560` | V1 分支均为 BF16 矩阵乘法，Q/K/V 旋转输出 BF16 |
| clip 顺序 | `store_aligned_int2.cpp:39-67` | `oscar_rotation_clip_int2_kv.py:103-126,180-185,387-399` | 先对绝对值排序阈值裁剪，再取 min/max；默认 .96/.92 的 D256 下标分别 245/235 |
| scale / zero | `store_aligned_int2.cpp:64-70` | 同上游 `:182-185,211-223` | FP32 `scale=max(max-min,1e-8)/3`，`zero=-min/scale`；normalized-zero 正负号一致 |
| 量化整数舍入 | `store_aligned_int2.cpp:73-79` | 同上游 `:193-196`；`kv_quant_kernels.py:45-64` | `x/scale+zero+.5` 转无符号整数；不是 nearest-even。当前多出的 clamp 在官方 Torch helper 也存在 |
| metadata 存储 | 144B slot 的四个 FP32 值 | `environ.py:602`、`eval_oscar_gpqa.sh:338`、`unified_kv_pool.py:1166-1178` | 当前官方启动环境默认 FP32；pool 构造器参数默认 BF16 不能替代启动配置证据 |
| prefill 反量化 | `attention_aligned_unpack.h:133-158`，mode 0 | `quantized_kv_prefill.py:243-272` | FP32 `(code-zero)*scale` 后输出 BF16 |
| decode 反量化 | `attention_aligned_unpack.h:121-158`，mode 1 | `decode_attention.py:1960-1986,2155-2182` | metadata 先 BF16，BF16 crumbs 减 BF16 zero 后乘 BF16 scale；当前显式中间舍入符合该上游路径声明 |
| W8A8 权重参数 | `aligned_math.py:43-59` 和 writer ABI | `unified_kv_pool.py:1582-1660` | 本地 writer 没有权重 scale/offset 参数；scale/zero 仅由旋转后的 FULL K/V 算得。上游此 pool 的 k_scale/v_scale 签名也未进入 INT2 算式 |

物理字节布局不同：官方每字节按四个连续 quarter 的 2-bit crumbs 打包；本地采用自己的 striped 布局，读写配套。不能拿物理字节不同直接归因为反量化错误。

## 两项确证的浮点组织差异

1. **inverse-Rv 与分源合并顺序不同。** 当前每个 history partial 先归一化，转 BF16 后乘 BF16 Rv，得到 FP32 原空间 partial，再由 `merge_lse_out` 合并；exact 窗口/current 使用原空间 BF16。证据：`attention_cv_aligned_decode.cpp:648-667`、`oscar_ascend/integration/impl.py:305-311`。上游 HP 和 INT2 都在旋转空间，stage1 partial 存 FP32，stage2 先合并到 BF16 输出，最后统一 BF16 逆旋转。证据：`triton_backend.py:1221-1225,3777-3797`，`decode_attention.py:2533-2590`，`quantized_kv_prefill.py:503-529`。
2. **online softmax/P 的 tile 划分不同。** 当前固定 KV256，P 在每个 tile 转 BF16，分母仍消费 FP32 P。上游 Qwen GQA6 的 INT2 tile 默认 64（batch>=16 时 32），HP tile 32，且可由环境变量覆盖。证据：`decode_attention.py:1207-1241,2190-2206,2293-2296`。FP32 max/sum/alpha 不消除每个 tile 的 BF16 P 舍入差异。

`codec_rounding_probe.py` 不导入本工程 aligned_reference，直接 AST 提取固定上游 Torch quantizer/clip-index，另写两个隔离的舍入顺序公式。结果见 `codec_counterexamples.json`：

- 4 个 clip ratio 的码值、scale、zero 与源码转写公式逐项一致。半整数示例 `[0,.5,1.5,2.5,3]` 官方量化结果 `[0,1,2,3,3]`。
- 两个 partial、同一 BF16 旋转矩阵与同一合并权重，仅改变 inverse/merge 次序，最终 BF16 输出最大差 `0.001953125`。
- 同一份 128 个 QK scores 与 V，仅改变 online tile 256/64，最终 BF16 输出最大差 `0.00146484375`。

这些 CPU 反例只证明浮点运算组织不等价；没有运行 Triton GPU 或 AscendC 指令，不能据此认定用户的 5 个百分点损失由这两个差异造成，也不能宣称某条顺序更准。

## 需要避免的归因错误

- 当前没发现 FULL INT2 codec 绑定 W8A8 权重 scale/zero 的代码链；W8A8 仍可能改变输入 QKV 分布，但那与误用权重量化 metadata 是两回事。
- 最新上游 `q_rotation/o_rotation/k_mean` 与 per-head clip 支持是否被 Qwen3.8 默认 PT 启用，取决于实际 RotationZoo artifact；不能凭代码中的可选分支或 V1 fallback 判断默认没有使用。
- 归一化 LSE 接近、主机 oracle 通过、CPU 公式相同均不能代表模型级精度对齐。档案 #13-20、#126/#145 的输出/布局/所有权教训保持有效。
- 本报告只做固定 tile 数值对照，不采用 D.4 中失败的全历史反量化生产结构；旧 dequant 6499.8–6655.1ms 与 store 209–215ms 不是本轮性能测量。
