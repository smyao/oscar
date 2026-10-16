# 三版本注意力/KV 全链路数值 dtype 对照（2026-10-16,rev2)

rev2 说明：按用户指正重做原生基线调研——Qwen3.5-27B 是**开源模型**（[HF 官方 config](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json) 直证），旧版「闭源未知」的表述收窄为**仅 torch_npu/aclnn 稠密 FIA 算子二进制内部不可证**；补齐 Embedding/norm/RoPE/GDN/logits/采样等原生阶段；原生列全部改为可读源码级证据。

三个口径：

- **不用 OSCAR 的原生基线**：原生 vLLM + vLLM-Ascend，冻结 argv（APC 关、`--max-num-batched-tokens 16384`、原生调度器、TP4、MTP3、FULL_DECODE_ONLY，见 `reports/stable_32k_64k_20261015/frozen_launch.json`）。
- **当前正式 OSCAR**：main `7047ce36`(= `codex/aisbench-stable` 同步）candidate 组合。
- **新增 BF16 历史版本**:`codex/oscar-bf16-attention`(`7047ce36` + `b3c83069`/`d49a7233`/`7b170738`)。

## 0. 模型与部署事实（钉版）

- **模型结构**(HF 官方 config):`text_config.dtype=bfloat16`;64 层 = **48 GDN 线性注意力 + 16 FULL 注意力**(`full_attention_interval=4`);head_dim 256;GQA 24 q / 4 kv(6:1);hidden 5120;`model_type=qwen3_5_text` → **dense MLP，非 MoE**;GemmaRMSNorm(eps 1e-6);`attn_output_gate=true`(FULL 层输出 sigmoid 门控);`tie_word_embeddings=false`;MTP 1 个 FULL 层（`mtp_num_hidden_layers=1`,3 个推测 token 复用同一层）;GDN:k 头 16 / v 头 48、head_dim 128、conv kernel 4。运行时旁证：每 TP rank GQA6、16 个 FULL 层（`docs/performance_findings_2026-09-23.md:29`)。
- **部署变体** `Qwen3.5-27B-w8a8-mtp`:w8a8(Linear int8)+ MTP3;argv 显式 `--mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16`——**覆盖** HF base config 的 `mamba_ssm_dtype: float32`;rope 由 hf-overrides 强制 yarn / factor 4.0 / mrope_interleaved / mrope_section [11,11,10] / partial_rotary_factor 0.25 / theta 1e7(rotary_dim=64)。
- **软件栈**:vllm-ascend v0.23.0 + PR12607(`docs/reference_manifest.md:8`)、torch-npu 2.10.0.post4、torch 2.10.0、triton-ascend 3.2.2(`references/vllm-ascend/pyproject.toml:21-23`)、CANN ≥ 9.1(`references/vllm-ascend/csrc/version.info`)、SOC ascend910b4(A2)。
- **OSCAR 两版只接管 16 个 FULL 层的注意力计算与其 KV 存储**;48 个 GDN 层、norm、RoPE、w8a8 投影、logits、采样等所有其余阶段三版共用同一原生路径。

## 一、主表

### A. 三版共同阶段（原生路径，OSCAR 不涉及）

| 阶段/数据 | 不用 OSCAR 的原生基线 | 当前正式 OSCAR | 新增 BF16 历史版本 |
|---|---|---|---|
| Embedding 权重 | BF16(checkpoint 量化描述若单独覆盖则不可证） | 同左 | 同左 |
| QKV / o_proj / MLP 投影（w8a8) | 激活 BF16→int8 动态量化；int8×int8 matmul + int32 bias + FP32 deq_scale → **BF16 输出**(aclnn 内部归约不可证） | 同左 | 同左 |
| input/post RMSNorm(fused add+norm) | BF16 进出；**自写开源核**内部建 FP32 buffer 计算 | 同左 | 同左 |
| q/k 头 norm + RoPE | **可读 triton 融合核**:q/k、cos/sin 升 FP32,norm+旋转乘法 FP32，输出回 BF16;partial 0.25(64 维）;mrope interleaved | 同左 | 同左 |
| FULL 层输出门控（sigmoid) | BF16（模型代码） | 同左 | 同左 |
| GDN conv1d | 权重 BF16 不量化；conv 状态 BF16(argv)；自写开源核 **FP32 累加器** | 同左 | 同左 |
| GDN 门控 g | A_log/dt_bias FP32；自写 gating 核**输出 FP32** | 同左 | 同左 |
| GDN q/k L2 norm | triton l2norm(vllm fla)，接口 BF16（内部未逐项核实） | 同左 | 同左 |
| GDN decode/spec 递推（recurrent_gated_delta_rule) | **自写开源核全源码可读**:q/k/v/β/state BF16 进，g/gk FP32,BF16 输入 CAST_NONE 精确升 FP32 计算，输出/状态 **CAST_RINT(RNE)** 回 BF16 | 同左 | 同左 |
| GDN prefill chunk(chunk_gated_delta_rule) | 自写开源核：out BF16,state 同部署 dtype(BF16)；内部逐段未核实 | 同左 | 同左 |
| GDN 出口 RMSNormGated | triton 核：x/z/w 升 FP32、rstd FP32，输出 BF16 | 同左 | 同左 |
| GDN conv / SSM 状态缓存 | **BF16 + BF16**(argv 显式，覆盖 HF base 的 FP32 SSM 默认） | 同左 | 同左 |
| LM head / logits | LM head 走 w8a8→BF16;**本版 LogitsProcessor 不做 .float()**（仅 soft_cap/scale) | 同左 | 同左 |
| Sampler / MTP 验收采样 | logits 在 Sampler `.to(torch.float32)`;softmax/log_softmax/top-k/top-p 全 FP32;rejection sampler FP32 | 同左 | 同左 |

### B. FULL 注意力层（三版真正分叉的部分）

| 阶段/数据 | 不用 OSCAR 的原生基线 | 当前正式 OSCAR | 新增 BF16 历史版本 |
|---|---|---|---|
| QKV 进入注意力 | BF16（融合 triton 核输出，可证） | BF16(ABI 强制校验，非 BF16 拒绝） | BF16（同 main) |
| KV cache 物理存储 | 全量 **BF16**(kv_cache_dtype auto=模型 dtype) | 历史区 INT2 codes + FP16 scale/zero(D256 下 136 B/token/KV head)；精确窗口 BF16 | **与 main 完全相同**（存储协议未动） |
| 量化 store（写历史区） | 无（`reshape_and_cache` 原样写入） | BF16 进 → FP32 旋转/百分位裁剪/scale-zero 计算 → codes FP32 截断为 INT2 打包，scale/zero 舍入 FP16 落盘 | **与 main 完全相同**(store 源文件未改） |
| 精确窗口 / recent KV | 不适用 | BF16 原样字节，旋转/量化不触碰 | BF16 原样（合同 `exact_windows_BF16_unchanged`) |
| Q 旋转（CV/Hadamard) | 无（仅 RoPE) | BF16 进 → **FP32** Hadamard/FP32 旋转矩阵 → **FP32** 出 | **与 main 完全相同**(rotate 源文件未改） |
| 历史区解包/反量化 | 无 | INT2→int16 位操作→FP32 精确 Cast;FP16 metadata→FP32;FP32 `Mul(scale)` 后 `Add(zero)` | **与 main 完全相同**（两个 unpack 头未改） |
| 发布到 GM/L1 的 K/V 操作数 | 直接读 BF16 cache | **FP32** | FP32 重建+finite 检查后 **RNE→BF16** 发布（新增舍入点 ①) |
| Q 操作数（进 Cube 前） | BF16 | FP32（旋转结果直接用；当前/窗口 BF16→FP32 Cast) | FP32 finite 检查后 **RNE→BF16**（新增舍入点 ②) |
| FIA/CV 算子传参 | TND、sparse_mode=3、int8 mask、scale=head_dim^-0.5;**不传 inner_precise/antiquant**（仅 C8 路径用，本部署 C8=False) | 自定义 CV kernel(tasks/block_table/window_tags) | 同 main（算子不同，协议相同） |
| QK matmul | 闭源 FIA:BF16 接口可证，**内部操作数/累加不可证**（旁证见 §三） | FP32×FP32 → FP32 累加，`SetHF32Mode(false)` 禁 HF32/TF32 | **BF16×BF16 → FP32 累加**(ResultMatrix=float),`SetHF32Mode(false)` |
| online softmax(max/sum/exp/alpha) | 闭源 FIA 内部不可证 | **FP32**(`SoftmaxFlashV2<float>`) | **FP32**（未改） |
| P（注意力概率）送 PV 前 | 闭源 FIA 内部不可证 | **FP32，不做任何降精度 cast** | 每 tile 未归一化 P **RNE→BF16**（新增舍入点 ③) |
| softmax 分母 | 闭源 FIA 内部不可证 | FP32 running sum | FP32 running sum(**用未舍入的 FP32 exp**，不消费舍入后的 P) |
| PV matmul | 闭源 FIA 内部不可证 | FP32×FP32 → FP32 累加（cluster 模式 atomic FP32) | **BF16 P × BF16 V → FP32 累加** |
| 归一化 | 闭源 FIA 内部不可证 | FP32 `Muls(1/sum)` | FP32（未改） |
| partial 输出 | 无（FIA 直接出最终结果） | **FP32** | **FP32**（未改） |
| LSE | 图模式对外 buffer 按 query.dtype=BF16 分配（结果未使用）;**内部计算不可证** | **FP32**(FP32 `Log` + running max) | **FP32**（未改） |
| 跨分区 merge | 无 | **FP32**(`merge_lse.cpp`) | **FP32**（未改） |
| 归一化 partial 送逆 Rv 前 | 无 | FP32 直接进 Rv matmul | **RNE→BF16**（新增舍入点 ④) |
| 逆 Rv 旋转 matmul | 无 | FP32×FP32 → FP32 | **BF16×BF16 → FP32** |
| Rv 矩阵存储 | 无 | FP32(natural + NZ 重排仅改地址） | **natural ND BF16** 预分配；缺失显式失败无回退 |
| 算子最终输出（回 vLLM) | BF16(FIA 直出；融合输出量化 NotImplementedError) | FP32 workspace → **BF16** `copy_` | 同 main（出口转换不变） |
| padding 行 / 错误域 | 无此概念 | 输出 +0、LSE −inf、status 0；出错行 NaN；非法 scale/zero→code 3、末成员 NaN→code 2 | **逐字相同**（生成器 sha 证明） |
| metadata(status/tasks 等） | vLLM 标准 | status int32、tasks int64、block_table int32、window_tags/slots int64、packed uint8 | 与 main 完全相同 |
| 当前块（kind=2)prefill | FIA BF16 进 BF16 出 | **仍走原生 FIA**，输出 copy 上行到 FP32 partial | 同 main |
| decode 当前/窗口 KV 消费 | FIA 内部 | BF16 → FP32 Cast 进入 FP32 计算 | BF16 → FP32(finite 检查）→ RNE→BF16 与历史区同路发布 |
| MTP draft(1 个 FULL 层，eager) | draft 继承 BF16;fc 层 w8a8;FIA(draft 分支），全程 eager 不捕图 | q1 later-MTP 同主路径全程 FP32(MDL kernel OwnerTile=1);first MTP 当前块走原生 FIA | q1 独立 BF16 kernel(`decode_q1`),dtype 方案同 q4 |
| 图捕获（FULL_DECODE_ONLY）对 dtype | 无影响（图/eager 同一闭源 FIA、同批张量） | 本轮未上 NPU 图（not_run) | 未测（not_run) |

## 二、一句话总结三版差异

- **原生**：开源模型全链路可读——BF16 接口 + BF16 KV cache + w8a8 Linear;GDN/norm/RoPE/采样均有开源实现（内部 FP32 计算、RNE 回 BF16)；唯一黑盒是稠密 FIA 算子本体（BF16 进出可证，内部不可证）。
- **当前正式 OSCAR**:16 个 FULL 层改为「BF16 接口 + INT2/FP16 存储 + **全程 FP32 计算**(`SetHF32Mode(false)`)+ FP32 partial/LSE/merge + 出口 BF16」；其余 48 GDN 层及所有外围阶段与原生逐行相同。
- **BF16 分支**：在 main 基础上只把三个 Cube 乘法（QK、PV、最终逆 Rv）改为 **BF16 操作数 + FP32 累加**，新增 4 个 RNE 舍入点（K/V 发布、Q 进 Cube、每 tile P、归一化 partial 进 Rv)，分母仍用未舍入 FP32 exp,Rv 预生成 natural ND BF16；存储/量化/旋转/解包/softmax/LSE/merge/错误域全部不动。

## 三、FIA 内部精度：证据分级

**A. 接口层可证（本部署，910B/A2)**
- Q/K/V 以 BF16 原样进入 `torch_npu.npu_fused_infer_attention_score`（上游零提升点：triton 融合核输出 BF16,`split_qkv_rmsnorm_mrope.py:322-325`;attention 前向只切片不 cast,`attention_v1.py:1266-1273`)。
- KV cache BF16;`enable_c8_quant` 仅 ModelSlim `kv_quant_type=="C8"` 置位（`attention_v1.py:434-436`、`modelslim_config.py:942`)→ 本部署 **False,`inner_precise`/antiquant 确定不传**(`inner_precise=1` 仅见 C8 分支 `attention_v1.py:841-848`)。
- 传参全集：TND、sparse_mode=3、int8 0/1 mask、scale、block_table、actual_seq_lengths；图模式调同一算子 `.out` 变体（`attention_v1.py:1336-1357, 958-977, 1087-1106`;`device_op.py:55-100`)。

**B. 开源旁证（本地可读，非稠密 FIA 本体）**
「BF16 操作数 + FP32 累加 + CAST_RINT 输出」是官方同硬件开源实现的既有做法，三份独立证据：
1. vllm-ascend 自带 A2 sparse(MLA)attention:BF16 模板实例化 `references/vllm-ascend/csrc/attention/sparse_flash_attention/op_kernel/sparse_flash_attention.cpp:92-100`；中间计算 FP32(`// 中间计算数据类型为float, 高精度模式`)`.../arch22/sparse_flash_attention_service_cube_mla.h:117-122`;BF16 输出 `RoundMode::CAST_RINT` `.../arch22/sparse_flash_attention_service_vector_mla.h:1265-1268`。
2. GDN 自写核（全源码）:BF16 输入精确升 FP32 计算、CAST_RINT 回 BF16,`references/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule.h:238-240, 261-264, 395, 398`。
3. 同 SDK MMAD 微测：`csrc/experiments/fp32_bf16_mmad_ratio/kernel.cpp` + `reports/fp32_bf16_mmad_ratio_20261010/decision.json`;dav_c220 Mmad 白名单 `Tuple<float, bfloat16_t, bfloat16_t>`(`artifacts/cann_sdk_cube_public_20261010/impl/basic_api/dav_c220/kernel_operator_mm_impl.h:344-345, 368-372`)。
旁证只证明做法存在，**不**据此断言闭源稠密 FIA 逐位相同（合同 `reports/bf16_attention_20261016/native_dtype_contract.json:63-67` 的 not_claimed 边界不变）。

**C. 部署二进制不可证**
稠密 FIA 本体在 torch_npu/aclnn 闭源 .so;P 舍入、归约树、softmax/LSE 内部精度、`inner_precise` 默认语义本地无源可读。`artifacts/cann_sdk_*` 各目录经检索**不含任何注意力/FIA 源**（仅 matmul/L1/fixpipe 头文件快照）。

## 四、分阶段证据（文件：行号）

### 1. 原生基线（模型开源，栈内除 FIA/aclnn 外均有源码）

- 结构:`references/vllm/vllm/model_executor/models/qwen3_5.py:137-153`（层型二选一）;`references/vllm/vllm/transformers_utils/configs/qwen3_5.py:89-96`(interval=4 默认）、`:59, 86`(head_dim 256);GemmaRMSNorm `qwen3_5.py:39-41, 173-178`；输出门控 `references/vllm/vllm/model_executor/models/qwen3_next.py:240, 242-250, 294-302`。
- Embedding/LM head:`qwen3_5.py:224-227`(VocabParallelEmbedding 无 quant 实参）、`:473-482`(ParallelLMHead 带 quant_config;modelslim 映射 `modelslim_config.py:569-579, 665-677`)。
- LogitsProcessor 不 .float():`references/vllm/vllm/model_executor/layers/logits_processor.py:89-104, 54-73`；唯一 FP32 是 TP argmax 对收集 `:146-149`。Sampler FP32:`references/vllm/vllm/v1/sample/sampler.py:96`(注释 "Use float32 for the logits")、`:306`;top-k/p `ops/topk_topp_sampler.py:148-149, 201-206, 249`;rejection sampler `references/vllm/vllm/v1/sample/rejection_sampler.py:150, 214-216, 472`。
- RMSNorm:fused add+norm 自写开源核 FP32 buffer `references/vllm-ascend/csrc/moe/add_rms_norm_bias/op_kernel/add_rms_norm_bias.h:68-72, 98-116`(A2 默认启用 `references/vllm-ascend/vllm_ascend/utils.py:409-441`)；独立 Gemma norm 走闭源 aclnn 但 **rstd 输出 FP32** `references/vllm-ascend/csrc/torch_binding.cpp:660-687`(:683 `at::kFloat`)；主路径 fused 调用 `patch_qwen3_5.py:98-102, 136`。
- q/k norm+RoPE 融合 triton 核：q/k 升 FP32、norm FP32、cos/sin FP32、输出按 qkv.dtype=BF16 存回 `references/vllm-ascend/vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py:102-109, 139-160, 201, 240, 322-325`;patch 取 cos_sin_cache 后 `.to(qkv.dtype)` `patch_qwen3_5.py:44-64`;cos_sin_cache FP32 算后转模型 dtype `references/vllm/vllm/model_executor/layers/rotary_embedding/base.py:59-63, 89, 97`;rotary_dim=64 `rotary_embedding/__init__.py:66-72`。
- w8a8:BF16 激活 `torch.ops.vllm.quantize`(→`torch_npu.npu_quantize`,div/qint8)`references/vllm-ascend/vllm_ascend/ops/register_custom_ops.py:182-191, 277-279`;`npu_quant_matmul`(int8×int8,int32 bias,deq_scale,output_dtype=params_dtype=BF16)`w8a8_static.py:44-51, 66-67, 74-140`。
- GDN：状态 dtype 计算 `references/vllm/vllm/model_executor/layers/mamba/mamba_utils.py:84-96, 108-116`(argv 两个 bfloat16 → conv/SSM 皆 BF16；挂接 `qwen3_5.py:678-686`);conv1d 自写核 FP32 累加器 `references/vllm-ascend/csrc/moe/causal_conv1d/op_kernel/causal_conv1d.h:229, 297-333`（调用 `vllm_ascend/ops/gdn.py:199-303`);gating A_log/dt_bias FP32 + 输出 g FP32 `qwen_gdn_linear_attn.py:519-524`、`csrc/attention/fused_gdn_gating/op_kernel/fused_gdn_gating.h:80, 88-95`;recurrent 核接口与内部 dtype `csrc/attention/recurrent_gated_delta_rule/op_host/op_api/aclnn_recurrent_gated_delta_rule.h:21-32`、`op_kernel/recurrent_gated_delta_rule.h:238-240, 261-264, 395, 398`;chunk 核 `csrc/attention/chunk_gated_delta_rule/chunk_gated_delta_rule_torch_adpt.h:52-56`、`gdn.py:426-444`;RMSNormGated triton 内部 FP32 `vllm_ascend/ops/triton/layernorm_gated.py:56-67, 135-136`。
- FIA 接口与传参：`attention_v1.py:1303-1357`(eager 三分支）、`:1246-1254`（图分发）、`:806-983`(full_graph_fia,softmax_lse buffer 按 query.dtype 分配 `:833`);`device_op.py:47, 55-100, 2112-2115`;KV 写入 `attention_v1.py:1430-1456` + `device_op.py:49-52`。
- MTP draft:`references/vllm/vllm/model_executor/models/qwen3_5_mtp.py:80-97`(fc 走 quant_config→w8a8)、`:99-106`(1 个 full_attention DecoderLayer)、`:146`(spec_step_idx 复用）;Ascend patch 对 draft 同样生效 `patch_qwen3_5.py:153-196`；配置改写 `references/vllm-ascend/vllm_ascend/patch/platform/patch_speculative_config.py:104-113`。
- 版本钉版:`docs/reference_manifest.md:8`;`references/vllm-ascend/pyproject.toml:21-23`;`references/vllm-ascend/csrc/version.info`。

### 2. main（当前正式 OSCAR)FULL 层 FP32 链

- ABI 锁 BF16:`oscar_ascend/integration/impl.py:78-79`;`csrc/striped_attention_bindings.cpp:48-50`。
- 存储：槽布局 `csrc/kernels/attention_striped_unpack32.h:24` + `oscar_ascend/ops/contracts.py:78, 86-89`;store FP32 量化/FP16 落盘 `csrc/kernels/rotate_clip_store_striped.cpp:70-212` + `striped_store_pack.h:63-106`；精确窗口原样 `:388-409`。
- Q 旋转 FP32:`csrc/kernels/rotate_vector_init.cpp:65-69, 85-106, 238-239` + `csrc/rotation_bindings.cpp:39-42, 78` + `oscar_ascend/rotations.py:74-77`。
- 解包 FP32:`attention_striped_unpack32.h:98-197`(C16/C4)、`attention_hybrid_l1_unpack.h:118-209`(q4)、`attention_striped_unpack_padded.h:112-187`(q1)。
- FP32 计算链：K/V 发布 `attention_cv_l1_reuse_cluster16.cpp:816-826`（及 hybrid `:496-504`、mdl `:449-457`);Cube 类型 FP32 + 禁 HF32 C16 `:272-273, 350, 532`、hybrid `:81-82, 158, 237`、mdl `:56-57, 132, 211`;softmax FP32 C16 `:959-973`;P 不 cast C16 `:978`;PV FP32 C16 `:572, 603-607, 1023-1034`;LSE FP32 C16 `:1141-1150` + `runtime.py:202`;merge FP32 `merge_lse.cpp:24-27, 67-93`；逆 Rv FP32 C16 `:614-618`、hybrid `:787`、mdl `:250-252`;NZ 仅改地址 `runtime.py:766-772`；出口 FP32→BF16 `impl.py:292`;padding/错误域 C16 `:1142-1167`、`merge_lse.cpp:52-63`、`rotate_vector_init.cpp:271-274`。
- MTP/q1 同构：`attention_cv_mdl_decode.cpp:846-847`;first MTP 当前块走原生 FIA `oscar_ascend/integration/current_attention.py:40-42, 167-202`。

### 3. BF16 分支（仅列与 main 不同处）

- Cube 类型 BF16 进 FP32 出：`csrc/kernels/attention_cv_bf16_decode.cpp:44-46, 60-62, 122`、C4 `attention_cv_bf16_cluster4.cpp:238-240, 482`、C16 `attention_cv_bf16_cluster16.cpp:293-295`。
- RNE 转换头 `csrc/kernels/attention_bf16_cast.h:20-27`(normal-mask、≤64 元素、CAST_RINT);decode 7 处 `attention_cv_bf16_decode.cpp:270, 284, 458, 599, 664, 686, 797`;C4/C16 各 6 处（`attention_cv_bf16_cluster4.cpp:575, 601, 614, 790, 801, 925`;`attention_cv_bf16_cluster16.cpp:661, 687, 700, 876, 887, 1035`)；生成器强制计数 `tools/generate_bf16_decode.py:154-157`、`tools/generate_bf16_clusters.py:152-165`。
- K/V RNE 发布 decode `:451-460, 682-690`、C4 `:782-804`、C16 `:875-893`;Q RNE decode `:262-271`、C4 `:594-602`、C16 `:673-687`;QK/PV/Rv 乘法 decode `:205-216, 239-249, 647-650, 744-754`、C4 `:480-563`、C16 `:549-571, 599-649`。
- softmax 仍 FP32 + P RNE + 未舍入分母：decode `:565-601, 657-658`、C4 `:816-983`、C16 `:1013-1093`;合同 `configs/acceptance_bf16.json:9-15`。
- Rv natural ND BF16:`oscar_ascend/runtime.py:760-765`;缺失显式失败 `oscar_ascend/integration/impl.py:240-243`。
- LSE/partial/merge/padding/status 未改：decode `:815-847`;`merge_lse.cpp`（未改）;`tools/generate_bf16_clusters.py:174-179`(sha 证明）;`configs/acceptance_bf16.json:27-36`。
- 独立 oracle 同方案：`tools/bf16_reference.py:34-40, 65, 68-93, 97-99`（注释 :81「The denominator consumes FP32 P. Only the Cube PV operand rounds.」)。

## 五、仍未可证项（收窄后）

| # | 项 | 说明 |
|---|---|---|
| 1 | 稠密 FIA 内部 QK/PV 操作数与累加、softmax/LSE、P 舍入、输出舍入模式、`inner_precise` 默认语义 | torch_npu/aclnn 闭源 .so;§三-B 三份旁证只证明做法存在，不断言逐位相同 |
| 2 | `_npu_reshape_and_cache` / `npu_quant_matmul` / aclnn rms_norm 等 aclnn 算子内部归约树 | 接口 dtype 可证，内部不可证 |
| 3 | Embedding 是否被 checkpoint `quant_model_description.json` 单独量化 | 该 json 不在本机（`modelslim_config.py:665-677`) |
| 4 | GDN chunk 核与 fla l2norm 的逐段内部精度 | 接口 dtype 可证（自写/triton 开源），未逐项核实 |

## 六、BF16 分支未变更面（防误读）

`git diff 7047ce36..HEAD` 下与 dtype 相关部分**完全未动**:Q 旋转 kernel 与绑定、量化 store、两个 unpack 头、`merge_lse.cpp`、全部旧 FP32 `attention_cv_*` kernel、`oscar_ascend/rotations.py`、`oscar_ascend/ops/reference.py`、`configs/acceptance.json`（旧 FP32 合同逐字节保留，仅作差异诊断）。host 侧改动仅 `impl.py`、`contracts.py`、`cv_dispatch.py`、`meta.py`、`runtime.py` 与绑定/CMake 注册；生成器 `--check` 确认 `FP32_sources_modified: false`。

精度门：旧合同 `configs/acceptance.json`(fused_attention atol/rtol 0.005、pack/unpack 逐位、store_dequant 0.002）对 main 继续冻结；BF16 分支独立合同 `configs/acceptance_bf16.json`(`oscar_bf16_attention_v2`,atol/rtol 0.005 对独立 BF16 oracle，测前冻结，容差不事后放宽）。BF16 分支 NPU 执行/图/服务/模型质量/性能均 `not_run`，本表 dtype 结论为源码与 CPU-debug 层证据。
