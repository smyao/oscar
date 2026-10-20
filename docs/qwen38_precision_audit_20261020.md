# Qwen3.8 原版与 W8A8 的 OSCAR 精度差排查

日期：2026-10-20。审计 HEAD：`b388590f`，另有审计前已存在的未提交 C16/运行时改动，未改动这些内容。

用户观察：W8A8 比默认原版的 OSCAR LongBench 高约 5 个点；模型、quantization 和旋转配置同时变化；W8A8 PT 由一键脚本为该模型重新生成。尚无两次实际 effective-target、成功 PT 的 capture manifest、逐题结果或对应 native 分数，不能将 5 点全部归因于某一行代码。

排查阶段只读源码、运行主机回归和人工数值反例；用户随后授权修复。最终改动涉及模型身份与校准产物校验，默认数值设置和内核不改；未连接真机、未测速。数值证据见 [evidence.json](../reports/qwen38_precision_audit_20261020/evidence.json)。

档案参照：#86 的 FULL/MTP 校准边界，#97/#165/#166 的校准准备与缓存，#126/#145/#164 的数值及设备证据边界，#148 的模型质量回归。源码发现不追加为已经定位的真机故障。

## 0. 手动启动的实际模型与 OSCAR/PT 身份未绑定，已修复

用户补充手动启动 W8A8 时指定 `quantization=ascend`，原版时删除该参数。对真正的原版权重，这种选择是正常的；量化是否启用最终仍由 vLLM 根据参数和模型元数据解析。

修复前，`runtime.py` 从 `OSCAR_TARGET_CONFIG` 或默认 target.json 读取模型，并据此选择 PT；未与 runner 实际加载的 `model_config.model/quantization` 核对。主机替身已执行真实 runner 包装器及 `_prepare_rotations`，复现“原生命令指向 B、OSCAR 配置指向同形状 A，仍选择 A 的 PT 而没有拒绝”。这是一项确定的源码缺陷，但现场两套有效配置尚未取得，不声称已经证明全部 5 点由它导致。

现在 `plugin._patch_runner` 在原生 `load_model` **之前**调用 `integration/model_identity.py`，核对实际模型规范路径、解析后的量化模式、activation/cache dtype、revision、HF overrides。检查必须早于模型构造，因为 Attention 构造期间就会加载旋转。配置漂移立即报错，既不加载错 PT，也不自动篡改原生模型配置。

一键入口正常生成一致的 effective-target；手动 vLLM 命令若与 `OSCAR_TARGET_CONFIG` 不一致，错误会指出差异，并要求通过统一模型入口或对齐配置后准备对应 PT。检查仅读取主机配置，不扫描原生源码、不访问权重或同步设备，也不增加默认部署审计流程。

## 1. 已纠正：模型声明 FP32 不表示当前原生应切换 FP32

| 位置 | 当前行为 |
|---|---|
| `reports/official_alignment_20261018/official_qwen38_config.json:92` | 保存的官方配置声明 `mamba_ssm_dtype="float32"` |
| `tools/target_cli.py` | 正式服务使用 BF16 conv/SSM cache |
| `tools/calibrate_model_rotations.py` | 校准 LLM 也使用两个 BF16 cache dtype |
| `references/vllm/vllm/model_executor/models/qwen3_5.py:678-686` | 状态 dtype 取自实际 cache_config |
| `references/vllm/vllm/model_executor/layers/mamba/mamba_utils.py:84-96` | 显式 BF16 确实解析成 BF16 状态；并非只改变打印标签 |

后续核实的决定性证据是 `references/vllm-ascend/csrc/attention/chunk_gated_delta_rule/op_host/chunk_gated_delta_rule_tiling.cpp:360-363`：当前 chunk-prefill 对非 Ascend950 的 FP32 state 直接返回 `GRAPH_FAILED`。本项目目标为 910B4，原生当前路径使用 BF16 状态。recurrent decode/MTP 支持 FP32，不等于整模型支持。

**撤回“BF16 SSM 是应优先修复的配置错误”的判断。** HF 模型声明与当前原生运行能力不是一回事。本轮不切换 FP32，不改 GDN 数学，也不接入另一条未经目标验证的原生分块路线。

本轮实际修复是：服务、校准、PT 身份共用同一原生 BF16 精度声明；显式不支持的 FP32/FP16 cache 配置报错，不再无声忽略。默认 argv 仍保持原生 BF16。原先的主机反例保存在 evidence.json，并标注结论更正，不能将其视为原生精度故障的证据。

## 2. 默认校准的 30000 tokens 不代表长上下文覆盖

`tools/calibrate_model_rotations.py:58-85` 累计多条完整 GPQA prompt。`:158-165` 使用单请求、eager、关闭 chunked prefill，`:176-177` 每题单独 `generate(max_tokens=1)`，只执行 prefill，每题位置重新从零开始。

已保存日志 `reports/rotation_process_fix_20261019/target_failure.txt:9,42` 给出 99 条、29867 tokens、引擎 max_model_len=848。按当前 `max(prompt_len)+1` 规则，该次输入最长为 847 tokens。**该历史运行随后因线程池报错，没有完成校准，不能把它写成当前成功 PT 的采样结果。** 这里用于说明默认配方的输入性质。

当前默认配方未验证长位置 RoPE、暖 prefill、跨 chunk 的状态重用以及 decode/MTP 分布。它能正确拟合所采短题的统计量，却没有证据证明这些矩阵在原版模型的 LongBench 上足够好。分别为 W8A8 和原版重新生成 PT 只能消除直接串用，不能保证两套量化误差相同。

## 3. 固定裁剪与压缩 oracle 不能替代模型质量验收

两套模型默认共用 sink=64、recent=256、K clip=0.96、V clip=0.92。`integration/aligned_math.py:59` 将裁剪参数传给 writer；`tools/aligned_reference.py:38-57` 定义当前按每行绝对值排序后裁剪的算法。重新学习 Rk/Rv 不会自动为原版调整这些参数。

人工 CPU 反例：旋转域 D256、Q=8e0、历史 K=[10e0,0]、V=[全1,全-1]。默认 K 裁剪删除唯一非零通道，注意力首坐标从 0.9866143465 变成 0。它只证明该算法依赖实际分布，**不是 Qwen3.8 的真实激活，也不证明当前 clip 应改成某个值**。

已有 v3 oracle 主要验证“算子是否按声明的压缩数学正确执行”，没有给出原版模型压缩前后的 logits/LongBench 损失。不能以这些测试通过宣布原版模型质量已验收，也不能未经对照放宽容差或修改裁剪。

## 4. 已排除或未发现的旧 W8A8 假设

- `install_serve -> apply_model_options -> variant_config -> effective-target.json` 将同一模型、量化模式和 HF overrides 传给校准与服务。原版默认不传 ascend quantization，旧 YaRN 覆盖已经移除。
- 正常自动 PT 身份绑定模型目录、config/index、权重 stat、tokenizer/ModelSlim、量化模式和 HF overrides；换模型不会仅因架构相同直接复用旧 Qwen3.5 PT。
- OSCAR 接收的是原生 projection 后的 BF16 Q/K/V；原生 Q/K norm 和 RoPE 在它之前，FULL gate 和输出投影在它之后。未发现仍需 W8A8 scale 却没有处理原版的分支。
- Q/K @ Rk、V @ Rv、历史输出 @ Rv 转置的方向与 TP4 头分组未发现错误。
- `output_gate_type=swish` 属于 GDN；原生 GDN 读取它，Ascend 实现执行 SiLU。FULL attention 的 sigmoid 是独立门控。审计中的初步冲突猜测已经撤回，不能据此修改 FULL。
- MTP 的 compact 路径接受 `AscendUnquantizedLinearMethod`，没有发现仅支持 W8A8 的限制。

## 5. 其他边界问题

**显式导入 PT 的身份检查缺口，已修复，和本次自动生成不同。** 修复前，对缺失 fingerprint 的外部官方 pair 使用当前 fingerprint 默认值，合法旧模型矩阵可能被贴上新模型身份。现在 K/V 必须各自声明匹配的模型指纹和完整执行 profile，缺失或不一致直接拒绝，不能从目标配置补造来源。这仍不能解释用户明确说明的自动重采结果。

**PT 身份版本和采样绑定，已修复。** profile 增加本地 recipe_version=2 与有效 BF16 conv/SSM dtype。采样 manifest、拟合 K/V、combined artifact 和运行时均核对同一 profile。旧自动缓存切到新路径生成，完整的新缓存可复用；旧显式缓存不被覆盖或接受。recipe 是本地采样协议版本，不代表已经逐字节绑定真机所有原生二进制。

**生成配置可能变化。** 服务不指定 `--generation-config`，原生默认 auto，可读取模型目录的 generation_config；chat template 同样随模型目录变化。若 LongBench 请求没有显式覆盖相关项，两个模型的对照还需核实这些差异。

## 验证与后续归因

本轮执行以下主机回归，73 passed：

```bash
.venv/bin/python -m pytest -q tests/test_model_config.py tests/test_official_rotation_calibration.py tests/test_rotation_capture.py tests/test_aligned_reference.py
```

另独立执行 SSM 参数提取/覆盖反例和裁剪反例，结果落盘。Torch 报本机缺少 NumPy 的初始化警告，张量测试正常完成。以上均不代表 NPU、图、模型质量或性能验收。

后续修复回归覆盖实际 runner 身份与加载前拒绝、共享状态精度、direct/observe/probe 配置传播、原生冻结 argv、两层 spawn、PT 拟合/导入/复用及运行时校验。运行记录见 `reports/qwen38_precision_audit_20261020/repair_validation.json`。

固定一键入口仍为：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

recipe v2 的首次启动会为当前模型自动生成新 PT，之后复用；无需删除原文件。显式指定的旧 PT 或不含所需身份的外部 PT 会明确报错，避免贴上当前模型身份继续运行。此修复保持 BF16 GDN、INT2 数学、裁剪、MTP=3、APC、graph、预算、设备和端口；没有修改现有 timer。

现场归因应先读取两次已有 effective-target、MODEL_SELECTION、成功 PT 的 calibration_profile/capture_manifest，以及已有 native 分数和生成参数。特别要比较：

- 原版的 `native分数 − OSCAR分数`；
- W8A8 的 `native分数 − OSCAR分数`。

只比较两个 OSCAR 分数无法把原模型差异与 OSCAR 引入的额外损失拆开。用户随后要求不再做对比而直接修复，因此本轮没有请求或运行新的模型对照；修复已证实的实际模型绑定、配置拒绝和 PT 身份漏洞。长位置校准及裁剪的模型质量影响仍未确认，不修改其数学或容差。保持 MTP/APC/graph/预算，不要求重跑 AISBench。当前没有宣布 5 点损失已定位或已修复。
