# 三版本注意力/KV 全链路数值 dtype 对照（2026-10-16）

三个口径：

- **不用 OSCAR 的原生基线**：相同冻结 argv（`reports/stable_32k_64k_20261015/frozen_launch.json`：APC 关、显式 `--max-num-batched-tokens 16384`、原生调度器、TP4、MTP3、FULL_DECODE_ONLY）下完全禁用 OSCAR，原生 vLLM-Ascend + 闭源 FIA。
- **当前正式 OSCAR**：main `7047ce36`（= `codex/aisbench-stable` 同步），candidate 组合（striped + decode bundle + hybrid L1 q4 + MDL q1 + C4 member pipeline + C16 L1 reuse）。
- **新增 BF16 历史版本**：`codex/oscar-bf16-attention`（`7047ce36` + `b3c83069`/`d49a7233`/`7b170738`），四个新算子 `attention_cv_bf16_{decode, decode_q1, cluster4, cluster16}_out`。

每格只写源码/合同可证的内容；闭源不可见处明确标「闭源未知」，不以硬件能力冒充算子行为。证据（文件:行号）集中在第三节。

## 一、主表：阶段 × 三版本 dtype

| 阶段/数据 | 不用 OSCAR 的原生基线 | 当前正式 OSCAR | 新增 BF16 历史版本 |
|---|---|---|---|
| 模型激活 / QKV 进入注意力 | BF16（dtype auto→模型 BF16；w8a8 Linear 输出 BF16） | BF16（ABI 强制校验，非 BF16 直接拒绝） | BF16（同 main，绑定未改） |
| Linear 层（w8a8） | int8×int8 matmul → BF16 输出；不碰 attention/KV | 同原生（OSCAR 不涉及） | 同原生 |
| KV cache 物理存储 | 全量 BF16（kv_cache_dtype auto=模型 dtype） | 历史区 INT2 codes + FP16 scale/zero（D256 下 136 B/token/KV head）；精确窗口 BF16 | **与 main 完全相同**（存储协议未动） |
| 量化 store（写历史区） | 无（`reshape_and_cache` 原样写入） | BF16 进 → FP32 旋转/百分位裁剪/scale-zero 计算 → codes FP32 截断为 INT2 打包，scale/zero 舍入 FP16 落盘 | **与 main 完全相同**（store 源文件未改） |
| 精确窗口 / recent KV | 不适用（全部 KV 皆 BF16） | BF16 原样字节，旋转/量化不触碰 | BF16 原样（合同 `exact_windows_BF16_unchanged`） |
| Q 旋转（CV/Hadamard） | 无（仅 RoPE，BF16） | BF16 进 → **FP32** Hadamard/FP32 旋转矩阵 → **FP32** 出 | **与 main 完全相同**（rotate 源文件未改，输出仍 FP32 query_rot） |
| 历史区解包/反量化 | 无 | INT2→int16 位操作→FP32 精确 Cast；FP16 metadata→FP32；FP32 `Mul(scale)` 后 `Add(zero)` | **与 main 完全相同**（两个 unpack 头文件未改） |
| 发布到 GM/L1 的 K/V 操作数 | 直接读 BF16 cache | **FP32** | FP32 重建并做 finite 检查后 **RNE→BF16** 发布（新增舍入点 ①） |
| Q 操作数（进 Cube 前） | BF16 | FP32（旋转结果直接用；当前/窗口 BF16→FP32 Cast） | FP32 finite 检查后 **RNE→BF16**（新增舍入点 ②） |
| QK matmul | 闭源 FIA：BF16 接口，**内部操作数/累加精度闭源未知** | FP32×FP32 → FP32 累加，`SetHF32Mode(false)` 显式禁 HF32/TF32 | **BF16×BF16 → FP32 累加**（ResultMatrix=float + L0C float 断言），同样 `SetHF32Mode(false)` |
| online softmax（max/sum/exp/alpha） | 闭源未知（通行实现标注 FP32，无直接代码证据） | **FP32**（`SoftmaxFlashV2<float>`，FP32 running max/sum/alpha） | **FP32**（同一 V2 float 路径，未改） |
| P（注意力概率）送 PV 前 | 闭源未知 | **FP32，不做任何降精度 cast** | 每 tile 未归一化 P **RNE→BF16**（新增舍入点 ③）；仅 Cube PV 操作数舍入 |
| softmax 分母 | 闭源未知 | FP32 running sum | FP32 running sum（**用未舍入的 FP32 exp 求和**，不消费舍入后的 P） |
| PV matmul | 闭源未知 | FP32×FP32 → FP32 累加（cluster 模式 Cube atomic FP32） | **BF16 P × BF16 V → FP32 累加**（atomic=1 仅作用于 FP32 C） |
| 归一化 | 闭源未知 | FP32 `Muls(1/sum)` | FP32 `Muls(1/sum)`（未改） |
| partial 输出（跨分区中间结果） | 无（FIA 直接出最终结果） | **FP32** | **FP32**（未改） |
| LSE | 对外 buffer 按 query dtype=BF16 分配；内部计算精度闭源未知 | **FP32**（FP32 `Log` + running max） | **FP32**（未改） |
| 跨分区 merge（merge_lse） | 无 | **FP32**（Exp/加权和/Log 全 FP32） | **FP32**（`merge_lse.cpp` 未改） |
| 归一化 history partial 送逆 Rv 前 | 无 | FP32 直接进 Rv matmul | **RNE→BF16**（新增舍入点 ④） |
| 逆 Rv 旋转 matmul | 无 | FP32×FP32 → FP32 累加（rv 为 GM float） | **BF16×BF16 → FP32 累加** |
| Rv 矩阵本身存储 | 无 | FP32（natural + NZ 重排，仅地址变换不改值） | **natural ND BF16**，初始化预分配缓存；缺 BF16 Rv 显式失败、无回退 |
| 算子最终输出（回 vLLM） | BF16（FIA 直出；融合输出量化直接 NotImplementedError） | FP32 workspace → **BF16** `copy_`（唯一出口舍入） | 同 main：FP32 → BF16 `copy_`（合同「existing final output conversion to BF16」） |
| padding 行 / 错误域 | 无此概念（原生逐请求有效槽） | 输出 +0（FP32 域）、LSE −inf、status 0；出错行 NaN；非法 scale/zero→code 3、末成员 NaN→code 2 | **逐字相同**（PublishEmpty/PublishRows/PublishStatus 生成器 sha 证明未改；合同 status/padding 精确域） |
| status/tasks/block_table 等 metadata | vLLM 标准（block_table int32 等） | status int32、tasks int64[16]、block_table int32、window_tags int64、slots int64、packed uint8 | **与 main 完全相同** |
| 当前块（kind=2）prefill | FIA BF16 进 BF16 出 | **仍走原生 FIA**（BF16 进出），输出 copy 上行到 FP32 partial | 同 main（native-current 路径未改） |
| decode 当前/窗口 KV 消费 | FIA 内部 | BF16 → FP32 Cast 后进入 FP32 计算 | BF16 → FP32 Cast（finite 检查）→ RNE→BF16 与历史区同路发布 |
| MTP draft/verify | draft 继承目标模型 BF16 + FIA（eager，enforce_eager） | q1 later-MTP 同主路径全程 FP32（MDL kernel OwnerTile=1）；first MTP 当前块走原生 FIA | q1 有独立 BF16 kernel（`decode_q1`），dtype 方案与 q4 相同；其余不变 |
| GDN/mamba 状态缓存 | BF16（argv 显式 `--mamba-cache-dtype bfloat16`） | 同原生（OSCAR 不涉及） | 同原生 |
| 图捕获（FULL_DECODE_ONLY）对 dtype 的影响 | 无影响（图/eager 同一闭源 FIA、同批张量） | 本轮未上 NPU 图（not_run） | 未测（not_run） |

## 二、一句话总结三版差异

- **原生**：BF16 接口 + BF16 KV cache + 闭源 FIA；内部 QK/PV/softmax 精度不可见，只声明接口 dtype，不声明内部逐位行为。
- **当前正式 OSCAR**：「BF16 接口 + INT2/FP16 存储 + **全程 FP32 计算** + FP32 partial/LSE/merge + 出口 BF16」。唯一两处舍入：store 时 scale/zero 落 FP16 与 INT2 量化、出口 FP32→BF16；计算侧 `SetHF32Mode(false)` 故意锁死 FP32 MMAD。
- **BF16 分支**：存储、量化、Q 旋转、解包重建、softmax/LSE/merge、错误域**全部不动**；只把三个 Cube 乘法（QK、PV、最终逆 Rv）改为 **BF16 操作数 + FP32 累加**，为此新增 4 个 RNE 舍入点（K/V 发布、Q 进 Cube、每 tile P、归一化 partial 进 Rv），分母仍用未舍入 FP32 exp，Rv 矩阵预生成 natural ND BF16，出口仍 BF16。

## 三、分阶段证据（文件:行号）

### 1. 进入注意力的 Q/K/V

- 原生：argv 无 `--dtype` → vLLM auto 解析链 `references/vllm/vllm/config/model.py:149, 2017-2018, 2049-2062`，平台首选 BF16 `references/vllm/vllm/platforms/interface.py:153-158`；w8a8 `params_dtype` 默认 BF16、int8 matmul 输出 BF16 `references/vllm-ascend/vllm_ascend/quantization/methods/w8a8_static.py:48, 133-139`；q_norm/k_norm/rotary→attention 无 cast `references/vllm/vllm/model_executor/models/qwen3_next.py:306-315`。
- OSCAR（两版共用绑定）：`oscar_ascend/integration/impl.py:78-79` 非 BF16 Q/K/V 直接 `OscarReadinessError`；`csrc/striped_attention_bindings.cpp:48-50` `Check(query, ..., at::kBFloat16)`、`Check(queryRot, ..., at::kFloat)`。

### 2. KV cache / 存储协议

- 原生 KV cache BF16：`references/vllm/vllm/config/cache.py:75-76`（auto=模型 dtype）→ `references/vllm/vllm/model_executor/layers/attention/attention.py:276-278, 603-604` → 物理分配 `references/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:4628, 4637, 4642`；写入无 Python 侧 cast `references/vllm-ascend/vllm_ascend/device/device_op.py:49-52`。旁路排除：FP8 需 kv_cache_scheme（`attention.py:237-243`）、C8 需 checkpoint 声明（`modelslim_config.py:941-942`）、fa quant 门控 A2 上 False（`quantization/utils.py:217-220`）；attention 断言 k/v scale=1.0 `attention_v1.py:1511`。
- OSCAR 槽布局 `[Kcode64B, Vcode64B, Kscale2B, Kzero2B, Vscale2B, Vzero2B]`：`csrc/kernels/attention_striped_unpack32.h:24`（注释）、`oscar_ascend/ops/contracts.py:78, 86-89`；D256 下 136 B/token/KV head。
- 量化 store（两版相同，`rotate_clip_store_striped.cpp` 未改）：BF16 进、FP32 旋转/百分位裁剪 `:70-181`；`float scale=(max-min)/3.0F` `:187`；scale/zero 舍入 FP16 并回读 FP32 作除数 `:192-194`；codes `Cast(..., CAST_TRUNC, ...)` int32 `:204`、2-bit 打包 `csrc/kernels/striped_store_pack.h:63-106`；FP16 scale/zero 原始 2 字节写入 `:207-212`。
- 精确窗口 BF16 原样：`rotate_clip_store_striped.cpp:388-409`（注释 :390-391 “No rotation or quantizer rounding can touch these bytes”）；BF16 分支合同 `configs/acceptance_bf16.json:30` `exact_windows_BF16_unchanged`。

### 3. Q 旋转（两版相同）

- `csrc/kernels/rotate_vector_init.cpp:65-69` BF16→FP32 Cast；Hadamard 全程 FP32 `:85-106`（float signs :91、FP32 归一化常数 :87-88）；旋转矩阵 FP32 `:238` + 绑定强制 `csrc/rotation_bindings.cpp:39-42` + artifact 校验 `oscar_ascend/rotations.py:74-77`；输出 FP32 `:239`、`rotation_bindings.cpp:78`、buffer `oscar_ascend/runtime.py:194`。

### 4. 解包/反量化（两版相同）

- C16/C4 用 `csrc/kernels/attention_striped_unpack32.h`：int16 位操作 :98-113，INT2→FP32 精确 Cast :115-119，FP16 metadata Gather :124-126，raw-bit 守卫（非法 scale/zero→error 3）:140-177，FP16→FP32 :180，FP32 `Mul`+`Add` :187-197。
- q4 用 `csrc/kernels/attention_hybrid_l1_unpack.h`：同构 :118-209；q1 用 `csrc/kernels/attention_striped_unpack_padded.h`：同构 :112-187（注释 :172-187 “Preserve old FP32 reconstruction exactly: q*scale first, then +zero.”）。
- BF16 分支未改这两个头文件与 `store_int2.cpp`（`git diff 7047ce36..HEAD` 证据见 `tools/generate_bf16_clusters.py:166-172` 逆变换恢复原母版、`--check` 输出 `FP32_sources_modified: false`）。

### 5. main 的 FP32 计算链（当前正式 OSCAR）

- K/V 发布 FP32：`csrc/kernels/attention_cv_l1_reuse_cluster16.cpp:816-826`（`DataCopy(work[...], dequantBuf.Get<float>(), ...)`，`work` 为 `GlobalTensor<float>` :334）；q4 `attention_cv_hybrid_l1_decode.cpp:496-504, 718-724`；q1 `attention_cv_mdl_decode.cpp:449-457`。
- 窗口/当前 BF16→FP32：C16 `:750, 783, 809`；hybrid `:430, 463, 489`；mdl `:383, 416, 442`。
- Cube 类型 FP32：C16 `:272-273`、hybrid `:81-82`、mdl `:56-57`（`MatmulType<..., float, ...>`）；禁 HF32：C16 `:350, 532`、hybrid `:158, 237`、mdl `:132, 211`（注释 “no implicit HF32/TF32 relaxation of the frozen oracle.”）。
- softmax FP32：`SoftmaxFlashV2<float,...>` C16 `:959-961`、hybrid `:622-624`、mdl `:557`；FP32 alpha C16 `:966-973`；score×scale FP32 C16 `:858`。
- P 不 cast：C16 `:978`（FP32 scores 直接 DataCopy 进 GM P 区）。
- PV FP32：C16 `:572, 603-607`（atomic FP32 累加）、跨单元 `Add` FP32 `:1023-1026`、归一化 `:1034`。
- LSE FP32：C16 `:1141-1150`、`runtime.py:202`、绑定 `striped_attention_bindings.cpp:56`。
- merge FP32：`csrc/kernels/merge_lse.cpp:24-27, 67, 78-84, 93`。
- 逆 Rv FP32：C16 `:614-618`（rv 为 `GlobalTensor<float>` :318）；hybrid `:787`（rv float :131）；mdl `:250-252`（rv float :106）；NZ 重排不改 FP32 值 `runtime.py:766-772`。
- 出口 FP32→BF16：`oscar_ascend/integration/impl.py:292`（`output.view(n,h,d).copy_(workspace.output[:n])`，workspace FP32 `runtime.py:203`，output 为原生 BF16 buffer）。
- padding/错误域：C16 `PublishEmpty :1155-1167`、`PublishRows :1142-1148`；merge 端 `merge_lse.cpp:52-63`；旋转端无效槽 `rotate_vector_init.cpp:271-274`。

### 6. BF16 分支的计算链（仅列与 main 不同处）

- Cube 类型：`csrc/kernels/attention_cv_bf16_decode.cpp:44-46`（A/B=`bfloat16_t`、Result=`float`）、C4 `attention_cv_bf16_cluster4.cpp:238-240`、C16 `attention_cv_bf16_cluster16.cpp:293-295`；L0C float 断言 decode `:60-61`；禁 HF32 decode `:122`、C4 `:482`。
- 共享 RNE 转换头 `csrc/kernels/attention_bf16_cast.h:20-27`（normal-mask、≤64 元素、`CAST_RINT`）；调用点：decode 7 处（`attention_cv_bf16_decode.cpp:270, 284, 458, 599, 664, 686, 797`，生成器强制 `tools/generate_bf16_decode.py:154-157`）、C4/C16 各 6 处（`attention_cv_bf16_cluster4.cpp:575, 601, 614, 790, 801, 925`；`attention_cv_bf16_cluster16.cpp:661, 687, 700, 876, 887, 1035`）。
- K/V：FP32 重建+finite 检查后 RNE→BF16 发布 decode `:451-460`（注释 “FP32 dequant/finite checks precede the only BF16 RNE publication.”）、`:682-690`；C4 `:782-804`；C16 `:875-893`。
- Q：FP32 载/检查 `:262-267`，RNE→BF16 `:270-271`；C4 `:594-602`；C16 `:673-687`。
- QK/PV/Rv 乘法：MMAD 实现 decode `:205-216`（`SetTensorA(operands[...])` BF16、`IterateAll<true>(work[c])` FP32 GM）；QK `:239, 744`；PV `:242, 747`，FP32 累加消费 `:647-650`；C4 atomic 仅作用 FP32 C `:503-511`（注释 :508）；逆 Rv `:249, 754`、C4 `:485, 526, 563`、C16 `:645, 649`。
- softmax 仍 FP32：V2 float decode `:577-581`、alpha FP32 `:586-593`；P RNE→BF16 `:598-601`；分母用未舍入 FP32 sum `:657-658`；合同 `configs/acceptance_bf16.json:10-13`。
- 逆 Rv 输入：归一化 partial RNE→BF16 decode `:662-665, 796-799`、C4 `PublishNormalizedQueries :568-580`、C16 `:654-666`。
- Rv 预生成 natural ND BF16：`oscar_ascend/runtime.py:763-765`（`inverse_value_transposed.to(dtype=torch.bfloat16)`，FP32 源 `oscar_ascend/rotations.py:41, 83`）；缺失显式失败 `oscar_ascend/integration/impl.py:240-243`。
- LSE/partial/merge FP32 与 padding/status 精确域：decode `:815-847`、`merge_lse.cpp`（未改）、生成器证明 PublishRows/PublishEmpty/PublishStatus 逐字未改 `tools/generate_bf16_clusters.py:174-179`；合同 `configs/acceptance_bf16.json:14-15, 32-36`。
- 独立 oracle 同方案：`tools/bf16_reference.py:34-40`（RNE）、`:65, 83`（Q/K/V 与逐 tile P 舍入，注释 :81 “The denominator consumes FP32 P. Only the Cube PV operand rounds.”）、`:68-70, 77-85, 89`（FP32 状态/alpha/LSE）、`:90-93`（逆 Rv）、`:97-99`（FP32 merge）。

### 7. 原生 FIA 侧

- prefill/decode 均走 FIA：路由 `references/vllm-ascend/vllm_ascend/attention/attention_v1.py:1468-1475`；MTP 下 paged attention 被禁用 `references/vllm-ascend/vllm_ascend/attention/utils.py:104-106`，故 decode 也走 FIA；图模式分发 `attention_v1.py:1246-1254`，调同一闭源算子 `.out` 变体 `:958-977, 1087-1106`。
- BF16 进 BF16 出：输入承接第 1 节（Python 层零转换，`device_op.py:92-100`）；输出写框架 BF16 buffer，融合输出量化 `raise NotImplementedError` `attention_v1.py:1508-1509`。
- 对外 LSE buffer 按 query dtype 分配（即 BF16）：`attention_v1.py:833, 1002`。
- 合同边界：`reports/bf16_attention_20261016/native_dtype_contract.json:56-68`（confirmed 五条 / not_claimed 四条）。

## 四、闭源未知项（原生列明确不声称的部分）

| # | 未知项 | 说明 |
|---|---|---|
| 1 | FIA 内部 QK/PV 的操作数与累加精度 | `torch_npu.npu_fused_infer_attention_score` 闭源；dav_c220 Mmad 白名单含 `Tuple<float, bfloat16_t, bfloat16_t>`（`artifacts/cann_sdk_cube_public_20261010/impl/basic_api/dav_c220/kernel_operator_mm_impl.h:344-345, 368-372`）只是**硬件能力**，不证明 FIA 使用；合同 not_claimed `native_dtype_contract.json:64-65` |
| 2 | FIA 内部 softmax / online max/sum 精度 | 无接口暴露；审计报告按通行实现标注 FP32 但自注无直接代码证据 `docs/attention_precision_report_20261015.md:175` |
| 3 | FIA 输出回 BF16 的舍入模式 | 开源 A2 sparse attention 用 CAST_RINT（`native_dtype_contract.json:41-44, 61`），dense FIA 是否相同无证据（同 `:65`） |
| 4 | FIA LSE 的计算精度 | 仅见对外 buffer 按 query dtype 分配（`attention_v1.py:833, 1002`） |
| 5 | `_npu_reshape_and_cache` 内部写入语义 | 闭源；Python 侧无转换（`device_op.py:49-52`） |

另注：模型 checkpoint config 不在本机，「主干 BF16」由 auto 解析链 + w8a8 输出 dtype + 合同边界共同支撑（`native_dtype_contract.json:66`）。

## 五、BF16 分支未变更面（防误读）

`git diff 7047ce36..HEAD` 下，以下与 dtype 相关的部分**完全未动**：Q 旋转 kernel 与绑定（`rotate_vector_init.cpp`、`rotation_bindings.cpp`）、量化 store（`rotate_clip_store_striped.cpp`、`store_int2.cpp`、`striped_store_pack.h`）、两个 unpack 头、`merge_lse.cpp`、全部旧 FP32 `attention_cv_*` kernel、`oscar_ascend/rotations.py`、`oscar_ascend/ops/reference.py`、`configs/acceptance.json`（旧 FP32 合同逐字节保留，仅作差异诊断）。host 侧改动仅 `impl.py`、`contracts.py`、`cv_dispatch.py`、`meta.py`、`runtime.py` 与绑定/CMake 的能力注册；生成器 `--check` 确认 `FP32_sources_modified: false`（`tools/generate_bf16_decode.py`、`tools/generate_bf16_clusters.py`）。

精度门：旧合同 `configs/acceptance.json`（fused_attention atol/rtol 0.005、pack/unpack 逐位、store_dequant 0.002）对 main 继续冻结；BF16 分支独立合同 `configs/acceptance_bf16.json`（`oscar_bf16_attention_v2`，atol/rtol 0.005 对独立 BF16 oracle，测前冻结，容差不事后放宽）。BF16 分支 NPU 执行/图/服务/模型质量/性能均 `not_run`，本表 dtype 结论为源码与 CPU-debug 层证据。
