# AscendC 算子 ABI、数值与真实缺口

已实现七个独立AscendC符号，真实CANN9.1/Ascend910B4编译、链接和官方CPU-debug通过。完整CV与窗口/current路径见 `cv_implementation.md`，旋转融合见 `rotation_pipeline.md`，runtime见 `runtime_implementation.md`。本文件保留基础quant/merge ABI与硬件路线分析；真实NPU数值、图和性能均未通过本轮验收。

## 档案与 PR 证据

算子开发前亲读 `issue1_full_record.md` 开头症状表，G1–G13/#3/#21、G25–G34/#4–#22、#53–#73、#79–#83/#85，以及 #87–#93/#98–#116/#118–#122 的报错/根因；并分两块读完启动文档 §0.7 和 D.4 的全部原始 JSON 打点。没有读取或复用任何旧 OSCAR 失败实现；`references/` 仅只读。

- G2–G13/#87–#90：使用一个普通 C++ launch 声明头，kernel 参数仅 primitive/GM 指针，不跨 host stub 暴露 AscendC 类型、不同步复制 tiling POD。采取 direct launch，所以不产生未注册的 OPP tiling 函数。
- #3/#21/#81–#83/#91：目标固定 `ascend910b4`；无 `float↔unsigned` 数值强转；用向量 `Exp/Log`；归约采用 native 四参 `ReduceMax(dst,src,tmp,count)`，min 使用 `ReduceMax(-x)`；所有输出缓冲按最大 D 分配。
- G26/G27/#13–#20：精度 gate 不变；K/V 元数据顺序固定；源向量保持 FP32，只对 scale/zero 先 FP16 舍入。旧档案 #111 提到整向量 cast FP16，不能用于本 PR 的逐位兼容实现。
- G30–G34/#12：merge 每个输出/LSE行均有写入，全空行写 `0/-inf`，空 split 的 poisoned output 不参与乘法。
- #53–#69：不依赖 FIA 的可选 LSE 返回形状，不把空 LSE reshape 成有效结果；没有构造未知的 FIA 调用形态。
- #85/#98–#122：构建/安装/加载是分离判据，manifest 不是 `.so`，`.so` 存在不是 NPU执行成功。源码采用单一 direct-launch 接口，不加 OPP 备用路线。

语义源：`references/oscar-vllm-pr46774/vllm/v1/attention/ops/triton_oscar_store.py` 与 `triton_oscar_decode.py`。每向量一组，`scale_fp16 = fp16(max((max-min)/3,1e-8))`、`zero_fp16=fp16(min)`；编码是 `clamp(int((x-fp32(zero_fp16))/fp32(scale_fp16)+0.5),0,3)`。INT2 为四个符号一字节、低位先行。K/V 分别拥有独立 scale/zero。旋转空间是 `Q@R_k`、`K@R_k`、`V@R_v`，输出最后乘 `R_v.T`。

**PR 零 scale 域问题**：`1e-8` 舍入到 FP16 是 0，常量/极窄向量会触发除零。CPU oracle 和本候选算子都显式拒绝这个未定义转换域，禁止暗中提高 epsilon。算子 status=3，缓存行不发布。这项必须在目标 PR CUDA 与 NPU 的具体转换行为上完成定案，尚不声称覆盖全部输入域。

## 单一工程路线

采用 `ascendc_library(oscar_ascend_kernels SHARED ...)`，host PyTorch 扩展通过同流直接 launch。先例为 `references/vllm-ascend/CMakeLists.txt:41–81`、`csrc/kernels/get_masked_input_and_mask_kernel.cpp:351–377`、`csrc/torch_binding.cpp:302–305`。这是启动文档允许的独立 AscendC 库形态，不是一次 OPP 加载失败后的备用分支。

构建参数：

```bash
python -m tools.build_ops --soc ascend910b4
```

产物：`liboscar_ascend_kernels_<source-signature>.so`、`_oscar_ascend_ops*.so`、`build_manifest.json`。manifest 记录 ABI=1、接口=direct_launch、绝对产物路径和七个实际符号；所有 NPU数值/图捕获/图重放/性能标志初始 false。CMake configure 会写 manifest，只有后续真实文件检查、模块加载、NPU同步与数值断言才能升级对应检查项。禁止把 manifest 当构建通过证据。

loader现在要求构建工具写入 `build=passed`，校验runtime manifest与同目录 `oscar_build_signature.json`一致，重算configuration JSON的SHA256，再分别重算extension和kernel库的SHA256。仅保留mtime、缺构建签名记录、或只执行configure均不能通过加载。签名代表构建记录完整性，不能替代实际NPU或图验证，也不自动证明当前runtime版本与构建时相同。

原生 header/API 对照：

| 接口 | 原生先例 |
|---|---|
| ReduceMax 四参 | `csrc/moe/moe_gating_top_k/op_kernel/moe_gating_top_k_generalized.h:185` |
| V_S / S_V 事件 | 同文件 `:186–204` |
| S_MTE3 / MTE3_S | `csrc/moe/add_rms_norm_bias/op_kernel/rms_norm_base.h:257–264` |
| DataCopyPad | `csrc/kernels/bgmv_expand.cpp:173`，A2 非对齐 GM/UB copy |
| UB→GM准确非对齐copy | `csrc/moe/scatter_nd_update_v2/op_kernel/scatter_nd_update_no_sort.h:121` |
| NPU stream/guard | `csrc/torch_binding.cpp:302–305` |
| FP32累计真实Cube | `csrc/batch_matmul_transpose/op_kernel/batch_matmul_transpose_kernel.cpp:44–49` |
| A2 CV flag协议 | `csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h:78–172` |

## Store ABI 与 D.4 对照

**二次审查修正**：status/LSE的相邻标量不能由多个core用 `GlobalTensor.SetValue` 写入。该API先改本core DCache，后续按64B cacheline回写，会随机覆盖同cacheline别的core结果；scalar GM读取还可能缓存旧metadata。当前两kernel已全部移除scalar GM读写：slot和LSE经MTE2读到UB，status/LSE经准确字节长度的DataCopyPad写回，配对MTE/S/V事件；UB内scalar pack不受该GM缓存问题影响。使用changed-input replay和poison sentinel验证此修正仍待NPU。[官方GlobalTensor缓存说明](https://www.hiascend.com/doc_center/source/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0007.html)

`torch.ops.oscar_ascend_ops.store_int2_out(key_rot, value_rot, slot_mapping, raw, status, physical_block_tokens, physical_num_blocks, raw_ssm_offset, physical_page_stride) -> ()`

- `key_rot/value_rot`：连续 FP32 `[N,H,D]`，已旋转/裁剪，D∈{64,128,256}。
- `slot_mapping`：连续 int32/int64 `[N]`，slot<0 跳过，越界 status=1；调用方必须保证同一次并行 store 的非负 slot 唯一。
- `raw`：一维 uint8，原生 GDN SoA raw allocation 的 FULL SSM 区。
- `status`：连续 int32 `[N,H]`，每行写入；0正常，1地址超界，2非有限归约量，3量化元数据非法。探针必须 `synchronize` 后检查；热路径不做 `.item()`。

每head字节数 `P=2*(D/4+4)`，D256时P136。偏移精确是 `{K_codes:0,K_scale:64,K_zero:66,V_codes:68,V_scale:132,V_zero:134}`。若 `C` 是一页conv字节、`M` 是一页SSM字节、`nb` 是physical块数，则 `raw_ssm_offset=nb*C`、`physical_page_stride=M`。对 `slot=s`：

`offset=nb*C + floor(s/B)*M + (s%B)*H*P + head*P`。

原生 virtual128 子页映射 `v=b*(B/128)+i`，因此 `s=v*128+token_in_virtual` 自然还原 physical 页。SSM剩余空间与conv区不写。禁止追加完整 BF16 history allocation。

**D.4 四问**：相位是 phase1_stores；失败量级209–216ms/16K，原因是重复搬运和小算子堆积；当前 component 在一个 launch 内完成FP32归约/量化、UB位打包与一次准确136B scatter，不产生中间量化HBM张量；复杂度O(NHD)，UB固定<8KiB/core，目标显著低于215ms/16K但无实测。源代码尚用每向量16/32/64字节的有界scalar-UB pack，必须用profiler证实或替换为native vector gather/shift；它不能因写成C++就自动获得“高性能”结论。完整旋转/percentile clip已在rotate_clip_store.cpp融合；此standalone quant component仍可独立验证。

## Merge ABI 与 D.4 对照

`merge_lse_out(partial_out, partial_lse, output, lse, status) -> ()`

输入 FP32 `[R,S,D]`、FP32 `[R,S]`；输出 FP32 `[R,D]`、FP32 `[R]`、int32 `[R]`。S∈[1,128]，D同store。输入为已经归一化的各分段输出与自然对数LSE，所有分段必须在同一个V旋转空间。

`m=max(lse_s)`；`w_s=exp(lse_s-m)`；`out=sum(w_s*out_s)/sum(w_s)`；`lse=m+log(sum(w_s))`。空split用 `lse=-inf`；全空行返回0与-inf；NaN/+inf LSE为错误，不能伪装空split。非空partial_out必须有限；该条件由Stage1数值gate保证，Stage1由attention_cv.cpp提供，目标NPU验证仍独立进行。

**D.4 四问**：对应materialize/merge；旧materialize约0.5ms，但host prepare约725ms；本实现只读有界split输出，一个launch无历史恢复、无host请求循环；复杂度O(RSD)、每core UB<4KiB，workspace是调用方预分配输出，S不随历史L无界增长；目标低于decode FIA的0.6–1.1ms参考，尚无NPU实测。每row循环S个tile依然可能带来延迟，需要实测决定是否改分层向量归约。

## C04 fused INT2 Cube/Vector：实现与硬件边界

调度单元为(request,kv_head,query_tile,source_kind,history_split)，Mq64、KV tile32。一次压缩tile读取供GQA×MTP各query共用，QK、PV与历史输出逆旋转由真实Cube MatmulImpl执行，INT2 SIMD解包与FP32在线softmax由Vector执行，禁用HF32近似。源码在attention_cv.cpp，设备任务表在attention_tasks.cpp，完整ABI见cv_implementation.md。

A2的Vector→Cube使用每Cube固定 `(256*D+4096)*4` B GM通信（D256为278528B），显式AIV UB预算161024B。容量与历史L无关，没有[L,D]完整历史buffer；总通信流量仍线性，不能把有界tile staging说成零HBM。此差异如实保留。

D.4对应dequant+FIA：旧6500ms全历史dequant被流式解包/矩阵乘替换，不做历史逆旋转；输出逆旋转只对query partial做。目标是接近原生18.7ms相位，但当前只有真实CANN编译与官方CPU-debug证据，没有NPU时延证据。

CPU-debug覆盖GQA6/Q1/2/3/4/17、Hkv2、mixed batch、无序物理页、空历史/分段、非法tag/metadata/NaN，以及MTP后续draft从slot/BT纠正seq_lens。自适应S只依赖捕获shape，在同一arena内分块；S2/S20真实对拍通过，未宣称提速。

H18字面亚线性与任意输入的精确full attention矛盾：不读取某个V就不能对其任意变化产生正确输出。因此本工程不伪称o(L)，仍需需求口径澄清，最终性能必须配对实测。

## 尚须完成的目标验证

每一个候选kernel：target干净编译 → dlopen/schema注册 →同流最小调用+设备同步 →源/目标位级或冻结容差对拍 →poison输出/边界/物理页隔离 →图捕获返回 →输入长度/slot改变后的真实重放 →profiler及D.4预算。以上是独立证据，不互相替代。

Store probe需覆盖FP32 bin边界、同向量整数编码4^4组合、负slot、i32/i64、virtual128边界、physical B边界、页末尾、scale与zero位模式、非法元数据不发布。Merge probe需覆盖S1/3/128、全空/部分空、被mask的NaN输出、极差LSE、不同batch/head展平、所有输出行写入。测试阈值来自 `configs/acceptance.json`；不得事后调宽。
