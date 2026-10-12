# CV score 有限性检查：high16 充分条件研究

## 状态与范围

**research：尚未实现、未编译、未做 CPU-debug/CAModel/NPU 验证。** 本文记录数学证明和只读源码/API 审查，不是数值或性能验收报告。没有修改生产 AscendC、SDK、GDN、服务候选开关或冻结精度门。

**后续决策：单独优化方向已淘汰，不实现。** C4 Q336插桩已完成数值与SDK task计时；两个actor的最终写回日志截断，整trace门仍失败，但所有`finite_reduce`叶子区间均完整、次数已逐actor对真实task/profile核对。core3两AIV的history分别为140861/133275 cycles，window9545/9362，总task为2660850 ticks；即使不计新增guard成本，也没有足够证据支持1.5倍完整核收益。详情见`reports/c4_phase_profile_q336_20261009/finite_leaf_evidence.json`及`partial_diagnostic_summary.json`。不将有限子范围证据扩写为完整相位验收。

档案依据：#126 的 UB 复用所有权；#140 的客户端时间不可冒充内核归因；#145 的 MTE3→MTE2 依赖；#154 的 CPU 顺序模拟不能排除真 NPU 竞争。已重读启动文档完整 D.4 原始日志。D.4 的 6499.8–6655.1 ms 历史恢复、18.5–18.9 ms FIA、约 209 ms store 属于历史失败对照，不能用作当前相位时间。

本研究只尝试省掉**已经能够证明结果为有限值的检查**。score、P、alpha、softmax、INT2 读写、FP32 计算和错误契约保持原样；不降低精度，不用逐元素 finite 冒充“求和不会溢出”。

## 1. 当前代码确实存在的成本

- `csrc/kernels/attention_cv_window_range_cluster4.cpp:694–807`：每个 Softmax block 最多 32 行；对有可见 KV 的行，在 mask 前执行 `ReduceSum(tmp, scores[row*256], tmp[16], 256)`，随后 `V_S`、Scalar `Finite`，失败置 `error=2`。
- `csrc/kernels/attention_cv_decode_bundle.cpp:441–554`：同一检查，**每个 block 最多 16 行**，不能按 32 行估算 decode 收益。
- 整行没有可见 KV 时，原代码写入负无穷并跳过 finite 检查；有可见 KV 时，检查域是 scaled、mask 前的全部 256 项，不仅是可见列。
- `csrc/kernels/oscar_common.h:18–20`：`Finite(x)` 为 `x==x && x<=FLT_MAX && x>=-FLT_MAX`。

### 原 ReduceSum 的内部同步

只读安装的 CANN 9.1 头文件得到：

1. `tools/tikcpp/tikcfw/impl/kernel_operator_vec_reduce_intf_impl.h:1162–1181`：四参数 count overload 在 `__NPU_ARCH__ == 2201` 下调用 count 专用实现，未使用传入的 `sharedTmpBuffer`。
2. `tools/tikcpp/tikcfw/impl/dav_c220/kernel_operator_vec_reduce_impl.h:153–179`：`set_mask_count` → `vcadd(..., mode=1)` → 内部 `V_S` Set/Wait → `get_acc_val()` → Scalar 写 dst → `S_V` 与 `S_MTE3` Set/Wait → 恢复 normal mask。

因此，把原 32 次 ReduceSum 留着、最后才统一读取结果，不能消除内部逐行同步。`WholeReduceSum/ReduceRepeat` 的 mode=0 和高阶 AR 行归约的 Add/PairReduceSum/BlockReduceSum 路径又没有原归约树等价证明，已否决直接替换。

本方案不生成替代行和；只在原行和必然有限时省掉这个不影响输出的检查，否则执行原检查。

## 2. high16 判据与位级证明

对 scaled、mask 前每个 FP32 score，令其原始 32-bit 位模式为 `u`。取高 16 位并清除符号：

```text
h = (u >> 16) & 0x7FFF
safe_element = h < 0x7B00
```

`0x7B00 = 31488 = 246 << 7`。令低 16 位为 `l∈[0,65535]`，则绝对值位模式为 `h*65536+l`。所以：

```text
h < 0x7B00
⇒ (u & 0x7FFFFFFF) ≤ 0x7AFFFFFF
⇒ |score| < 2^119，且 score 有限。
```

这里使用 IEEE binary32 正有限数的位模式单调性：`0x7B000000` 恰为 `2^119`。所有 ±Inf/NaN 的清符号高 word 均 ≥`0x7F80`，不会被误认安全；NaN payload 位于低 word 也不例外。±0 和 subnormal 可以通过。恰为 ±`2^119` 的值保守进入原检查，不影响正确性。

高 word 经 And 后一定是 `[0,32767]` 的非负整数，按 int16→FP32 转换完全精确。后续 `Compares(...,31488.0F,LT,...)` 的输入始终是有限整数，**不依赖 Float-NaN 比较是否 ordered**。

按 uint16 lane 选择奇数索引对应 FP32 高 word，需要在后续 tiny probe 用明确的 32-bit payload 验证目标数据布局。不能用普通随机有限数掩盖高低 word 选反问题。

## 3. 任意原 FP32 加法归约次序都不会溢出

设一行最多 256 项，每项绝对值不超过 `B=2^119`。采用保守舍入误差界 `E=2^104`：在下述界内，FP32 一次加法舍入误差不超过一个该量级 ULP；更小指数及 subnormal 的误差也被包含。

对含 `k` 个叶节点的任意二叉加法子树，归纳证明：

```text
|rounded_sum(k)| ≤ k*B + (k-1)*E
```

- `k=1`：输入界直接成立。
- 两子树分别有 `k1/k2` 项时，舍入前绝对值最多为 `k*B+(k-2)*E`。
- 因为 `k≤256`，该值最多为 `2^127+254*2^104`，远离 FP32 溢出边界。此次舍入保持有限，加入至多 `E` 后即得归纳式。

整行及任何中间节点均满足：

```text
|rounded_sum| ≤ 2^127 + 255*2^104
             < 2^128 - 2^104 = FLT_MAX。
```

因此，无须改变或知晓原 ReduceSum 的加法树，只要该实现执行其规定的 FP32 求和，安全输入就不会使原 `Finite(sum)` 失败。该证明留有接近两倍的幅度余量，未假定完全无舍入误差；有限输入的加法也不会产生 NaN。将阈值改成 `2^120` 则不成立：256 个正阈值的精确和已为 `2^128`。

检查必须发生在原 Muls 之后。不能从未缩放的 QK、Q/K 有限性或原始输入幅度直接替代本判据。

## 4. 已核实 API 与原生先例

### GatherMask：使用整数位搬运

安装头路径均以 `/usr/local/Ascend/cann-9.1.0/` 为根：

- `tools/tikcpp/tikcfw/impl/dav_c220/kernel_operator_vec_gather_mask_impl.h:107–123`：固定 pattern API 支持 `half/bfloat16_t/uint16_t/int16_t/float/uint32_t/int32_t`；16-bit 分支统一 reinterpret 为 uint16 指针，调用 `vreducev2`。没有浮点 Cast 或算术。
- `tools/tikcpp/tikcfw/impl/kernel_operator_vec_gather_mask_intf_impl.h:81–106`：公开固定 pattern overload 到上述 A2 分派。
- `asc/include/basic_api/kernel_struct_gather.h:64–78`：参数次序是 src0BlockStride、repeatTimes、src0RepeatStride、src1RepeatStride。
- `references/vllm-ascend/csrc/attention/lightning_indexer/op_kernel/arch22/lightning_indexer_vector.h:398–412`：原生 A2 明确注明 pattern 2 筛选奇数索引，并使用 `false, 0` 及连续 repeat stride 8。

原生先例的数据类型是 uint32。16-bit 支持及纯整数分派的证据来自 SDK，不能写成“已有本 high16 用例的原生真机验证”。实际候选应直接用 `scores.ReinterpretCast<uint16_t>()`，不需要先变成 half。

8 行对应 2048 个 FP32，即 4096 个 uint16：

```text
src1Pattern = 2
reduceMode = false
mask = 0
GatherMaskParams = {1, 32, 8, 0}
输入读取 = 32 × 256 B = 8192 B
期望输出 = 2048 个 uint16 = 4096 B
```

尾块 `r∈[1,8]` 行时 `repeatTimes=4r`，期望保留 `256r` 个 word；不读取补齐到 8 行的 padding。

公开接口还会读 `AscendCUtils::GetRsvdCnt()`，其实现为 `get_rsvd_cnt()`。头文件没有显式 V_S Set/Wait，但这**不证明该 Scalar 寄存器读取没有等待成本**。编译及计时门必须包含这部分，不假设它免费。

只读安装头 SHA256：

| 头文件 | SHA256 |
|---|---|
| dav_c220/kernel_operator_vec_gather_mask_impl.h | `cf306aead716ada7cef8f448226e5655de917809f9adcc803c6a2ba3afddb0f7` |
| kernel_operator_vec_gather_mask_intf_impl.h | `ab2168cdc9de3bda454107717e44f67395a7299795fe02793fa72f0fb7b2cebf` |

### And、精确 Cast、packed Compares

`csrc/kernels/attention_striped_unpack_padded.h:99–109` 已使用 uint16 And 的块广播；`123–166` 已使用 int16 raw bits 精确 Cast 为 float、`Compares<float,uint8_t>` 和 packed-mask Scalar 读取。因此高 word 清符号后可以复用这些接口契约。

比较 count 为 `256r`，为 64 的整数倍；packed 输出为 `32r` 字节，为 32 B 的整数倍，读 all-ones 时没有未覆盖尾位。

## 5. UB 分区及存活性

以下是最大 8 行切片的候选分区，**尚未分配或实现**。全部起点 32 B 对齐，放在现有 C4 D256 dequantBuf 的 16384 B 中：

| 字节范围（左闭右开） | 用途 | 字节 |
|---|---|---:|
| `[0,4096)` | Gather 的 high-word，原地清符号 | 4096 |
| `[4096,12288)` | int16→FP32 精确转换结果 | 8192 |
| `[12288,12544)` | packed comparison mask | 256 |
| `[12544,12576)` | 16 个 uint16 的 `0x7FFF` 广播常量 | 32 |
| **合计** | | **12576** |

And 使用现有 block-broadcast 方式，不另分配 4096 B 常量数组。每次 Softmax 的四个 8-row 切片复用同一分区。decode 的 dequantBuf 更大，但其 Softmax block 只有 16 行，需要单独计时，不套用 C4 的四片收益。

必要所有权条件：

1. 此处之前的 K/V GM 发布必须已完成读取 dequant UB，不能删除现有 MTE3→V 依赖。
2. Gather→And→Cast→Compares 的生产/消费顺序必须由相应 Vector 依赖保护；source score 全程只读。
3. Scalar 读取比较 mask 完成后，下一片或 SoftmaxV2 才能复用对应 UB。不能因头文件未显示某个等待就假定所有权已转移。
4. guard 必须在 SoftmaxV2 使用 dequantBuf 作工作区之前结束。
5. 不能覆盖 decoder 尚待 PrefetchValues 消费的 packed/packedNext、持久 unpack indices，或 acc/max/sum 状态。
6. #145 的 scoreBuf MTE3→MTE2 和 #154 的其他 V→MTE2 依赖均保持；不能把 CPU 顺序执行当依赖正确的证明。

## 6. 原错误与输出契约如何保持

建议首先研究最易审计的控制流：**先检查完整 active block；所有切片都安全才跳过该 block 的逐行 ReduceSum，否则整个 block 执行原 loop。** guard 自身不更新 `error`，也不改任何 score。

- 全安全：对于原本会执行 finite 检查的每一行，已证明原 ReduceSum 有限，原代码不会置 `error=2`；省略这些检查不改变结果。
- 任一切片不安全：完整执行原 loop，原 sum 溢出、NaN、错误覆盖顺序和 mask 前检查域均保留。
- 全不可见行含坏值：最多导致 guard 不通过；原 loop 仍跳过该行，不引入错误。
- padding 行：不得纳入 guard，也不改变原 padding 的负无穷/零填充。
- 已有 query/KV/page 错误：guard 不清除错误；原 ReduceSum 若会用 `error=2` 覆盖已有错误，必须由不安全路径继续作出同样更新。
- C4 `memberErrors`、kvError 和两个 AIV lane 的状态独立性不变；不能把一个 member 的 guard 结果写成另一个 member 的错误。
- ReduceSum 输出只供 finite 判断，不参与 P/alpha/LSE/输出计算。跳过的是一个已证明为真的谓词，不是用较低精度近似它的和。

不允许用 guard 失败直接判 `error=2`：阈值外可能有很大的有限值相互抵消，而原归约仍有限。

## 7. 后续最小验证门（全部未运行）

先等 C4 phase31 完整、扰动合格的 `finite_reduce` 数据决定是否继续。值得实施时才做以下必要验证：

1. **独立 Gather/guard tiny probe**：固定 uint32 payload 经 uint16 view，核对全部 high-word、And 后整数、Cast 精确值、packed mask、keptCount 和 source 字节不变。覆盖 1/6/8 行和连续切片复用。
2. **强制高低 word 区分**：至少包括 `0x7AFFFFFF`、`0x7B000000`、`0x7B000001`、对应负数、±0、subnormal、±FLT_MAX、±Inf、qNaN/sNaN，尤其低 word 非零的 `0x7F800001/0xFF800001`。防止拿低 word 当高 word而错误接受 NaN。
3. **原有限性检查对拍**：安全域的极值、正负混合、阈值邻居；阈值外但各元素有限的求和溢出，以及大值抵消。对不安全输入比较原检查实际结果，不人为假定某个归约次序。
4. **完整错误域**：全不可见行坏值、部分不可见列坏值、padding poison、已有 metadata/query 错误及 C4 member/lane 混合错误。原/新 status 必须逐位一致；有效输出、P/partial、LSE 按原冻结门及适用的逐位门验证。
5. **实际异步/图门**：真实 NPU 上同地址改变输入，使安全/不安全分支交替，检查 mask/临时 UB 无旧数据污染；CPU-debug/CANN 通过不得冒充此证据。
6. **性能门最后执行**：同形输入、同精度、同图/运行方式比较正常原核与正常候选核；guard microprobe 时间只能解释成本，不能替代完整 kernel 时间。

不新增真实用户数据集，不要求再次运行 AISBench；后续需要的目标验证仍是短时合成 probe。

## 8. ≥1.5 倍收益门与淘汰条件

当前无相位份额或候选耗时。若要成为实质优化，必须用完整正常算子的同形 A/B 证明至少 `old_time/new_time ≥ 1.5`；不能拿“32 次归约变成四片检查”直接宣称八倍。

方案新增 Gather、整数 And、Cast、Compares、packed-mask 读取、寄存器计数读取及必要依赖；这些成本都应进入完整候选计时。插桩结果须先通过既定扰动门，不能把不完整 actor trace 当作占比。

若旧检查在可删除的真实关键路径上占比为 `f`，其局部加速比为 `s`，理想整体上界为：

```text
speedup ≤ 1 / ((1-f) + f/s)
```

要达到 1.5 倍，至少需要 `f*(1-1/s) ≥ 1/3`；即使检查免费，`f<1/3` 也达不到。这里的 `f` 必须是完整执行可消除成本，不能从多个并行 actor 的 inclusive 时间相加得到，更不能把 kernel 占比当作服务占比。

若 C4 实际 `finite_reduce` 数据已经否定足够收益，或完整 A/B 不过 ≥1.5 门，**淘汰本方向，不因接口可行而继续小幅演进或推送**。本研究本身也不能证明整轮相对原生的 4–5 倍目标；服务目标仍需要独立、同口径证据。

## D.4 四问结论

1. 对应相位：fused FIA 内部、scaled score 的 finite guard，位于 mask 和 SoftmaxV2 之前。
2. 历史失败：D.4 的全历史恢复约 6.5 秒、host enqueue 背压，以及 #145/#154 的缓冲复用竞争；不把其耗时套给本方案。
3. 结构约束：仅固定最多 8-row/2048-score 临时位检查，不物化历史、不改 score/P、不重写归约树；阈值外原检查完整保留。
4. 收益边界：当前未测。先用真实 C4 相位裁决是否值得做，再过原精度/错误/异步门和完整正常核 ≥1.5 倍门。
