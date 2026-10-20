# C16 BF16 转换候选的范围与寻址证明

对应现有 timer：四个代表 AIV 的 `bf16_cast` 为 7.476–7.628 ms，约占
9.60%–9.80%。这不是纯算术时间，也不是所有层的平均；新增候选只压缩发射
相同转换指令的次数，尚无完整 C16 或模型提速结论。

## 改动边界

- 新 helper `csrc/kernels/attention_c16_cast.h`，同签名 `CastRne(dst, src, count)`。
- `tools/c16_cast_transform.py` 在全部原 include 之后引入新 helper，用本文件内
  的 namespace 名称宏将六处调用绑定到它，文件尾解除宏。六处调用原文本和
  原 include 都保留，现有 timer 的匹配点不变。headers 先完成解析，因此
  unpack header 内的 mode1 仍调用原 helper。转换改动保留全部浮点数据、RNE、
  FP32 softmax/accumulator、错误检查、DMA fence 及 Cube/Vector flag。
- 原核另存 `attention_cv_aligned_cluster16_control.cpp` 作校验对照；生成后的生产
  `attention_cv_aligned_cluster16.cpp` 保留原 API/文件名。共享
  `attention_bf16_cast.h` 和 timer 源码不修改。
- q1/q4/C4 不调用新 helper。未尝试删除 BF16→FP32→BF16 往返：
  当前这里还做逐行 finite/overflow 检查；移除转换必须另证 NaN、异常状态及
  既有缓冲所有权一致，不能借此次发射合并顺便改动。
- 不增加任何生产 UB/GM 分配；自然顺序 BF16 目标、FP32 源、tile 及地址全不变。

## 为什么不是 #164 的大 count 转换

已亲读档案 #164 及本机 CANN 9.1.0：

`aarch64-linux/asc/impl/basic_api/dav_c220/kernel_operator_vec_vconv_impl.h`
L452–459 的 BF16/FP32 `CastIntrinsicsImpl` 将显式 repeat/stride 直接交给
`vconv_f322bf16r`；L800 起四参数 count 重载会启用 `set_mask_count()`，
而 L852 起显式 mask/repeat 重载只设置 mask，不改变 mask 模式。

新 helper 首先 `SetMaskNorm()`，每条指令：

- mask=64，表示每次 repeat 最多 64 个 FP32 元素；
- repeat 在 1–255；超出 255 时在相邻区间继续下一条指令；
- `UnaryRepeatParams{1,1,4,8}`，dst/src repeat stride 分别为 128/256 字节；
- 最后不足 64 个元素仅以余数 mask、repeat=1 转换，结束 `ResetMask()`。

同库原生先例：
`references/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/utils/block_epilogue_pertoken_v2.hpp:182`
使用 FP32→BF16 RINT、显式 repeat、相同 `{1,1,4,8}`。

对一次 full-repeat 指令，若起点为元素索引 `o`，repeat 数为 `r`：

- 源读范围：`[4o, 4(o + 64r))` 字节；
- 目标写范围：`[2o, 2(o + 64r))` 字节；
- 两个范围都不超过调用方声明的 `count`，相邻 repeat 无重叠、无空洞。
- 循环中 `o ≤ count`，因此 int32 正 count 的 offset 加法不会越界；
  余数 `<64`。零 count 不发 Cast，仅恢复正常 mask。
- 每次 offset 为 64 的倍数，保持原有 32 字节对齐。全程不缩短生命周期或
  删除 #126/#145 所要求的 MTE3→V/MTE3→MTE2 屏障。

生产 C16 最大调用 `count=32×256=8192`。原 helper 发射 128 次 repeat=1；
候选发射一次 repeat=128。`count=256` 从 4 次变 1 次。
这是源代码发射次数变化，**不是 128 倍/4 倍设备提速**。

## D.4 四问

1. 相位：fused FIA 内部 BF16 操作数发布。
2. 前代失败：全历史恢复耗时 6499.8–6655.1 ms/card，FIA 本身 18.5–18.9 ms；
   短 decode FIA 0.6–1.1 ms，写路径约 209–216 ms。不能靠扩大恢复区换发射减少。
3. 结构：保留固定 tile，只合并独立转换 repeat，不新增历史恢复/中间历史 arena。
4. 上限与证据：每条指令最多 255×64 元素、生产最多 8192；收益仍须用户
   设备比较完整 C16，而非把 helper 发射次数当完整耗时。

## 功能探针

`cast_probe/` 使用真实官方 tikicpulib::Ascend910B4，源代码调用原/新 helper。
66 个病例覆盖零长度、64 元素 mask 边界、8192 最大生产长度、255-repeat 上限
16320 及其两侧、20000 元素的两批指令。输入含正负 tie、signed zero、normal
边界、subnormal、Inf、带 payload 的 NaN 及确定性随机 FP32 位型。

校验旧新全部输出逐位相同、输入不变、整个 inactive 输出保持 sentinel，
并独立校验 normal finite/zero 的 round-to-nearest-even 及环境溢出模式；
subnormal/NaN 的设备特殊处理按原 helper 逐位对照，不能改成其他软件 NaN 约定。
probe 专用 UB 合计 161024 字节，不进入生产源码树或服务。

CPU-debug 无法验证异步流水和真实 VCONV 硬件执行，因此仍需 CANN 目标编译及
完整 C16 原/新逐位与设备净耗时对照；未执行部分不能写作通过。

### 独立 oracle 的溢出模式修正

首次 CPU 功能运行在 `input_bits=0x7f7f8000` 失败：原 helper 与新 helper
均得 `0x7f7f`，最初软件 oracle 得 `0x7f80`。旧新逐位比较并未失败；
问题是 oracle 默认有限输入舍入溢出必为 Inf，没有建模 CANN 的饱和控制。
首次及增强位型日志由主验证流程保留，不能删除失败记录后称初次全通过。

已只读核对 SDK 源码
`aarch64-linux/asc/impl/basic_api/dav_c220/kernel_operator_common_impl.h`
L155–181：`GetSaturationModeBit<FLOAT>()` 返回 48，
`SetSaturationFlagImpl(true)` 清除此位，注释明确 **bit=0 means saturate**；
`GetSaturationFlagImpl()` 返回此位的逻辑非。

探针现在读出原/新转换前后的 `CTRL[48]`，要求三者严格一致。独立 oracle
仍逐个检查所有原先的 normal finite/zero 元素；有限输入舍入溢出时，
若该位为0则返回对应符号的最大有限 BF16 `0x7f7f/0xff7f`，为1则返回
对应 Inf。新增正负 BF16 max、halfway 和上下相邻 FP32 位型；
**没有跳过失败输入，没有放宽比较，没有更改生产 helper。**
