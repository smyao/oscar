# 旋转、精确裁剪与双写流水线

档案依据：G27（窗口必须保存原值），#13–22（metadata 舍入与位序错误），#37–49（prefix/MTP materialized KV 的生命周期），#70–71/D.4（写入与全量恢复延迟），#81–83/#91–92（A2 API 与 UB 边界），#111（旋转结果必须保留 FP32）。语义来自指定 PR `oscar_attn.py:235–243` 与 `ops/reference.py` 独立 oracle；未参考失败项目代码。

## 接口与数值

`rotate_out(input, rotation_transposed, output, status, hadamard=False, slots=None)` 只写入预分配 FP32 输出与每行状态。输入 `[N,H,D]` 可以是 BF16、FP16 或 FP32；`D=64/128/256`。矩阵是 FP32 的 `R.T.contiguous()`。Q 使用 `Rk.T`，最终 value 输出使用 `Rv`，分别实现 `Q @ Rk` 与 `out @ Rv.T`。全程不降低乘法输入精度。可选 `slots` 是 NPU int64 `[N]`；负 slot 对应 token 的全部 head 在 UB 清零，输出和状态均为零，避免原生 padding 的 NaN 错触发有效行的数值闸。有效行的非有限值仍返回状态 2；`slots=None` 保持原有完整旋转语义。全 padding tile 直接跳过矩阵变换。

`rotate_clip_store_out(key, value, rk_transposed, rv_transposed, slots, positions, packed, raw_key, raw_value, raw_tags, status, block_tokens, blocks, ssm_offset, page_stride, sink_tokens, recent_capacity, k_clip, v_clip, hadamard=False)` 在一次 kernel launch 内完成新 token 的 K/V 旋转、绝对值百分位裁剪、INT2 量化和压缩页写入，并保存未旋转 BF16 原值到物理页 padding 的精确窗口。

裁剪使用 `Sort32` 与最多两级 `MrgSort`，得到真正的相邻阶统计量，再按 FP32 rank 做线性插值；不是最大值乘比例，也不使用近似二分阈值。零比例按 PR 禁用裁剪。scale 和 zero 先舍入到 FP16，重新转 FP32 后参与除法；4 个 INT2 code 以 LSB-first 存入一个字节。FP16 scale 下溢为零时状态为 3；这保留已有的显式失败契约，不偷偷改 epsilon。

状态 0 表示成功或 padding；1 表示 slot 越界；2 表示非有限数值；3 表示无效 FP16 quantizer；4 表示负逻辑位置。每一 token/head 必须明确写状态。K 和 V 都量化成功后才发布压缩行。探针同步后必须断言全部状态为零，生产边界由运行时检查。

## 生命周期与寻址

`slots/positions` 是 device int64 `[N]`。压缩地址仍使用原生 SSM SoA 区域及当前物理页几何。raw K/V 为 `[physical_pages,S+R,H,D]` BF16 strided views；tags 为 `[physical_pages,S+R]` int64，保存页内 token 位置。`R` 包含 speculative slack。

sink 仅当逻辑位置 `< S` 时写入；recent 仅对逻辑位置 `>= S` 写入 `S + in_page % R`。两个区域构成不重叠的逻辑窗口，sink token 不会在 ring 内保留重复副本。原生 cache manager 的可写 partial 页不共享，当前 batch 每物理页只有一个连续写区间。因此若 `slots[t+R] == slots[t] + R` 且仍是同一页，后面的 token 才是该 raw ring 槽的唯一写者。这个 O(1) 规则防止大 prefill 多核同时写相同 raw 槽，同时保留最后 R 个非 sink 位置。只读 prefix 页没有写入，MTP reject 的处理和 prefix 生命周期由 runtime 管理。

bindings 检查 shape/dtype/device/stride、signed-int64 溢出，以及实际活跃写区间不重叠。raw K/V/tags 可是同一原生 allocation 的 interleaved page views；压缩写区间必须与这些 view 的活跃区间分离。kernel 内不调用 torch、CPU oracle 或设备同步。

## D.4 四问

1. 对应 `phase0_store/phase1_stores`，query/output 旋转属于 attention 边界。
2. D.4 原始日志 16K 写入为 215.1–216.0 ms；32K 续写为 208.6–209.3 ms。全历史 dequant 达 6499.8–6655.1 ms，CPU prepare 达 724.7–726.2 ms，不能复制这些访问结构。
3. 本实现只访问本次新 K/V；旋转、排序、裁剪与量化在 UB 中完成，直接写终态压缩行。Hadamard 使用 `log2(D)` 级向量 butterfly；任意已校准 FP32 矩阵按 16 个输出通道分 tile，每次 matrix tile 被 8 行复用。设备内部行循环不会产生 Python token 循环或串行 torch kernel 链。精确 raw 窗口只写 sink 与每页本批最后 R 行，从不恢复历史。
4. Hadamard 工作量为 `O(N H D log D)`；任意 dense 旋转为 `O(N H D²)`；UB 在两条分支都小于 64 KiB。目标是 decode 低于 D.4 的 0.6–1.1 ms 量级、16K 写入显著低于 215 ms。**这些是目标而非实测结论**；CPU 数学测试、CANN 编译、CANN CPU 仿真、NPU 数值与耗时分别记录，dense throughput 必须由真机打点验收。

`prepare_device_rotations` 仅在初始化时准备矩阵转置，并验证 Hadamard 的实际 Sylvester 符号模式后才允许 fast path。不得仅凭 artifact 的字符串标签启用不同数学操作。图回放使用持久 tensor 地址，不在热路径重新分配这些矩阵。
