# BS16→28：算子工作量与无效行消除

本轮基底 `c89cae5a500667926e5eb4fdf2c675baa1edf22e`，分支 `oscar-new-7`。用户要求在 PC 命中率相同的条件下，从算子侧解释扩大 batch 未增加吞吐，并优化可证明冗余的工作。上一轮主机 prepare/prefix 合批保持；本轮不改变启动参数、timer、PC、卸载、GDN/MTP 状态推进或量化配方。

## 为什么更大的 batch 不必然带来吞吐

当前 decode 每个历史 tile 仍经过 packed DMA → UB INT2 解包/FP32 反量化 → BF16 固定 GM stage → Cube QK → FP32 softmax → Cube PV → FP32 partial/LSE → merge → 单次 BF16 逆旋转。INT2 降低缓存容量，但不会消除每个请求的 attention 工作；PC 命中省掉的是对应前缀的重算，恢复后的历史仍要参与 decode。

同等历史长度和每请求输出数下，BS28 比 BS16 多 75% 请求工作。现有 q4 在 TP4、每 rank Hq6/Hkv1、D256、20 Cube、连续四行请求布局下的静态计数如下。这里采用实际 selector；N64/N112 都在现有 capture 列表内，没有据此归因成 BS28 未命中图。

| 项目 | BS16 | BS28 |
| --- | ---: | ---: |
| q4 输入行数 | 64 | 112 |
| split 数 | 5 | 4 |
| history leader 数 | 80 | 112 |
| 最忙核 32K history tile 循环数 | 104 | 192 |
| 最忙核 64K history tile 循环数 | 208 | 384 |

原始计算和源码绑定见 [静态工作量](../reports/batch_scaling_20261112/workload.json)。这是逻辑循环和字节计数，不是设备时间或物理 HBM 流量。最忙核负载增幅和逐请求反量化成本足以说明“PC 相同、BS 更大”不能直接推出吞吐增加；没有本次设备计时，不能把某一项认定为现场唯一瓶颈，也不能宣布访存已饱和。split/调度本轮不改。

## 已实施的两个改动

### 负 slot 的 store 行不再量化

原 `AlignedStore::Run` 对 tile 中全部行先做 K/V `ClipQuantize::Process`，随后才读取 slot；负 slot 将错误码置零并丢弃所有结果。新实现先用原有带 fence 的 `Read` 保存最多八个 slot，只跳过负 slot 的 K/V 量化，发布时使用相同快照。

有效行的排序、clip、FP32 metadata、INT2 编码、错误码及其优先级逐字保持。非法正 slot 仍执行原量化后按原次序报错；不把非法地址推断成 padding。整个输入 DMA/Cast、外层 fence、packed/window/tag 发布和 `CopyValidateQuery` 保持。原生 slot 在单次算子执行中不可变，回放每次重新读取，不跨调用缓存。

MTP 后续轮可能保留 N=4B 的输入张量，而仅 B 个 slot 有效。在 Hkv1 的该情形下，BS16 的量化调用从 128→32，BS28 从 224→56；少 75% 调用。无 padding 时没有该收益，并增加最多八个 int64 的局部快照及整数分支。这不是 store 或整轮耗时降低 75%。精确 recent/sink 窗口仍从原 BF16 输入写入。

### qcount=0 的 decode 任务少读十个标量字段

原 `LoadTask` 读取全部 11 个字段；`Process` 遇到 `qcount==0` 只发布零 status，不读其他字段、不参与 attention 交接。新实现先读 qcount，严格等于零时返回；其余任务仍读取全部 11 个字段，包含负 qcount 的非法/padding 任务。

AIV 的完整 128B task DMA 和 MTE2_S fence、AIC cache invalidation、完整 `Process/TaskValid`、错误路径、浮点计算、跨核 flags 和 partial 发布全部保留。该优化省的是 GM/UB 标量 `GetValue`，不声称减少 AIV DMA 字节。

上述 q4 布局中 BS16/28 分别有 720/1008 个零任务。遍历一份任务的标量读取从 10560→3360、14784→4704；双 AIV lane 的总计各乘二。q1 仅在实际 qcount=0 时受益，不能套用 q4 follower 比例。

## 源码与构建保护

旧 `attention_cv_official_decode.cpp` 完整保持已保护的 232b 源码，旧 `store_aligned_int2.cpp` 同样不改。新增两个 `*_active.cpp`，同 public ABI；CMake 各只编译一份实现，不增加运行时 dispatch 或额外算子。

`tools/active_decode_transform.py` / `tools/active_store_transform.py` 以完整母源 SHA 拒绝漂移，并能逆变换恢复整份原文件。`tools/generate_active_kernels.py --check` 在 build/reuse 前校验母源、派生源、工具及 manifest；整棵 csrc 的现有构建指纹拒绝复用旧库。旧冻结测试与哈希不放宽，C4/C16、Cast、softmax、量化 helper、PT、配置与 argv 保持。

两个原启动入口 `bash scripts/start_node93_oscar.sh` 和 `bash scripts/install_serve.sh --variant candidate` 使用原 op ABI，编译后即进入派生实现，无需额外开关。探针 source binding 包含新源及生成链。正常直启是否运行 probe 仍按原入口策略，不加入新的部署前置流程。

## 验证范围与待验证项

主机测试提取实际新旧 C++ 控制流编译运行，覆盖全部活行、混合/全 padding、多头、尾 tile、非法 slot/position、非有限输入、独立 original 输入及同地址改变 mask。store 的量化和 DMA 是记录型替身，用于比较有效调用序列、输入、packed/window/tag/status；不把它冒充真实 CANN 量化精度或 DMA 并发验证。decode 同样使用实际 `LoadTask/Process/TaskValid`，核对 AIC/AIV dispatch、状态、访问集及 fence。

已有 NPU writer 功能门补充固定地址 eager/graph 的四种活行 mask，inactive K/V 放入 NaN/Inf，逐位核对独立 codec、精确窗口和 canary；报告必须带完整新证据才通过。它不采集性能。本机没有 CANN SDK/Lima，因此本轮未运行此真 NPU 门，也没有 CANN 编译、官方 CPU-debug 或设备图通过的证据。完整回执见 [验证记录](../reports/operator_active_rows_20261112/validation.json)。

未启用 packed page 跨 tile 复用、提前 V 解包、跨请求共享或 softmax/归约改写：前两者会改变 DMA/Vector 交接与 UB 生命周期，后两者涉及额外工作区或数值顺序。历史 #145/#154 的所有权错误和 #172 的短输入退化说明，不能以省指令推断它们安全或更快。

本轮相关组合共 1024 项通过、1 项 Linux ELF 测试因 Mac 跳过。初次 60 项 IPC bind 被沙箱拒绝，解除该限制后 metadata 文件全部 72 项通过。新增文件包含 62 项测试，另增加 2 项不完整 probe 凭证拒绝用例。扩大检查还发现旧 aligned 测试的 8 项 v3/v4 fixture 冲突；已在独立 `c89cae5` 快照复现同样 8 failed / 41 passed，保留原始日志，未放宽旧断言或改生产策略。相关组合通过不表示全仓测试零失败。

## D.4 四问

1. 相位：current-token store 和融合 decode CV 的任务装载，未新增全历史恢复阶段。
2. 原始失败依据：已亲读启动档案 D.4 完整日志；旧全历史恢复约 6499.8–6655.1ms、store 约 209–215ms 是历史结构教训，不是当前算子比例。
3. 规避方式：仅移除原本不发布的负 slot 量化和零任务字段读取，保持固定 tile、算术、错误语义及全部交接 fence；无历史长度级缓存、主机回读或输入长度特判。
4. 成本与证据：局部 slot 快照/分支如上，q4 标量与条件 MTP 量化调用减少已由源码/主机控制验证；真实 CANN、图、质量和吞吐分别待验证，不用局部工作量声称 BS28 已获得端到端收益。
