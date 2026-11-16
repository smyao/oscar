# Decode 相邻 token 的 KV 复用与写入优化

审计基底：`oscar-new-7` / `f3d618d`。档案 #126/#145/#150/#154/#155/#164/#172、启动文档 D.4；本轮无新增真机错误，不向真机档案追加主机推论。

## 当前实际行为

1. `integration/impl.py` 在 attention 完成后调用 `store_current`，仅写当前输入的有效 slot。`official_math.store_rotated` 复用本轮已经计算的旋转 K/V，传入 writer 的两组输入为同一个 `kr, vr, kr, vr`。稳定历史不会在每个 decode step 重新量化。
2. 每个有效当前 KV 行的 K/V 各量化一次，并保存 INT2 副本；需要的 sink/recent 槽同时保存 clipping 前的 BF16 原值。当前输入和精确窗口的 attention 使用 BF16；不是把新 token 量化后立即反量化再作为当前输入。
3. 历史 source0 在每次 attention 中重新分 KV256 tile 解包，直接供融合 attention 消费，不保留整段展开历史。q4 的全部 query 行及同 KV head 的 GQA 行共用该次 K/V 解包；q1 也在 GQA 行之间共用。已经存在的这项复用不能再计一次收益。
4. MTP 的一个 forward 不等于一个最终输出 token。q4 可以校验多个位置，后续 draft 仍依赖前面的结果。被拒绝的位置可以由新值重写；不能依据相同 position、slot 或 window tag 跳过量化。target 层与 MTP 层的 KV 不同，也不能互用。

注意：source0 的候选范围包含随后被精确窗口 mask 掉的 recent 位置，当前 reader 在 mask 前读取并校验 packed metadata。因此不能局部改为“recent 离窗时才量化”：这还涉及错误域、packed 有效性、MTP 覆写及卸载格式，而不只是延迟一次调用。

## 为什么不直接跨 step 保留完整反量化结果

同层的稳定历史确实可以在数学上复用，但现有 workspace 是跨层共享的临时区。固定 graph 地址不代表其中内容跨 step 有效。native 页回收、共享前缀分叉、MTP 同位置重写及 MemCache GET 直接写回都会改变内容；现有请求 generation/version 不是每层每页的内容版本。

按 D256、每卡 Hkv=1、16 个 FULL 层加 1 个 MTP 层计算，BF16 K/V 每 token 每层共 `2×256×2 = 1024 B`。以下是额外镜像容量，尚未计原 INT2、GDN、tag 与管理状态：

| 历史长度 | BS16，17层/卡 | BS28，17层/卡 |
|---|---:|---:|
| 32,768 | 8.5 GiB | 14.875 GiB |
| 65,536 | 17 GiB | 29.75 GiB |

只固定保留一个 KV256 tile，BS28/17层仍需119 MiB，但只覆盖32K/64K历史的约0.78%/0.39%。这是可省解包的覆盖率，不能换算成吞吐收益。普通小 LRU 还会被每轮正向全历史扫描冲掉。完整方案需要独立缓存、设备可见页代际和覆盖所有写入/恢复路径的失效协议。本轮不启用没有容量与净收益依据的跨 step BF16 历史缓存。

## 本轮实现：复用本次量化已加载的 BF16 原值

原 writer 每8个 KV 行批量读入 K、V 进行量化，随后 `WriteRaw` 又为每个需要发布的精确窗口行读取原 K/V 各一次。官方路径这两组源地址完全相同，且 binding 已禁止任何输入与 packed/status/window/tag 输出重叠。

派生 writer 仅在 `key == originalKey && value == originalValue` 时：

- 将原有 K 批量 DMA 的目的地设为扩大的 `rawBuf`；V 仍放入 `inputBuf`，没有额外 UB copy。
- 量化仍通过原有 BF16→FP32 Cast 和完整 quantizer；clipping 只改 `floatBuf`，保留的 BF16 位不变。
- sink/recent 发布直接读取这两块 UB 中对应的行，省掉重复 GM→UB 读取。
- 两个原输入中任意一个地址不同，继续使用原有 original-K/V 读取路径。旧 aligned 合同仍可保留原空间的精确窗口。

这是**同一次写入内部的输入复用**，没有新增跨 token、层、step 或请求的持久数据状态；历史解包次数没有因此减少。

资源和工作量：每个实际发布的 D256 KV 行省2次 DMA、1024 B逻辑 GM读取；INT2编码、packed写入、raw写入与tag写入次数均保持。UB由23424 B增至27008 B，即每核增加3584 B；HBM和graph持久分配不增加。减少的读取可能原本命中L2，不能把这些字节当实测HBM节省。增加的UB活跃容量及分支成本也需要在设备上验证；不据此宣称整轮吞吐提升。

## 数值、所有权和接线

- `active_store_transform` 保留原样；新增 `store_raw_reuse_transform` 钉住其完整SHA，通过 `official_active_store_transform` 组合，逆变换必须还原完整母源。
- 所有 quantizer、Cast、错误优先级（padding→slot→position→quantizer）、sink/recent owner、公开ABI与clip保持。
- 原有 `MTE2_V`、`V_MTE3`、`MTE3_MTE2` fence全部保留。最后一个约束raw发布完成后下一tile才能覆盖UB；量化阶段的`V_MTE2`同样保留。
- `csrc/CMakeLists.txt` 原有writer目标编译新派生源，无额外算子或推理参数。生成清单把两个新transform带入构建身份，拒绝复用旧产物；candidate直启、observe和node93启动所使用的同一writer ABI均获得此实现。
- q4/q1/C4/C16 reader、PT、FP32 metadata、精确窗口布局、PC/MemCache、GDN/MTP与timer没有修改。

## 验证边界

主机测试提取前后完整 `AlignedStore` 类，比较packed、窗口原始位、tag、status及量化调用序列，覆盖同址/部分同址/不同原输入、padding、错误、尾行、多头跨tile、多核及同地址改输入；并核对只消除了预期的原输入重读。量化使用可追踪stub，这不等于CANN数学或异步DMA执行验收。

真实NPU功能probe保留原有独立原输入用例，并新增与生产一致的同址输入、固定地址改变KV值及padding的eager/graph逐位门。报告必须包含该门，不能以旧的四个独立clone用例证明新分支。此处只交付probe及主机接线验证，没有运行NPU。

最终323项主机检查通过：writer/功能probe/lifecycle160项，生成/构建身份/正式candidate路由/原有runtime163项。完整结果记录在 `reports/decode_kv_reuse_20261116/validation.json`。初次运行发现8项旧v3测试继承当前v4 candidate配置的既有失败，在未改基底快照复现；只修正测试内的历史政策选择，生产FP32 metadata/v4合同不变，初始失败及最终回执均保留。

本机无可用CANN工具链，CANN编译、官方CPU-debug、NPU、graph设备回放、模型质量和性能均未验收；保留所有真实边界。
