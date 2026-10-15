# Host metadata 与 prepare 复用复审

结论：prepare 复用保留为默认关闭的 research 开关。71 项主机契约检查通过，未连接真机；没有完整设备路径 1.15 倍、模型质量、32K/64K 相对 native 超过 30% 的证据。

## 现状核实

- `metadata.from_common` 已按 native common 对象缓存 CPU cumulative，不能重复申领“移除每层CPU列表重建”的收益。
- `draft_discard` 已在 builder 处理原生显式 discard 和逻辑拒绝边界；本轮未改变它、MTP或GDN。later-draft 的冻结 prepare 仍会在设备检查虚拟块表，未以跳过活行检查换速度。
- `GraphWorkspace` 按 device/geometry 共用；普通 FULL 层重复生成的任务仅依赖 native metadata、Hq/Hkv、sink/recent 和 split 数，不读取各层物理 raw 或旋转内容。每层 Q/K/V 投影、rotation、attention、FIA、merge、KV 写入和 status_guard 均保持。
- 除 `impl.forward`，另一个生产 prepare writer 是 `current_only_dispatch`；mixed 分区递归进入 `impl.forward`。这两条提前返回路径均在入口与 finally 使租约失效，分区自身禁止复用。

## 生命周期安全边界

原生 `model_runner_v1.py` 的真实 forward 经 `set_ascend_forward_context`，该作用域调用 `vllm.forward_context.set_forward_context`，由 `create_forward_context` 新建对象。因此租约同时绑定 **ForwardContext 对象身份、OscarMetadata 对象身份、workspace、stream 与几何**，不以复用的 tensor 地址推断同一步。

仅用 metadata 身份和“首层重复”不能覆盖同一 metadata 换值后从不同 FULL 层开始的非标准重入。复审已加入 ForwardContext 身份，并执行 pinned native `create_forward_context` / `override_forward_context` 的嵌套作用域回归：同一 metadata 的内层调用从 full7 开始也失效；返回外层 full1 再失效。重复任何层的额外检查保留，覆盖一个 context 内重复调用完整模型。

相同 stream 上设备执行异步，但 prepare、source2 suppression、各层消费保持顺序；host 不回读任何 metadata 值。新 stream 不复用旧任务。该改动不引入多 stream 同时写同一 scratch arena 的支持：原 `GraphWorkspace` 本身就是单 stream 顺序复用契约。capture、dummy、draft、实际 mixed/current-only 分区仍执行原有 prepare；地址不变。

CPU forward harness 使用 16 个不同物理 raw 验证：32K/64K context 和 int32/int64 slots 下，prepare/suppress 16→1，任务/位置/输出字节保持，其余所有操作各执行 16 次。错误任务字节未被复用逻辑清除。另验证真实 q4 dispatcher 只给 hybrid 算子传预分配 NZ Rv，缺失即报错。

## 成本与裁决

N=16384、Hkv=1、S=1 的任务表每次为 `16384×3×16×8 = 6 MiB`。在一个 metadata group 连续经过16个FULL层的条件下，生成任务表写量由96 MiB降至6 MiB，减少90 MiB；位置写量减少1.875 MiB。这里是源码字节账，**不是设备访存计数或速度**。main native slot本已int64，不申领其转换收益。

实际 mixed/current-only 分区不会获得这项复用，默认decode graph与later-MTP也未改变。当前缺少它在完整路径中的时间占比、设备配对和端到端收益，故不足以加入默认稳定候选。

## 独立 C16 探针

`tools.probe_c16_member_pipeline.py` 比较当前C16与普通member-pipeline C16：Q672/C337、全部2016 tasks、成熟16-member簇、16 grouped leaders/16 skips、全672 query独立oracle、bad80/bad96/bad336旧16-row交错lane错误域。

图门在固定3个BLOCK512物理页、10个虚拟128列容量中将C337改为C593，KV round由2变3。扩容只发生在CPU探针fixture构建阶段，未使用尾部保持A5/NaN/-1；不改服务分配。两臂计时均包含CV+merge+BF16copy，共同current-source FIA明确排除，不称完整模型收益。当前源哈希、产物签名、变Q/变任务/变packed metadata和相同地址均纳入门。

探针本身未在NPU运行。其host回归验证拒绝C4/不完整/未变输入图证据、固定容量、故障lane、完整计时调用链和无NPU显式失败。

档案依据：#34/#36 决定固定图地址和容量；#37–49 保留物理页所有权；#129 与D.4防止history恢复/host重建；#142 排除dummy/capture假通过；#157 保持显式discard和冻结guard。FE0七文件哈希全部匹配。本地脚手架曾因测试变量名遮蔽而失败，修正后回归通过；没有新增真机错误，未追加故障档案。
