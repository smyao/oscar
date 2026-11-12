# BS16 → BS28 的工作量与优化

2026-11-12，基底 `87a8959`，分支 `oscar-new-7`。用户反馈同一数据集、相同 PC 命中率下，BS28 相比 BS16 吞吐没有增长。本轮根据当前生产源码消除两处确定的重复工作，保留现有推理数值链。没有连接真机或运行性能实验；下面的工作量不能当作现场耗时或吞吐增幅。

## 为什么容量增长未必转化为吞吐

PC 命中减少重复 prefill，但后续每条 decode 请求仍需计算其可见历史的 attention。相同历史长度下，28 条请求的历史读取和有效反量化工作量是 16 条的 1.75 倍；现有 q4/q1 内核按请求处理历史，增加请求数不会启用同一请求的 C16 多查询历史复用路径。

对于实际执行的、可比较的纯 decode 步，近似 `输出 tokens/s = B × 每请求平均有效输出 token 数 / step 时间`。若平均 MTP 接受量不变、step 时间也增加到 1.75 倍，BS28 的吞吐便与 BS16 相同。Running 峰值不等于每步实际执行的 B；纯 decode 步数占比也不等于耗时占比。因此现有日志不能单独证明全部瓶颈在访存、MemCache 或某个算子。

以下按当前 TP4、每 rank Q/KV 头数 6/1、head dim 256、20 个 Cube、MTP3、均匀 decode 请求计算，使用实际 `WorkspaceGeometry.splits_for_tokens`、candidate selector 和 kernel 的 owner/split 公式。32K/64K 是静态比较场景，不是现场请求长度测量。

| main q4 工作量 | BS16 | BS28 |
| --- | ---: | ---: |
| token 行数 | 64 | 112 |
| 每 source 的 split 数 | 5 | 4 |
| 每层任务描述符数 | 960 | 1,344 |
| 任务表字节数 | 122,880 | 172,032 |
| history leader 数 | 80 | 112 |
| 最忙核的 32K history KV256 循环数 | 104 | 192 |
| 最忙核的 64K history KV256 循环数 | 208 | 384 |
| 相同长度的 packed 历史读取比例 | 1.00 | 1.75 |

64 和 112 都是现有 graph capture size，不能直接归因于 BS28 额外 padding。更多请求同时增加队列中的 kernel 工作；当前 split 从 5 变为 4 后，每个 history leader 的区间也更长。循环计数包括各 split 的 tile 尾部，不能直接换算为执行时间。later MTP q1 的 padding 与有效行分开计入报告。packed 计数基于有效页表下实际请求的复制载荷：当前 history loader 仍读取 `[sink, context)`，包括随后在 softmax 被掩掉的 recent 区域；current 不在此范围内。

当前已经有 K/V 共用 packed 读取、预取流水以及 FP32 merge 后一次 BF16 逆旋转，本轮不重复计为新增优化。BF16 K/V tile 仍需通过固定 GM 暂存区再读入 Cube L1；固定暂存区可能命中 L2，所以逻辑 GM 字节量不是实际 HBM 流量，也不能据此宣称已证明 memory bound。

源码入口与八组可复现计数见 [workload.json](../reports/batch_scaling_20261112/workload.json)。报告绑定源码 SHA，内嵌复现脚本。没有以调整 split、改变 softmax 归约顺序或输入长度特判来换取速度。

## 已完成的两项优化

### 主模型 decode 共用一次 prepare

此前 prepare 复用只覆盖符合条件的 eager prefill。主模型的 16 个 FULL 层在 decode 中仍分别生成相同的 tasks/positions，FULL 图也记录这些重复 prepare。

现在 [prepare_reuse.py](../oscar_ascend/integration/prepare_reuse.py) 和 [impl.py](../oscar_ascend/integration/impl.py) 将复用扩展到注册在同一个原生 ForwardContext 中的主模型 eager decode，以及均匀 FULL 图捕获。已有 `experimental_prepare_reuse` candidate 开关直接启用，无需新增参数。

- 同 metadata、context、stream、geometry、phase、descriptor 才共用。descriptor 保留强引用并比较身份；重复 layer、新 context、不同组/geometry 或其他 writer 使旧记录失效。
- FULL 图中的第一层仍记录真实 prepare，后续层省去重复 launch；每次 replay 都先执行图内 prepare，重新读取本次 metadata。没有把某次 Python 缓存当成跨 replay 的有效设备内容。
- 原生非 MLA 的 MTP 捕获会标为 `ChunkedPrefill`，因此 capture 资格使用实际 FULL/uniform/native context，不能用 query 长度猜测是否为 decode。独立 dummy/profile、PIECEWISE、DBO 和 draft 不进入新增路径。
- 每层 rotate、CV、merge、store、guard 继续执行；MTP 各轮保留原有 prepare。主模型 prepare **16→1**；主模型加三轮 MTP 的示例为 **19→4**。这是 prepare launch 数变化，不是总算子数或 step 时间下降比例。

任务表和 positions 在这些读端只读，物理 KV、状态、输出和每层旋转独立。原有混合 prefill 资格及真正 warm prefill 的 GDN/MTP 保护保持。

### 同批 prefix 保存合并计数回读和屏障

稳态纯 decode 若没有 prefix 保存、抢占或回载，本来就不执行逐步 MemCache PUT/GET/保存同步。本项优化针对同一步完成多个不同 prompt、需要发布 prefix 的情况，不能当作所有 decode 步的收益。

[state.py](../oscar_ascend/offload/state.py) 新增调用内的惰性 counter 读取：同一 counter snapshot 的完成 event 只等一次，base/accepted 按有效行批量回读。后续仍逐请求解析、PUT、ACK，下一条失败不抹掉已经成功的前一条。不会提前解析全部请求，不缓存到下个调用，不读取 buffer 未使用的容量尾部。需要 speculative next-accepted 的情况保留原选择规则。

[connector.py](../oscar_ascend/offload/connector.py) 在共同 target/draft producer 完成后，只在该批第一次实际保存前执行 current-stream 屏障。后续 canonical GDN 使用原有 `clone=False` 视图，不提交新设备写入；PUT 仍同步完成后才产生该请求 ACK，全 TP 收齐后才发布。私有 swap 保留原同步行为。

同一 snapshot 中全部 16/28 条非 speculative prompt 均保存时，counter D2H 从 **32/56 次标量读取→2 次列读取**，counter event 等待和保存 stream 屏障各从 **16/28→1**；PUT、ACK 数及字节格式不变。单条保存仍走原 scalar 路径，无 prefix 时直接返回，不构造批量读取对象。

代价也保留记录：若实际保存 S 条、snapshot 有 B 条有效行且 `1 < S < B`，批量回读两列的 B 行，约 `12B` 字节（int64 base/int32 accepted），旧路径约 `12S` 字节；以少量额外元数据字节换取更少 D2H 调用。没有把它描述为无条件减少传输字节。

## 验证与交付边界

联合主机回归 **834 passed，0 failed，0 skipped，29.28 秒**，命令及源码 SHA 见 [validation.json](../reports/batch_scaling_20261112/validation.json)。新增 15 项 decode prepare 测试与 29 项 prefix 批量计数测试，包括：

- BS16/28 的 N64/112、同地址换新内容、不同图交替 replay、MTP3 覆盖、EAGER→FULL replay→新 EAGER context、非法 task 仍被 guard 拒绝。
- 实际原生 ForwardContext/ACLGraphWrapper 的捕获回放编排，配合 CPU recording graph/operator doubles 对照输出、tasks 和 positions；这不是 NPU 图执行证据。
- none/align、counter 槽复用/异流 producer、有效行边界、迟到失败、逐请求 ACK、真实本机 IPC 与全 TP 发布及异址字节恢复。
- CANN/PT/clip/量化、BF16舍入、split/softmax顺序、GDN/MTP、PC/chunked prefill、图地址、timer、argv 与配置的现有保护。

原 232b impl SHA 只反向移除精确列出的已授权主机接缝后继续校验；expected SHA 与全部 CANN hash 未放宽。没有修改 `csrc/`、`configs/`、`scripts/`、`tools/` 或原生参考树。

原入口 `bash scripts/start_node93_oscar.sh` 和固定 candidate 直启命令会实际启用本轮优化。没有更改 BS、设备、端口或服务参数。当前交付证明移除了上述重复工作；模型精度、NPU 图和 BS28 相对 BS16 的端到端吞吐仍未实测，不能标为已拿到收益或已消除全部开销。
