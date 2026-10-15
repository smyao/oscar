# C16 task 扫描的只读成本边界

档案 #129/#140/#148 与完整 D.4；没有新增设备实验、没有修改 prepare/status。
母源 `csrc/kernels/attention_cv_window_range_cluster16.cpp` 的 `Process` 两遍扫描保留
`scheduleTile=4`；每次 `LoadTask` 在 AIC 读11个 int64，在每 AIV 做128B DMA并等待
MTE2→Scalar。第一遍连 qcount0 follower 都读完整描述符；每个候选成熟leader再次
复核16个成员，第二遍又逐token找request-relative的336-token anchor。

以本轮真实CPU prepare保存的Q672/C337/recent256、S1/Hkv1的2016行任务计算：

| 全核工作（不是每核） | 数量 |
| --- | ---: |
| 第一遍基本 LoadTask | 2016 |
| 第二遍逐token LoadTask | 672 |
| 第一遍20个成熟域候选的16-member复核 | 320 |
| 第二遍1个完整簇的16-member复核 | 16 |
| AIC LoadTask总数 / int64标量读取数 | 3024 / 33264 |
| AIC逻辑元数据读取 / 两AIV逻辑DMA合计 | 266112 B / 774144 B |

同样的单请求、GQA6、S1、充分长context、N16384条件式为49152+16384+12096+752
=78384次 LoadTask。它说明固定token任务开销真实存在，但不是已测HBM流量或耗时。
第一遍额外检查还受真实request边界、padding、split和可见域影响，不能把该条件数
用于未知混合步。

可用旧闭合范围：`reports/c4_phase_profile_q336_20261009/profile-trace/instruction_samples/`
的 `core3.cubecore0.instr_log.dump.markers.json` actor完整，field20（metadata）151个
闭合区间合计56098 ticks，中位379、最大626；旧该核总task为2660850 ticks，约2.1%。
这只是旧C4、Q336范围。其两个AIV完整trace门失败，只能得到前68次metadata的闭合
区间，不能补出余下比例或提升成当前C16性能归因。

结论：存在可进一步降低的重复元数据工作，但现证据不足以将它置于长历史FP32
矩阵/B装载之前。Q672 CPU120秒和CAModel900秒超时只说明本地验证未完成，不能
证明死锁、目标NPU扫描瓶颈或任何相位占比。当前先完成单B1的数据流候选裁决。
