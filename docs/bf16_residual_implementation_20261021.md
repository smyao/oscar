# BF16 残差补偿实现与验证（2026-10-21）

用户已授权引入残差补偿；这是对档案 #173 数值问题的源码修复，不是新的真机通过记录。冻结 oracle 与 acceptance 不变，不恢复旧 FP32 QK/PV 生产路由。

## 数学与数据生命周期

每个需要补偿的 FP32 值拆成 `hi=RNE_BF16(x)`、`lo=RNE_BF16(x-float(hi))`。历史 QK/PV 按 HH、HL、LH、LL 顺序作四次 BF16 输入/FP32 输出矩阵乘。非原子路径首项覆盖输出、后三项累加；C4 已有在线 accumulator 的四项全部累加。每项重新设置 shape，全部结束后才发布原来的完成信号。两份 BF16 是近似表示，不宣称能精确恢复任意 FP32 值。

- 历史 K/V：INT2 物理格式不变，解包仅生成每 token 四级 FP32 小表，再生成高/低 BF16 表并 gather 到有界 KV tile；没有全历史 FP32 展开。
- 历史 Q：已有旋转算子输出 FP32 `[N,H,D]`，reader 内拆分。C++ binding 拒绝旧 BF16 `[N,H,D+16]` 输入，避免跨版本误解释。
- sink/recent/current 的 Q/K/V：保留原 BF16 值，不读未初始化低平面。QK 一次；P 拆高/低，PV 两次。原生 current FIA 分支仍用原生实现。
- P：原 FP32 概率不原地修改，softmax 分母仍使用原 FP32；16KiB scratch 分块生成高/低 P。
- FP32 softmax/LSE、在线累加、逆旋转、merge、状态/错误优先级及持久缓存格式保持。
- MDL 双缓冲预取同时发布高/低 K/V；保留 #171 的 MTE3→MTE2 共享 UB 依赖。C4 H32 双lane仍映射到原两个H16域，高低起点分别校验。

## 成本（源码计数，不是 NPU 性能结果）

| 项目 | 单份BF16版 | 补偿版 |
|---|---:|---:|
| 历史QK，每tile | 1次乘法 | 4次乘法 |
| 历史PV，每tile | 1次乘法 | 4次乘法 |
| 精确源QK/PV | 1/1 | 1/2 |
| 持久INT2 KV字节 | 不变 | 不变 |
| 临时BF16操作数平面 | 1 | 2 |
| 旋转Q，每D256行 | 544B | 1024B |
| MDL每核workspace | 655360B | 1212416B |
| general每核workspace | 786432B | 1179648B |
| C4每核workspace | 1970176B | 2887680B |
| C16每核workspace | 3883008B | 5259264B |

没有增加 UB 分配，已有 scratch 分块复用；新增 Cast/Sub、gather、GM写/读、累加必须计入性能。持久压缩收益仍在，但不能据此声称更大batch能弥补所有计算成本。runtime 共享工作区必须覆盖其可能选择的每个reader，不得假设general天然大于MDL。

## 本地证据与目标验证

`reports/bf16_residual_screen_20261021_v2.json`：11个固定CPU案例，四项补偿的partial/output/LSE/最终BF16/relativeL2全通过冻结门；三项仅作独立对照，不用于本版生产。抵消用例最大输出误差从3.6953125降至约1.91e-6。旧失败报告保留。

`reports/bf16_residual_probe_numerics_20261021.xml`：首12个有效NPU fixture的CPU数学回归及q1负Q状态，共13项通过。候选从实际striped物理页解码，不从期望结果反推。C++函数主机测试覆盖实际四项调用、原子标志、双平面地址、source不变、2048分块和尾块、reader越界防护。它们均不是CANN或设备执行。

C4四项直接累加到online accumulator，与CPU先求四项和再加acc存在FP32次序差异；设备门仍必须通过。当前宿主机无CANN/NPU，不宣称设备/图/完整服务/质量/性能通过。

固定candidate入口实际使用此实现，无需新增开关。目标机先用 `bash scripts/diagnose_bf16.sh` 编译并探测；精简报告自动写到 `logs/bf16_diag_latest/result-key.log`。失败不得跳过验收门。完整服务仍需后续短probe；不要求重跑AISBench。

最终本地回归：`bf16_residual_combined_valid_20261021.xml` 417项通过；`bf16_residual_runtime_20261021.xml` 129项通过、1项按既有测试设计跳过；两个集合不重叠，共546项通过。另有安装入口专项回执。`bf16_residual_combined_20261021.xml`是未配置本机Xcode编译器路径的失败中间记录；`bf16_residual_probe_tests_20261021.xml`是生成器修改期间的中间记录，均保留且不作为最终通过证据。最终测试显式使用本机Xcode clang与SDK，无CANN/NPU。
