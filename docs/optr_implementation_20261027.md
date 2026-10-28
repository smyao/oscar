# OptR 离线旋转精修：源码复核、改造与验收边界

2026-10-28最新用户授权：允许生成实验PT，先测试OptR。固定candidate直接启用recipe5，无需新增参数；独立test执行并记录退化，但不阻断实验PT/后续服务。validation逐请求非劣和数值合法性仍严格，test不参与矩阵重选。新证书schema2声明experimental和发布政策，逐层原始test指标及汇总失败状态随PT保存，加载时显示`OPTR_ARTIFACT`；不能将[历史失败](optr_independent_test_failure_20261028.md)改记为通过。下文旧独立test禁止发布描述是历史合同，由本授权替代；训练数学、线上核/runtime/timer/argv未改，未验收模型质量或性能。

拟合/trace profile保留历史v1字节身份，避免重复原生采样；其旧`test_policy`字段不是schema2的发布判定，后者由证书`publication_policy=validation_gate_independent_test_report_v1`明确声明。schema1旧PT继续要求独立test非劣，不能无标识放行。只兼容审核过的父版本拟合实现：矩阵优化和trace数值源码哈希、模型/基底/trace/profile/维度/test-only身份均匹配才复用旧头/完整层；原文件不改，哈希损坏仍失败。未完成层的联合检查重新执行，不把旧失败报告当完成断点。

2026-10-27用户部署反馈已确认原生capture完成，但全层CPU拟合被原通用1800秒时限终止。准备时限、真实进展及断点续作已在[后续修复](optr_preparation_timeout_20261027.md)中更新；下文首次交付的“phase timeout不改”记录不再代表当前行为。

本轮基于工作区 `4f4ef7ef589741c3866ee3aac62448ac142fe813`，把输出感知优化接入固定 candidate 准备流程。线上只替换现有 Rk/Rv 常量；不是新增 attention 算法，也没有把优化器放入服务。分析依据见 [OptR 集成分析](optr_integration_analysis_20261027.md)。上游算法来源固定为 [66d4656](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/tree/66d46561b9bdf5d715f591e0f5845a4cbd927dee)，不依赖运行时下载或执行上游工程。

## 复核后的取舍

原样移植上游不成立：当前系统有 Qwen gate、GQA/TP、D256 单组量化、perhead_center、固定精确窗口、FP32 metadata 和 v4 BF16 运算链。上游独立 K/V 代理及其默认配方不能直接充当当前系统的质量保证。已采用以下约束：

1. 基底保持经过认证的 recipe3，K mean、K=.96/V=.92、sink/recent、INT2 编码、144B 布局不变。recipe4 仍不作为 candidate 默认。
2. 仅离线训练正交修正 `R = R_base @ exp(A-A.T)`，最后保存合并后的 FP32 Rk/Rv。服务仍使用已有 BF16 矩阵乘和已有逆旋转，不新增矩阵乘、反量化、KV 分支或状态。
3. 以真实原生 FULL attention 的 post-RoPE Q/K、V、sigmoid gate、gate 前输出和分片 W_O 为依据。先按原生 BF16 语义执行 gate 乘法，再度量 FP32 W_O 投影误差。
4. K、V 先后优化，但两阶段都使用联合 KV 输出误差。逐头选择后，还必须把全部 TP rank 的投影相加再计算整层误差，保留跨头交叉项；禁止用“各头 SSE 之和”替代。
5. 验证同时覆盖单分片拟合模型和采样时真实 frame 的动态分片。后者绑定实际 AIC/AIV 数、padding 后输入形状、workspace 容量和 qtile；完整历史先分片再做窗口 mask，current 源使用完整 forward 的 producer 边界。

## 理论收益与限制

令 `L(Rk,Rv)` 为已固定验证样本上的联合投影误差。候选集合包含原 recipe3，选择要求每个请求、每种被验证分片均非劣，且整层总误差严格下降才替换该层。因此对这组样本及 CPU 数学合同，有 `L_selected <= L_recipe3`；没有改善的层明确保留原矩阵。

另有独立 test 数据，既不选择 step，也不选择矩阵或超参数。选定矩阵若在 test 上退化，准备过程失败，禁止发布 PT 和启动该候选。不能在看过 test 后静默退回原矩阵并宣称 test 通过。

这比 qqt/sst、score MSE 或逐头代理更接近目标输出，但不是任意输入上的支配关系，更不是最终任务得分定理。各层以原生 teacher 输入独立校准，未覆盖量化误差跨层传播、全部混合 batch/MTP/graph 形状及自回归分布变化。FP32 W_O 投影和 CPU 归约也不声称逐位复刻 NPU BF16 投影/TP all-reduce；native FIA 内部保持不可验证边界。不得据此宣称原先约 3 点差距已恢复或已达到论文标准。

## 首次准备与缓存

- 固定 `candidate` 在原生 BF16 权重、qqt_sst/perhead_center、显式 token budget 路径启用 `rotation_refinement=optr`、recipe5。显式 Hadamard、W8A8 和 native-budget 研究入口保留原有适用路径；native/baseline 清除该配置。
- 原 recipe3 的认证、已有原始统计复用保持。其准备在独立子进程执行，退出后才启动新的原生采样，避免父进程 NPU 上下文占用采样显存。
- 老 qqt/sst 统计没有必要的 query、gate 和 teacher，不能冒充 OptR trace。首次缺少 trace 时自动采样：train 4K；validation/test 各 4K、32K、64K；每 frame 抽取 8 个 query，另采 8 个真实 decode 输入。采样进程仅自身关闭 OSCAR/APC/MTP，采用 eager、单请求，保留 16384 chunk budget。
- 数据先按完整源记录划分 train/validation/test，再拼接到目标长度。默认来源是已打包 GPQA；重复拼接会明确记录。这是校准长序列，不能当作 LongBench/GPQA benchmark 成绩。
- trace 按 rank/层/请求保存，拟合逐层加载。完整 manifest、实际 token、prompt/source 分区、frame、矩阵、配置与文件哈希共同校验。不完整采样保留诊断，下次用新的 generation 目录；完整 trace 可复用。
- recipe5 与 recipe3 使用不同路径和发布 manifest。配置、模型权重身份、clip/window/TP/分片策略变化令缓存身份不同；产物内均值/矩阵改动令证书或文件哈希校验失败。已发布 recipe5 自带基底哈希，复用时不重新读取后来重生成的同 profile recipe3。哈希损坏、漏层、旧 PT 重新标记、非正交矩阵和错误证书均失败。
- 已有完整 recipe5 PT 时跳过采样及优化。首次准备增加 NPU 采样、CPU 优化和磁盘 trace 成本；不能声称首次启动零开销。现有 phase timeout/失败清理机制不改，失败不继续正式服务。

## 改动位置

| 文件 | 职责 |
|---|---|
| `oscar_ascend/integration/optr_capture.py` | 仅隔离采样 worker 使用的可撤销 hook；读取真实 gate/QKV/teacher/位置及 W_O |
| `tools/calibrate_optr_traces.py` | 原生子进程、互不重叠的源记录计划、逐请求采集与完整 manifest |
| `oscar_ascend/optr_traces.py` | trace/TP 身份认证、动态分片重建、整层联合误差 |
| `oscar_ascend/optr_calibration.py` | CPU 离线 STE/正交优化及部署数值近似；不被 serving forward 导入 |
| `oscar_ascend/optr_policy.py`、`optr_artifacts.py` | 独立 recipe 身份、不可放宽的选择策略、原子发布和证书验证 |
| `tools/prepare_optr_rotations.py` | 基底→采样→拟合→验证→独立 test→发布；固定入口自动调用 |
| `rotations.py`、`prepare_rotations.py`、`serving_variants.py` | 初始化加载及 candidate 接线；保留旧路径合同 |

## 性能与功能边界

`runtime.py`、所有 attention/codec/prepare/merge CANN 核、线上 integration、`prepare_device_rotations`、timer、接收/拒绝 mask、MTP/GDN/prefix/graph 生命周期、缓存预算和服务 argv 均保持。本轮固定同一输入形状的在线算子数、矩阵尺寸、buffer 和 KV 字节量；没有在线校准或额外 GEMM。候选与原 recipe3 的 `serve_argv` 逐项相等。

原 FP32 metadata overlay 的 recipe3 描述仍作为旧合同保留；recipe5 的矩阵身份、离线目标和非劣规则由独立 `optr_policy`/certificate 定义，继承现有 v4/FP32 metadata 算术与容差。没有把新 PT 重新标记成旧合同，也没有放宽任何旧验收阈值。

这些是“未增加在线工作量”的源码和主机证据，不是端到端吞吐实测。生成内容、输出长度、MTP 接受率会受数值影响，仍可能改变端到端表现。遵守项目禁连真机、禁止测速的约束，本轮不能给出“实际性能一定不低于当前版本”的验收结论。

## 本地验证

核心测试已验证：GQA6 的 S1/S20 与独立 v4 runtime oracle 一致；跨 KV256、21-query producer、BF16 gate 舍入、FP32 metadata、因果窗口、microbatch、正交性以及“逐头改善但联合输出退化”的反例。独立合成 train/validation 的小维度例子约降低 29.8% 输出误差，仅用于证明优化器可以工作。

另外验证 trace 完整性/TP 对齐/源分区、证书篡改/非劣判定、缓存失效/复用、固定入口接线、显式旧模式兼容与在线 argv 不变。确切命令和结果见 `reports/optr_implementation_20261027/verification.json`。没有新增真机档案，不运行 AISBench，也不把 CPU 测试写成 NPU/graph、模型质量或性能通过。

固定入口参数保持：`bash scripts/install_serve.sh --variant candidate`。设备 `0,1,2,3`、端口 `7878`；显式后四卡仍使用 `--rear-cards`。本轮代码在当前工作区，未改动 main/stable 或远端分支。
