# BF16 开销优化与启动失败修复（2026-10-19）

母版本 `8b1695a`，分支 `oscar-new-3`。本轮默认候选同时接入四项算子改动，并补齐正式服务进程监督。依据是当前执行源码、真实用户日志及本地主机测试；未使用失败项目代码，没有修改原生参考树、FE0 七文件或冻结精度阈值。

## 本次日志的首错

原始附件按字节保存在 [target_startup_memory_20261019.txt](../reports/target_startup_memory_20261019.txt)。原生 `worker._init_device` 报可用 **1.19/29.49 GiB**（另两个 worker 1.20），低于 `gpu_memory_utilization=0.9` 所需 **26.54 GiB**。源码调用顺序为 `init_device()` 后才 `load_model()`；此次尚未执行 OSCAR Attention。`ERR99999`、EngineCore 失败是后续结果，不能据此修改 BF16 算术。日志没有占用者、目标 git SHA 或本轮 NPU 内存释放证据。

发现并修复的独立生命周期缺陷：直启原先用 `execvpe` 交出父进程，缺少正式服务的统一退出清理。现在由 `tools.serve_process.run_service` 监督本次新 session，正常运行无截止时间；退出、SIGINT/SIGTERM/SIGHUP 和异常后，有界清理本次进程组。API leader 先退出而 worker 仍占用日志 fd 的情况也覆盖。所有输出实时打印并完整落入 `serve.log`；原始 rc、signal 对应 shell rc、清理结果和最终状态均保留。清理失败不会变成成功；只在观察到进程组不存在时确认清理完成。

不降低 0.9，不新增默认环境扫描/前置审计/测试请求，不按名称杀进程，不清理旧 run 或其他项目服务。失败后从本次日志打印内存不足说明和只读检查命令。**修复未来本次进程组的生命周期，不等于已释放用户机器当前约 28 GiB 的占用。** 真机应先通过 `npu-smi info` 确认占用者，再正常停止已确认可停止的旧服务；未知占用不能自动终止。

## 默认候选的四项改动

| 项目 | 修改前 BF16 候选 | 本轮实现与新增成本 |
|---|---|---|
| CV 工作区 | 原 FP32 arena 加完整 BF16 同偏移镜像 | 分别紧排 BF16 Q/K/V/P 与 FP32 score/PV/归一化/在线状态；无新增搬运。C4/C16 最终逆旋转按既有逐 member 握手复用单个归一化输入 tile。 |
| 历史 Q | 旋转写 FP32，各任务/分片检查、重复舍入到 BF16 | 新 `rotate_bf16_query_out` 保留 FP32 旋转与原 ReduceSum，舍入一次。输出 BF16 `[N,H,D+16]`；末尾32B中的首个 FP32 值保存原 sum，供 reader 保持舍入前有限性检查。新增最大4352B UB；D256 每行 GM 从1024B减到544B。 |
| 精确 K/V 检查 | 每行 BF16→FP32 Cast，然后原 FP32 ReduceSum | 在已有 natural scratch 中最多4行合并 Cast；每行 ReduceSum、tag→finite 错误覆盖顺序和相关 fence 不变。没有新增 UB/GM；连续有效4行的 Cast 从4次减到1次，尾部和非法行不承诺相同比例。 |
| 首次 Matmul | 初始化 FP32，然后第一次 QK 切换 BF16 | 初始直接初始化 BF16，省掉首次 FP32 初始化及对应 End/Reset。历史输出逆旋转仍用 FP32，之后必要的切换与屏障保留。 |

Q trailer 必须保存原 FP32 sum，不能重新对舍入后的 BF16 Q 求和替代：溢出、抵消和 nonfinite 判定可能不同。此处不是 CRC；受检查的生产者和消费者遵循明确 ABI。普通 sink/recent/current Q/K/V 仍按原始 BF16 位发布。INT2 缓存、FP16 scale/zero、查表解包、FP32 softmax/PV 累加/LSE/逆旋转/合并、最终 BF16 输出的精度顺序不变。

P 在 FP32 softmax 后转成 BF16 仍是精度契约要求，没有删除。未采用不明 SDK 重载、未删除必要同步、未将逆旋转拆成额外 kernel；索引构造等剩余成本不能凭指令数量宣布可以全部规避。

| D256 每 Cube CV arena | 修改前字节 | 修改后字节 |
|---|---:|---:|
| range / balanced | 1,376,256 | 786,432 |
| MDL q4 / q1 | 1,720,320 | 655,360 |
| C4 member pipeline | 3,741,696 | 1,970,176 |
| C16 | 7,495,680 | 3,883,008 |

例：N16384/Hq6/Hkv1/D256/S1/20 Cube/C16，每卡 CV **142.96875→74.0625 MiB**，Q **96→51 MiB**；整个共享 workspace **729,711,872→610,272,512 B**，减少 **113.90625 MiB**。这是固定分配量，不是每层累乘、HBM 实测流量或吞吐增益，也不足以解释此次启动前约25 GiB的空闲缺口。

## 验证与交付边界

- 组合回归 **1127 passed / 1 skipped**（重复大形状；C16 已覆盖），额外新 Q probe ABI **15 passed**。可复跑 argv/env 保存在 `reports/bf16_overhead_combined_v2_command_20261019.json`。
- 初轮组合 **1124 passed / 3 failed / 1 skipped**：两个旧 workspace 字节断言及一个旧 rotate mock；按新 ABI 修正，未放宽数值或错误断言。初轮 XML/log 保留。
- 实际生产 RotateOnly 与五份 LoadQueries 源体被主机编译执行，覆盖舍入位、舍入前非有限/溢出、padding、尾行、多 core/多 head、输入变化；删除 checksum 发布的负控必须失败。旋转引擎、DMA和异步 fence 为明确 stub，不能当成完整算子 CPU-debug 或 NPU 通过。
- precise 共享头实际主机执行 **25 passed**，包括 tag/finite 错误优先级和溢出。六路实际寻址/Cube 控制流、C4 归一化区串行所有权、Matmul 初始化/切换正负控均覆盖；mode2跨核事件模型仍是契约假设。
- 源生成与反向恒等检查通过；FE0 **7/7** 未变；冻结 acceptance SHA256 为 `b044a5f485cd8cfed81dc76c7199717952853167b4e1d8e0aa9bff499b290811`。
- 两入口计划均选择 `query_rotation=rotate_bf16_query_out`、`cv_workspace_layout=compact_typed_v2`、BF16 q4/q1/C4；三处直接 BF16 探针和完整 attention 路径也实际调用新 producer。CV-only计时明确不含 producer；完整路径计时包含它。真实 graph 门仍检查固定地址换输入。
- 本机无 CANN、Lima 或 NPU。新代码 **CANN/官方CPU-debug/NPU/图/模型质量/性能未验收**。本次日志的 build-ops 成功属于此前目标代码。既有 BF16 数值集 **7/11 失败（含2例最终输出）**及 #165 未归因32K Trap 保持未解决，不能用本轮主机通过覆盖。

汇总：[bf16_overhead_delivery_20261019.json](../reports/bf16_overhead_delivery_20261019.json)。本轮不要求 AISBench。

代码已同步到目标机、并确认后四卡空闲后，直启仍为：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate --rear-cards
```

它只安装/构建/准备旋转/正式服务，无探针，不提供验收证据。短时算子与图门使用现有 `bash scripts/install_observe_serve.sh --variant candidate --probe-only`；任何门失败仍停止。仅复查编译可用 `python3 -m tools.deploy --only build-ops`。默认配置、后四卡模式、原生调度器、APC策略及16384预算均未改变。
