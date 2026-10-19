# 2026-10-19 全面本地验证

档案依据：#94/#95/#125（实时输出、真实结果）、#135/#136（拥有的进程与资源清理）、#154/#167（生产BF16 query ABI与服务内部fatal）；新目标机内存错误单独登记#168。母版本`5ca84395da4a6e2bf29d398d1664d6fdda0652c0`，分支`oscar-new-3`。分析当前源码与本项目只读原生reference，没有读取失败OSCAR项目。

## 结论和修复

**本地发现并修复了服务生命周期缺陷，当前仍不能判定OSCAR整体无bug或可以通过真机验收。**

1. `managed_server`用于服务probe、synthetic和observe，原来只在body中检查存活。最后一次检查后worker fatal、API退出0仍可能记录finished/measured。现与直启共用严格原生fatal识别，在清理前固定日志边界，提前退出也失败；保留真实进程码，不把监督器发TERM后的关闭日志当作首错。
2. 原来清理失败只写生命周期字段，synthetic上层可能继续给出measured。现清理失败抛出错误，顶层报告不能覆盖生命周期失败；body已有异常则保持原异常，同时落盘清理/分类失败。
3. `Popen`已创建OS进程但尚未返回Python句柄时，TERM/INT/HUP会中断赋值，使清理误以为没有子进程。现在仅在取得句柄的短窗口延迟Python信号异常，随后立即清理本次进程组；保留首次信号、原始进程码与启动异常。没有给子进程继承blocked signal mask。
4. managed service增加自己的信号处理与清理期重复信号保护，退出恢复调用方handler。日志/观测失败也必须执行拥有的进程组清理；故障注入用真实子进程，不以模拟进程替代。
5. 过期能力测试遗漏生产必需的`rotate_bf16_query_out`。按实际reader的`D+16`输入ABI补上producer断言，保留研究request-window与生产能力分界。

上述逻辑由现有直启/observe/部署共同调用，不需要新的开关。不改变算子、KV格式、精度阈值、oracle、原生调度、图开关或默认probe顺序；无新增逐步NPU同步/主机回读。额外日志判定发生在服务结束时，未测性能、不申领倍率。

## 测试范围与真实结果

| 验证 | 结果 | 能支持的结论与限制 |
| --- | --- | --- |
| 全仓初轮pytest | 1899 passed / 52 failed / 14 errors / 134 skipped；另29 subtests passed，105.37s | 保留完整XML、日志与66个失败/错误节点；不是全仓通过 |
| 修改后的相关25文件最终交叉回归 | 451 passed | 服务、候选入口、路由、BF16 producer/reader/arena契约，含33项真实进程故障注入；保留前一轮443项回执 |
| 用户授权后的本机HTTP回环回归 | 70 passed，15.93s；此前受阻33项全部通过 | 实际127.0.0.1 HTTP与拥有的测试子进程，覆盖启动、流式请求、超时、故障、中断、清理与日志；NPU/模型资源为测试夹具，不是实际NPU服务 |
| 实际解包源码尾段H16/H32/H64 | 9 passed | 每形2048组FP16元数据×256值，四级表/舍入/Gather与独立位oracle比较；错行偏移和覆盖活元数据两个负对照均被检出。SDK行为、Matmul与硬件同步不在此范围 |
| 实际Python forward跨轮workspace | 8 passed | target/first-MTP × none/align × warm8232/16384；首16K→历史→q4并发128→拒绝3→later-q1，污染旧缓冲、逆序物理页，检查原生current抑制、写读顺序、padding与地址保持。NPU算子为明确主机替身 |
| 完整BF16 CPU数值筛查 | rc2，4/11通过、7/11失败，其中2例最终输出失败 | 旧FP32对照11/11通过；保留原空间oracle和冻结阈值，不能宣布精度通过 |
| 真NPU probe入口的无NPU负对照 | rc1，torch_npu缺失，服务未启动 | 证明无NPU时显式失败，不会CPU替代；不证明NPU算子可运行 |
| CANN、官方CPU-debug、NPU、图捕获/回放、真实模型服务 | 本轮未执行 | 本机macOS arm64，无torch_npu/bisheng/npu-smi/Lima，无法启动目标模型服务 |

全仓66个节点分为：33个本机HTTP绑定被沙箱EPERM阻断；17个缺少历史原始报告/指令trace；13个已退出生产的request-window研究配置/旧错误顺序；2个旧实验静态规范门；1个producer能力断言已修复并通过。没有删除失败用例、伪造历史输入、绕过生产校验或修改冻结精度来制造全绿。

HTTP回归前两次沙箱外自动审批超时，均未执行。用户随后明确回复“允许运行本机回环测试”，在提交`79d3ae9`上成功执行5个测试文件，真实rc0、70项通过、0失败/错误/跳过。逐节点匹配确认初轮被EPERM阻断的33项全部在通过集合中；完整[日志](../reports/full_local_loopback_20261019.log)与[XML](../reports/full_local_loopback_20261019.xml)保留。本次不需要新增生产代码修复，不连接目标机或外部服务。

初轮的33个权限节点及1个能力断言现已分别复验通过；剩余17个缺原始材料、13个旧研究契约与2个静态规范节点保持未通过。这是逐节点更新，不重写初轮全仓结果，也不宣称全仓已经重新跑绿。回环HTTP服务、model/tokenizer/遥测和NPU资源夹具只验证主机流程，不证明真实模型推理或设备显存释放。

## 数值失败为什么不能宣称已修复

当前history四级FP32重建后RNE到BF16，sink/recent保留BF16，所有CV QK/PV用BF16输入；softmax/累加/LSE/合并/逆旋转仍FP32。源码检查没有发现本次新的、可证明的解包位错误。

但BF16输入与旧FP32重建oracle不是数值恒等变换。合法抵消用例中Q=0、P=1、旋转为identity，FP32重建的相反V对仍有残差，BF16舍入后该残差为0；最终一个坐标期望`-0.021484375`，候选为`0`，冻结允许误差只有`0.0051074218936264515`。因此**同时要求现行BF16操作数舍入与所有此类输入通过旧冻结门，存在已经复现的冲突**。不以修改oracle/容差、恢复旧FP32生产路径或隐式补偿绕过用户精度契约。

另外5例只有partial_output失败；大值例还有最终输出失败。原生FIA内部P类型/归约树未由开放调用接口证实；CPU筛查也不完整模拟原生current FIA产生BF16 partial的分支。上述证据既不证明与dense原生逐位相同，也不证明这就是目标机Trap的原因。

## 最新目标机日志

运行`20261019T084558.865108Z`在08:47:47、Worker500971的原生`_init_device`失败：free13.03GiB < required26.54GiB，发生在load_model之前。此次supervisor正确返回failed/rc1，不能给#165/#167提供新的guard归因。显存占用者未知；`cleanup_complete=true`表示拥有的进程组清理结果，不等于NPU显存释放已测。

## 可复现命令与证据

以下是已配置`.venv`的本地开发机命令，不是真机部署前置条件。当前Mac主机C++测试需显式选择实际Xcode工具链：

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
export OSCAR_TEST_CLANGXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/test_full_service_audit.py tests/test_full_runtime_audit.py tests/test_full_bf16_audit.py
.venv/bin/python -m pytest -q tests/test_benchmark_measure.py tests/test_benchmark_mixed_passive.py tests/test_live_output.py tests/test_plog.py tests/test_service_workflow.py
.venv/bin/python -m tools.experiments.bf16_all_sources_screen --output /tmp/oscar-bf16-local-screen.json
```

最后一条当前预期真实rc2；pytest中验证“门能拒绝错误”的测试通过不表示此数值筛查通过。详细命令、结果、哈希及最终增量回归：[汇总JSON](../reports/full_local_validation_20261019.json)。

原始材料：[全仓日志](../reports/full_local_audit_before_20261019.log)、[逐节点失败](../reports/full_local_audit_before_failures_20261019.json)、[交叉回归](../reports/full_local_final_20261019.xml)、[数值筛查](../reports/full_bf16_audit_screen_20261019.json)、[源码/精度分析](../reports/full_bf16_audit_findings_20261019.json)、[最新真机日志](../reports/target_diagnostic_startup_memory_20261019.txt)。42个稳定checkpoint不提升，FE0七文件与冻结阈值保持原字节；不要求重跑AISBench。
