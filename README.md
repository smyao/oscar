# OSCAR Ascend 外部适配工程

已实现 FULL 层 INT2 缓存、真正的 Cube/Vector attention、融合旋转/裁剪/写入、精确窗口、MTP 位置修正、固定图缓冲、外部插件及一键安装/探针/服务流程。GDN 使用原生状态和 reshape。新增代码不覆盖原生 vLLM/Ascend 源文件。

在 node93 的本项目目录执行这一条命令：

```bash
git pull --ff-only && bash scripts/install_probe_serve.sh
```

默认读取 `configs/target.json`：物理 NPU **0,1,2,3**、`ascend910b4`、TP=4、端口 **8989**，模型为 `/softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp`。这些值来自本次启动文档附录 A/F；全部子进程使用同一配置，不继承其他任务卡号。脚本使用当前 `python3`，也可通过 `OSCAR_PYTHON` 指定已有解释器；node93 无需创建 `.venv`、安装 Lima 或运行 `validate_vm.sh`。

默认流程：安装构建依赖/插件 → 编译算子 → 真 NPU 算子对拍及 CV/旋转探针 → 自动准备旋转文件 → TP4/MTP/长输入/图服务探针 → 回收本任务进程及 NPU 资源 → 正式服务并验证真实请求。**探针保留且失败即停止**；启动流程不执行环境清单、原生源码扫描或 readiness 前置审计。安装、编译、数值、服务或资源清理失败都会保留日志和退出码，不改走原生 FULL 或 CPU。每个阶段的输出、服务启动日志及 traceback 都实时显示在当前终端，同时写入 `logs/<本次运行>/`；失败时打印阶段名、退出码和日志路径，无需打开文件才能看到错误。该流程尚未在目标 NPU 全程执行。此次启动修订对应档案 #74–76/#94–98/#107/#117/#123–125；最新真机CV数值失败记录见 #126。

不启动设备即可查看计划：

```bash
bash scripts/install_probe_serve.sh --plan
python3 -m tools.target_cli --print-command
```

需要单独调试某个阶段时：

```bash
python3 -m tools.deploy --only build-ops
python3 -m tools.deploy --only probe-ops
python3 -m tools.service_probe --config configs/target.json --output reports/service.json --log-dir logs/service
```

**最新真机结果：128、16K、32K、50K四个串行请求均已生成16个token，并开始混合并发探针（#130）。** 32K和50K仍很慢，完整并发/质量/性能尚未验收。本轮将历史INT2与当前BF16的逐token搬运改为有界分块DMA，保持计算公式、精度阈值与固定UB/GM预算；提速幅度待目标实测。日志会明确debug_sync是否启用，并区分客户端等待、host事件及设备同步检查点。ArgSort转AiCPU告警已由用户确认基线也有，本轮未改该原生路径。[最新原文](reports/target_serial_50k_progress.txt)。

当前运行的采证（只读本轮进程、日志、trace和所属PID的CANN日志，不启动模型、不接触设备、不发信号）：

```bash
python3 -m tools.inspect_run --run-dir logs/20260922T053137.509986Z
```

采证结果写到该目录的 `stall-inspection.json`。停止本轮后，可用显式调试入口重跑完整流程：

```bash
bash scripts/debug_service.sh
```

它保留目标参数、全部探针和图模式，对tokens≥1024的非捕图调用增加NPU流同步检查点，逐rank记录 `phase_begin`、`waiting_for_device`、`device_completed`。`waiting_for_prior_work` 表示还在等待此前原生工作。捕图期间不会插入同步；普通入口默认关闭这些检查点。同步调试会改变调度和耗时，不能当作性能验收数据。

以下仅用于本地开发，不是 node93 的部署步骤。已配置本项目 `.venv` 的开发机可运行 CPU 测试：

```bash
.venv/bin/python -m pytest -q
```

Mac 宿主机安装 Lima 且 `oscar` VM 已启动时，可运行：

```bash
bash scripts/validate_vm.sh
```

此命令在 Lima `oscar` VM 的独立 `/home/sunao2000.linux/gpt_new_oscar` 目录编译、链接，并运行同一份 AscendC kernel 的官方 CPU 调试器。首次配置可加 `--bootstrap`，仅安装该目录独立虚拟环境的依赖；不访问 VM 中旧项目代码。报告在 [reports/vm/validation.json](reports/vm/validation.json)，最新一轮共61个算子用例通过，另含 MTP/padding 元数据检查。

一键性能诊断已并入 `git pull --ff-only && bash scripts/install_probe_serve.sh`：它自动依次启动原生和 OSCAR 服务，对同一组 **synthetic** 的 20K/23K/27K/30K 四路并发、每路64输出 token 测量 TTFT/TPOT/E2E 与吞吐，确认两轮资源释放，再按冻结的速度比值决定是否进入正式服务。终端仅打印少量阶段/资源状态、故障与最多8行配对 `PERF_*` 摘要；完整输出仍在本轮 `logs/`。只需把这些摘要或失败行直接贴回。`bash scripts/probe_concurrency.sh` 是跳过安装/编译/功能门的单命令快路径，也自动跑原生→OSCAR两轮，无需第二条 `--native` 命令。正式服务就绪后等待 `OBSERVER_READY`，再用自己的压测程序发送32并发长请求；被动报告只观测该流量，不生成数据集或请求。单批 synthetic 速度门不是全工况性能验收。详见[服务探针](docs/service_probe.md)。

一键配对中的原生阶段只用于明确的基线测量，OSCAR失败绝不会切换到原生服务。HTTP测量不伪造设备时间或MTP query长度；硬件、数值、显存、profiler证据齐全后才能做最终验收。[性能协议](benchmarks/README.md)与[分相位profiling](docs/profiling.md)给出数据格式和命令。

算子采用独立 `ascendc_library` + 同流 direct-launch 的原生已有工程方式；交付两个共享库和带签名manifest，不安装custom OPP vendor。这一实现方式的差异见[算子设计](docs/ascendc_design.md)。Python wheel本身不含NPU二进制，目标入口首次使用本机CANN编译。扩展保留相邻 `lib` 的运行搜索路径，loader 使用 manifest 中校验过的 kernel 绝对路径加载，不要求手工设置本项目的 `LD_LIBRARY_PATH`（档案 #125）。后续源码、工具链、参数签名及产物指纹一致时复用已有产物，跳过 CMake 配置和编译；有变化或产物损坏时重新构建。

默认旋转为论文明确支持的data-free Hadamard，在NPU生成、绑定模型/PR指纹，不能冒充样本校准。严格artifact加载、样本统计和旋转构造API另在 `rotations.py`；缺层不能静默identity，MTP draft采用档案#86的明确identity策略并告警。模型级质量容差仍须在目标实测前定义。

详细状态：[检查清单](docs/checklist.md)、[全局设计](docs/design.md)、[原生接缝](docs/native_integration.md)、[运行时](docs/runtime_implementation.md)、[参考语义](docs/semantics.md)、[算子清单](docs/operator_inventory.md)、[服务探针](docs/service_probe.md)。`references/` 只读且不入新工程Git；目标部署不依赖它，本地完整native源码契约测试需要这些已提供的参考树。
