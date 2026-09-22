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

**最新真机结果：已进入TP4/MTP模型加载，止于关闭前缀缓存时的KV布局初始化（#127）。** 修复区分GDN请求状态块与FULL物理页：保留原生GDN块长262144和none模式，FULL页按SSM字节容量取2816 token；原生源码、启动参数和探针不变。此前75项store/merge原语NPU探针已有直接通过记录；本次日志没有前序CV/旋转探针明细，完整服务、图及性能仍待真机复验。[最新原文](reports/target_uncached_layout_failure.txt)。CANN编译及官方CPU调试已有通过记录，不能替代真实模型验收。H18严格亚线性历史读取与精确全注意力存在冲突，当前实现线性读取INT2、固定大小tile通信，不生成全历史BF16副本。

以下仅用于本地开发，不是 node93 的部署步骤。已配置本项目 `.venv` 的开发机可运行 CPU 测试：

```bash
.venv/bin/python -m pytest -q
```

Mac 宿主机安装 Lima 且 `oscar` VM 已启动时，可运行：

```bash
bash scripts/validate_vm.sh
```

此命令在 Lima `oscar` VM 的独立 `/home/sunao2000.linux/gpt_new_oscar` 目录编译、链接，并运行同一份 AscendC kernel 的官方 CPU 调试器。首次配置可加 `--bootstrap`，仅安装该目录独立虚拟环境的依赖；不访问 VM 中旧项目代码。报告在 [reports/vm/validation.json](reports/vm/validation.json)，最新一轮共61个算子用例通过，另含 MTP/padding 元数据检查。

显式原生基线与测量：

```bash
python3 -m tools.target_cli --native --config configs/target.json
python3 -m benchmarks.measure --help
python3 -m benchmarks.compare --help
```

原生基线仅由 `--native` 明确选择，绝不由 OSCAR 失败自动触发。HTTP测量不伪造设备时间或MTP query长度；硬件、数值、显存、profiler证据齐全后才能做最终验收。[性能协议](benchmarks/README.md)与[分相位profiling](docs/profiling.md)给出数据格式和命令。

算子采用独立 `ascendc_library` + 同流 direct-launch 的原生已有工程方式；交付两个共享库和带签名manifest，不安装custom OPP vendor。这一实现方式的差异见[算子设计](docs/ascendc_design.md)。Python wheel本身不含NPU二进制，目标入口首次使用本机CANN编译。扩展保留相邻 `lib` 的运行搜索路径，loader 使用 manifest 中校验过的 kernel 绝对路径加载，不要求手工设置本项目的 `LD_LIBRARY_PATH`（档案 #125）。后续源码、工具链、参数签名及产物指纹一致时复用已有产物，跳过 CMake 配置和编译；有变化或产物损坏时重新构建。

默认旋转为论文明确支持的data-free Hadamard，在NPU生成、绑定模型/PR指纹，不能冒充样本校准。严格artifact加载、样本统计和旋转构造API另在 `rotations.py`；缺层不能静默identity，MTP draft采用档案#86的明确identity策略并告警。模型级质量容差仍须在目标实测前定义。

详细状态：[检查清单](docs/checklist.md)、[全局设计](docs/design.md)、[原生接缝](docs/native_integration.md)、[运行时](docs/runtime_implementation.md)、[参考语义](docs/semantics.md)、[算子清单](docs/operator_inventory.md)、[服务探针](docs/service_probe.md)。`references/` 只读且不入新工程Git；目标部署不依赖它，本地完整native源码契约测试需要这些已提供的参考树。
