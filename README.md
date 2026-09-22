# OSCAR Ascend 外部适配工程

已实现 FULL 层 INT2 缓存、真正的 Cube/Vector attention、融合旋转/裁剪/写入、精确窗口、MTP 位置修正、固定图缓冲、外部插件及一键安装/探针/服务流程。GDN 使用原生状态和 reshape。新增代码不覆盖原生 vLLM/Ascend 源文件。

**验证边界：CANN 9.1.0 / Ascend910B4 编译和官方 CPU 调试已通过；真实 NPU、图捕获/回放、模型质量、实际显存收益及性能尚未验收。** VM 没有 NPU 设备或模型权重，不能把编译和 CPU 调试当作这些结果。H18 的严格亚线性历史读取与精确全注意力存在冲突，当前实现线性读取 INT2、固定大小 tile 通信，不生成全历史 BF16 副本。

本地/VM 检查：

```bash
.venv/bin/python -m pytest -q
bash scripts/validate_vm.sh
```

第二条在运行中的 Lima `oscar` VM 的独立 `/home/sunao2000.linux/gpt_new_oscar` 目录编译、链接，并运行同一份 AscendC kernel 的官方 CPU 调试器。首次配置可加 `--bootstrap`，仅安装该目录独立虚拟环境的依赖；不访问 VM 中旧项目代码。报告在 [reports/vm/validation.json](reports/vm/validation.json)，最新一轮共61个算子用例通过，另含MTP/padding元数据检查。

目标 NPU 环境先在 `configs/target.json` 明确本次物理设备号（当前为null，防止继承其他任务设备），并核对模型路径。完整入口：

```bash
bash scripts/install_probe_serve.sh
```

流程为：环境记录 → 源码检查 → 安装构建依赖/插件 → 编译算子 → 二进制签名检查 → 真 NPU 算子对拍 → 旋转artifact准备 → TP4/MTP/长输入/图服务探针 → 所属进程组及NPU资源回收 → 原生完整性复核 → 正式服务并校验真实请求。任何阶段失败都保留日志和退出码，不改走原生 FULL 或 CPU。该流程已做本地故障注入测试，尚未在目标 NPU 全程执行。

不启动设备即可查看计划：

```bash
bash scripts/install_probe_serve.sh --plan
python3 -m tools.target_cli --print-command
```

单独运行环境/构建/算子/服务阶段：

```bash
python3 -m tools.deploy --only environment
python3 -m tools.deploy --only build-ops
python3 -m tools.deploy --only probe-ops
python3 -m tools.service_probe --config configs/target.json --output reports/service.json --log-dir logs/service
```

显式原生基线与测量：

```bash
python3 -m tools.target_cli --native --config configs/target.json
python3 -m benchmarks.measure --help
python3 -m benchmarks.compare --help
```

原生基线仅由 `--native` 明确选择，绝不由 OSCAR 失败自动触发。HTTP测量不伪造设备时间或MTP query长度；硬件、数值、显存、profiler证据齐全后才能做最终验收。[性能协议](benchmarks/README.md)与[分相位profiling](docs/profiling.md)给出数据格式和命令。

算子采用独立 `ascendc_library` + 同流 direct-launch 的原生已有工程方式；交付两个共享库和带签名manifest，不安装custom OPP vendor。这一实现方式的差异见[算子设计](docs/ascendc_design.md)。Python wheel本身不含NPU二进制，目标入口会在本机CANN重新编译。

默认旋转为论文明确支持的data-free Hadamard，在NPU生成、绑定模型/PR指纹，不能冒充样本校准。严格artifact加载、样本统计和旋转构造API另在 `rotations.py`；缺层不能静默identity，MTP draft采用档案#86的明确identity策略并告警。模型级质量容差仍须在目标实测前定义。

详细状态：[检查清单](docs/checklist.md)、[全局设计](docs/design.md)、[原生接缝](docs/native_integration.md)、[运行时](docs/runtime_implementation.md)、[参考语义](docs/semantics.md)、[算子清单](docs/operator_inventory.md)、[服务探针](docs/service_probe.md)。`references/` 只读且不入新工程Git；目标部署不依赖它，本地完整native源码契约测试需要这些已提供的参考树。
