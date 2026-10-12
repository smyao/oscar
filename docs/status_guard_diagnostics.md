# status_guard 首错诊断（档案 #157）

## 手工复制的小日志

对于 `logs/npu-check-时间/` 目录（含 `service-report.json`、
`diagnostic-report.json`、`service/server.log` 和 `build/`），执行：

```bash
python3 tools/compact_failure_log.py logs/npu-check-时间
cat logs/npu-check-时间/key-errors.txt
```

省略目录时读取 `logs/latest-npu-check.txt`。默认最多 6000 字符，优先保留
阶段结果、首个设备状态、错误摘录和少量日志末尾；保留来源及行号。
合成诊断的预期 Trap 会单独提示，缺失/不完整报告不视为成功。
该脚本只依赖 Python 标准库，可以单独传到真机执行，不改变服务、配置、
算子或原始日志，不读取请求响应文件。已存在输出文件不会覆盖，重新提炼时
指定 `--output` 新文件名。摘录可能截断，不能替代完整日志或验收证据。

本轮只增加诊断，不宣称已经修复 507035。2026-10-12 的四 rank plog 把故障符号定位到 `oscar_status_guard_kernel_8+0x258`，但尚无四组状态的实际非零值；也没有该地址的实机反汇编，不能仅凭通用 MTE 文案断言是哪一处访存。

## 启用与测试

默认关闭。`OSCAR_STATUS_DIAGNOSTICS=1` 开启，其他非 0/1 值报错。固定直启入口自动继承该环境变量、重新编译绑定并继续启用完整 candidate：

该开关应在启动前设置；关闭后也必须重启服务，已捕获的图不会被进程外环境变量修改。

```bash
git pull --ff-only && OSCAR_STATUS_DIAGNOSTICS=1 bash scripts/install_serve.sh --variant candidate
```

直启不提供 probe 验收证据。需要测试时先停止原服务，在已通过原脚本安装的真机项目根目录、原 Python 环境执行以下短门：

```bash
git pull --ff-only && \
python3 -m tools.deploy --only build-ops && \
python3 -m tools.probe_status_diagnostics && \
OSCAR_STATUS_DIAGNOSTICS=1 python3 -m tools.service_probe \
  --variant candidate --prefix-only \
  --log-dir logs/prefix-diagnostic --output reports/prefix-diagnostic.json
```

保留当前 target.json 设备/端口（本次实际 4,5,6,7 / 7878），不从其他任务继承卡号。合成诊断仅使用其中第一张可见 NPU，不加载模型；随后 prefix 服务短门仍为 TP4、prefix、MTP、原图模式，保持全部门限和失败即停止。不是 AISBench，不要求重新跑完整 benchmark。

合成诊断为四组各启动一个隔离子进程：健康捕图→两次健康回放→改变原地址上的第 1300 个状态→同图回放。只有实际打印该组 index=1300/code=7、保留原 Trap 导致非零退出、且未超时/中断/清理失败才通过。这里的 code=7 是人工非零注入，不解释为真实服务的错误语义。四次预期设备异常保存在 `logs/status-diagnostic-probe/`；总报告为 `reports/status_diagnostic_probe.json`。无 NPU、编译/ACL callback 不兼容、无首错日志、吞掉 Trap 均失败，不退回 CPU/eager。

## 新日志及边界

- `OSCAR_STATUS_DIAGNOSTICS_ACTIVE`：每个进程的第一条实际设备流回调，确认诊断链已执行。不是模型或图验收证据。
- `OSCAR_FIRST_ERROR { ... }`：每个非零状态组一行，最多四行。包含 PID、逻辑 NPU、层名、guard 阶段、CV 算子、首个错误 index/code、该组非零总数，以及对应 task、slot、position、query 起止、seq_len、虚拟块、预期 slot 和物理页。
- “首个”指每组最低非零索引，不保证它在时间上最早出错；组间输出顺序也不是设备事件的时间顺序。
- 状态分组为 attention/rotate/merge/store。attention 的 index 除 2 得到 task 行、余数为 lane；任务可能覆盖多 token/head，所以记录的是 task leader，不谎称已经定位到 task 内唯一坏元素。其他三组按本地 Q/KV heads 映射 token/head。非 attention 行的 task 仅提供 source0 的代表性请求上下文。
- `current_only_prepare` 的 attention 组是零历史/prepare 的布尔守卫，不是 CV 原始错误码；`current_slots`、`*_padding`、`mixed_boundaries` 同样按阶段解读，不能套用 attention 数学错误码。
- `task[10]` 为 prepare 元数据错误，`task[7]` 为 source（0 历史、1 精确窗口、2 当前），其余字段保持原 16 列，不重新解释或修改生产任务。later MTP 优先用 checked prepare 修正后的 position/context，不用含 rejected tokens 的原生 seq_len 推导位置；slot-context 查找失败时输出未知值 -1。
- `context` 只描述记录的调用点/静态容量，实时请求数据来自设备 packet。没有把 capture 时的主机请求长度、请求数或 draft index 当作 replay 的当前状态。`callsite_invocation` 只是该 callback 的调用次数，不是全局模型步数。

设备上对全部状态归约，包括索引 1024 之后的状态，再通过同一个 mutable 自定义 op 执行 pinned D2H→host callback（立即 flush stderr）→原 `oscar_status_guard_launch`。原 guard/全部 AscendC 计算内核字节不变；既不清零坏状态，也不跳过 Trap。正常路径原样调用旧 guard，不做额外设备算子或 D2H。参考依据仅为只读原生树 `references/vllm-ascend/csrc/torch_binding.cpp` 的 device_print 实现。

每次记录的调用独占 1 KiB pinned packet，并保持到进程退出，避免图回放回调悬空和不同捕获图共享快照。每 worker 最多 65536 份（64 MiB packet，另有上下文/分配器开销），超限显式失败；这是短时诊断，不应长期常开。增加的归约、设备到主机复制和回调会扰动性能，不能用于吞吐对比。

日志同时实时显示并进入既有服务/相位文件，compact 终端模式也不隐藏上述两类消息。服务短门仍收集自有进程 plog、保留真实错误和退出码并清理资源。若更早的设备 fault 已阻止流推进，callback 可能根本执行不到；没有 `OSCAR_FIRST_ERROR` 不代表状态健康，仍需结合 plog，禁止把未打印解释为通过。

本机完整 CPU 回归通过；本机无可用 NPU/CANN、macOS C++ toolchain 也不可用，因此新增绑定编译、ACL callback 图捕获/回放和真实首错输出仍待上述真机短门，不能写成已通过。checkpoint/量化/数学/精度阈值、max_num_batched_tokens 和 candidate 选择不变。
