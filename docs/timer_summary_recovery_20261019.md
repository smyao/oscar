# 从已有32K日志恢复timer摘要

本轮修复最后汇总时的 `can only concatenate str (not "float") to str`。
用户日志已推进到native/candidate请求结束、服务清理及离线解析之后；本次优先读取
已经保存的导出文件，不要求重新运行模型。

```bash
git pull --ff-only && bash scripts/test_timer_32k.sh --summarize-only /workspace/gpt_new_oscar_kimi/logs/timer32k-20261019T092907.267068Z
```

此模式不读取模型配置、不安装/编译、不拉起服务、不发送请求、不导入torch_npu，
也不会再调用已经耗时约360秒的SDK原始数据解析。它重新检查现有CSV/trace、修复仅由
摘要解析器造成的窗口状态误判，再输出不超过200行/50KiB的断点摘要。
仍复制 `OSCAR_TIMER_SUMMARY_BEGIN` 到 `OSCAR_TIMER_SUMMARY_END` 即可。

## 根因与修复

官方torch_npu `TraceEventManager.create_x_event` 本来就用
`convert_ns2us_str(event.ts)` 导出时间戳、`convert_ns2us_float(event.dur)` 导出时长。
旧代码用转换后的值检查有效性，却在scope包含关系中仍拿原始字符串与数值相加。
这不是用户环境问题，是本工程没有按官方序列化格式处理。

现在在解析入口统一规范化有限时间值和标识符，后续比较/排序/端点运算只使用规范化
数值；支持空白和科学计数法。非法字段逐类报告缺失，不将CPU事件或坏记录伪填成
设备耗时。CPU/NPU flow采用官方 `torch_to_npu` / `HostToDevice` 的id与精确端点，
将BF16旋转等host scope关联到对应设备kernel；有歧义时保留未归因。

断点输出包括 `HOST_EXCLUSIVE`、`DEVICE_PHASES`、`PHASE_PMU`、`TOP`、
`COMPARE` 和 `DEVICE_COMPARE`；按rank/window与实际形状区分，不把四rank求和为wall，
不虚构融合CV内部子阶段。

## 旧数据保护

- 原始CSV、trace、step JSONL和请求测量不改动。
- 原run、离线状态和已有摘要先保存在同目录 `.before-summary-recovery` 文件；重复恢复不覆盖这份原件。
- 只有“捕获成功、后续解析出错”的状态可依据有效导出恢复。模型/捕获失败及未完成采样不能升级为通过。
- `summary-recovery.json` 记录本次恢复结果；旧错误保留为历史。
- 若实际没有设备导出，仍输出已有断点与缺失原因并返回2，不伪造成功数据。

## 验证

113项针对性主机回归通过。包含原错误最小复现、官方字符串时间戳/浮点时长类型矩阵、
混合CSV字段、官方flow、非法记录、三组×四rank×四窗口（48窗口）的完整保存目录、
真实CLI输出及重复恢复；验证断点/PMU/热点/对照均能打印且原始导出字节不变。
恢复测试显式禁止模型、HTTP和SDK解析调用。另有2项未改动的SSE测试本轮未重复。

没有连接真机或重新测速。测试使用合成的官方格式文件，不能代替用户实际计时数据；
收到恢复后的摘要后才分析OSCAR热点与计算/访存瓶颈。

官方格式依据：[Ascend PyTorch TraceEventManager](https://github.com/Ascend/pytorch/blob/master/torch_npu/profiler/analysis/prof_common_func/_trace_event_manager.py)。
验证记录见 [summary.json](../reports/timer_summary_recovery_20261019/summary.json)。
