# 32K timer 首轮报错修复

本次报错是诊断脚本的两处缺陷。用户日志对应 `fce13f9d`（现场报告dirty=true），
原生BS4的control和profile长批均完成，随后128→1收尾请求的统计中断；OSCAR服务未开始。
原文保存于 [target_failure.txt](../reports/timer_capture_parse_fix_20261019/target_failure.txt)。

## 已确认的问题与处理

1. **单token没有TPOT，空样本median崩溃。** SSE客户端正确返回 `tpot_ms=null`，但
   `benchmarks.mixed.run_batch` 过滤null后仍调用 `statistics.median([])`。
   修复为空指标返回 `p50/p95=null, sample_count=0`；真实0ms仍统计，缺失值不伪造。
   flush异常另记failed/error，不再遗留running。
2. **daemon worker不能解析profiler数据。** 官方 `NpuProfiler.analyse` 明确拒绝daemon
   进程；此前worker中的同步callback虽保存raw，却未产生设备任务表。
   worker现在使用 `analyse_flag=False`，只采集并标记 `captured_pending_analysis`。
   两个服务退出且资源释放后，一键脚本自动启动普通Python子进程做离线解析。
   有界600秒、输出落盘；实际非空且包含有效设备任务的CSV/trace才允许标记complete。
   SDK仅返回None/0、info-only或空表均不算解析成功，PMU与rank覆盖仍由最终摘要逐项验证。

离线阶段不加载模型、不发送请求，也不计入任何control/profile批次wall。
原始step记录保持当时的pending事实；解析更新window及run-summary，保留先前失败状态。
原生图、MTP、BF16、三组BS4/4/8、32K×256和模型配置均未改变。

```bash
git pull --ff-only && bash scripts/test_timer_32k.sh
```

仍只需复制 `OSCAR_TIMER_SUMMARY_BEGIN` 到 `OSCAR_TIMER_SUMMARY_END`，
最多200行/50KiB。新增OFFLINE_ANALYSIS/错误摘要，无须手工调用SDK或翻设备trace。

## 本次已有证据与边界

- 用户日志：原生BS4 control完成4/4，输入131072、输出1024，wall约26.2秒、输出39.08 token/s，preemption为0。
- profile长批也完成4/4，wall约29.83秒；它是插桩轮，不能拿来替代control。
- 四rank的采样窗口覆盖已达到，目标decode B4也出现；但当时设备任务数均为0，不能从host包络推断NPU瓶颈。
- OSCAR尚未运行，没有本轮native/OSCAR比值，更没有compute/HBM-bound结论。
- 105项主机回归通过：真实localhost SSE单token及完整客户端流程、真实spawn daemon不解析、普通进程创建解析子进程、有效导出/静默失败/空表、同一键入口清理与汇总。
- 没有连接真机、运行NPU模型或性能测试；没有改内核、精度或调度。

官方API核对来源：
[profiler callback与analyse入口](https://github.com/Ascend/pytorch/blob/master/torch_npu/profiler/profiler.py)、
[daemon限制及离线解析实现](https://github.com/Ascend/pytorch/blob/master/torch_npu/profiler/analysis/_npu_profiler.py)。
源码hash与验证回执见 [summary.json](../reports/timer_capture_parse_fix_20261019/summary.json)。
