# OSCAR 分相位真实 profiling

档案 #70–73、启动文档 D.4 是本入口的判据：历史prepare约725ms是host时间；dequant约6500ms、FIA约18.7ms、store约209ms才是设备时间。不能拿快速host launch或HTTP成功证明设备吞吐。

`oscar_ascend.timing.phase(name, **host_fields)` 包住 `prepare / rotate / fia / merge / phase1_stores / status_guard` 调用；`history_window` 是融合CV读kernel的别名，解析时归入fia。默认 `OSCAR_TIMING=0`、`OSCAR_PROFILER=0` 返回复用nullcontext，不import torch、不读时钟、不创建NPU event、不同步stream。标签只接受现有host标量，拒绝Tensor以避免隐式在线回传。

显式 `OSCAR_TIMING=1` 输出 `oscar-timing` JSON host相位日志；其中`device_ms=null`、`device_evidence=requires_npu_trace`，永不把host launch时间改名device时间。`OSCAR_PROFILER=1` 开启record_function标签；相位标签本身不启动/停止全局profiler。

服务沿用原生vLLM Ascend `TorchNPUProfilerWrapper` 的启动/停止方式与torch profiler配置；只在需要的实际请求窗口收集。不得为了测量改变 `FULL_DECODE_ONLY`、MTP、TP4、异步调度或量化配置。调用链中的标签形如 `oscar::fia::{"layer":"...","tokens":...}`；graph capture里的host标签不等于后续graph replay执行证据，必须看实际硬件kernel事件。

独立NPU probe可使用显式会话：

```python
from oscar_ascend.timing import profile_session, phase
with profile_session("reports/npu_profile", "worker-rank0"):
    with phase("fia", tokens=4, layer="model.layers.3.self_attn.attn"):
        run_existing_cv_probe()
```

该会话要求 `OSCAR_PROFILER=1`、`configs/target.json` 当前任务devices已选择，且`ASCEND_RT_VISIBLE_DEVICES`完全一致。使用原生同栈 `torch_npu.profiler.profile` 的CPU+NPU活动、Level1与PipeUtilization，并采内存。显式stop/export的profiler自身开销只存在于测量窗口；没有向默认生产路径增加同步。无NPU或设备尚未选择会落`state=not_run`并抛`ProfilingUnavailable`，不会继续执行CPU替代工作。

真实trace生成后：

```bash
python -m tools.summarize_profile \
  reports/npu_profile/<worker>/ASCEND_PROFILER_OUTPUT/trace_view.json \
  --rank 0 --output reports/oscar_profile_rank0.json \
  --jsonl logs/oscar-timing-rank0.jsonl --require-complete
```

传入明确文件路径，不自动捞取可能陈旧的其他实验trace。支持gzip JSON。默认Chrome trace时间单位为us；若导出方明确使用ns，传`--time-unit ns`。`displayTimeUnit`只是查看器格式，不用于暗中改换timestamp单位。无输入trace时写`status=not_run`、device时间与恢复bytes都是null，退出码2。

`tools/summarize_profile.py` 仅统计有NPU硬件process/设备属性/AI_CORE等明确证据的完整device事件。CPU包装函数即使同名`oscar_attention_cv_kernel`也不计设备时间。通过下列证据归因：

1. 可识别真实device kernel名：prepare、rotation、CV、merge、store、guard；graph replay没有重新执行Python时仍能归因。
2. trace显式flow将host record_function连接到device kernel。
3. 唯一明确的correlation id连接host launch scope与device kernel。

没有关联证据的opaque kernel不按时间接近猜测相位。每phase分别输出device busy区间并集、kernel时长之和、kernel数、p50/p95、host scope总时长和归因方法。并集避免重叠Cube/Vector任务重复计时；跨多个硬件process时分别求并集再相加。整份trace按rank独立保留，不能拿TP多卡总和同单卡D.4数字比较。

`--require-complete` 要求真实NPU事件、prepare/rotate/fia/merge/store均有device证据、没有未归因的OSCAR核或独立全历史restore核。通过只代表trace归因完整；同配置原生基线、每case数值验收、32K/50K性能、MTP接受率和内存收益仍是独立门。

**dequant的零字节结论不能伪造。** 源码没有独立全历史恢复路径，CV固定tile通信容量与L无关；但A2固定GM通信总流量仍为Θ(LD)。trace若出现`full_dequant/history_dequant/full_history_restore`会明确标`forbidden_kernel_observed`。只看Chrome duration trace没有独立restore核，至多证明`no_standalone_restore_kernel_observed`；`full_history_restore_bytes`仍为null，因为该格式没有足够的byte计数。要证明0 bytes，必须补同轮device内存/指针范围/流量证据，不能凭源码硬填0，也不能把未观察相位写成`device_ms=0`。

`tests/test_profile_evidence.py` 的合成trace带`oscar_test_fixture=true`，永远不能通过真实设备归因门。测试覆盖725ms host与6500ms device区分、CV overlap、flow/correlation、独立restore识别、NPU缺失、坏duration与默认零event/sync路径。测试结果不生成NPU性能结论。
