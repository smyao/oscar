# 2026-10-19 请求期 Trap 与服务退出状态修复

本轮基于 `19d157a`；用户确认目标机包含最新代码，未提供目标 SHA。对应真机档案 #167。本轮修复服务故障被记为成功退出的问题；**NPU Trap 的具体来源仍未确定**，没有据异步错误栈猜改 Attention 算子。

## 日志与源码能证明的范围

原附件逐字节保存于 `reports/target_runtime_trap_20261019.txt`（65537 bytes，SHA256 `834fec8944f9ffc4c247195c85e565c278c2641368618042be9a9d1522ed82b1`）。片段在 scheduler stats 中途截断，不包含完整退出日志。

| 证据 | 结论与边界 |
|---|---|
| 06:21:25 空闲29.12/29.49GiB；06:22:06 Application startup complete | 本次越过上次 #166 的初始化内存不足；不能将两次错误混为一谈。 |
| POST HTTP 200，随后06:39:20 vector exception 507035 | HTTP响应头不能证明生成已完成；当前失败属于请求执行期。 |
| 各rank可见PC差0x264，mask `0x6500020bd00028c` | 与历史guard特征一致，但缺当前符号绑定与site/family/code，不能唯一归因。 |
| `acl_graph.py:256`同步报错，`:257`才调用replay | 本次图回放尚未提交，错误可以来自此前设备工作；“Replaying aclgraph”在同步前打印。 |
| dump中computed16384、本轮8232、无spec token | 这个批次大于capture上限512，原生dispatcher必选NONE；若执行OSCAR历史分支，会选BF16 C16。它不能等同同步处正在进入的小批次图。 |
| `EngineCore.step_with_batch_queue`先提交新批次，再从队列取最老future收结果 | dump绑定被等待future的scheduler，不保证是worker此刻执行的批次。附件不足以判断最先失败的是历史prefill、首draft、后续draft或其他已提交工作。 |
| 06:39:32 `Worker proc … died unexpectedly`，外层却rc0/stopped | 已有明确内部故障证据，不能只按API进程rc0判服务正常停止。 |

以上引用只读原生实现：`references/vllm-ascend/vllm_ascend/compilation/acl_graph.py:243–257`、`references/vllm/vllm/v1/cudagraph_dispatcher.py:276–285`、`references/vllm/vllm/v1/engine/core.py:512–564`。

## 修复内容

原生 `launcher.terminate_if_errored` 在引擎错误时可只设置 `server.should_exit=True`；`serve_http`正常返回后，CLI可返回0。`multiproc_executor.monitor_workers` 在非主动关闭状态下检测worker死亡，才打印这次明确的fatal记录。

`tools.serve_process`现在向`run_phase`传入仅用于正式服务的结束判定器。它流式读取本次日志，精确匹配原生文件来源、ERROR格式和完整的worker/engine终止消息；普通请求错误、任意Traceback、TBE EOF或关闭期SIGTERM提示不满足条件。先固定日志字节边界，再清理本次进程组，避免把监督器自己终止残留worker产生的日志倒算为原始故障。

结果明确分为两个字段：`process_returncode`保留实际子进程退出码；`returncode`用于脚本/服务结果。原始0但有上述fatal证据时，后者为1，`status=failed`，并附`failure`证据；原始非零、超时、用户信号的有效退出语义保留。读取分类日志失败也明确失败。终端和相位JSON在最终判定后才输出，不先打印PASSED再更正。

额外成本仅为服务结束时一次线性日志读取，每次最多64KiB；没有新增在线设备同步、NPU回读、请求、Attention计算或显存。BF16/INT2、旋转、sink/recent、MTP、图选择、候选默认、设备/端口和原生源码均未修改。此判定不是在线健康监控，也不能覆盖未来未知的原生日志格式。

## 本地验证与保留边界

- 旧代码真实子进程反例：打印这次原生fatal记录后exit0，旧代码确实输出PASSED，保存在 `reports/service_fatal_false_success_before_20261019.log`。
- 生命周期、部署、实时日志和诊断回归：179 passed，`reports/service_fatal_outcome_after_20261019.xml`。
- 候选入口、BF16 q4/q1/C4/C16路由及原生异步边界：152 passed，`reports/runtime_trap_routes_20261019.xml`；与上一套存在重叠，不相加为独立用例数。
- 真实native AST执行验证sync异常时replay未执行、8232超过图上限、dump等待最老future、诊断主eager同时阻止draft图。算术/设备队列未模拟。
- 8232-token边界专项4 passed：真实任务生成/C16调度在8232与8256 padding下覆盖23簇和24个独立历史组；真实producer→LoadQueries连接验证末组21 queries、126活行及checksum/padding，旧D步长负控失败。`reports/runtime_8232_contract_20261019.xml`仅支持主机地址/owner/控制流，不验证SDK算术或异步事件完成。
- FE0七文件字节一致，冻结 `configs/acceptance.json` 不变。CANN/NPU/图完成/模型质量/性能未在本机验证；已有BF16数值集7/11失败（含2例最终输出失败）及 #165/#167未归因Trap保留，不以主机测试消除这些欠账。

重现命令以汇总回执 `reports/runtime_trap_diagnosis_20261019.json` 为准。

## 目标机最小定位

已有诊断入口同时开启主模型eager和`OSCAR_GUARD_HOST_SYNC=1`，原生MTP也不会选择图，不需要更改冻结guard。同步本轮代码后运行：

```bash
bash scripts/install_serve.sh --variant candidate --rear-cards --diagnose-trap --log-dir logs/trap-20261019-diagnostic
```

就绪后仅发送一条原先触发错误的请求，最大输出设16 tokens，并发1。使用完整原请求保留输入特征；不要求重跑AISBench。这是归因运行，不作性能验收。若失败，优先保留首条完整的 `OSCAR status guard trapped ... site=... family=... code=...` 或 `OSCAR device failure preceded status guard ...` 异常链。

将该目录里的 `serve.log`、`serve.json`、`status.json`、`effective-target.json` 作为文件发送；本次粘贴有64KiB截断，继续截取尾部可能丢掉最初归因信息。正常启动仍用 `bash scripts/install_serve.sh --variant candidate --rear-cards`，不会默认开启诊断同步。
