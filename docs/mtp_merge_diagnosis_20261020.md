# 2026-10-20 later-MTP merge 故障定位

一键收集入口：`bash scripts/diagnose_bf16.sh`，默认复用本次失败目录的有效配置；
可追加另一次运行的日志目录。终端实时显示探针输出，另一个终端用
`tail -f logs/bf16_diag_latest/bf16-probe.log` 查看，
结束后用 `cat logs/bf16_diag_latest/summary.txt` 查看退出码及压缩包位置。
失败仍收集已有结果及原服务日志，不重启服务、不改配置、不绕过精度门。
脚本通过共用phase管理自己的子进程，900秒截止；超时不是设备通过。

档案 #157/#165/#167/#169；本地母源 `157ba09`，目标提交未提供。结论是**故障族已定位，产生坏值的位置未确定，尚未修复**。

## 已确定的事实

用户回传三个日志片段和运行 `20261020T064430.498396Z` 的状态。片段原字节、长度和 SHA256 见[回执](../reports/mtp_merge_diagnosis_20261020.json)，不能当成完整 serve.log。

首个明确归因为：

```text
OSCAR status guard trapped at site=forward:mtp.layers.0.self_attn.attn family=merge code=2
```

调用栈经过原生 `_run_merged_draft` 和 `_compact_call` 的实际 `original(model, **call)` 行，确认是后续 draft 且执行了 compact。诊断配置为 eager、无图；因此不能将本轮归因于 graph replay。

冻结 `merge_lse.cpp` 中 code2 的两条来源是：

1. 输入 partial LSE 为 NaN/+Inf 等非法值；合法的空源 `-Inf` 不报错。
2. 合并输出的 FP32 ReduceSum 非有限；可能是输出含 NaN/Inf，也可能是有限元素相加溢出。

这不是 slot 错误码，也不能证明错误始于 merge 内核。`guard_diag.py` 先同步已有工作，再按 attention、rotate、merge、store 检查；此次前两个族通过，但尚未检查 store 族。CV 的 PV 累加、逆旋转读回和最终 partial 发布并没有完整的结果有限性校验，因此前两个族通过不能推出 partial 正常。

## 主机和内核审计

compact 将输入行数缩为 B，保留原 N 给 split 选择。实际进入 compact 意味着原 N≥1024；当前 Hq6/Hkv1、至多32 Cube 的配置下得到 S1。`runtime.partial_views` 展平后截取 B 对应的连续头部，merge 使用相同的 B/H/S；没有发现误读原大 N 尾部的证据。

`MTP_DISCARD_BRIDGE requests=1` 是一次性日志，不能证明错误步仍为 B1。scheduler dump 的8232同样不能直接当成出错算子的 N。dump 当时一个运行请求、KV占用约1.98%，不支持“大 BS 显存容量不足”的判断。

q1 活行由 leader 发布，padding 经 PublishEmpty 写零和 `-Inf`。M6 逆旋转虽然读回更大的固定 Vector 块，最终只发布活行。源码中的读回在 kRotReady 和 MTE2_V 之后；本轮未复现可证明的漏写、越界或缺失同步。SDK Matmul 的真实异步行为、dtype 切换及设备上的坏值仍未核实。

## 本地验证与交付边界

执行：

```bash
.venv/bin/python -m pytest -q tests/test_guard_diag.py tests/test_compact_later_mtp.py tests/test_bf16_history_pipeline.py --junitxml=reports/mtp_merge_host_20261020.xml
```

结果68通过、1.84秒；这是主机契约回归，不是设备数值或服务通过。FE0七文件逐一哈希一致，精度门及独立 oracle 不变。既有 BF16 数值7/11失败也不因此消失。

本轮不修改生产代码、候选开关、默认启动命令或冻结 guard，不以跳过检查、清零异常值或关闭 MTP 掩盖问题。服务监督器已正确将原始进程 rc0 + native fatal 判成有效 rc1；没有重现假成功。

后续决定性证据是 merge 前三源的 partial/LSE，以及对应当前 B、原 N、split 和实际算子：先区分非法 LSE、非有限 partial 与 merge 归约异常，再定位 PV 或逆旋转。现有日志不包含这些值，本机没有 CANN/NPU，不能据此声称数值故障已经修复；不要求重跑 AISBench，也不要求重复现有同粒度 guard 启动来获取相同报错。
