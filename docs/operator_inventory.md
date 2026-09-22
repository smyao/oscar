# 算子清单与验证状态

命名空间 `torch.ops.oscar_ascend_ops`，所有算子采用调用方预分配输出，无CPU数值执行路线。七个符号均已通过VM中的CANN9.1/Ascend910B4编译与链接。同一kernel body的官方CPU调试结果在 `reports/vm/guest/reports/`；这些不等于NPU验收。

| 符号 | 契约/阶段 | 实现 | 验证 |
|---|---|---|---|
|store_int2_out|FP32[N,H,D]→136B/head/token(D256)，fp16 metadata，SSM SoA scatter|store_int2.cpp|D64/128/256，i32/i64，页边界/负slot/越界，CPU-debug逐位 |
|merge_lse_out|FP32 partial[R,S,D]+LSE[R,S]→FP32输出/LSE，所有空行写0/-inf|merge_lse.cpp|S1/3/128、poison空split，CPU-debug；修复真实Log禁止inplace问题 |
|rotate_out|BF16/FP16/FP32当前query→FP32旋转结果，原始/逆旋转均由R转置契约决定|rotate_clip_store.cpp|Hadamard蝶形与dense FP32，CPU-debug |
|rotate_clip_store_out|新K/V→旋转、精确percentile clip、quant/pack、BF16 page snapshot和tag|rotate_clip_store.cpp|26种旋转/融合写用例：精确码/meta/guard/tag；非法行不发布 |
|prepare_attention_tasks_out|device qstarts/seq_lens/slots→固定tasks/positions；后续draft从BT恢复真实位置|attention_tasks.cpp|dummy、padding孔洞、stale MTP seq_len、重复/缺失页，CPU-debug |
|attention_cv_out|INT2 history、BF16窗口/current→FP32分段输出/LSE；Cube QK/PV/逆旋转，Vector解包/softmax|attention_cv.cpp|Q1/2/3/4/17，GQA6、Hkv2、mixed batch、物理页置换、bad tags/metadata/NaN，CPU-debug |
|status_guard|同stream合并检查四组int32状态，错误触发AscendC::Trap，无D2H|status_guard.cpp|真实CPU-debug正例与故意坏status触发Trap负例 |

生产forward为 prepare→query rotate→三段CV→LSE merge→新KV融合写→status guard。先读取旧窗口，再写当前chunk；不恢复全历史到HBM，不重复量化旧chunk。窗口snapshot位于原生padding，随物理页共享/回收。所有dims、dtype、stride、capacity和输入输出alias由C++与metadata契约校验。

CV固定GM通信为每Cube `(256*D+4096)*4` B；Mq64同时容纳目标GQA6×MTP4=24行，历史tile跨query复用。A2的Vector→Cube需要此有界通信，总流量仍线性；不能宣称零HBM解包流量或H18字面亚线性。

每个kernel头与[CV设计](cv_implementation.md)、[旋转设计](rotation_pipeline.md)、[主设计](design.md)均有D.4四问。性能目标未实测，不从压缩比或CPU调试时长推导NPU吞吐。

独立NPU gates：`tools.probe_ops`、`tests/test_cv_contracts.py`、`tests/test_rotation_npu.py`（显式opt-in），随后完整TP4服务probe。NPU、graph capture/replay、32K/50K性能和模型精度分别保持未运行直到实际证据到位。
