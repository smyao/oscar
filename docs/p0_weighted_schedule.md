# P0：q4 长历史按 KV256 权重分配

首轮先完成独立 A/B 判决；node93 的
`observe-20260929T081412.853398Z` 已通过逐位、冻结 oracle、changed-input
图和五组 Event，weighted 中位数 10.7628ms、fast 11.2804ms，五组均更快。
因此 candidate 生产路由现已启用 `attention_cv_fast_weighted_out`。依据档案
#150 的 q1 调度收益与后续 q4 逐核负载复核，调度权重取任务表中的真实
`ceil((kvend-kvbegin)/256)`，而不是把 `kind==0` 写成固定常数。空任务、
坏任务和 window/current 的地址、状态发布、数学及 ABI 均不改变。

## D.4 四问

1. 对应融合 history CV/FIA 内的工作归属，不改变 dequant、QK/PV、softmax、
   merge 或 store 数值路径。
2. D.4 的失败路径曾把完整 History 恢复到 HBM，耗时 6499.8–6655.1ms；
   本项目 #150 又证明，等权 index stride 会让不同 KV 长度的任务集中到少数核，
   wall 由最重核决定。
3. 新符号在 N≤128 的 q4 图形内，由 AIC/AIV 从同一任务表确定性重放
   least-loaded 分配；权重来自实际 KV 区间，工作区公式、输出行、错误状态、
   FP32 数学和外部 INT2 布局完全不变，不增加工作区，也不物化 History。
4. probe 必须记录 20 核的预测 KV256 权重、max/mean，以及旧 fast 与 weighted
   各 2 次预热、5 次 AB/BA 交替 Event。只有 weighted/fast 严格小于 1，且逐位、
   冻结 oracle、merge、同地址 changed-input 图 capture/replay 全通过，P0 才算
   设备侧成立。本次真 NPU 比值 0.954119、预测 max/mean=1.11715；这是
   N128/S3 q4 单算子证据，不外推完整服务吞吐。

## 验收与路由边界

入口仍为：

```bash
git pull --ff-only && bash scripts/install_observe_serve.sh --variant candidate --probe-only
```

`fast-unpack-report.json` 的 `q4_decode32_n128_s3.p0_weighted` 保存权重和 Event
样本。生产选择严格限于已验证的 Hq6/Hkv1/D256、N≤128、max_query_len=4；
candidate 配置须同时开启 history reuse、fast unpack、weighted q4，否则显式失败。
其他 q 长度、维度、GQA 或更大 padding 继续走原已验证 fast/C4/q1 符号，不把
本次证据范围外推。固定直启命令会实际启用本轮优化：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

## P1.5：q4 split 扫描

同一 `32*q4/Hq6/Hkv1/D256/N128` 输入现在额外扫描 S1、S2、S3、S4。
四侧均调用 weighted 算子，逐项检查状态和冻结 oracle，记录实际任务表推导的
20 核 KV256 权重；计时按 2 次预热、5 次正式测量并在正序/逆序间交替，避免
固定调用位置偏置。报告位于 `fast-unpack-report.json.q4_split_scan`，终端摘要为
`PERF_Q4_SPLIT_SCAN`。

扫描不会直接改变生产路由。只有某个非 S3 候选中位数最低且五次均严格快于
对应的 S3 样本，才报告 `promote_sN_after_graph_gate`；否则报告 `retain_s3`。
即使出现稳定候选，也必须先为胜出 split 补同地址 changed-input 图门，才允许
修改正式服务的 split 策略。
