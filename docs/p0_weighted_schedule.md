# P0：q4 长历史按 KV256 权重分配

本轮只实现独立 A/B 判决，不切换生产路由。现有
`attention_cv_fast_out` 继续服务；新
`attention_cv_fast_weighted_out` 仅由 `probe_fast_unpack` 调用。依据档案
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
   设备侧成立。当前无真 NPU 结果，不宣称速度收益。

## 验收与路由边界

入口仍为：

```bash
git pull --ff-only && bash scripts/install_observe_serve.sh --variant candidate --probe-only
```

`fast-unpack-report.json` 的 `q4_decode32_n128_s3.p0_weighted` 保存权重和 Event
样本；缺符号、缺 20 核权重、任一精度/图门失败或 Event 无严格改善都会使相位
失败。probe 通过之前不把新符号加入 `cv_dispatch.py`，因此普通
`install_serve.sh --variant candidate` 只会编译它，不会启用未经测量的调度。
