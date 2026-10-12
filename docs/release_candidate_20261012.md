# 2026-10-12 一键候选交付

按用户指令停止新增检查和验证，交付已完成关键本地验证的候选。

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

观察入口使用相同候选逻辑：

```bash
git pull --ff-only && bash scripts/install_observe_serve.sh --variant candidate
```

- 默认显式开启 prefix caching，使用物理页精确快照；未启用请求私有窗口研究布局。
- 省略 `--max-num-batched-tokens`，保持原生 GDN、调度预算及默认0–3卡/8989；直启可叠加 `--rear-cards` 使用4–7卡/7878。
- 启用合格量化/旋转组合和 MDL q4 数据装载；后续MTP q1及C4沿用原路径。
- MDL完整本地配对1079788→931042 ticks，1.1597629倍；CPU与CAModel逐位/oracle通过；完整40项CANN/Torch构建通过。此为局部算子结果，不是整轮2–4倍验收。
- Prefix及启动接缝已完成主机验证，真实NPU prefix、模型质量和整轮速度尚待目标机认证。
- direct入口不跑probe。observe保留真实NPU/graph/速度门，失败阻止服务。
- 未完成的direct-NZ与C4 MDL不进入默认候选；C4 member pipeline仅显式research，默认关闭。
