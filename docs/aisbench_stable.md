# AISBench stable 启动策略

2026-10-13：同步 `codex/oscar-ascend` 的 `488d5542` 全部已提交逻辑。

```bash
git switch codex/aisbench-stable && git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

直启和 observe 共用候选配置，默认显式 `--no-enable-prefix-caching --max-num-batched-tokens 16384`。设备0–3、端口8989；直启 `--rear-cards` 仍为4–7、7878。原生调度器不变，未启用诊断同步或 eager。关闭 APC 不是已修复此前 Trap 的证明，本分支名称也不代表新组合已获真机认证。

## 采样边界适配（2026-10-14）

两个一键入口自动加载外部采样边界修复：greedy输出按请求隔离，累计draft计数视图带32字节前导零区。启动仅打印一次/worker `PERF_SAMPLER_FIX`。正常命令无需 `--diagnose-trap`，默认仍关闭prefix caching、预算16384；显式开启prefix caching也加载同一适配。

本地已对拍原生采样规则和后续MTP位置，保留随机算法、RNG与严格guard；源码执行模型复现code5并验证适配后通过。真实NPU与两次完整服务故障消失未在本地验证，不能用主机结果冒充真机验收。
