# AISBench stable 启动策略

2026-10-13：同步 `codex/oscar-ascend` 的 `488d5542` 全部已提交逻辑。

```bash
git switch codex/aisbench-stable && git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

直启和 observe 共用候选配置，默认显式 `--no-enable-prefix-caching --max-num-batched-tokens 16384`。设备0–3、端口8989；直启 `--rear-cards` 仍为4–7、7878。原生调度器不变，未启用诊断同步或 eager。关闭 APC 不是已修复此前 Trap 的证明，本分支名称也不代表新组合已获真机认证。
