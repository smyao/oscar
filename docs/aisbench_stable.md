# AISBench stable 启动策略

2026-10-13：同步 `codex/oscar-ascend` 的 `488d5542` 全部已提交逻辑。

```bash
git switch codex/aisbench-stable && git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

直启和 observe 共用候选配置，默认显式 `--no-enable-prefix-caching --max-num-batched-tokens 16384`。设备0–3、端口8989；直启 `--rear-cards` 仍为4–7、7878。原生调度器不变，未启用诊断同步或 eager。关闭 APC 不是已修复此前 Trap 的证明，本分支名称也不代表新组合已获真机认证。

## 显式开启 prefix caching

在 `codex/aisbench-stable` 分支执行：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate --enable-prefix-caching
```

observe 入口使用相同开关：

```bash
git pull --ff-only && bash scripts/install_observe_serve.sh --variant candidate --enable-prefix-caching
```

开关只对本次启动生效，实际传入 vLLM `--enable-prefix-caching`，预算仍为16384。省略开关仍默认关闭；直启可叠加 `--rear-cards`。本地主机已核对两入口的启动参数；开启后的真实NPU服务稳定性尚未确认。

## 采样边界适配（2026-10-14）

两个一键入口自动加载外部采样边界修复：greedy输出按请求隔离，累计draft计数视图带32字节前导零区。启动仅打印一次/worker `PERF_SAMPLER_FIX`。正常命令无需 `--diagnose-trap`，默认仍关闭prefix caching、预算16384；显式开启prefix caching也加载同一适配。

本地已对拍原生采样规则和后续MTP位置，保留随机算法、RNG与严格guard；源码执行模型复现code5并验证适配后通过。真实NPU与两次完整服务故障消失未在本地验证，不能用主机结果冒充真机验收。

## MTP 逻辑缓存位置修复（2026-10-14）

两个入口自动以本轮逻辑长度与原生拒绝数构造后续MTP缓存槽，模型RoPE坐标不变。修复已物化历史与偏移RoPE坐标不一致时的越界/错误回写；纯文本零偏移保持槽位一致。冻结guard、GDN及AscendC算子均未修改。主机反例通过不代表最新APC真机Trap已确认消失，归因边界见档案#161。
