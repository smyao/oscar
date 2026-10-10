# 2026-10-07 candidate 真机认证交付

档案依据：#129/#140/#142/#145/#148/#151–155，以及启动文档 D.4。按用户最新指令停止扩大本地测速，交付已完成主要数值检查的组合方案；整轮 8 分钟与新版本真 NPU 精度仍由本轮真机结果认证。

## 一键入口

带真实算子数值/图门并启动观察服务：

```bash
git pull --ff-only && bash scripts/install_observe_serve.sh --variant candidate
```

编译安装后直接启动：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

两条命令共用 `tools/serving_variants.py`，实际启用相同组合。直启仍不运行 probe，不能把直启成功视作精度验收。本轮默认物理设备 `4,5,6,7`、端口 `7878`；兼容参数 `--rear-cards` 选择相同设备。启动时一行 `PERF_RUNTIME_CONFIG` 显示实际开关、token 预算和 scheduler。

## 本轮启用

| 路径 | 改动 |
| --- | --- |
| 目标 q4 / 后续 MTP q1 | 原 FP32 历史数学顺序；两块有界 KV 工作区流水、INT16 临时行布局、精确窗口批量读取、区间 mask、精确源无效列裁剪 |
| 长/混合 C4、C16 | 同数学窗口批量读取及区间 mask |
| 零历史请求后缀 | 仅当前未压缩 K/V 用原生 current FIA，随后仍完整写入 INT2；设备 guard 验证零历史事实 |
| 首轮 MTP | 经原生调用链限定的 current 部分使用 FIA；历史保持 INT2，身份旋转不改 |
| 后续 MTP 大 padding | 仅已审计的 eager dense MTP 路径收缩到真实请求行；保留原 source split 与位置/slot 语义 |
| 混合批次 | FULL attention 内按请求边界分开短 decode 前缀与长后缀，保留原 split，不改 GDN |
| 20–30K 首次准入 | 外部 scheduler 将 token 预算设为至少 32768，合格长 prompt 尽量整段准入；继续沿用原生异步/GDN执行 |

冻结 `configs/acceptance.json` 未修改。独立 oracle 不进入生产路由，不恢复 BF16 全历史，不改原生源码。尚未完成数值验收的 `decode_weighted` 试验未加入生产源码、算子选择或 candidate 开关。

## 已有证据与边界

- 完整部署 CANN 库和 Torch 扩展已在 Lima/CANN 9.1/ascend910b4 编译链接成功；能力表和绑定一致。
- 组合 decode 官方 CPU 20 个有效/错误/窗口/跨请求用例通过逐位与冻结 oracle；base/balanced/C4/C16 四条 reader 的独立 CPU 对照通过。
- 32 个合成请求，历史长度 20/23/27/30K 循环，N128/S3/Hq6/Hkv1/D256、20 Cube、全部 source/task：当前 SIMD **16,290,667** → 组合 **9,589,788** CAModel 周期，**1.69875×**；partial/LSE/status 逐位一致，独立 oracle 两臂通过。同一实际部署库和模拟配置。
- 这不是用户数据集、真实 NPU 时延或整轮 14→8 分钟结果。按用户收口要求停止了尚在运行的完整 K32 q1 性能模拟；不将未完成结果列为通过。
- 观察入口保留真实 NPU 数值、错误域、同地址变更输入图回放、缓存字节与后续 q1 检查，任何门失败阻断正式服务。无需再次提供数据集或运行 AISBench。

精简证据索引：`reports/decode_candidate_delivery_20261007.json`。

## 2026-10-10 prefix/GDN checkpoint 兼容修订

固定直启命令不变，安装的插件自动启用共同页边界checkpoint修复，不能只更新探针。`candidate`目前仍为至少32768的token预算，目标配置本值16384未改；它不是必须设成某个2304整倍数的参数。实际GDN页保持768，prefill切点采用FULL/GDN共同边界2304，并兼顾最后logits token和MTP丢弃页。原生decode、状态复制、q4/q1/C4与其他候选开关保持不变。

已安装最新代码后，必要的有界短服务验证命令为：

```bash
python3 -m tools.service_probe --variant candidate --prefix-only
```

该命令自行起停服务，需先停止同端口正式服务。除128-token起始请求外，执行隔离的4736冷请求/重放、4864续写，以及独立6912整页冷请求/重放；必须有实际调度策略、原生cached_tokens、TP4/FULL/MTP/图和资源释放证据。不会运行AISBench，不替代完整模型质量或吞吐验收。

本机完整回归1046 passed、147 skipped、6 subtests passed；没有本轮NPU/CANN执行证据。源码复现旧切分下4736/6912/20000 prompt的联合命中为0，修订后有可复用checkpoint；是否带来实际吞吐提升仍取决于工作负载命中率与设备时间，不能从这些CPU结果宣称。
