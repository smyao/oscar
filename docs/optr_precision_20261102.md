# OPTR 精度诊断与 recipe3 基线修复（2026-11-02）

诊断基底：`oscar-new-6@7c4a6212932fd32176164c0a28172b4cafbead6c`。依据档案 #86/#97/#148/#165/#166/#170/#172/#173/#174：校准来源必须可认证；代理改善不能冒充模型改善；保护线上算子、原生调度、PC/GDN/MTP、预算、timer 和已验收的性能来源。

用户确认比较对象为迁移前的 **perhead_center recipe3 OSCAR**。本轮回传 LongBenchv2 324293：accuracy 42.52、easy 53.49、hard 36.90、short 42.52；此前 OSCAR 平均 45+。这是约 2.48 个百分点以上的 overall 差距，不是相对下降 2.48%，也不是两侧各十轮配对统计。没有现场 PT、证书、服务加载 SHA、accepted_layers 或逐请求结果，不能断言以下哪一项解释了全部模型分差。

## 1. 已确认问题及本地证据

| 问题 | 旧代码行为 / 证据 | 本轮处理 |
|---|---|---|
| 比较基线被替换 | v9 把 recipe3 改为每层共享、K 均值为零、train 前512行的 recipe6 seed；非劣门只与这个新 seed 比较 | 新策略读取并认证原 recipe3 K/V pair 与 combined PT，保留逐头矩阵、K 均值和标量 clip |
| 独立 test 失败仍发布 | schema2 把 test 退化写成报告，`publish → load → prepare cache` 均可成功 | 新 schema3 必须通过 validation 和 independent test；test 失败写报告后抛异常，不根据 test 重选 |
| K 候选过早丢弃 | FP32 proxy 先固定 K，BF16 联合部署门最多检查其6个 V 方案 | 保留已有 K/V 检查点，至多12个全层方案；验证集严格通过者中选 SSE 最小者 |
| recipe3 数据进入保留集 | 原 recipe3 不是新 v9 train-only seed；直接复用旧三分采样不能证明与原拟合独立 | 从原 prompt SHA/token count/原样本边界重建 recipe3 用过的记录；先排除，再按记录分 train/validation/test |
| 多长度样本是同一串前缀 | 原4K/32K/64K从同一拼接流的第0 token开始 | 各长度使用确定且不同的完整记录起点，记录实际消费区间、重复情况与来源 SHA；这不等于自然长对话独立性 |
| 完成 PT 不绑定拟合实现 | 部分 checkpoint 绑定代码摘要，但完成 PT 的 cache early return 未检查 | 新完成证书绑定实现摘要、原 recipe3 文件和各层常量；代码变化自动从已认证原始 capture 重拟合 |

本地固定反例使用合成 BF16 输入、真实拟合和部署数值回放，不是现场激活：

- D64、2 KV heads、默认64+256精确窗口：同一组512行拟合后，共享零均值 seed 的 validation/test SSE 分别为 recipe3 的 **4.992 / 15.390 倍**。说明“换 seed 再改善”不能建立对原 OSCAR 的优势。
- 同一套真实80步优化产生的候选，在另一个明确标记为诊断搜索的合成测试输入上，通过旧非劣门仍为 recipe3 的 **1.6143 倍**误差。该搜索不是独立质量验收。
- 另一合成例中，validation 相对 recipe3 改善到0.803倍，但 test 为1.504倍；旧发布政策允许这类 test 失败的 PT 发布与复用。证书路径另有最小可重放反例。
- 新候选召回的固定 D8/H2 例：primary 与全部固定 K 的 V 候选未通过；已有 `k_step_1 + 原V` 通过逐 query/frame/terminal、两种历史分区、三种投影视图，validation SSE **0.317226 → 0.169302**；选定后独立合成 test **1.911906 → 0.373431**。这只证明修复能找回被旧筛选丢弃的有效候选。

原始报告与复现脚本在 [reports/optr_precision_20261102](../reports/optr_precision_20261102)。旧策略反例明确针对7c4a621/v9，不能把它们写成新默认实现仍然允许失败发布。

## 2. 拟合和部署数学的差异

逐头初始矩阵为原 recipe3 的 `Rk[h]`、`Rv[h]`，均值 `mu[h]` 固定。仍按 K 后 V 顺序优化，以 `R = R0 @ exp(A - Aᵀ)` 保持正交；80步、Adam lr=0.02、已有评估步1/21/41/61/80不变。每次只驻留一个 KV 头的训练/验证历史，按全局 KV 头数维持原梯度缩放。MTP identity/零均值规则和线上头切分不动。

上游 FP32 loss 仍包含 attention KL / output loss，不能代替现有 BF16 联合输出门。D256 实际 recipe3 fitter 的合成例证明：FP32中心化与部署 `BF16(K - BF16(mu))` 相对 RMS 差0.187%；仅中心化舍入改变约0.1167%的 INT2 codes，完整 BF16 旋转链改变约0.1995%。无压缩历史的精确 prefill 在代理里 loss/梯度为零，但 BF16旋转往返仍有输出误差。因此保留原 BF16 部署回放和全部严格门，未放宽容差、未重启已退役 native_exact，也不声称改代理就必然提高任务分数。

## 3. 发布、复用与失败处理

固定 candidate 使用 `optr_baseline=recipe3`、策略 `optr_recipe3_centered_candidates_ascend_v10`，仍为 recipe5/原 perhead_center 存储 ABI。旧无 marker 的 v9 仅保留明确的历史研究/回归路径，默认 candidate 会设置新 marker；旧 v9 PT 的 profile 不能冒充新 PT。

每个候选与**同一原 recipe3 常量**比较，validation 必须逐请求、逐帧、逐 query、terminal、三种投影视图均非劣且总 SSE 严格下降。没有合格候选的层保留该层原 recipe3；整份 PT 至少一层严格改善，覆盖全部 FULL 层。选定后只验一次 independent test；退化即终止，不借 test 换候选。新证书对发布和启动复用都严格执行这些条件。

PT、base pair、capture 和拟合 checkpoint 分别认证。新采样源须能重建原 recipe3 prompt SHA/count；无法认证、有效剩余记录不足、文件损坏或 test 失败均明确报错。旧 recipe3 PT **不要删除**，它正是比较基线。新 v10 自动使用独立缓存目录，不需先清空整个 artifacts。仅拟合实现变化时复用合法原始采样再重拟合；旧 v9 trace 缺少基线排除证据，不自动复用。

现有直启与 probe 共享 `serving_variants.py`，本地命令仍是：

```bash
bash scripts/install_serve.sh --variant candidate
```

发布并拉取本轮代码后，固定更新命令仍是 `git pull --ff-only && bash scripts/install_serve.sh --variant candidate`。本轮仅修改本地工作区；未提交或推送，远端 pull 目前不会得到这些修改。默认前四卡、7878、TP4、16384、PC默认关闭和MTP配置不变；`--rear-cards` 可选模式不变。启动摘要应显示 v10、`layer_kv_head`、`key_centering=true`、`independent_test_blocks_publication=true`；最终收据必须为 `independent_test=passed` 且 accepted_layers 非空。

## 4. 性能与兼容边界

没有修改 CANN、INT2 pack/unpack/反量化、QK/PV/逆旋转、KV布局、窗口/clip、精度容差、GDN、prefix cache 页引用、MTP位置、调度、vLLM argv 或 timer；没有新增长度/数据集路由。除了命名的离线准备与配置接缝，既有生产文件逐字对比7c4a621，q4/q1/C4/C16继续保留既定性能源码来源。

离线成本会增加：每层从共享的一对轨迹变为逐头轨迹，全层回放最多12组而非6组；新采样证据不足的旧 capture 需要重新生成。线上操作数量不增加，但新常量可能影响生成长度或 MTP 接受率，所以源码身份不能保证端到端耗时完全不变。按用户约束，本轮不连接真机、不运行 NPU/性能或 AISBench。

## 5. 尚未被本轮证明的事项

采样仍为 BS1、eager、PC关闭、无MTP，每请求8步decode、每帧8个query；4K/32K/64K长度虽覆盖，默认理想调度每split仅80个query。新起点消除了严格嵌套前缀，仍有拼接、截断、重复和自然问答分布偏差。原生逐层teacher也未包含多层同时量化后的隐藏状态/生成漂移；CPU的 W_O/TP舍入视图不等于NPU闭源GEMM/HCCL逐位输出。详细来源审计见 [capture_audit.md](../reports/optr_precision_20261102/capture_audit.md)。

后续若仍退化，优先级为：在独立数据上增加自然终止位置和长decode覆盖；补真实post-W_O teacher及PC/MTP/mixed-batch离线轨迹；用真实加载的recipe3与OPTR进行端到端配对，而不是继续只优化局部SSE。未使用LongBench题目或反馈分数调整单题/长度规则。

本轮已修复可复现的基线、筛选和发布缺陷；**尚不能证明 LongBenchv2 超过45+，不能承诺所有输入或模型均超过OSCAR**。本地主机回归、真实NPU、模型质量和端到端性能的状态分别记录于 validation.json，不混写为passed。

最终本地验证：联合回归 **1326 passed / 1 optional skipped / 0 failed**；补充证书回归 **16 passed**（包含联合运行后新增的1项生产/测试基线许可检查，不将重复项相加）。**879个**既有生产文件逐字保持，9个既有文件仅变更离线接缝；wheel的211个Python源文件逐字核对、5个新增模块从隔离安装目录导入通过。`git diff --check`通过。详见 [validation.json](../reports/optr_precision_20261102/validation.json)。
