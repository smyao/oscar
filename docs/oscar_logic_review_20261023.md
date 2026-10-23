# OSCAR逻辑复核与recipe4回归撤回

用户反馈最新a2d9b42f差距扩大至约10点，此前e2a1a97a约3点。本轮将其作为回归处理，不继续调参，不要求用户执行诊断。

## 已证实的问题

recipe4选择器优化协方差加权K重构误差，不能保证实际softmax(P)V误差更小。新增test_clip_proxy_attention_counterexample.py调用真实selector，并独立重写uniform codec：D256，两行互为相反数，11个坐标幅度100，其余为1；Q=sqrt(D)·e0，V=[1,-1]。

- 原K clip .96：attention输出1.0，与未量化一致。
- selector选择.94：四项代理995.372314→421.136719，全部通过原非劣门；attention输出却为0.761594。
- prefill和decode反量化均出现该结果。

这是实际选择器的确定性反例，说明将未获模型验收的局部代理策略默认化是错误决定。它不是用户真实激活，不证明10点差距的全部因果占比，也不能把校准误差降低当成质量恢复。

## 默认修复与PT核对

candidate恢复perhead_center recipe3及原K=.96/V=.92，并清除配置中遗留的rotation_k_clip字段。即使磁盘同时存在recipe3与recipe4，真实prepare必须选择recipe3路径与profile；显式指向不匹配PT仍拒绝。recipe3不存在时按原一键流程自动生成，不要求用户手动删除文件。显式研究配置仍可读取recipe4，正常candidate不再启用。

e2a1a97a→a2d9b42f没有改CANN、模型配置、采样collector、Q/K norm、RoPE或V裁剪；新增推理数值变化是每层每头的K裁剪值。同一捕获下，Rk/Rv/均值逐位相同，协方差没有被后续trace归一原地改坏。recipe4使用新的profile/hash路径，代码不允许把recipe3 PT伪装成已更新recipe4；但没有现场日志，不能宣称已经验证用户加载的实际文件。

固定命令保持：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

## 内核与生命周期复核

源码和主机复核未确认重复scale、重复RoPE、clip污染精确窗口、跨TP头错配、MTP逻辑位置或prepare reuse静默错读。store只改独立UB，精确窗口重读原始旋转域BF16；read-before-write、ring259及原生物理页所有权不变。上述结果不等于证明目标NPU无错误。

发现旧v4 CPU矩阵的实质覆盖缺口：C4/C16历史很短，未直接覆盖多个KV256块时的在线alpha与旧累积值更新。正在补充解析长历史输入的本地AscendC CPU验证；真实结果另落reports/oscar_logic_review_20261023。无真机/性能测试，模型1点以内目标尚未达成。
