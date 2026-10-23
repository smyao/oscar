# 剩余精度差的细粒度对齐：逐头 K 裁剪

用户报告e2a1a97a已把LongBench差距从5–6点缩小到约3点，要求控制在1点以内。本轮基于该提交继续修复；这条反馈是质量进展，不把主机检查当作达到1点目标的证据。

2026-10-23通过GitHub分支API重新确认官方sglang-rebase仍为38680852e52c10bd6b9e9a64ed9bc815e742859b。

| 核对项 | 结论及本轮处理 |
|---|---|
| 逐头K clip | 本地所有头固定.96；官方pool按全局层/头表切TP分片，prefill和decode flush共用。补齐PT→TP rank→writer传递，保持V=.92与CANN ABI。 |
| center的V统计 | 官方S_v由原始K的能量计算，center不重算权重；本地相同，保持。 |
| decode BF16 bit trick | 官方仅用128+c的BF16位表示精确恢复0/1/2/3；metadata、减法、乘法的BF16舍入与本地mode1一致，保持。 |
| prefill反量化 | 官方FP32减乘后写BF16，对应本地mode0，保持。 |
| QKV变换和输出合并 | per-head/center、全部来源同一旋转域、FP32 partial合并后一次最终逆旋转已在当前版本；未发现本轮新增重复RoPE、漏gate或Q scratch覆盖K/V问题。 |
| 窗口生命周期 | 本地fixed-per-forward与官方8-token flush分配器仍不同；官方mixed decode不支持MTP。本轮保持已工作的native物理快照/MTP3/APC，不能称完全同构。 |
| tile/归约及原生current-only | Ascend KV256与官方自适应32/64 tile、Ascend FIA与CUDA内核仍有有限精度差异；没有证据把其占比归因成3点，未盲改内核。 |
| 发布PT和校准分布 | 本地自动拟合的公式已对齐center，但不是官方未公开的Qwen3.8 PT/逐头clip表，也没有其校准语料的逐值对拍证据。短GPQA是公开startup共同默认，不能宣称已验证长上下文泛化。 |

## 裁剪表如何自动产生

官方公开了per-head clip运行接口及标量等价测试，没有公开Qwen3.8的实际表或自动选择程序。本轮选择器明确标为**本地预先固定的校准目标**，不是声称复制官方未公开算法。

recipe4在每头原生校准K中取最多1024个等距样本。先按生产顺序BF16中心化与旋转；用现存uniform INT2公式模拟prefill/decode反量化。候选.80/.85/.90/.92/.94/.96/.98/1.0加当前基准，按实际writer的float32量化下标去重。

设旋转域量化误差为e，Q协方差为M_q，R为实际BF16矩阵，使用eᵀ(RᵀM_qR)e/D衡量注意力分数的线性误差代理，同时计算原始误差和按维去均值误差。两种反量化模式的四项指标必须全部不高于原基准，才允许候选替换；在合格候选中最小化最差相对误差，平局保留基准。

**不能只看去均值误差**：历史INT2会有量化偏置，精确窗口/current的误差却为0；历史共同偏置不再是整个softmax的共同常数。两个key真实均为0、V=[1,-1]时，仅历史key增加1会把输出0改成tanh(0.5)。独立测试冻结该边界，防止选择器靠忽略跨来源偏置得到虚假的“改进”。

该指标没有逐query的BF16旋转舍入、实际因果softmax或验证集LongBench分数；同一校准样本上的非劣不等于真实模型质量保证。完整候选分数、选择、样本数和目标随PT保存，不隐藏未改善头（这些头保留原基准）。

## 一键部署与兼容边界

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

candidate默认recipe4。新缓存身份绑定候选集合、目标和基准clip；旧recipe3/2不能直接复用为新PT。优先认证recipe3保存的原始样本（包括它从recipe2继承的样本），其次认证对应recipe2样本，自动重拟合并生成裁剪表；样本缺失才按原流程自动校准。旧PT、样本、manifest不修改，来源profile不伪造。已有匹配新PT正常复用，无需用户运行诊断或管理文件。

每头clip为初始化时读取的host常量，TP4切对应全局KV头；目标层缺表失败，MTP保留配置默认值。所有生产reader共用的writer接入这一个标量，V的参数不变。CANN、timer、冻结oracle、native argv、GDN、graph、预算、设备与端口不变。

功能检查与源绑定见reports/head_clip_alignment_20261023/validation.json。本轮不连接真机、不测速；1点以内仍须由真实模型结果验证，不能由校准代理宣布达标。

官方证据：固定版本README的结果配方注释；unified_kv_pool.py的parse_k_clip_per_head、TP分片及clip_ratio_to_index；oscar_rotation_clip_int2_kv.py与gpu_flush_int2.py的逐头索引；decode_attention.py的_int2_crumbs及实际fused stage1；fit_oscar2_variants.py的center/S_v分支。以上均为用户指定仓库的只读源码。
