# OptR 精度退化检查与修复（2026-10-29）

用户反馈当前OptR比此前OSCAR掉点更严重。本轮没有收到现场SHA、PT身份和逐项评测分数，不能把下面的合成反例当成现场全部掉点的归因。修复继续默认启用OptR，不撤回recipe3默认或增加绕过开关。档案依据：#170的代理改善不等于输出改善、#173的独立test退化、#172的拟合与断点身份。

## 已复现的问题与改动

1. **同一可见前缀因未来数据而改变训练损失。** 旧代理用完整轨迹最终长度减recent确定所有query的精确窗口，而采样来自多个原生forward。本地相同可见前缀追加被causal mask遮住的数据后，旧K loss从0.0248553641变为0.0295659546；修复后两者均为0.0249128863。新值与旧值不是同一精度目标，不据其绝对大小声称精度收益；关键是同一可见输入不再受未来行影响。精确区现在按每次forward的context固定，保持sink、cached recent及current，首轮context=0没有量化历史。
2. **拟合与部署舍入不一致。** 旧代理在FP32原空间用逆旋转近似K；部署执行BF16均值/中心化、BF16 Q/K/V旋转、FP32 metadata重建后BF16操作数。新代理按这些舍入计算K概率及V输出，BF16使用STE反传，保持FP32矩阵累加。K仍优化KL与输出误差，V仍固定选中的K概率；官方Adam、80步/lr0.02及验证时机保留。逐头FP32投影目标和dense softmax仍是代理，最终选择继续用原有BF16 tile/split重放；不能称为NPU逐位模拟。
3. **请求总量掩盖单次decode退化。** 实际CPU重放反例中，请求SSE从3.88287648降到3.17502054，但最后一次decode从1.57917923升到1.76134034（+11.53%）。新证书保留每个捕获forward的误差、teacher能量及query位置，validation要求逐请求、逐forward及两个分片均非劣。每帧先汇总所有TP/GQA头再平方，不重复计算attention，也不重复计入总量。
4. **全部头一起接受/拒绝会丢掉有效候选。** 每个头只重放一次seed/proposal，缓存投影后的少量query输出，枚举4头最多16种组合。仅在逐forward门通过且总SSE严格改善时选择最佳组合。反例中原方案全头替换SSE36、seed16，而单独保留有效头可达4；这证明选择器能保留有效组合，并非模型分数证据。独立test不进入组合选择。
5. **只有4K训练却用于32K/64K。** train新增32K和64K，与validation/test仍按完整源题目分离，默认请求数从7变9。此变更增加首次原生采样和离线拟合工作量，不能承诺准备更快；线上仍只加载相同形状的旋转矩阵。输入仍为GPQA拼接/可能重复、稀疏query，尚未解决全部长上下文泛化问题。

官方参考：[固定提交的OptR拟合代码](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/blob/66d46561b9bdf5d715f591e0f5845a4cbd927dee/optr/calibrate_optr.py)。本轮是针对Qwen gate、D256、分块prefill和昇腾BF16链的适配，不声称复刻官方CUDA/SGLang全部实验条件。

## 生效、恢复与验证

- policy升级为`optr_frame_bf16_ascend_v3`，默认PT进入新的`optr_<digest>/refined.pt`目录，旧完整PT及拟合断点不会被视为本轮结果；显式指定不兼容旧PT仍拒绝。
- 默认旧trace只有4K训练，不能冒充新增长训练。固定启动自动采集9请求的新轨迹；已有完全匹配的新轨迹/新PT可复用，recipe3种子认证复用不变。显式研究采样计划若与历史完全相同，仍可严格认证复用旧原生trace。
- `[oscar] OPTR_ARTIFACT`现在打印`refinement_policy_version`及`fitting_objective`，可确认实际加载的是v3产物。
- 独立test继续执行并记录逐forward退化，按用户2026-10-28已授权的实验发布政策不阻断实验PT；这不代表模型质量通过。validation逐forward门没有容差放宽。
- 新文件仅由离线准备调用；CANN算子、forward/runtime、MTP/GDN、timer、vLLM argv和默认前四卡/7878不变。没有增加每token计算或常驻模型缓存。

固定入口仍为：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

CPU损失/梯度独立公式、因果不变性、D256/GQA6、帧证书、缓存组合与真实小样本拟合接缝、旧PT隔离、启动配置及wheel打包检查见`reports/optr_accuracy_fix_20261029/verification.json`。没有连接目标机或获得本轮模型评测结果，不能承诺掉点已恢复到论文标准或吞吐实测提升。
