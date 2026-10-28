# 合法 chunk 划分与 prefix seed 的数值反例

档案 #37–49/#126/#145/#170–172；这是合成 CPU 数值反例和原生合同主机重放，不是模型测试、设备错误记录或性能实验。没有修改生产窗口、调度或内核。

复现命令，在仓库根目录运行：

```bash
.venv/bin/python reports/precision_stability_20261028/chunk_dependence/reproduce.py
```

脚本执行当前 production checkpoint policy 和指定 references 内原生 splitter/manager/hash 合同；全局 token budget 都是16384，只声明其他请求留下的剩余预算不同。

| 同一个5000-token prompt | 剩余预算16384 | 剩余预算768 |
|---|---|---|
| 实际重放 chunk | 2304 / 1536 / 1160 | 768 / 768 / 768 / 768 / 768 / 768 / 392 |
| 最终共同 prefix hit | 2304 | 2304 |
| 计算query position2000时的context | 0 | 1536 |
| 此query读取token64的精度层 | 精确current | INT2 history，cut=1281 |

数值输入为明确构造的 D256/Hq6/Hkv1 BF16 张量，使用identity旋转、零key mean这一合法特例，固定 K=.96、V=.92 和 mode0。token64的前128维V为BF16(.2)，余下维度为±1；Q/K的13个非零维度使该token具有高attention权重。输入详细定义与当前源码SHA均在脚本和 `result.json`。

| 独立 v4 oracle | 第一输出元素 |
|---|---:|
| 一个精确current源 | 0.19921875 |
| 相同history/window/current边界，但全部使用原BF16 | 0.19921875，全部元素逐位相等 |
| history使用当前144B codec和mode0重建 | 0.33203125 |

最后一行相对第一行 max_abs=0.1328125，768个元素不同，所有输出有限。全BF16分源消融为0误差，排除了本例中仅由source划分/merge舍入解释差异的可能。代码对应 `attention_official_tasks.cpp:149–152` 的任务区间、official reader 的固定forward cut，以及 `tools/official_reference.py` 的独立codec/attention oracle。

这证明当前精度层选择依赖合法调度chunk。相同token前缀和相同queryable checkpoint可由不同计算路径产生；注意力输出差异可以进入后续层KV/GDN，从而成为后续重复请求的不同缓存种子。本例**没有执行完整模型或GDN张量运算，没有证明物理页覆盖、缓存内容串请求或设备竞态，也不能作为LongBench低分/波动的唯一根因或份额归因**。

当前terminal checkpoint策略可使相同充足预算下冷热请求的末尾chunk一致，因此也不能说所有cold/hot重复都会不同。关闭PC、改预算、改模型参数或添加输入长度特判都不是这份证据实施的修复。
