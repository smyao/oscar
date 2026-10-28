# APC联合状态与MTP分组反例

档案 #37–49/#126/#145/#157/#170–173。复现：

```bash
.venv/bin/python reports/precision_followup_20261028/reproduce_protocol.py
```

所有源码SHA及结果位于 `protocol_evidence.json`。生产文件未修改；全局预算仍为16384。下面是合成输入、真实原生主机合同和指定纯torch数值参考，不是NPU、完整Qwen、LongBench或性能验收。

## 1. Native hybrid lookup混合两个计算轨迹

当前 `PrefixCheckpointLeases` 保持开启，每活跃请求额外GDN页不超过3：

1. A的5000-token prompt以2304/1536/1160执行，形成2304联合checkpoint。
2. X的8000-token prompt共享A前5000 tokens，实际命中2304；X继续到4608/6912/8000。X一直引用A的第一FULL页，而自己的最终lease为4608，原继承的A2304 GDN页随后可释放。
3. A完成；使用真实 `get_new_blocks` 消耗free queue再归还，表示缓存压力。仍被X引用的FULL页不能被驱逐，A旧GDN checkpoint被回收。
4. B与A拥有相同prompt hash，联合命中为0；B用合法剩余预算768重算。
5. B自己的第一FULL页是43，但后续原生联合lookup返回FULL页1（A）及GDN页46/50/54（B），命中长度2304。

原因是原生 `BlockHashToBlockMap.insert` 保留重复hash的先后顺序，`get_one_block` 选第一个存活块；FULL和GDN按group独立选择。对原生近似确定的同token状态，这是既有假设；OSCAR精度层随chunk改变时，同token hash可对应不同状态，独立选择便能拼出不属于任何单次计算轨迹的联合状态。

执行来源标签是符号；脚本另外给出真实数值可区分性：位置2200位于2304页最后259个snapshot行。两种合法chunk的独立attention输出分别0.19921875/0.33203125；同一固定线性投影后，当前codec的FULL slot相差64字节、精确snapshot相差128元素。把相同隐藏差异输入指定Ascend纯torch GDN golden，104-token后到checkpoint2304的BF16最终state分别0.08984375/0.1494140625，128元素不同。

这不是完整模型端到端payload重放；它把“原生真的会选不同轨迹”和“该轨迹确实可能产生可区分FULL/GDN数值”分别证明。不能据此声称用户全部分差都由APC混配造成。

## 2. MTP verify与serial target不等价

同一逻辑query position770，q4 verify的context768固定cut513；serial q1的context770固定cut515。因此旧token514在前者用精确BF16，在后者用INT2。当前144B codec、mode0、相同Q/K/V输出为0.2001953125/0.333984375。

同一个固定两类线性head `logits=[.25-x,x-.25]` 给出verify前三个target argmax `[0,0,0]`，serial q1为`[0,0,1]`。当前真实greedy adapter对draft `[0,0,0]`接受全部三个token；全BF16分源消融两路都是`[0,0,0]`。采样器没有比较错误，问题是被比较的target数学随verify分组发生变化。

这个例子没有执行完整Qwen logits，也没有证明未提交token被错误hash发布。它证明当前精度层语义不足以保证speculative decoding相对serial target的等价性；它与APC联合状态混配是两个独立问题。
