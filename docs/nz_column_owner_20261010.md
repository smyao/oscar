# NZ发布的D列分工备选：只读审查

2026-10-10；只讨论当前striped source0，未写核/改精度/运行VM。档案#13–22/#70–73/#126/#145/#148/#154–156、startup D.4。等待预pack-B微测，再判断是否值得实现。

## 可成立的布局与收益范围

两lane每轮都载相同64行packed；lane0只解bitplane0–3（D前128列），lane1解4–7（D后128列）。四轮覆盖KV256。每lane每轮输出`[16 C0,64rows,8]`，两次8-C0 Gather/2D发布：GM float起点`(lane*16+c0Base)*256*8 + chunk*64*8`，其中c0Base=0/8；DataCopy2D为`{8,64,0,192}`，长度/stride使用32B单位。

Host纯坐标枚举证明两lane/四轮覆盖65536个NZ float恰一次。它不证明设备数值或净周期。

- C4 H32旧lane16布局转NZ：512burst/K → 本方案128burst/K，确有4倍burst数减少；payload未减少。
- decode H64上一页四个8-C0块方案已经是128burst/K，本方案仍128；Gather/发布调用次数也相同。**不能把C4的burst收益套给decode。**
- 两lane均读全packed，GM有效packed payload从每KV34816B增至69632B（含同一row的K/V和metadata），metadata解析也重复。物理事务/缓存命中尚未知。

## 虚拟旧lane与错误优先级

旧owner必须按**相对当前KV256起点**的rowOffset计算：C4 `(rowOffset//16)%2`，decode `(rowOffset//64)%2`。不能用绝对position；splitBegin非对齐时会错。对8种非对齐起点、17种live长度、两个lane的集合枚举，272个lane-specific比较/旧width均相同。

每lane虽解所有64行的半D，但metadata错误只用`liveMask & virtualOwnedMask`更新自己的error；**不属于自身错误域的行也必须完整解码**，因为该lane仍拥有这些行的一半D值。

`LoadPacked/Unpack`当前source0的可观察错误组合：

1. 原Physical失败置1，并留下清零packed行。
2. owned live行的零scale随后被K metadata guard置3，V guard同样保留/置3。
3. 所以该有界load+K/V组合若有owned坏地址或坏metadata，最终置3；没有则保持输入error。Unpack本身不产生score错误2。

新loader可用局部address-error及行mask，只有失败run确实包含owned live行才写1；随后照原K/V guard变3。不能让未拥有行的Physical失败污染全局error。新64row DMA run可能跨多个旧16row域，不能只看run第一个row的owner。C4仍在所有四块完成后将kvError按`memberErrors→kvError→Softmax error`应用；旧非有限Q和后续score检查不变。

Decode还必须保留原提交时点：`PrefetchKeys/Values`保存未来错误，**当前PV消费后**才`CommitPrefetchErrors`。新物理chunk为0/1/2/3，lane0只把chunk0/2的error写旧pending slot0/1，lane1只写chunk1/3；未拥有chunk不得用0覆盖已经记录的owned错误。旧两个slot仍按K0,V0,K1,V1提交，四个physical-live计数另存。

## 完整UB与隐藏依赖

| 路径 | 保留全部既有路径后的显式UB | 原因 |
| --- | ---: | --- |
| C4 H32→64row×half-D | **165312B** | packed+5120、natural+1024、metadataIndex+256、metadataHalf+128、Brcb scratch+1024；dequant32KiB不变 |
| decode H64列分工 | **179136B** | 保留原64KiB dequant、34KiB natural，packed从两份64row增为四份，+20480B |

Decode不能仅保留两份packed：K预取→Softmax→V预取之间需要全下一KV的四个64row包；否则V要再读GM，整体packed流量将从原方案的2倍进一步变为4倍。也不能把decode dequant/natural按half-D减半：旧source1/current仍读写64×D全宽，且Softmax使用原64KiB临时view，缩减会越界或改变tiling。C4旧precise只有16row，全宽8KiB自然/16KiB FP32可容入新缓冲，原16KiB Softmax view仍固定。

物理量化格式仍是D256、slot136B、padded stride160B、wordCount32、metadata offset128/132。**不能直接实例化现有`Unpack<D=128>`**；只缩小产出的维度和bitplane数量，存储格式参数不变。FP16→FP32、Mul→Add次序保留。

NZ scratch继续用dead natural/score，不借acc/stats/packed。每次复用score输出之前按上一页处理MTE3_V，尤其decode的V预取不能只依赖旧MTE3_MTE2；每块发布后再复用输出也需完成MTE3。两lane都必须发全KV ready，哪怕该chunk没有自己的虚拟metadata owner或q1只有lane0 query行。

## 判定

**C4：结构可行，但需完整helper/错误域/UB/新DMA成本资格后再算净收益。Decode：目前没有优于既有H64→NZ行分工的发射量优势，却增加packed/metadata工作，不作为默认扩展。** 查询行归属、原softmax、跨核flag数量与完整KV零尾不变。以上不证明NPU、graph或整轮速度。
