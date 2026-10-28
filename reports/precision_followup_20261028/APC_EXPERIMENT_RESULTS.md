# APC cohort 两种修复实验与未放行原因

档案 #37–49/#127/#128/#157/#171/#173。原生参考树、production plugin、算子和启动参数均未改；文件仅在本目录，未接 candidate。所有测试运行真实 pinned native AST 合同，合成 token hash/请求，不执行模型、NPU或性能实验。

复现：

```bash
.venv/bin/python -m pytest -q reports/precision_followup_20261028/test_cohort_gate_experiment.py reports/precision_followup_20261028/test_cohort_lookup_experiment.py
```

本次 11 项通过。这里的通过包含**确认反例确实发生**，不是放行该算法。

## 原问题确认

`reproduce_protocol.py` 重新运行得到同一结果：B 的真实 FULL 首页为43，native 查询却返回 A 的 FULL 页1和 B 的 GDN 页46/50/54，联合命中2304。每个活跃请求既有 GDN lease ≤3页保持；不是关闭PC的问题。

## canonical 发布/驱逐门

`cohort_gate_experiment.py` 保存实际 FULL 链、全部GDN页和分配代际。失去任一依赖时整组撤销GDN hash广告。原生hash计算、lookup函数体、tensor、引用计数、request block table和状态计数不改，**hash广告集合改变**。

| 策略 | 原混配 | 有效分叉 X 的命中 | 短 B/C 重算之后 | 结果 |
|---|---|---:|---:|---|
| 无完整cohort时推广新FULL | 可改正 | 6912→4608 | 2304 | 撤掉合法长命中，不放行 |
| 任一有效descendant依赖旧FULL就拒推广 | 避免非法混配 | 保持6912 | B=0、C=0 | 存在持续重算仍miss，不放行 |

详见 `cohort_tradeoff_results.json`。此外，重复B的2304 GDN被保留A策略拒绝后，B后来推广4608可导致原A2304失效，短prompt命中2304→0。全局一组 `get_one_block` 首entry不能同时表示两个对共同hash使用不同物理FULL链的有效分叉。

## matched-cohort lookup overlay

`cohort_lookup_experiment.py` 保留全部真实cohort及依赖，不撤销任何广告。先运行原native最长命中/对齐/EAGLE drop，然后寻找同长度、同各组hash的完整存活cohort。原结果已完全一致时原对象返回；混配时替换返回的物理FULL链和必要的GDN tuple。

已证明的主机合同：

- 原 A→X→B 反例，B仍命中2304但改选它实际使用的FULL_B；X4608仍是原结果对象、原长度。
- 一个GDN group驱逐导致原生first-entry混组时，选择另一个完整存活GDN tuple及对应FULL链。
- 原native的EAGLE lookahead只作为已有hash存在证明；它不是 `[0,n)` 状态依赖，也不返回给计算，不要求同producer。
- 各GDN尾页的group/hash不变，后续原生 `cached_blocks_this_step` admission仍拒绝同step读取。
- lookup不修改hash广告、refcount、free queue、manager counters、状态推进；共享fork、驱逐、同ID再分配、失败reset与成功reset覆盖。

**两条实质边界，不能包装成性能无损：**

1. B结束后FULL_B已在free queue，而FULL_A仍由X引用。native混配路径选A，正确路径必须选B；同2304命中却多占一个free FULL页。原生admission会真实计入，不能隐藏。`cohort_lookup_capacity.json` 记录adopt消耗4页(1FULL+3GDN)，原路径是3页。`review_apc_capacity.py` 的独立native检查还证明紧容量时A路径可调度而B不可调度。
2. 若GDN_B仍有hash，但它的实际FULL_B已被native回收，而同hash的FULL_A仍在，原生仍可报2304命中，却没有任何完整cohort可恢复。实验显式抛 `NoCoherentCohort`，没有假造原长度成功。生产实现必须安全缩短或miss；不能保证保留这种本来不一致的hit长度。

该overlay也明确改变原native返回的物理页选择，不满足“lookup返回结果完全不变”的字面约束。当前保存的是可审查实验，不是默认修复。原生hash函数/最长长度算法未改不等于lookup行为未改。

## 资源与实施范围

两个实验均无额外设备页分配、无copy/NPU算子、不扩大既有3页lease。但matched方案可改变既有free页被touch的数目；canonical方案可改变命中/重算量。二者都有CPU sidecar维护成本。当前实验直接保存每checkpoint完整FULL依赖列表，空间为所有有效cohort的prefix页数之和，单长请求可呈平方增长；尚未做持久共享链优化或host成本测试，不应接入生产。

执行覆盖为当前FULL2304单组+三组GDN768。matched实验显式拒绝非FULL/align-Mamba类型；不同FULL block_size的多组原生lookup可能返回未trim至最终end的后续FULL组，此时实验严格形状检查会拒绝，没有声称这些拓扑已通用验证。

没有根据这些合成结果宣称用户全部分差归因已闭合、四指标已达标或32K/64K性能不退化。
