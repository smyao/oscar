# APC cohort设计独立审查

2026-10-28；源码审查，不是NPU/LongBench/性能结果。档案#37–49/#127/#128/#157/#171/#174。旧`APC_COHORT_DESIGN.md`中的撤hash gate不是这里建议的默认方案。

## 全局canonical发布门的限制

`BlockHashToBlockMap.get_one_block`对相同group/hash只选一个物理块。若两个分叉checkpoint分别依赖同一公共token前缀的不同FULL物理块，两者都完整且仍驻留时，全局canonical选择无法同时服务两者。仅在“无valid dependency”时promote可以保护既有canonical cohort，但会拒绝新cohort；它不等于保留所有可用的联合命中。

原A→X→B复现已经包含必要对照：X的4608 checkpoint依赖A的首FULL块，B的2304 checkpoint依赖B的另一首FULL块。把A首块撤hash以提升B会破坏X，撤B的GDN则使B从2304变成0。任何声明“publication-only且全部合法命中不减”均需要额外机制，否则不成立。

## 推荐实验：按完整cohort修正混配lookup

先调用原native longest-hit。若返回的全部FULL/GDN物理身份与一个仍有效登记cohort一致，原tuple/list对象直接返回。如果native返回混配，在同一个命中位置和相同token/group hash下选择一个完整仍驻留的cohort，改用其实际FULL链与完整GDN tuple。

这个overlay明确改变物理lookup选择，不再是“native lookup行为完全不变”。它可以保留native hash表、最长命中算法、页分配、引用计数、LRU和状态推进的实现；必须在交付说明中指出与旧约束的区别，不能通过名称掩盖。

当同长度完整cohort存在时，该替换不增加设备页、KV/GDN copy、已命中的token重算或算子任务。新增的是host记录和选择成本。仍需检查物理选择改变的后续引用/LRU结果，不可把一次lookup零device-work当作所有长期调度相同。

不存在同长度完整cohort时，不可能安全保留一个混配hit：例如GDN仍存、其真实FULL已被回收，仅相同hash的其他FULL存活。应显式回到更短的完整cohort；不能把原不合法hit的长度作为可无条件保持的性能承诺。这和删除合法cohort是不同边界。

## MTP lookahead与admission

- `references/vllm/vllm/v1/core/single_type_kv_cache_manager.py:552–564`：FULL按hash向后匹配，再为EAGLE pop一页。用于计算的FULL数据依赖是`[0,n)`；`[n,n+B)`只提供原生lookup需要的hash存在性证明。
- `references/vllm-ascend/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:295–305`：FULL允许多匹配一页，Mamba不允许这项margin。overlay应保留原native返回的n，不再对已pop结果重复drop。
- lookahead页不进入返回的计算前缀，不必与GDN同轨迹，不能因为producer不同撤其hash。撤掉唯一证人页会把n降为n−B。
- `references/vllm/vllm/v1/core/single_type_kv_cache_manager.py:1065–1074`：同step未完成屏障在`get_num_blocks_to_allocate`按GDN尾页hash判断；不在AscendMamba的lookup里。替换应保持group/hash和list长度/空页布局，由原admission继续拒绝，不能用refcount推断设备已完成。
- `SingleTypeKVCacheManager.allocate_new_computed_blocks`正常touch传入的页；不要求它们恰好是hash表首entry，因此同hash有效alternate页可沿原生分配路径复用。

## 最小记录与失效

保存每个真实checkpoint的FULL各group物理页tuple、GDN tuple和每页generation/hash即可；不需要额外producer/request lineage。不能只保存末页或canonical标志：任一FULL页可被独立回收，末页仍在不代表实际前缀完整。

普通tuple比先构建持久化DAG更易审查。FULL2304下32K约14页、64K约28页；实际host开销必须按有效记录数和页引用数计数，不能声称零成本。

仅request free不能删除仍可缓存命中的record。原native真实eviction/reallocation与成功reset才使相关身份失效；失败reset不清。`BlockPool.reset_prefix_cache`还会直接遍历reset_hash，应显式覆盖，不能只hook普通evict。

## 必须交给实现的控制用例

1. 原A→X→B：B保持2304且FULL/GDN来自B；X保持原4608及原对象，hash/refcount/freequeue在lookup前后不变。
2. coherent native hit直接返回原对象；无重复页路径不得重建所有lists。
3. GDN三group首entry分别来自不同记录时，选择完整tuple；不可只替换FULL后仍留混合GDN。
4. FULL仅lookahead页来自其他轨迹时仍可合法命中，不误判数据依赖。
5. 同step publication后原admission继续拒绝；new_step后可接受。
6. 任一真实依赖被回收、同ID同hash重新分配、失败/成功reset均不认领旧record。
7. 更短完整checkpoint选择保持原native hash/EAGLE上界和空页布局，不重复drop一页。

以上仅处理联合状态一致性，不证明用户的全部分差或分钟差来自此处，也不修复另一个MTP精度分组问题。
