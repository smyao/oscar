# 零额外设备页的APC联合状态发布门：待实施设计

档案 #37–49/#127/#128/#157/#171；依据本目录的真实native合同反例。这里只提出外部插件方案，尚未编辑production，未证明模型质量或性能通过。

## 目标与不变量

原生hash算法、`find_longest_cache_hit`、FULL/GDN状态张量、模型算子和已有3页/活跃请求lease保持。新增host sidecar只决定一组GDN checkpoint能否继续出现在原生prefix hash表。

对每个可见joint checkpoint `n`，必须存在一个真实生产记录：

1. 该记录来自当前请求实际使用的FULL prefix物理页链，覆盖 `[0,n)`；不是按相同token hash猜测等价。
2. 记录的全部FULL页 `(group,id,generation,hash)` 仍有效，而且每一页都是原生hash表当前 `get_one_block` 将选中的那个entry。
3. 该checkpoint的原生可见GDN tuple必须完整来自同一登记记录；不能因各group独立淘汰而拼接不同记录的剩余成员。

FULL页链内部可以继承先前请求的页：后续GDN正是基于这条实际继承链计算的。因此记录绑定实际输入页链与GDN tuple，不能简单要求所有block拥有相同request ID。

## 记录内容与空间

记录键使用已有native prefix hash和合法joint token边界；record保存FULL各group的实际prefix block身份序列、各GDN group在 `n / block_size - 1` 的真实非null/hash匹配页，以及生成代际。记录只包含host对象/整数/现有hash，不复制任何KV/GDN字节，不新增NPU算子或设备页。

物理generation在原生 `BlockPool.get_new_blocks` 返回后递增；同一个block ID重新分配即变代。原生分配前的eviction通知负责撤销旧记录。generation与原有hash共同检查，避免“同ID、同token hash、不同计算结果”重新出现时被旧record认领。

需要反向索引：physical block身份→依赖records，FULL `(group,hash)`→依赖records，GDN成员→所属records。其host空间随有效缓存checkpoint和prefix页链增长，必须纳入资源报告；没有新增设备页不等于没有CPU/host内存成本。失效记录应删除索引，不能随请求历史永久积累。

## 外部接缝

### A. 原生coordinator完整发布后检查

包装当前OSCAR+APC+align coordinator的 `cache_blocks`：先执行原函数，待全部manager的发布和 `cached_blocks_this_step` 更新结束后，再读取该请求可实际发布的joint checkpoint并建立/验证record。

只能记录真实非null、已赋预期hash的GDN页；不能把长chunk跳过的中间状态补成record。沿用原生FULL/MTP lookahead、prompt−1与共同对齐限制，不改变native cache counter或GDN状态推进。

如果新GDN tuple的FULL链与当前native可见FULL链不一致，调用原生 `BlockPool.evict_blocks` 撤销该tuple的prefix hash资格。若FULL链一致但可见GDN首entries不属于同一record，也撤销不完整或冲突tuple，使可见tuple只来自一个完整record，或者无joint hit。

必须在整个coordinator发布返回后执行。提前在某个manager内部清hash会违反 `MambaManager.cache_blocks` 随后的 `assert block.block_hash is not None`。即便稍后撤销，`cached_blocks_this_step` 仍保持原生值；不能为提高命中删掉这道同step未完成屏障。

### B. 原生驱逐/重分配触发整组失效

包装 `_maybe_evict_cached_block`，在原调用前保存old hash/身份，调用后处理受影响records：

- FULL关键页失效或其hash的canonical第一entry变化：撤销依赖它的GDN完整tuple。
- 任意GDN成员失效：撤销同record其他GDN成员，防止不同tuple残片在各group重新拼接。
- 不修改FULL hash优先级、活跃tensor、block refcount或free queue。

使用原生public `evict_blocks(set(ids))`。该API明确允许refcount>0的block，只移除prefix hash，不归还或写动活跃页。回调需用有界待处理队列/重入保护，避免撤销tuple时递归修改正在迭代的sidecar集合。原有lease引用继续由已有request-free逻辑释放，每请求上限仍3页。

`reset_prefix_cache` 成功时清sidecar；失败时不得假定cache已清。plugin卸载/服务新建绑定也要清对应sidecar，不跨coordinator共享。

## 反例中的预期结果

原反例B发布GDN `[46,50,54]` 时，记录的FULL首块是43，原生canonical FULL仍是A的1。发布门应撤销B三页的hash资格；原生lookup自身不改，其结果从不兼容2304变为0。X仍继续拥有FULL页1和自己的GDN状态，B仍继续使用自己的活跃页；两者数据和refcount不变。

这是**改变cache可命中集合**，不是保持所有命中的透明优化：不兼容checkpoint不再可命中，可能增加recompute。`num_cached_block`已经前进的被撤销checkpoint不会由原manager自动重新发布；若以后FULL canonical变化，也不应擅自恢复旧record。初版接受保守miss，是否需要安全再发布应另行设计和证明。

因此，这个方案不等于关闭PC，也不构造BF16回退；它仍复用符合不变量的原生缓存。但不能承诺本次所有32K/64K性能与命中率原样保持，必须分别记录新增host成本、失效数量和设备/整轮未测状态。

## 未采用的替代方案

按GDN cohort主动改选对应FULL duplicate blocks（例如返回B的43而非A的1）有机会保留2304命中，但它明确改变lookup返回的物理块链和选择策略。当前不选该方案，也不通过额外pin全部FULL链或把GDN lease扩大到6页来掩盖一致性问题。

## 必须先过的本地协议门

- 原混配反例：门关闭返回FULL_A/GDN_B；门开启返回0，活跃tensor符号、refcount、free queue位置和原生state推进不变。
- 正常同record完整命中、部分命中、MTP lookahead/drop仍通过；所有有效FULL与GDN成员来自已登记的同一实际输入链。
- 只驱逐一个GDN group成员、只驱逐一个FULL prefix页、canonical duplicate晋升、同ID再分配后同hash重现，均不留下可混搭残片。
- 新request继承已有prefix后产生更长checkpoint，允许真实继承链；不要求所有FULL页来自同一request。
- 三组GDN逐个发布时没有中途断言故障；同step admission屏障、decode后postprocess/hash时序保持原生。
- 完成/取消/抢占/free/reset/unregister之后无sidecar无限增长；原有每请求3页lease上限不增加。

此门仅处理联合checkpoint一致性。MTP q4-vs-serial q1的fixed-cut反例是另一项数学问题，不会因APC发布门自动消失。
