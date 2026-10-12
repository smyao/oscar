# OSCAR prefix caching 兼容闭环

档案依据：#26（前缀窗口必须精确）、#37–49（窗口生命周期及 MTP 接受边界）、#127（none/align 原生几何）、#128（128 虚拟块与物理页分离）、#129（host 通过不等于设备完成）。本页从属于 `skills/oscar-quant-optimization/SKILL.md`；不取代性能主目标。

## 当前源码事实与本轮修复

本工程原有物理页快照已为 prefix caching 保存完整页末尾的 BF16 精确窗口，不能说此前完全没有前缀设计。但实际启动/性能路由缺少贯通验证。用户报告开启时报错，尚无错误正文；以下是源码确认的问题，不是对那次未知报错的根因判定。

1. `tools/target_cli.py::serve_argv` 原先忽略 `enable_prefix_caching` 配置。按用户2026-10-12最新要求，所有拉起脚本默认显式传 `--enable-prefix-caching`：target默认true、`variant_config`缺失时补true、即使绕过variant直接调用`serve_argv`也默认true。禁止把字符串 `"false"` 当真值。
2. `OscarMetadataBuilder` 原先要求 `enable_prefix_caching is False` 才保留 mixed decode split。开启 APC 后，小 decode 与长 continuation 又被全局大 query 的 CV 路径绑在一起。原生仅共享完整只读 FULL 页、分叉写入私有 suffix，故请求边界分流可保留。已取消此不必要的禁用，PCP/DCP、dummy/capture、later-MTP 等既有约束保留。没有改变量化公式或内核。
3. direct/observe 都接受原生同名 `--enable-prefix-caching` / `--no-enable-prefix-caching`，共享 `variant_config` 与 `serve_argv`。既有命令无需加参数即可默认开启；日志明确 `prefix_caching=true`。native/baseline/candidate默认一致，显式false仅供用户主动诊断；不能自动关闭以掩盖不兼容。builder 看的是原生解析后的布尔值。

## 原生协议及所有权

只读依据：

- `references/vllm/vllm/model_executor/models/config.py::MambaModelConfig.verify_and_update_config`：开启 APC 时原生从 `none` 派生适用的 `align`/`all`；Qwen3.5 当前合约为 `align`。
- `references/vllm-ascend/vllm_ascend/patch/platform/patch_mamba_config.py::verify_and_update_config`：保留原生 GDN 状态形状/dtype，align 时 mamba block 跟原生 FULL block 对齐。当前 D256/TP4 示例为768，OSCAR FULL 物理页为2304；native MTP conv 宽度例为6，page817152 B。
- `references/vllm/vllm/v1/core/kv_cache_utils.py::resolve_kv_cache_block_sizes`：hash granularity 由实际组 gcd 得出；scheduler granularity 为 lcm。不能拿虚拟128当物理/哈希页。原生 chunked prefill 仍按原生 `cache_config.block_size` 分块，不提高 token budget。
- `references/vllm-ascend/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py::AscendHybridKVCacheCoordinator`：FULL 与 GDN 必须都命中共同合法边界；保留 MTP lookahead drop。
- `references/vllm/vllm/v1/core/single_type_kv_cache_manager.py::FullAttentionManager` / `SingleTypeKVCacheManager`：只缓存完整页；两个请求命中后共享其只读页面，后续 `allocate_new_blocks` 分配不同可写尾页。这里无需对共享完整页做 COW，不能凭空实现新的共享半页协议。
- `references/vllm/vllm/v1/core/block_pool.py`：touch/free/refcount/hash eviction/reset 都保持原生实现。OSCAR 快照由物理页拥有，随该页保留或重用，不绑定临时 batch 行号。

`configs/target.json` 没有 `mamba_cache_mode` 覆盖，插件也不强制改 GDN mode。显式 prefix 参数传进 vLLM 后由上述原生模型配置流程选择；不能在 KV pool 已建好之后再切 mode。

## exact window、分叉与 MTP

`SnapshotLayout` 将每个 FULL 物理页的精确 sink/recent 数据放到 native padding 的独立区间，INT2 放 SSM 对应的 FULL payload 区间。它不与 GDN SoA 视图重叠。命中任意完整页边界时，sink 来自第一逻辑页、recent 来自包含对应位置的页；不得从 INT2 重建精确窗口。

命中后两条请求只读相同 INT2 和精确 snapshot，在各自尾页写当前 K/V。release/preemption 不清除仍有 refcount 的共享页；eviction/reset 撤销 hash 后，旧数据不会再被匹配为当前 prefix。新页在读之前必须通过真实 prefill/store 写入可见区域，不能把 tag 中碰巧相同的旧位置当新请求状态。

分流只按请求边界切 query/current/slot/seq/table，保留历史正 context；`current_only_plan` 对正 context 为 `None`。原生 block table storage 仍共用原内存的切片。main/first-MTP 的位置由 prepare 按 context+row 推导；later-MTP 保持原接受位置 slot 修正并不进入该分流。attention metadata 的 `positions` 并非生产 prepare 的位置来源，不能用 mRoPE 数组替代接受位置。

## 当前验证与边界

`tests/test_prefix_caching.py` 执行未改写的 pinned native AST 合约：完整/部分命中、MTP 最后一页 drop、共享 refcount、私有 suffix、释放/抢占所用 free、驱逐/物理 ID 复用、running reset 拒绝及释放后 reset。额外检查真实 `SnapshotLayout.views` 中两条分叉写不改变共享精确窗口。

`tests/test_mixed_decode_split.py` 覆盖相同 computed prefix 下 main/first-MTP split 与 unsplit 输出一致、缓存写槽一致，以及 context2304、共享虚拟页18..35、私有尾物理页2/3的真实元数据切片。当前 positive-context cache-hit 不会误路由 current-only。`tests/test_runtime_native.py` 继续验证原生配置派生、GDN 对象/字节范围、物理2304/2816与虚拟128、MTP拒绝数0–3等契约。

这些是 host 数据流及字节/数值证据，**没有新增真 NPU prefix 设备执行、图回放或模型输出验收**。完整共享前缀真实服务用例仍须包含冷/热 cache、长短 suffix 混合、MTP拒绝、抢占/复用/reset及图内metadata改变；短通过也不能宣称全模型质量或2–3倍性能。

研究开关 `experimental_request_windows` 当前默认未启用，它把 snapshot 改为请求 sidecar，尚无跨请求 prefix 恢复协议，仍显式拒绝 prefix。不能为宣称兼容而开启该组合、关闭用户 prefix 或降级全历史 BF16；以后接入该研究路径必须另行解决发布/恢复/共享生命周期。

## 请求私有窗口研究布局：具体缺口与恢复协议

### 是否影响当前默认使能

`FEATURE_FLAGS` 不含 request_windows；它仅在 `RESEARCH_FEATURE_FLAGS` 中，target 未设置，candidate 不会自动打开。当前默认 route 是 `SnapshotLayout`。但若显式打开研究标志，`AscendRuntimeProvider.seal_request_windows` 要求 prefix=false/mamba=none，`DualRegionPageLayout.__post_init__` 也只允许 none。因此当前研究布局与默认 prefix=true **确实不能同时运行**；这是尚未实现的功能，不能保证没有设计缺陷，更不能仅删除两处断言。

### 最小正确数据协议

1. **不可变 checkpoint 与运行窗口分开**：可共享 prefix 由原生完整 FULL page/hash拥有不可变原始BF16窗口；请求私有 sidecar只保存本请求当前窗口。Checkpoint身份必须包括层/模型/旋转/布局版本、原生prefix-chain hash、物理page ID、allocation generation、logical end。request ID不能作为共享前缀身份。
2. **只发布已提交边界**：完整FULL页末尾且原生已提交到该位置，才可使checkpoint有效。MTP尚未接受的写不能发布为可共享前缀。来源必须是原始BF16 current K/V或仍精确的旧窗口；禁止从INT2补回。跨越多个边界的长prefill必须在覆盖每个待发布边界时取其窗口，不能事后从最新ring寻找已覆盖的数据。
3. **命中恢复**：在native FULL/GDN共同命中后、第一次读取窗口前，把该checkpoint的sink及recent+spec原始字节复制到新lease的sidecar。目标tag用新lease epoch重新编码；不能复制来源请求的epoch。每层最大323行（当前S64/R256/M3），与历史长度无关。任一source generation/tag缺失即失败，不能用zero或解量化补齐。
4. **物理生命周期**：prefix查询与`touch`保留原生页refcount，restore ticket另保留checkpoint读引用。释放请求sidecar不删除共享checkpoint。native eviction/重分配更新generation且撤销旧hash/record；旧ticket或graph未完成时不得重用其checkpoint字节。reset先撤销可命中目录，再按原生running/reset规则排空/取消，之后允许复用；epoch不能回绕。
5. **设备/图顺序**：writer写BF16内容→publish checkpoint generation/ready→同stream或明确event依赖→restore校验并复制→发布目标tag→attention。host enqueue或CPU协议通过都不证明这个顺序成立。固定restore描述符、lease map、buffer地址必须覆盖batch重排/多请求同时hit/MTP接受位置变化。若restore放在graph外，也必须在graph读取同sidecar前完成且遵守native输入准备事件。
6. **scope一致性**：所有target FULL和draft FULL层的checkpoint及原生GDN状态必须对应同一最终命中长度与同一权重/旋转/布局版本。LoRA/cache_salt等身份继续通过原生prefix-chain hash，不可自创简化prompt hash。

### 容量不能漏记checkpoint

prefix align示例中，native B=768、P=817152 B；若双段研究布局简单从none放宽到align，则B=2×2304=4608，其FULL几何上限已是6倍。每页额外保存323行完整BF16 checkpoint需333336 B，FULL倍率上限变为 **4.26159倍**，尚未计请求sidecar、目录/pin、GDN和碎片。这个方案可以作为正确性基线，**不能作为6–7倍容量结果**。

这不是所有新布局的理论最优上界。进一步重排同一页两段，共786432 B；只保留256行recent BF16及2048 B逐行tag后剩522240 B，恰可存3840个136 B INT2 token，得到5倍页几何。**但这还没有支付sink64的65536 B成本**；当前FULL空闲conv30720 B不够。需要另一个预算内、随首prefix页生存的sink slab和引用协议，且冷prefix的根数量不能假定只等于active请求数。若每页统一预留sink+recent，则768对齐容量仅3072、约4倍。immutable已提交checkpoint能否只存R而非R+M，需要把native提交watermark/MTP末页drop与设备publish顺序连成证明；运行sidecar仍必须保留R+M。这些是待实现的新ABI分析，不能与当前可用布局混称。

减少checkpoint密度需要新协议：完整INT2页可保留原生hash，但最终允许的hit必须落在“所有FULL层均有exact checkpoint且原生GDN也命中”的共同边界。原生hybrid coordinator对FULL命中的 **downward-closed** 假设（命中更长即可直接裁剪到任意较短页）不再成立。必须加固定点求交，变短后重新确认原生GDN；还必须正确保留MTP的peek/drop margin，不能重复丢最后一页。不能随意降低native前缀命中粒度后仍宣称语义/性能无代价。

稀疏record pool要在KV预算前预留并限制容量。只有native完整页仍可命中且精确record可保留时才能广告那个边界；记录用尽不能静默恢复INT2窗口。Checkpoint淘汰必须同步影响可命中边界，不能出现native命中但设备缺窗口。密度/record预算的净收益（额外recompute、复制、目录、GDN checkpoint驻留）未测，不能仅将全页checkpoint除以某个采样因子就宣称达标。

### 当前默认的真实布局字节账

`tools/analyze_memory_64k.py` 已能计算默认APC/align的page-owned snapshot路径，同时保留显式none的历史条件账。`reports/memory_64k_prefix_align_20261012.json` 使用条件参考几何16+1 FULL/48 GDN、TP4，并非当前真机model config确认：65100输入+512输出+3 MTP reserve、0前缀共享时，每请求每rank native FULL为1194676224 B，OSCAR FULL为402855936 B，FULL物理页比 **2.96552×**。原生 `MambaSpec.max_memory_usage_bytes` 在align要求 **2+MTP=5页/层**，计交接previous state；不能继续沿用none的4页。加入该最大live GDN reservation后，native全部KV为1403049984 B，OSCAR为611229696 B，比 **2.29545×**。FULL达到6/7倍仍需进一步减少当前FULL字节约 **2.0233/2.3605倍**。

这不是整张HBM压缩比：权重、graph、workspace及原生临时峰值另列未知。缓存完成但refcount=0的GDN checkpoint留在同一固定BlockPool内，可驱逐、算free，不能再当额外tensor重复收费，也不能凭空当作前缀共享收益。真实命中/保留页数没有输入证据，报告明确未知与0共享比较基线。none/request-window的6.30881× FULL /3.82339×全部KV仍列在条件研究项，加入max_num_seqs=128的所有预留sidecar、status和device-map字节，明确不适用于当前默认APC。

### 本轮已实现与尚缺

独立host原型 `tools/experiments/request_window_prefix_protocol.py`（**不被服务导入**）实现：完整已提交边界的精确字节采集、带scope/hash/page-generation身份的有界不可变目录、in-flight pin及evict/reset拒绝、向新lease epoch恢复的固定行计划、FULL checkpoint与native共同hit的递减固定点、完整字节账。`tests/test_request_window_prefix_protocol.py` 6项通过，含MTP未接受边界拒绝、缺原始row拒绝、分叉位模式、旧epoch拒绝、GDN缺中间checkpoint时不能直接trim，以及6倍被额外checkpoint成本抵消。

**未实施的上线必需项**：native allocation/eviction/touch/reset与checkpoint目录的真实接缝；带MTP peek/drop的native hit adapter和正确computed block table返回；跨FULL/draft层的一致发布；budget-aware NPU checkpoint池；融合边界checkpoint writer、恢复copy/guard kernel与CANN验证；native异步事件和graph输入恢复；NPU命中/分叉/回滚/抢占/reuse/reset与模型质量/吞吐验收。因此研究开关仍不允许与prefix开启同时上线，本轮host原型不是兼容性完成证明。

## 一键入口

既有 candidate 命令不需新参数，两个入口默认均启用原生 APC：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
git pull --ff-only && bash scripts/install_observe_serve.sh --variant candidate
```

两条命令均显式传 `--enable-prefix-caching`，不添加 `--max-num-batched-tokens`。`service_direct` / `install_probe_serve` 等通过 `serve_argv` 的其他入口也使用同一默认。direct 仍无 probe；observe 仍保留真实 NPU 门。不得把 direct 可启动或 host 测试通过写成设备/模型验收。
