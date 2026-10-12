# 默认 APC 与 request-window 兼容设计复核

2026-10-12，只读代码审查；仅新增本报告。未运行 VM、Git、NPU或全量测试。档案依据：#26、#37–49（exact窗口、前缀和MTP生命周期），#127（none/align几何），#128（物理/虚拟块），#129（主机证据不能代替设备完成）。用户未知错误栈未被归因。

## 1. 默认使能 APC 的实际调用链

当前 `configs/target.json` 明确 `enable_prefix_caching=true`。`variant_config` 对省略字段也设 true，保留显式 false；`target_cli.serve_argv` 总是按严格布尔值生成原生正/负参数。direct 与 observe 写有效配置并把该文件传给所有配置接收相位及最终服务；observe 的 native 路径显式关闭 OSCAR 注册。

`experimental_request_windows` 在 `RESEARCH_FEATURE_FLAGS`，不在自动开启的 `FEATURE_FLAGS`。当前 target 没有此字段。candidate 保留显式研究配置，baseline/native 清除研究开关；常规 direct/observe、默认 provider、插件 scheduler 钩子均没有默认开启它的赋值。

默认 candidate 因而采用 `striped_v1`，baseline 为 `canonical_v1`。二者均走：

1. 原生 Mamba 配置从 APC 派生 align，原生 Mamba 状态和 manager 不变。
2. `transform_native_specs` 读取实际 MambaSpec，`HybridPageLayout` 以 mamba/128 的 LCM 对齐；当前 D256/Hkv1/TP4 的 FULL 页是2304，Mamba页768。
3. native coordinator 按实际组 gcd/lcm 得到 hash768、合法命中边界2304；MTP 最后匹配块 drop 保留。
4. `packed_view` 把INT2映射到所属物理页SSM区间；`SnapshotLayout.views` 把exact窗口映射到所属物理页padding区间。它们不依赖请求lease，跨请求命中不会需要从已经释放的请求sidecar恢复。
5. `NativeTokenBudgetRuntime` 只读取原生实际token/graph容量，为workspace预留；不改变APC、Mamba模式、页容量或哈希粒度。
6. `reshape_kv_cache_tensors` 把FULL交给插件，把原始GDN对象和原始分配交回原生reshape。目标reader/writer均使用实际 `state.spec.block_size`，没有在本次改动中硬编码2816。

未发现本次新默认造成默认服务必然进入 `request windows require prefix caching disabled` 的调用链。这里的结论是源码路由自洽，**不是启动、真实命中、NPU数值或图回放已经通过**。

## 2. 已排除的两种错误归因

- 原生scheduler的Mamba chunk对齐使用 `cache_config.block_size=768`；FULL物理页2304不等于必须一次调度2304。因此不能以2304大于可能的2048预算，直接推断deadlock。
- `patch_mamba_utils.py` 在A2替换了原生fused postprocess算子。`patch_mamba_config.py` 的历史注释不能单独证明当前APC/GDN必然无法编译。没有更改GDN的理由。

## 3. 实际未完成事项

### 显式 request-window + APC 仍不兼容

`AscendRuntimeProvider.seal_request_windows` 明确只接受 prefix=false、mode=none；`DualRegionPageLayout.__post_init__` 也只接受none。现有 `RequestWindowLeases` 的释放/重用以活跃请求为边界，没有跨请求持久checkpoint目录。当前双区把原per-page snapshot区用作第二段INT2，原精确状态已不在那里。删掉两处guard会让cache-hit的新epoch缺少合法exact窗口，不能称作兼容实现。

### 新默认不能沿用旧的none内存账

`tools/analyze_memory_64k.py:115` 明确拒绝 prefix=true / mode!=none。因此该工具读取新默认target会报错，其历史6.3倍FULL投影不能作为新align默认的容量结论。需要给APC/align单独核算FULL2304、GDN checkpoint驻留、native pool分组，以及真实prefix hit数量；不能为了让账本运行把新默认改回false。

## 4. 可实施的 request-window + APC 设计必须具备的状态

要让新请求在prefix长度L处恢复、且不重新计算/有损恢复精确窗口，每个仍可命中的合法边界必须保有：

- 每FULL层独立的、原始BF16 exact K/V：sink `[0,min(S,L))` 与 recent `[max(S,L-R),L)`；target和MTP层不可混用。当前INT2不能无损推回这些信息。
- 完整边界的有效性、实际逻辑L、物理页身份及重用generation；有效状态的发布晚于全部payload写完成。跨层仍按原生同一合法命中边界协调。
- 活跃请求sidecar仍保留 `S+R+spec`，服务新lease时把恢复内容写入新epoch；不可直接继承旧lease tag或把物理页ID当epoch。
- 全前缀的sink可由第一逻辑页/独立共享root对象持有，但其存活期必须覆盖所有引用它的已缓存prefix，不得只覆盖当前活跃请求。
- 缓存checkpoint由native页refcount/hash生命周期拥有；request finish、batch换位或抢占不能提前删除仍可命中的checkpoint。evict/reset/物理ID复用必须撤销旧generation；forced reset与在途graph/store还需要同一stream/event顺序。

### 发布和恢复时序

1. 冷prefill/store期间为**每个可能被native接纳的完整边界**发布checkpoint。一个chunk跨多个完整页时，不能只在chunk末保存当前ring；较早边界的exact尾部可能已被覆盖。
2. chunk跨边界但当前输入不足R时，组合旧exact sidecar与当前原始BF16输入；不能从INT2补足缺失部分。
3. 仅在满足原生computed/accepted水位后允许checkpoint命中；MTP暂存写和拒绝token不能提前发布。现有MTP完整页drop协议不应被替换。
4. 新命中请求获得私有lease和epoch，同stream恢复其sink/recent，再执行query attention；恢复只复制有界exact窗口。共享的压缩页和checkpoint保持只读，后续写入私有后缀与私有sidecar。
5. 使用固定地址、设备端动态metadata支持graph重放。并发命中相同prefix、首MTP/后MTP、请求复用、reset和部分命中均要验证。

## 5. 不可忽略的容量账

当前D256/Hkv1的exact K/V每行1024B，R256每边界至少262144B；sink64再65536B。tag/目录、双缓冲及活跃sidecar另算。缓存请求完成后这些状态仍可能需要保留，数量不能以active K32或max_num_seqs直接代替。

以当前原生SSM393216B+padding393216B两区为例：

- 两区总786432B；在每物理页保留R256 exact K/V及256个int64 tag后，余522240B可存3840条136B INT2，恰为768对齐。**这只是recent-only的5倍页容量上界**。
- 5倍方案仍欠sink存储。当前unused conv30720B容不下65536B sink；独立root slab必须显式预算并管理引用。
- 若每物理页统一保留sink+recent+tag，则剩余空间只能容3072条768对齐INT2，即4倍页容量；若还保留3个投机行也仍落在3072档。
- 当前未启用研究布局的5632-token/旧6.3倍投影不能在增加APC checkpoint后原样沿用。若改为独立checkpoint池或降低持久checkpoint密度，需要重新核完整FULL/全部KV账和原生hash可达性；不能省掉必要状态后继续声称兼容。

以上是布局成本推导，不是最优性证明、NPU性能或真实HBM结果。可以探索更紧凑的checkpoint目录/共享sink/有界保留政策，但改变native cache-hit承诺或主动少缓存边界须先明确语义和验收条件。

## 6. 合入前的具体证明门

- 当前传统物理快照路线：真实prefix冷/热命中、两个不同suffix、正context mixed split、MTP拒绝0–3、页复用、reset、相同graph地址metadata变化。
- 新RW/APC路线：在上述基础上加入chunk跨多个边界、边界左侧不足R的新token、checkpoint在原request结束后继续存活、恢复新lease epoch、所有layer状态一致，以及全部持久字节计入KV预算。
- 不应因本报告未发现默认路由冲突，宣布用户未知真机错误已解决。
