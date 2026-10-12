# 64K 独立紧凑 FULL 池与稳定请求精确环设计

状态：**源码级设计，未实现，未获得 NPU/图/服务验收**。参考基线为本工程 HEAD `4defe20` 与 `references/` 中指定原生树；本轮其他 workspace 优化可并行进行，本文按函数名与参考行号定位。未修改原生文件或 GDN 数学，未运行 VM、性能模拟或设备测试。

**评审修订：优先实施第11节的“双段FULL＋worker lease”方案。** 它保留唯一原生BlockPool、原生GDN spec、17张量共享布局，只扩充同一FULL物理id可用的packed区域。前10节保留为进一步去GDN padding的独立池备选；其多池协调器不是双段方案的前置条件。此前把EngineCore lease传输和coordinator重写视为必要的判断范围过宽，本节及第11节予以修正。

档案引用：#26/#37–49/#69 决定 exact ring 必须有稳定所有权和原生 MTP 提交/回滚语义；#34/#36/#127/#128 决定物理页、metadata128、图容量分开；#129/D.4 决定在线 metadata 更新在设备批量执行，不恢复全历史；#142 决定 dummy 与真实 capture 独立标记；#145/#154 决定异步所有权必须靠正确事件证明；#152/#155 决定字节收益与完整服务性能分别验收。D.4 的 dequant6499.8–6655.1ms、host prepare约725ms、stores约209ms 是禁止回归的历史失败，不是新方案速度估计。

## 1. 结论与范围

**若要按FULL/GDN分别计费并去除GDN自身padding，需要外部多池协调器，不能只改 worker tensor 分配。** 此独立池备选保留现有一个 FULL 组及三个 GDN 组的层身份/顺序，各组独立使用 `BlockPool`；FULL 使用紧凑 INT2 页，GDN 继续使用原生 `MambaManager`、BF16 conv/SSM shape、MTP3 状态块和原生计算。初版不合并 GDN 三组，减少 metadata group 重排风险；每层分配独立 raw tensor 已能消除17张量共享池中的空层槽与页 padding。较小的双段方案不去GDN padding，因此不需要这项重构。

当前目标限定 `prefix_caching=False`、`mamba_cache_mode=none`、TP4、PP=DP=DCP=PCP=1、无 KV transfer/CPU KV offload。维度、层数、dtype 从运行时 spec 读取。`max_model_len=262144` 和 MTP3 不改变；65100输入/512输出只用于池容量政策与本轮账本，不作为支持长度上限。

现有 prefix/align 路径仍是已有独立配置。显式开启 compact 后若遇 prefix/align、未支持 connector 或 cache spec，必须初始化时报错；不能悄悄换回旧池、关闭 prefix、丢掉快照或修改 GDN 模式。

## 2. 已确认的原生限制

| 接缝 | 原生事实 | 对紧凑池的影响 |
|---|---|---|
| `KVCacheConfig` | 仅一个 `num_blocks`，另有 tensor列表和group列表；`kv_cache_interface.py:855–875` | 不能直接表达 FULL有NF页、GDN各有NG页 |
| `KVCacheTensor` | `size`与`shared_by`是显式字段；同文件 `:830–836` | 可合法表达逐层不同size，不要求worker一定维持17个共享tensor |
| `UniformTypeKVCacheSpecs` | 必须同类型槽需求、同block_size；page_size为内部层字节之和；`:721–767` | 不是混合FULL/GDN的通用容器，不能靠包装绕过不同生存期 |
| 全局池 | `KVCacheCoordinator.__init__` 创建唯一BlockPool并传给所有manager；`kv_cache_coordinator.py:90–118` | 相同id只能属于一个活跃组，当前每个id在17张量中收费 |
| admission | manager需求按组求和；coordinator `:162–184`；`KVCacheManager.allocate_slots :354–398` 用一个free数比较 | 不同成本与不同池的空闲块不能求和后判断可分配 |
| 预算 | `get_kv_cache_configs` 先unify/group，再按统一页算块数、跨rank取单一最小值；`kv_cache_utils.py:2005–2085` | 必须替换这一特定配置的规划分支，否则tight FULL会被重新对齐或拒绝 |
| worker reshape | GDN通过raw/P得num_blocks，要求≥全局config.num_blocks；Ascend `model_runner_v1.py:4681–4714` | 小GDN池不能套大FULL池的全局num_blocks；需按组投影调用原生reshape |
| metadata | block table本来按group提供；GDN builder `ops/gdn_attn_builder.py:553–558,642–646` 只从自身group取状态id | 各group的局部id可以相同，只要地址域不混用 |
| 调度输出 | `NewRequestData.block_ids`与cached `new_block_ids`都是按组tuple；`sched/output.py:31–59,114–124` | 可继续使用原生传输格式，不需把请求id塞进attention tensor |

`get_kv_cache_groups` 的普通混合分支在 `kv_cache_utils.py:1680–1684` 明写单尺寸假设。`group_and_unify_kv_cache_specs` 当前只针对DeepseekV4（`:1431–1441`）；不得把Qwen3.5伪装成MLA或Deepseek来使用其特殊分支。

## 3. 为什么不保留同一个未改造的全局 BlockPool

如果保留全局id，且所有GDN原生计算直接以id索引状态tensor，每个GDN tensor必须能容纳最大FULL id。即使该id只属于FULL，GDN backing仍须为其留下位置，重新产生随FULL token容量扩大的空洞。给各组独立tensor但仍分配同样全局N也不能解决。

理论上可以保留全局**逻辑**id，再加每组 dense physical-id映射与加权容量管理。这样必须对FULL和GDN各自的metadata做地址翻译、实现全局多资源admission；实质上已经是多池系统，多一层映射和图状态。当前无理由优先选择它。

推荐每group保留一个未经修改的原生BlockPool实例：FULL和三个GDN组各自局部id从0开始，各有null block。**复用的是BlockPool实现及每组manager语义，不是现有唯一BlockPool对象或跨组共享17张量布局。** 相同整数id跨组重复合法；任何原来接受裸id集合的全局接口必须明确处理group，不得交给错误的池。

## 4. 外部实现模块与精确接缝

下列是拟新增组件，不表示代码已经存在。

| 外部组件 | 输入/输出与职责 | 应接入的现有位置 |
|---|---|---|
| `CompactFullAttentionSpec` | B128、每token真实PR INT2字节；不引用GDN C/M/P；独立cache-format版本 | 当前 `integration/specs.py`；经`KVCacheSpecRegistry.register`注册原生FullAttentionManager或明确子类 |
| `CompactPoolPlan` | `group_num_blocks`、每层page_bytes、ring容量/字节、metadata/null成本、rank共同plan签名 | 顶层可序列化dataclass，进程间传递，不放进未知VllmConfig字段（#31） |
| `CompactKVCacheConfig` | 原生KVCacheConfig子类＋显式plan；保留原生group/tensor字段 | 外部覆盖EngineCore使用的`get_kv_cache_configs`和`generate_scheduler_kv_cache_config`两个调用接缝 |
| `CompactKVCacheCoordinator` | 每group自己的原生BlockPool与manager；返回需求向量；逐组free/cache/new_step | 外部替换compact配置的coordinator工厂；不能只调父类构造后继续用原全局池 |
| `CompactKVCacheManager` | 保留请求token/MTP计数方式，替换单池admission与原子分配；统一对外usage/events | `Scheduler.__init__`中 `KVCacheManager(...)` 构造接缝；原生 `scheduler.py:228–244` |
| worker provider | 每层独立raw、FULL packed view、原生GDN state view、ring/映射固定地址 | 当前插件已经包装 `get_kv_cache_spec`、`_allocate_kv_cache_tensors`、`_reshape_kv_cache_tensors`；`plugin.py:105–143` |
| ring metadata和算子ABI | blocktable首FULL页→ring lease，完整逻辑位置tag；所有读写算子统一新格式 | attention任务准备、所有q4/q1/C4/C16 reader、store、current-only与MTP metadata |

EngineCore实调链在 `references/vllm/vllm/v1/engine/core.py:243–290`：worker specs → 原生可用显存profiling → config planner → scheduler config → worker initialize。新的planner必须保留原生profiling，再从其返回预算分配compact各池，不能事后在旧池外另分配一整份compact池。

外部hook应与现有插件一样延迟安装、保留原callable、检测冲突、失败恢复并可unregister（当前`plugin.py:309–330`）。EngineCore与Scheduler使用的是import后的局部名字，**只patch定义模块未必影响已绑定别名**；必须在这两个消费模块加载后安装确切绑定，worker与EngineCore均校验plan/schema版本。配置不含compact标记时只执行原实现；compact标记存在但能力不齐则失败，不做自动运行期降级。

`CompactKVCacheConfig.num_blocks` 不得被当作真容量。推荐仅为框架兼容保留明确的保守值，并要求所有compact容量决策读取plan。所有调用原生每组分配/reshape的局部config必须将 `num_blocks` 写成该组真实Ng。跨rank按**每组**取最小Ng，重新核算tensor.size和每组null；不能继续执行原生`:2072–2085`按一个N成比例裁剪所有tensor。当前TP4/PP1范围之外先失败，避免未经证明的PP投影。

## 5. 原子多池 admission 与保留 GDN 语义

保留现有四组的名字、顺序、MTP标记和每组manager类型。对GDN使用原生`MambaManager`对象，其 `get_num_blocks_to_allocate` 与 `allocate_new_blocks`继续执行原逻辑：mode=none增加三个完整state block，见 `single_type_kv_cache_manager.py:1073–1087,1125–1132`。不把它变成普通FULL、不改conv/SSM更新、接受token索引或state copy函数。

每次分配按以下事务执行：

1. 从同一request的原生computed/new/lookahead token数计算各manager需求 `d[g]`；ring仅在request首次获得FULL页时需要一个lease。full-sequence admission和普通一步分配都使用同一套向量规则。
2. 对每组检查 `d[g] <= free[g] − reserved[g]`，并检查ring free lease；任一不满足则返回原生“暂不能接纳”，不修改任何组。
3. 在EngineCore调度线程同一事务内按确定group顺序调用各原生manager分配。检查完后若出现意外异常，撤销本事务新增块与lease，不释放request旧块；异常向上传播。不能先成功分配FULL、GDN失败后留下半请求。
4. `free/get_blocks/remove_skipped_blocks/new_step_starts`逐group调用原方法。`KVCacheBlocks`仍是四组tuple。preempt/abort/finish必须同时释放lease；worker保持原生stream顺序，上一批计算完成前不得复用新owner。

原生 `reserved_blocks` 以及 `_request_remaining_blocks` / `_inflight_prefill_reserved_blocks` 是标量协议（`scheduler.py:2130–2151`）。初版无KV connector，必须在入口验证不走这些remote-load保留分支；外部manager若收到非零标量reservation必须拒绝，不能随意均分。未来支持connector时升级成group向量和对应scheduler接缝。

对外指标不能继续用一个pool的 `get_usage()`：报告FULL页占用、三个GDN池占用、ring lease、总**真实字节**占用。`evict_blocks(set[int])`、`reset_prefix_cache`、events/metrics不能直接合并重名id；prefix关闭时明确返回正确的无prefix行为，需记录的事件携带group。不存在合法的“用sum free伪装一个BlockPool”的捷径。

## 6. GDN 去 padding：可以不改计算，但需要诚实地改变存储 spec

原生 `MambaSpec.page_size_bytes` 在没有padding时恰好返回各shape×dtype之和（`kv_cache_interface.py:616–636`）。外部transform可对原MambaSpec做 `dataclasses.replace(page_size_padded=None)`，只改变物理padding字段；保留shape、dtype、block_size、mamba_type、mode、num_speculative_blocks等所有语义字段。**这保留GDN计算和状态内容，不保留原spec对象身份/旧P字节**，两者必须明确区分。若更高约束要求连对象身份和旧P都不变，则去GDN padding被该约束阻断。

每GDN层分配 `Ng × (conv_bytes+ssm_bytes)` 个byte；以原生SoA方式形成 `[Ng,*conv_shape]`、`[Ng,*ssm_shape]`连续张量。通过局部投影config，复用原生reshape `model_runner_v1.py:4681–4714`；调用期间只遍历当前group对应attention group，结束恢复原runner字段。仅改传入config而不限制runner的`_kv_cache_spec_attn_group_iterator()`仍会遍历其他组，是明确的实现陷阱。

也可在外部provider中逐字按已验证shape/dtype构造上述两种view，但必须与原生reshape对拍其shape/stride/offset，不重新推断GDN尺寸。原生GDN builder从自己blocktable中取前1+num_spec列，随后 `gdn.py:353,404,444` 按这些状态索引读写；这些计算不依赖FULL页字节，也不需要修改原生文件。

初版继续保留3个GDN组，每组16层各独立tensor。这样没有17宽padding槽；不必先把48层合并一个group。将来合并虽看似可行，但会改变builder/group组织，不能作为本版减少工作量的未经验证假设。

## 7. request-stable exact ring 的具体所有权方案

建议先采用**首FULL页锚点＋稠密lease映射**，不改FULL manager的页选择策略：

- prefix=false时原生FullAttentionManager保留请求的首FULL页直到free，且 block table第一列随row移动一起移动；request重排不改变锚点。依据：普通FULL不跳过前部块；`SingleTypeKVCacheManager.free :363–378`在释放请求时才释放这些块。
- 调度器按request ID分配稠密ring lease；worker从已有 `NewRequestData(req_id, block_ids)` / resumed数据建立相同映射，lease只在创建/释放事件变化。图内不读取Python request ID。
- 固定地址NPU表 `anchor_to_lease[NF]`、`anchor_generation[NF]` 把首FULL页映射到lease和generation；设备task准备通过该group blocktable第0列读取锚点，生成当前token的ring索引。入队/退出仅批量更新短metadata，不回读KV、不新增逐层Python逐request循环。
- 每个FULL层持有 `[lease_capacity, S+R+mtp, Hkv, D]` 的BF16 K/V，另有逻辑位置tag和generation。一个request的target16层和draft层分别保存自己的数据，不能共用数值内容；允许共用lease编号。
- 身份检查是 `(FULL group, anchor block, generation)`，不是batch row或仅裸物理id。block重用时generation改变；epoch不匹配立即报错。不得因为某个tensor地址相同就恢复旧request窗口。

跨进程lease一致性不应靠各worker独立“猜测下一个空slot”。推荐plan定义相同确定性分配器，按原生scheduler输出中的有序new/resumed/finished事件使用request ID处理，并每步校验事件/lease摘要；更稳妥的后续实现是外部扩展调度输出携带稠密lease和generation，再在worker同一update-states阶段批量写固定映射。两者需选定并完成序列化/异常契约测试后才能实现，不能同时维护互不校验的两份来源。

初版优先**显式扩展调度输出**，以EngineCore为唯一lease权威。外部定义可pickle的SchedulerOutput子类，保留全部原字段并增加ring admission/release事件；只在本插件worker之间消费，不能擅自把字段放到原生固定schema RPC而不验证传输。若目标executor拒绝该子类，则在同一已有RPC载荷的受控外部wrapper中传送顶层plan/event结构；协议未证明前这是落地阻断，不能以worker本地推测补上。

另一方案是限制首FULL块来自固定anchor id区，`lease=anchor−1`，可省映射传输，但需要修改FULL manager首次分配和pool free-list规则，admission还要分anchor/body两类空闲块。它不是“直接复用原生BlockPool”，故本版不优先。

### 精确值与 MTP 生命周期

环行索引必须使用**完整逻辑位置**，不能继续用旧 `position % physical_page_size` 的page snapshot公式。sink64固定一次；recent环保留256+3行。tag匹配完整position/generation。

prefill依然先完成history/window/current attention，再发布本chunk的新数据（现有`impl.py`的read-before-write顺序）。新writer为INT2编码全部需要持久化的新token，但exact ring仅写本次结束时应保留的sink与最后259行；避免16K并行writer对同一ring槽多写造成不确定最后写入。MTP拒绝最多3个ahead写入时，259行必须仍覆盖回滚后的256行。不能从INT2解码恢复“精确”BF16，不能通过扩大ring到全历史回避生命周期证明。

所有reader/store需要新ABI：普通q4/q1、C4/C16、current-only首块、first-MTP、later-MTP以及图捕获都必须消费同一lease/ring格式。旧 `SnapshotLayout` 要求ring能装入物理页（当前323>拟B128），因此直接把其 `page.block_size` 改128会报错；独立request-ring布局类与新kernel地址逻辑是不可跳过的实现项。

## 8. zeroing、dummy 与图的边界

原生新块清零输出是裸id列表：`KVCacheManager.take_new_block_ids :570–575`，Scheduler将其放到 `new_block_ids_to_zero`。Ascend zeroer `worker/utils.py:95–131`跳过Mamba，但假设FULL是独立K/V tuple且统一页stride。新的packed FULL不是该格式，必须从原生zeroer枚举中明确排除，并提供compact自己的packed/tag初始化。不能把某组local id送给会清所有FULL组的旧zeroer。

当前原生manager只对**精确类型** FullAttentionSpec/TQ/MLA追加zero id（`single_type_kv_cache_manager.py:284–289`），自定义FULL子类不会自动获得这一机制。因此不得依赖“原生应该会清”；compact初始化/重复分配必须自证所有可能被读取的packed metadata和ring tag有效。GDN仍使用原生计算的has_initial_state/accepted-state处理，不在其算子前擅自擦除有效状态。

raw pools、ring和映射在capture前分配，图中地址固定。dummy与capture未分配真实request lease，必须以显式dummy/capture标记写负slot和invalid lease，执行已设计的padding路径（#142）；不能访问真实ring0。回放前批量更新相同地址映射，重排/finish后不得改变旧graph的tensor地址。所有临时GDN group投影和runner字段包装必须在warmup前恢复。

## 9. 预算与64K容量政策

定义FULL全部层的单页成本 `PF=sum(full_layer_int2_token_bytes)×B128`；每GDN组单状态页成本 `PG[g]=sum(layer_conv+layer_ssm)`。选择ring/request-state容量R与FULL页NF后：

```text
NG[g] = 1 + R × (1 + mtp3)       # 每GDN组含自己的null block
pool_bytes = NF × PF + sum_g NG[g] × PG[g]
ring_bytes = R × sum_full_layers (323 × BF16_KV_row_bytes + tags/epoch成本)
total_new_bytes = pool_bytes + ring_bytes + anchor/lease表 + 其他明确metadata
total_new_bytes <= min_rank(本轮profiled_available_KV_bytes)
```

若ring在profiling前已经分配，budget中不再扣第二次；若在planner之后分配，就必须在以上预算中显式扣除。workspace/rotations按同一原则。NF包含FULL null block，三个GDN池各有自己的null，不再按旧全局池只计一次null。

R不能默默设为“Running26”或客户端32。可在显式compact容量政策下用本轮65100+512+3密度选出使 `min(R, floor((NF−1)/513))` 最大的R，同时不超过原配置max_num_seqs；还必须保证至少一个max_model_len262144请求可由NF容纳。R是状态池物理容量，max_num_seqs仍为原128的调度上限；池不足时等待，不修改请求长度或MTP语义。若政策要求保障128个短请求同时驻留，就必须真实支付128份GDN与ring成本，不能只按32份计账。

参考几何的活跃请求账见 `reports/memory_64k_analysis_20261009.md`：FULL约150.186MiB，全部原生GDN状态77.625MiB，总227.811MiB；该数没有包含本设计新增generation/lease表、allocator对齐、各池null与空闲预留，不能把它当实际分配总量。FULL相对逻辑BF16约7.253倍；全部KV相对原生物理字节约5.699倍，仍不能宣称全部显存6–7倍。

## 10. 必须完成的最小实施顺序与否决条件

1. **先做host资源契约**：独立planner/config序列化、四group blocktable返回、逐池admission、同一请求的成功/失败原子性、跨rank计划一致性；验证FULL满但GDN空/GDN满但FULL空都不会半分配。无设备成功声明。
2. **证明GDN原生等价**：tight raw经原生reshape得到相同dtype/shape/stride语义；MTP4状态索引、初态、accepted回滚、preempt/resume重算保持原生行为。这里不改GDN内核。
3. **实现新ring ABI与全部路由**：固定generation所有权，q4/q1/C4/C16/first/later MTP/current-only全覆盖，长chunk只写唯一保留槽，失效tag fail-closed。有任一路由仍期待旧page snapshot则不允许启用compact。
4. **短时真NPU验收**：仅用合成算子/图/短服务probe，验证所有权、非法输入、跨页、拒绝深度0–3、请求重排/复用、同址图变输入和真实分配字节；冻结oracle与精度不变。无需重跑已完成AISBench。
5. **交付开关**：新能力通过前不进默认candidate；通过后由`tools/serving_variants.py`唯一开关同时驱动两个candidate入口。显式compact与prefix/align冲突则失败，旧合法模式仍使用其显式原配置，不运行期暗切。

独立池备选的当前阻断是：多池config/EngineCore/Scheduler协议尚未实现、所有CANN reader/writer尚无新ABI、真实设备分配与精确回滚尚未证明。第11节进一步确认：仅worker可完成lease身份管理，不必新增EngineCore消息字段。已确认可复用的是原生每组BlockPool/Full与Mamba manager主体、按组blocktable传输、GDN张量view和原生GDN计算；没有找到必须修改原生源码或GDN数学的源码级要求。

## 11. 优先方案：双段 FULL packed + worker 稳定 lease

### 11.1 地址与原生池计费可以完整保留

对当前none模式、单层raw共N个物理页，设C为conv bytes、M为SSM bytes、T=P−C−M为原生尾padding。原生GDN继续使用：

```text
GDN conv(id) = [id*C, (id+1)*C)
GDN SSM(id)  = [N*C+id*M, N*C+(id+1)*M)
```

FULL只在自己拥有的物理id上写两个区间：

```text
segment0_base(id) = N*C       + id*M
segment1_base(id) = N*(C+M)   + id*T
B0 = floor(M / (slot_bytes*Hkv*128)) * 128
B1 = floor(T / (slot_bytes*Hkv*128)) * 128
B_full = B0+B1
```

当前条件几何M=T=393216B、slot136B，两个segment各容纳2816 token，总B5632；packed有效字节765952B，P仍817152B。C30720和两个10240B对齐尾部不写。所有segment按实际spec计算，显式拒绝不满足两段容量/128对齐的新几何。

原生唯一BlockPool保证不同活跃group不能同时拥有同一个物理id；FULL的SSM段和GDN的SSM段因此不会同时归不同请求所有。第二段是按id独立划分的原生padding，不是整个padding的任意共用区。每id仍在原17个共享tensor上付同样P字节，native `KVCacheConfig`、coordinator、Full/Mamba manager和GDN spec无需重写。**必须替换的是外部FULL spec的物理B、地址模型以及所有读写kernel；不能仅把旧单段B改大。**

该方案不删除任何GDN padding，只在FULL占用的id上利用它。因而第3节关于“独立池无法直接复用唯一全局id”的论证不否定本方案：这里没有独立池、不同id成本或新增按池admission。

### 11.2 metadata128 没有源码级阻断

原生 `vllm/v1/worker/block_table.py:47–68` 只要求物理B能被kernelB整除，然后设 `blocks_per_kv_block=B/128`。5632可表示为44个virtual128。外部spec的 `copy_with_new_block_size(128)`仍只建立metadata view，不改变P/两个segment。读写根据virtual block还原物理id、inpage，再以 `inpage<B0`选择segment，否则用 `inpage−B0`访问第二段。

每段边界2816本身是128的倍数；已有按virtual128边界分批的DMA不会跨segment误读。packed stripe的16行边界也被128整除。slot解析必须用新B5632，MTP后续slot→实际逻辑position仍通过本请求virtual128 table匹配，保留#128修复。

none模式原生GDN block_size262144保持原样。`lcm(5632,262144)=2883584`，与旧2816的LCM相同；这不是开启prefix的许可。prefix/align下的复用快照已被移走，故双段模式必须在初始化明确拒绝prefix=true或mode不为none。关闭双段的显式旧配置继续保留原页快照。

### 11.3 opaque kv_cache 在当前 FULL 接缝可行

两个segment在SoA raw中不连续，不能伪造一个跨5632行的普通4D strided packed view。可以绑定**真实存在、固定地址、与raw同storage的byte tensor**作为所有权token；例如raw的一维view。不能用无实际storage的假tensor或越界as_strided。

原生 `worker/utils.py:516–518`（及Ascend `patch_qwen3_next_mtp.py`的同等绑定）只把kv_cache赋给attention层；`attention/attention.py:670–688,749–761`取出tensor后直接传给外部impl，没有强制K/V矩形布局。本工程FULL `forward_includes_kv_cache_update=True`，也不执行普通原生KV写入。外部impl改为验证ownership tensor指向预期raw/format，再把真实raw、segment0/1参数传新算子即可。

必须审计并限制所有可能直接解释kv_cache形状的辅助消费者：native zeroer、KV connector、CPU/offload、cache共享捷径、调试dump。当前无connector/offload，不能由此宣称这些功能也已兼容。核心FULL前向没有发现必须改原生源码的障碍。

### 11.4 原生 zeroer：当前目标不启用，不构成硬阻断

较早第8节只看zeroer内部而未看Ascend worker入口，结论需要收窄。指定原生 `vllm-ascend/worker/worker.py:930–938` 只有 **method=eagle3且num_speculative_tokens>1** 才初始化 `_kv_block_zeroer`；当前qwen3_5_mtp（若规范化为mtp也一样）不满足。`GPUModelRunner._zero_block_ids :1103–1106`也只有对象存在才执行。另原生 `single_type_kv_cache_manager.py:284–289` 用精确type检测，当前OscarFullAttentionSpec不会自动生成普通FULL清零id。

因此当前目标中没有证据表明native zeroer会覆盖第二段padding。双段初始化仍应显式检查上述接缝/方法和opaque consumer资格；若切到eagle3或检测到已有zeroer，必须停止或接入经过验证的外部排除/专用清零实现，不能假装兼容。原生zeroer本身的K/V tuple与统一stride假设（`worker/utils.py:95–131`）确实不适用于opaque tensor，但不是本目标必经路径。

### 11.5 worker-only lease 协议，复用已有 SchedulerOutput

无需新增EngineCore传输。源码证据：

- `scheduler.py:940,945`每个正常schedule发布当轮常规preempted集合和finished集合；`:1038–1040`通过替换set而非原地clear保留输出引用。
- Ascend `_update_states :729–744`执行原生异步rewind修正后调用父类。父类在 `gpu_model_runner.py:1135–1149`删除finished；`:1160–1180`从batch移除未调度请求但保留cached request状态；`:1368–1379`在resume时替换block IDs；`:1436–1447`依次add、condense、reorder、refresh，随后返回。
- Ascend `execute_model :2025–2049`在 `synchronize_input_prep`范围调用 `_update_states`，然后`:2097`准备输入；零token步也在更新之后才`:2060–2074`返回。因此posthook处已取得稳定本步request行顺序，完成/抢占事件不会因零token步被跳过。

具体协议：

1. 每worker持久保存 `req_id → (lease_slot, epoch)`；lease数据只供本worker的FULL层使用，TP不同rank无需相同slot数值，只需各自映射正确。按有序事件做确定性分配有利于诊断，但不是新的跨rank通信要求。
2. 外部hook先保存事件，调用原生 `_update_states`，完整保留其返回的deferred-correction callable。随后先释放finished和常规preempted旧owner；resumed即使没有此前preempt事件，也必须使旧epoch失效并重新获得lease。若finished与同ID new同一步重叠，必须先结束旧owner再建立新owner，对齐父类`:1143–1147`。
3. **不能因某request不在input_batch就释放lease。** 原生明确区分暂未调度与真正抢占；暂未调度running请求仍持KV，必须保留ring。`scheduled_new_reqs`也可能是streaming continuation（父类`:1195–1199`），仍属同owner时保留lease，不机械重置。
4. 原生更新结束后，按 `input_batch.req_ids` 的最终顺序生成固定地址NPU `window_slots[rows]`、`window_epochs[rows]`，padding行为invalid。只更新小metadata，不拷贝ring K/V。attention层不逐request执行Python逻辑；新的metadata与current-only、mixed分段、first/later MTP子视图同步slice，不能继续使用父batch第0行映射。
5. lease池按max_num_seqs及真实可驻留owner边界预预算；不足必须明确失败并报告泄漏/超容量，不能借用另一request槽。普通常规抢占在下一步已有事件释放；强制reset另见下一节。

### 11.6 找到的协议边界：强制 reset 不是常规940事件

`Scheduler.reset_prefix_cache(reset_running_requests=True)`在 `scheduler.py:1953–1977`直接调用 `_preempt_request`，没有把这些请求加入下一次schedule内部的局部 `preempted_reqs`。因此“940一定覆盖所有抢占”不成立。已resume的请求可从 `resumed_req_ids`重建epoch，但仍在waiting的强制抢占旧owner可能暂时占据lease，导致容量泄漏。

初版可在双段模式下通过小型外部API wrapper明确拒绝 `reset_running_requests=True`/相应force-preempt管理操作，普通完成/取消/内存抢占不受影响；或者为该API添加明确的worker lease-reset通知（保持原生请求/状态复位顺序）。不能靠“缺席batch就释放”修复它，也不能默默增大lease池掩盖。该边界不要求改coordinator；当前65100/120/K32常规请求流程没有显示必须调用该管理API。

### 11.7 epoch 免逐次清空的条件，以及同stream顺序

ring tags在初始化时一次性设invalid。之后以一个64位tag编码epoch和完整logical position：本目标max_model_len262144需18位position，可在非负int64的其余45位存epoch；epoch从非零递增、溢出显式报错。这样维持现有每行8B tag成本；不要为省位直接截断position或允许epoch回绕。每次lease复用不必清空整环，旧tag会因epoch不匹配而失效。读者必须先验证tag再读取BF16值，store必须按既定依赖发布K/V与tag。

**epoch只能阻止旧数据被当作新数据，不能阻止旧in-flight writer覆盖新数据。** 安全条件是旧owner所有FULL/MTP读写、下一步映射copy、新owner读写按同一NPU current stream排序。现有CANN launch使用current NPU stream；外部hook不能把ring metadata复制到无依赖的旁路stream。

原生 `synchronize_input_prep`（`gpu_model_runner.py:3704–3716`）等待上一轮CPU→GPU copy事件，保护复用pinned CPU输入；posthook位于其范围内，应把新的小表H2D也纳入该事件。NPU同stream顺序则保护旧模型kernel与下一步metadata/KV的先后，不需要额外每步全局同步。指定Ascend graph replay在 `acl_graph.py:243–257`还保留自己的current-stream同步条件，但不能把该条件当旁路stream写入的保护。

若未来引入DBO、多执行线程、独立update stream或不同MTP stream，需显式event依赖或隔离metadata buffer后才能开启本方案。初版按当前单执行链、TP4/DP1/无CP/无KV transfer范围核实；不能把“async scheduling”误解为允许任意stream并发复用同一ring。

### 11.8 精确环与预算验收

精确窗口按第7节read-before-write、sink64/recent256+mtp3、完整logical-position tag执行；新writer只发布本chunk最终保留的唯一环槽，避免长chunk多个writer争同槽。所有q4/q1/C4/C16/current-only及MTP路径均须切到双段+sidecar reader/writer，版本标签阻止新读者配旧单段writer。

条件几何下，输出末FULL12个物理页/层/rank：池158.977MiB + 单份ring5.404MiB = **164.381MiB/request/rank**；相对原生FULL物理1139.332MiB为 **6.931倍**，相对BF16逻辑FULL约6.626倍。GDN池仍158.977MiB，因此全部KV约323.357MiB，相对原生全部1298.309MiB仅 **4.015倍**，不能称全部KV6–7倍。

固定sidecar若为128个slot预留，真实分配是 **691.737MiB/卡**（另加小metadata），必须在KV profiling前预分配或从预算中显式扣除，不能在KV池分完后追加。在K32时若按真实128槽预留摊销，而非按32个活跃窗口摊销，全部KV条件比降至约3.823倍；FULL对应物理分配比也应按相同占用口径计算。这不会否定方案，但禁止以单活跃request账冒充整个已分配pool压缩比。

**推荐实施顺序修订**：先实现双段地址host契约、worker lease事件协议及opaque资格；随后开发统一新ABI并验证精确窗口、跨segment、MTP拒绝与同址图复用；最后进行短NPU/图/服务与实际tensor字节门。无需先重写EngineCore/coordinator或GDN spec。独立池/去GDN padding保留为后续进一步收益方案。

## 12. 已实现的 sidecar host 描述与发布接缝

本节对应独立新增 `oscar_ascend/request_window_storage.py`、`tests/test_request_window_storage.py`，**尚未接入runtime/plugin/CANN**。已重读本轮指定档案#26–49/#69/#127–129/#137/#140与D.4原始全文；实现只涉及整数地址、容量、tag与小CPU metadata，不包含历史解压、窗口值处理或性能模拟。

### 12.1 固定布局与ABI

`RequestWindowLayout.from_config(config,num_kv_heads=实际本地Hkv,head_dim=实际D)`读取max_num_seqs、S/R/MTP和max_model_len，不从Running或客户端K32推容量。每FULL层的一份raw sidecar排列为连续K、连续V、连续tag三段；`views(raw)`只产生零复制view，不分配或初始化窗口内容。

| 数组/参数 | 当前条件值 | 类型/规则 |
|---|---|---|
| K、V各自shape | `[128,323,Hkv,D]` | BF16，slot→row→head→D连续 |
| tags shape | `[128,323]` | int64，每row一个epoch+position tag |
| request map shape | `[128,2]` | int64，列0=稳定lease slot，列1=epoch |
| request map padding | `(-1,0)` | 所有unused尾行均写invalid |
| ring行号 | `p<S ? p : S+p%(R+spec)` | p为完整逻辑位置，不是物理页内位置/mRoPE |
| position bits | 18 | `max(1,(max_model_len−1).bit_length())` |
| tag | `(epoch<<position_bits)+p` | epoch非零，最大`INT64_MAX>>position_bits` |

当前单层sidecar字节=`128×323×(1024+8)=42667008 B`；17 FULL层为725339136 B（691.737MiB）。二维request map每份2048B，通常每worker共享一份CPU staging与一份NPU map，各层消费同一身份表；每层K/V/tag内容独立。所有内存必须在预算里计费，不能把“描述类没有分配”当作运行期免付费。

`make_leases()`创建与布局位宽相匹配的RequestWindowLeases，避免默认int64 epoch上限超过packed tag能承受的45位。参数、地址总字节、epoch与position越界均显式拒绝；非2次幂max_model_len还拒绝位域中的未使用position。环末保留区间为sink加最后R+spec行，测试证明拒绝0–3个MTP token后需要的R行仍在其中，且同一次最终保留集合没有重复ring行。

### 12.2 固定图地址与CPU staging

`FixedWindowMap(layout,host_map,device_map)`绑定**调用者预分配**的CPU pinned与NPU int64二维tensor；构造检查shape、stride、dtype、device和源/目的不别名。`write_host(WindowBatch)`先检查全部lease、epoch、padding与重复slot，再批量填写完整CPU map；它检查两端对象、data_ptr、shape、stride均未改变。它不调用`.to(npu)`，不替换device tensor，不发布NPU数据。测试专用`testing=True`允许CPU/meta，默认生产检查仍要求NPU；显式非bool测试开关会失败。

Host图地址检查只是一项必要条件。函数不会也不能从Python推断旧kernel是否仍在使用相同sidecar；调用者必须满足下一节的实际流事件协议。

### 12.3 建议runner hook及两种不同的依赖

建议外部wrapper包 `NPUModelRunner._update_states`，在original返回后执行lease/update和map fill，再原样返回其deferred-correction callable：

```text
native synchronize_input_prep enter
  wait previous prepare_inputs_event                 # CPU pinned可重写
  native _update_states
    add/remove → condense → reorder → refresh
  external posthook
    leases.update(output, input_batch.req_ids)
    fixed_map.write_host(batch)
    device_map.copy_(host_map, non_blocking=True)      # 当前计算stream
  native _prepare_inputs / metadata构造
native input-prep exit: record prepare_inputs_event
main/MTP attention与store读取同地址device_map
```

源码锚点：Ascend `model_runner_v1.py:2025–2049`将该调用包在input-prep context中；父类 `gpu_model_runner.py:1436–1447`已完成condense/reorder；`:3704–3716`先同步上一轮copy事件、退出时record；`:694–697`在async模式创建该event。零token步在`_update_states`之后才返回，所以完成/抢占仍能回收lease。

必须区分两种保护：

1. **CPU pinned staging重用**：前一轮H2D结束后才能改CPU map。把新copy放在既有context内部，外层record覆盖它；下一轮enter等待该event。若此hook被从其他入口调用、context缺失，或async模式却没有prepare_inputs_event，必须拒绝该接入或提供独立copy完成事件，不能仅传一个“安全”bool冒充依赖。
2. **旧kernel与新owner的设备写入**：所有sidecar reader/writer、下一轮map H2D及新的reader/writer必须在同一current stream上排队。现有attention binding以`c10_npu::getCurrentNPUStream()`发射（`csrc/attention_bindings.cpp:65,132`等）。前一轮copy已完成不表示旧模型已完成；same-stream顺序才让新map/新owner写入排在旧模型工作之后。若改变为另一update stream，需要明确等待前一计算完成事件并使计算stream等待新copy事件，且要证明没有循环等待；本次没有实现或证明该多stream扩展。

epoch只使旧tag与新owner不匹配；**不能阻止较晚完成的旧writer覆写新K/V或tag**。因此不能删掉上述任一依赖。只读/写元数据的host测试也不能证明NPU异步执行安全。

在current-only、mixed分段和first/later MTP的metadata中，window map必须跟相同request切片一起取对应行；不能把子batch的row0误指父batch第0请求。graph capture/dummy保持固定map地址并走invalid slot分支，不能分配真实lease或在捕图时替换tensor。图回放同址改映射、MTP拒绝、physical-page复用必须另有短真NPU门。

### 12.4 本轮验证边界

命令：`.venv/bin/python -m unittest tests.test_request_window_storage -v`，本地Mac已配置venv，8项通过。覆盖128/323字节账与meta view、小CPU tensor字节offset、完整logical位置ring、MTP拒绝0–3、request重排、物理页/lease复用时epoch隔离、非法整数/越界/padding/同slot双owner、固定map地址不变，以及helper确实不写device destination。

首跑发现本地venv没有NumPy，CPU map fill曾失败；已改为显式小CPU int64 tensor的bulk copy，不安装依赖、不切换生产设备、不降低断言。Torch本身仍打印其可选NumPy初始化警告，最终测试均通过。该记录是本机组件测试，不新增真机档案条目，不代表设备/图/完整服务/性能已验收。
