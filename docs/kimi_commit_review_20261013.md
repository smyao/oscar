# 三次提交复审与当前优化主线

依据总 MD 第一指示、启动文档第 0 节、D.4、档案 #142/#148/#154–157；当前被审计 HEAD 为 c5b2ad40。源码发现与已有真机事实分开记录。

## 提交裁决

| 提交 | 实际作用 | 裁决 |
| --- | --- | --- |
| 07a27358 | 默认关闭的 guard 分族同步归因；冻结算子不变 | 保留故障归因用途，不计为性能优化；默认和捕图仍单次融合 guard |
| a0c4e3dc | 诊断时将状态内容回读到 host | 弃用内容回读，恢复按冻结 guard 的 site/family 同步定位；不通过挪出扫描目录规避生产路径约束 |
| c5b2ad40 | 扫描 later-draft 块表，将无法解析的 slot 置为 padding | 替换：非法 slot 不等于原生 discard；只接受原生明确 discard 掩码，活行错误仍须暴露 |

### c5b2ad40 的确定缺陷

1. `_unresolvable_later_draft_slots` 没有读取原生 discard 标记，将所有解析失败都当成可丢弃行。这会吞掉真实请求的损坏元数据。
2. `torch.div(lens + 127, 128)` 没有整数舍入模式。`lens=128` 时得到 `1.9921875`，使第 1 列也参与匹配；prepare 原语的有效列数是 1。`table=[10,10]`、`slot=1285` 是本来合法的一行，却因第二列陈旧值被判重复、清空。
3. 每个 later-draft attention 调用都建立 `B×columns` 张量并做匹配、归约与 argmax。它不是修复该语义所需的工作；应在元数据建立时使用原生已有的 B 行 discard 信息，避免每层重复扫描整张块表。

以上属于主机代码与最小反例证据，不代表用户真机上的具体错误码已经取得。

### 已完成替代与最少验证

已删除 c5 的全表扫描，回到 077548f 的设备严格 slot 检查；新增外部 proposer 作用域读取原生已有的 `discard_request_mask.gpu` 引用，由 metadata builder 在模型执行前只对明确 discard 行建立 padding 副本。副本为每 draft 组固定地址，主模型/首 draft 不变；活行、非法尾行不因内容异常被清空，作用域退出不遗留上一批掩码。没有修改原生文件、GDN 或 AscendC。

关键主机验证100项通过（discard错误域、guard、plugin、首draft与native runtime）；启动配置/直启及observe签名闸隔离运行22项通过。联合140项时出现1项测试模块重载导致的mock绑定失败（缺本机build manifest），该项隔离运行通过，记录为测试隔离问题，不伪报联合全绿。真NPU、实际graph与原崩溃复验尚未执行；本次修复不声明已证实原真机错误码或2–4倍性能。

## 真机归因边界

当前复制日志确认冻结guard Trap及MTP attention诊断同步观测点，未确认具体错误码；旧诊断未预先排空target图等先前设备工作，故观测family不保证就是故障来源。低 generation、prompt 信用、prefix hit=0 都不能直接排除内部 MTP、align 布局或自身窗口读取。档案 #157 顶部已注明对旧推断的修正，原回传保留。

## 大优化的主线

压缩减少的是持久 KV 容量；当前每历史 token 的 K/V 又展开成 2048 B FP32，并经有界 GM 工作区供给 Cube，写加读至少 4096 B 逻辑地址流量。这不等同真实 HBM 字节，但确实抵消了压缩存储的执行优势。当前完整多 KV 模拟显示 QK/PV 数据交付是主要关键链，不能只削减已被隐藏的解包时间。

继续基于当前 MDL 母源评估直接 NZ 解包与消费，计入真实 publisher、精确窗口/current、旋转、全部任务和错误域。只把完整相关路径 `t_base/t_new >= 1.15` 且精度/协议合格的项纳入组合。最终目标仍为原生 2–4 倍；单项模拟结果不替代整轮认证。

默认 prefix caching 开启，使用已有物理页拥有的精确快照；未完成跨请求恢复的 request-window 研究保持关闭。默认原生 token budget、GDN、冻结精度和两个 candidate 一键入口保持一致。

## 077 修复支线继续闭环（2026-10-13，源码与主机证据）

`tests/test_native_discard_pipeline.py` 直接执行参考树原生 `_prepare_inputs` 的 discard 发布片段及 proposer 的 `prepare_next_token_ids_padded`、`prepare_inputs_padded`、`attn_update_stack_num_spec_norm`，不是重新实现原生采样/位置算法。12组初始病例在第0/1/2请求的显式 discard、mRoPE及async-spec组合中复现 `anchor=qstart-1` 可达路径；新增4组连续调度病例按 `abc→cb→b→dba` 重排、缩容、扩容及discard转active。两轮later-MTP分别保持固定副本地址，旧批次mask尾部不污染新批次，活行非法slot不会被清空。与 `test_later_draft_slots.py` 联合 **28 passed（0.89s）**。本机缺NumPy，测试仅将CPU数组传输适配为torch；这些不是Triton/NPU或实际异步stream验证。

当前默认调用链核查：原生 `sample_tokens → propose_draft_token_ids → _propose` 在同次proposal中完成全部later metadata构造，OSCAR的mask副本排在模型调用之前；MTP `enforce_eager` 保持启用，target的FULL图与draft的eager链分离。原生 `_mamba_block_aligned_split` 使用 `cache_config.block_size`（当前参考几何768），不以OSCAR FULL物理页2304强迫chunk升到2304；省略服务token budget并使用引擎原值的约束不变。

### exact writer 的 keepRecent 所有权

`rotate_clip_store_striped.cpp` 仅在 `slot[token+ring] == slot[token]+ring` 且属于同一物理页时，省掉将被覆盖的旧recent写。默认原生FULL协议只共享完整只读页，可写尾页归单一请求；prefill/first-MTP在每个可写页内是连续slot区间，所以同一ring行最终恰由最后一位写者发布。later-q1的显式discard会形成padding洞，但每请求只有一个current slot且可写尾页不共享，不会产生同页相隔ring的第二写者。量化失败或无效position仍发布非零status并中止，不容许随后读取未发布窗口。当前未发现应修改该生产逻辑的源码反例；未实现的request-window研究不享有此证明。

### 官方CPU错误传播门

`csrc/cpu/cv_probe.cpp` 新增 `tasks_discard` 模式，无golden文件依赖。它在D256/GQA6上运行真实prepare、冻结fe0及当前默认bundle_q1：int32/int64两种slot、有效列外陈旧重复值、明确discard的-1、有效列内重复页、活行position越界及padding尾行。断言prepare的错误码与两条AIV的全部attention status一致、健康/padding的输出/LSE完整写入，并保留native slot/table字节。命令为 `oscar_cv_cpu tasks_discard metadata`。

官方CPU构建及此门均已完成，独占runner `child_returncode=0`、`supervisor_returncode=0`、`cleanup_complete=true`，耗时 **2.231387秒**，完成时间 `2026-10-13T08:04:40.472933Z`。VM控制记录为 `/tmp/oscar-fix077-cpu-20261013/control.json`，完整日志在同目录 `logs/`；紧凑报告为 [`reports/fix077_local_20261013.json`](../reports/fix077_local_20261013.json)。末行确认 `readers=[fe0,bundle_q1]`、`cv_status_propagation=passed`。本地已把“原生显式discard引出的非法anchor→metadata padding→真实prepare→默认q1 status”这一可达兼容缺口闭合，713c611d生产修复已推送；不额外修改冻结算子。

**边界仍保持**：#157原真机确切错误码未知，此次合成病例中的code5不等于取得了原故障的设备码。真实NPU、Triton分支、设备异步stream/实际图、全模型APC/质量和整轮性能均不能由该CPU门代替。未发现新的默认生产接缝缺陷，不据此宣称所有可能的077故障已获得真机100%排除。

## 新回传#158：原故障仍存在，收紧真实修复链的生效条件

用户随后回传新的四rank `status_guard` Trap（08:29:41；完整日志见#158），证明此前本地discard接缝闭环并未排尽服务故障。片段缺启动SHA和分族诊断，不能断言此次仍为attention族或code5。本次重新检查实际原生工厂及MRO：Qwen3.5 method规范化为`mtp`；工厂创建`AscendEagleProposer(EagleProposer, AscendSpecDecodeBaseProposer)`，Eagle不覆盖`_propose`，C3顺序落到被插件包裹的Ascend方法；参考A2树没有后续覆盖该方法。注册器在未来模块exec后安装，已导入模块也安装。**尚无证据认定真实回传的钩子确实漏装**。

发现并修补的确定防护缺口：`later_draft_slot_copy` 原先在scope缺失时直接返回原slot，即真实运行若被第三方覆盖钩子，会静默绕过discard兼容修复。现在method=mtp的真实later builder必须处于原生proposer作用域；缺失即明确host异常 `outside the hooked proposer scope`，不拖到融合device Trap。首次实际绑定打印一条 `MTP_DISCARD_BRIDGE ... source=native.discard_request_mask binding=queued`，只用主机已有的类型/shape/draft_index，没有读取mask、status或KV内容；每proposer仅一条，不把入队称为设备完成。

执行真实Eagle/AscendEagle类声明验证MRO，覆盖已导入模块callback与延迟loader callback两种安装，确认继承实例执行被包裹的`_propose`、固定副本及只打印一次；相关原生/firstdraft/mixed/slot回归 **136 passed（3.17s）**。这项收紧不能领取性能倍率，也不是#158已定位根因的声明；其余当前可达数值/窗口/流水原因继续排查，不能以本地通过宣布077完全修复。

同一诊断开关 `OSCAR_GUARD_HOST_SYNC=1` 增加设备侧code谓词：按family对原status在设备上计算 `status==1..5` 和非零补集，每次将0/1谓词交同一冻结guard再同步；异常链增加 `code=1..5/unknown`。原status不改写，无`.cpu()`/`.item()`/host张量bool和内容回读；默认及capture仍仅原来的一次融合guard，开关仍默认关闭。未知/负值及int32两端均被补集捕捉，不隐藏错误；多个code并存时报告先检查到的code，不能当作最早元素或时序根因。guard/slot/native三文件回归 **44 passed（1.32s）**。该诊断缩小下次明确启用时的未知范围，本身不解决当前未定位的Trap。

继续发现旧归因的队列边界缺口：原生ACLGraph在replay之前同步，replay之后直接返回；capture时OSCAR仍记录融合guard。因而先前target图内的Trap可能在后续MTP第一族guard同步时暴露。现在诊断模式在提交任何新guard之前先同步；若已经失败，异常标记 `family=prior_work code=unavailable`，不把它误归新family/code。该改动只对显式诊断生效；新增先前图错误病例及完整guard回归 **14 passed（0.70s）**。#157顶部已更正“观测点”与“故障来源”的区别。

### 实际align几何的CPU接缝补充（进行中）

新增 `oscar_prefix_q1_cpu` 使用FULL2304、virtual128、真实SoA raw偏移（conv30720、SSM393216、page817152）验证stripe writer→默认bundle_q1→merge。首次driver编译因常量`H`与官方SDK符号冲突失败（本机driver问题，非新真机错误）；全部几何符号改为`kProbe*`后构建通过。完整初始writer323行模式在120秒到限，child=-15/supervisor=124/cleanup=true/elapsed120.112s，原记录 `/tmp/oscar-prefix-q1-cpu-20261013/control.json` 保留，**不能声称完整初始writer通过**。

另外提供明确标记的 `--incremental` 模式：独立oracle种合法旧INT2/snapshot，将共享页末4行设为不同旧值，再由真实writer改回预期INT2整行及原始BF16字节；之后真实prepare/q1/merge在2304→2305→2304跨页/rollback中校验全部状态、完整原空间均值/LSE oracle及共享页不可变，并由真实writer更新私有页。K/V向量range均为3，避免常量scale下溢导致错误fixture。该模式已实际完成，**3病例通过、child=0/supervisor=0/cleanup=true，23.935674秒**；报告显式 `initial_seed=independent_oracle`、`real_writer_rows_before_queries=4`。完整323行seed仍未完成，此次增量通过不能替代它，也未复现/修复#158整服务Trap。紧凑证据见 [`reports/fix077_followup_20261013.json`](../reports/fix077_followup_20261013.json)。

原生GDN地址审查没有发现覆盖FULL的依据：`collect_mamba_copy_meta`只遍历mamba_group_ids及其原生layer_names，conv拷贝slice与SSM单页拷贝都限制在该state页内；dest是其当前GDN拥有的物理块。原生reshape按conv全池→SSM全池映射，OSCAR snapshot在其后的padding；FULL INT2复用SSM区但FULL/GDN活动物理ID由同一block_pool互斥分配（原生manager执行测试已有证据）。不能把gdn_attn_builder处的异步H2D报错当作其覆盖FULL的证明。

## 独立查实并修复的模型长度边界（非#158的64K根因）

只读独立复审通过真实原生方法发现：`max_model_len=M=262144`、target `seq_lens=M−1`、q4、valid_count=3/reject1、discard=false时，draft1为position262142/lens262144；draft2的accepted position262143仍合法，slot非负，但原生保留rejected行的 `seq_lens` 溢出后被置1。原生分别按length和accepted position判越界，二者确实不同；OSCAR据lens1只扫描第一列，从而对合法slot报code5。此条件在原生scheduler允许的M−1 target边界可达，但当前64K回传不能由它解释。

修复仅在metadata builder：同proposal的原生 `seq_lens_group[0]+draft_index` 确实超过M、当前native lens==1、slot非负且非discard时，在OSCAR专用固定地址length副本上恢复 `min(base+index,M)` 扫描上界。原生buffer和GDN不变；accepted位置仍由冻结prepare对完整块表唯一匹配得出，不读mRoPE，不从解析失败反推discard。负slot、discard、普通lens1或任何不满足窄门的行保留原native值；非法活行仍报5。每次proposal都按当次固定group0刷新副本，不缓存上一次batch的值。

新增32组真实native方法用例覆盖before-wrap的M−1/M/M+1、reject0/1/2/3、mRoPE、async-spec、discard及已越模型边界的负slot；5组额外反例确保窄门和坏活行错误不被吞。与原生/firstdraft/mixed/guard回归共 **187 passed（2.92s）**。这是已经本地复现、已有具体修法的独立兼容bug，不是#158已修复、真实NPU图已通过或性能达到目标的声明。

为避免远离边界时增加多个小TensorOp，后续加入有原生来源的快速条件：直接执行reference `_build_attention_metadata` 的max字段发布分支及异步GPU长度修正函数，确认其正数 `max_seq_len` 是第一draft设备length的上界；若 `max_seq_len+draft_index<=M`，仅保留固定destination copy，跳过设备谓词。0/未知或接近上限时继续完整5条件，不切换指针。包含此条件与全部真实wrap反例的两文件focused验证 **70 passed（1.82s）**。这仍是兼容修复的必要开销，不能把另一路算子18.8%收益视为已扣除此开销后的端到端收益。
