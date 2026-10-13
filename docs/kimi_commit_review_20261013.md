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

当前复制日志只确认 MTP attention 状态族触发冻结 guard，未确认具体错误码。低 generation、prompt 信用、prefix hit=0 都不能直接排除内部 MTP、align 布局或自身窗口读取。档案 #157 顶部已注明对旧推断的修正，原回传保留。

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
