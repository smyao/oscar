# C16 成功簇后的第二遍扫描跳步

档案 #34/#36/#126/#129/#144/#145/#148/#152/#154/#157，启动文档 §0 与完整 D.4 已读。仅独立研究，不改生产核、冻结guard或精度配置，不新增在线主机回读。

基底固定为 `c16_rv_reuse_20261015/c16_candidate.cpp`（SHA `9603817dddfba0e6357b42756a9956880137348ce8076c61aff58884acb6210b`）。Rv的独立数值/配对资格由它自己的报告裁决。后续性能只比较同库 **Rv vs Rv+finite+skip**，计划Q336/recent1/C320完整路径、原600秒/臂；不重跑旧stable/pipeline三臂，不乘旧倍率。组合由 `c16_b1_followup_20261015/generate.py` 按 `Rv -> apply_finite -> apply_skip` 唯一派生链生成，本目录不另占VM。

## 生产输入契约

`oscar_ascend/integration/impl.py` 先调用 `prepare_attention_tasks_out`，随后唯一的合法任务改写是 `current_attention.py::suppress_current_source_tasks` 对健康source2的end置为begin。`csrc/kernels/attention_tasks.cpp::AttentionTasks::Run` 逐token生成16字int64任务，行号固定为：

`token * (Hkv * 3 * splits) + (kv_head * 3 + kind) * splits + split_index`。

请求来自原生 `query_start_loc / seq_lens / slot_mapping`。非单调或越界qstart、非法length/context、原生显式负slot、later-draft错误映射仍由原prepare/第一遍错误分支处理。新代码不从`qcount==0`或slot解析失败推断discard；第一遍所有padding、错误、follower status写回逐字保持，状态和精度门不删改。坏metadata/QR与同地址换context的NPU测试仍需原输出/LSE/status逐位和冻结oracle门。

**这个契约不是“允许任意伪造raw任务表”。** 重叠request的人工任务表可以伪造两组相互覆盖的成熟簇，并构成跳步反例；原生单调qstarts不能产生这种表，原核对此类重叠写回也没有合法数值契约。Host测试主动保留反例，拒绝声称任意raw-row输入等价。不能将本证明当作接受伪造表的校验。

## 可达性证明

1. 原prepare只会让请求相对query-tile边界上的健康leader取得完整`qcount=queryTile`；组内显式slot hole会截短count，hole之后重新出现的leader最多取得本组余数。跨request边界同样被`min(request_end, group_end)`截断。
2. `IsClusterMember`逐一验证16个相邻完整组，且request/head/split/context/KV域相同。因此成功簇覆盖的半开区间`[token, token+16*queryTile)`内，每一行都属于该同一请求且无slot hole；同一head/split的下一个request-relative anchor只能在区间端点或之后。
3. 原第二遍已经执行这16成员，只需将for循环token推进`min(global_bucket_end, token+bucketWidth)`。代码使用`token += min(bucketWidth,end-token)-1`后由for自增，避免加法溢出，并严格留在当前global bucket。其他bucket、request、head、split保持原owner。VectorCluster结束时qbegin已变成最后成员，故不能用类成员qbegin算下一位置。
4. 只有整簇资格成功并执行后才跳步。非成熟、失败、padding及异常输入路径不跳；第一遍检查和所有数值函数、cross-core flags、workspace字节不变。

## 同源Host重放

`generate.py`抽取两臂的完整`Process`和原`CurrentTask/UseTask/TaskValid/IsClusterMember`正文，`host_template.cpp`只替换设备访问/计算为事件记录器。每逻辑核各重放一次AIC与不依赖lane的AIV控制流模型（两个AIV lane的Process/资格判断同文、不访问lane；不是实际设备actor执行），比较每个普通任务、成熟簇成员、错误、status调用的完整有序事件及counter；AIV stub按真实代码退出在最后member，覆盖qbegin变化的陷阱。这没有运行FP32计算或AscendC kernel。

已通过6153个明确有限的native可达案例、150300个actor重放：GQA1–16、Hkv1/2、S1/2/3/7、多个grid；每个336桶内的request起点offset；每一位置的单hole/相邻hole；GQA16每个8-token组的全部256种hole模式；短尾、上下文0/64/65/337/65536、16384形状、非法qstart/context及later-draft code5。这里的“穷举”仅指这些列出的有限族，不声称遍历无限输入空间。

`fixtures/`只保存三份已有真实AscendC CPU prepare的压缩任务表、输入metadata及来源/hash。当前转录与small65/small337/full337共64512个int64字段逐字相符；各在grid1/3/20/32重放。原full337 attention CPU超时不被改写：使用的仅是已完成的prepare检查点。

```bash
python3 csrc/experiments/c16_anchor_skip_20261015/generate.py --check
python3 csrc/experiments/c16_anchor_skip_20261015/host_replay.py
```

报告：`reports/c16_anchor_skip_20261015/host_proof.json`。`apply_skip(text)`可供组合generator调用，严格绑定原Process SHA `2bb45e7998fc855e4c4ac8e7e6e238df68931ca4f0ba9feb6f1f804876c088f1`，不修改Process外任何字节。

D.4：本项针对融合FIA内metadata扫描；旧host prepare约725ms/full-history恢复6499.8–6655.1ms是禁止结构，不是本轮相位占比。零新增GM/UB/L1，不读历史数据。只减少已证明无额外anchor的metadata读取，未测速度；CPU数学、CANN、CAModel、NPU、图与整轮30%目标仍独立。
