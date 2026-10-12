# 有界 NZ 完整裁决与剩余 L1 复用审查

更新：2026-10-11。B-NZ和Vᵀ-NZ都已完成同库完整C4配对并退出本次组合。本页保留实验前条件账及失败证据，不再把两项列为待跑计划。这些是本机CPU/CAModel结果；目标NPU、graph及整轮性能未验收。

| 完整实验，均为Q336/R1/C337/S1/20core、source0+1 | 同库基底→候选 ticks | 结果与CPU边界 |
| --- | ---: | --- |
| B-NZ | **1608607→1743703**，慢**8.40%** | CAModel全位/独立oracle/1008任务/60actor通过；publisher CPU通过；完整candidate CPU保留120秒超时。[裁决](../reports/c4_b_nz_20261011/decision.json)、[配对](../reports/c4_b_nz_20261011/completed/reports/camodel-valid/comparison.json) |
| Vᵀ-NZ | **1610898→1734415**，慢**7.67%** | CAModel同样全部通过；publisher CPU原始位/guards通过；完整candidate CPU未运行。[裁决](../reports/c4_b_nz_vt_20261011/decision.json)、[配对](../reports/c4_b_nz_vt_20261011/reports/vt-camodel-valid/comparison.json) |

两项各自两臂均在600秒期限内rc0、cleanup完成；真实K/V publisher已包含在周期内，常量Rv的256KiB位pack未计时。未采完整相位归因，不能把净回退唯一归到burst、Gather或某一同步。原报告、CPU超时和环境失败均保留；B裁决中历史“下一步Vᵀ”的文字由上述已完成Vᵀ裁决更新，不改写原报告。

## 1. 问题仍存在，但不是“全部时间在解包”

| 当前位置 | 已确认的真实工作 |
| --- | --- |
| `csrc/kernels/attention_cv_decode_bundle.cpp:46–47,628–654` | q4/q1已用HalfKv64/padded unpack和同task预取；解包后仍将自然FP32 `[KV,D]`写GM，Matmul输入仍ND。不能再把已有预取算新增收益 |
| `csrc/kernels/attention_cv_window_range_cluster4.cpp:203–204,460–477,901–920` | 一次KV准备已经供4成员复用；但4次QK和4次PV各自从GM搬B到L1。当前C4仍H16，不是C16；native小chunk下此路径真实可达 |
| `csrc/experiments/c4_member_pipeline/c4_candidate.cpp:216–217,463–468,729` | 已测H32+member流水研究基底同样发布自然K/V、使用ND Matmul。后续结构改动应在这个基底上计增量，不能另乘H32 helper2.346× |
| SDK `stage/copy_cube_in/copy_tile_to_cube/copy_tile_to_cube_common.h:459–487` | ND分支调用CopyND2NZ，NZ分支调用CopyNZ2NZ；现存收益方向是改变格式转换/切片，不是删除必要的GM读取。已有PV126 trace没有重复GM输入读取的证据 |

实验前核查的源SHA256：decode `d8171f0403cf176087e1c76c64e454e6fb1e74fd79252aab3b59e848cca19401`；生产C4 `14b28f9181220432a15bcc3f23679e539473b12e01d858bbb00ff3345522ba65`；研究pipeline `da59a923a44cdb7ac0916596cd89f802a24fc1ffef4426952402d90b5d2e7b31`。实现前重新核对，而非套旧副本。

## 2. 已闭合贡献与还不能证明的部分

[原C4完整闭合循环](../reports/c4_phase_profile_q336_20261009/partial_c4_pipeline_budget.json)的第二个full256 round：同一成熟簇比较边界 **390694 cycles**；KV准备 **53896**；4次QK共 **85620**，4次PV共 **106582**。这里只采用该闭合循环，整trace末尾截断的失败门不变；它不是64K/K32/整步占比。

- 即使这个循环的KV准备全部免费，其余成本不变，上限仅 `390694/(390694−53896)=1.1600×`。跨簇共享解包不能消掉各成员QK/PV/softmax，故不优先新增复杂跨core共享协议。
- 当前pipeline完整Q336/C337已测 **1849738→1610282，1.14870×**，保留为组合候选。该完整核结果没有拆出新pipeline的稳态各相位，不把旧390694循环当成其新分母。[配对证据](../reports/c4_member_pipeline_20261010/reports/c4-pipeline-camodel337/comparison.json)
- 同M的host免费预pack微测有收益：QK126 ND/B/AB **28760/22118/19436**；PV126 B-only **34742/20988**、AB独立配对 **34683/18820**。这些包括独立Matmul任务成本，不能把差值直接减出已初始化attention内Matmul，也不能跨库拼B→AB直接倍率。[matched形状](../reports/fp32_nz_matched_shapes_20261010/reports/nz-matched-camodel/comparison.json)、[B](../reports/fp32_nz_b_matmul_20261010/reports/nz-b-camodel/comparison.json)、[AB](../reports/fp32_nz_ab_matmul_20261010/reports/nz-ab-camodel/comparison.json)

仅作敏感性账：**假设**上述同M的比例能转移到旧闭合循环中的QK/PV、其他工作不变，得到：

| 假设布局 | consumer可省cycles | 新publisher增量为0时循环倍率 | 保持正净收益的新增成本上限 |
| --- | ---: | ---: | ---: |
| B-NZ | 61968 | 1.1885× | 小于61968 |
| AB-NZ | 76506 | 1.2435× | 小于76506 |

公式为 `Δ=sum(QK)*(1−rQK)+sum(PV)*(1−rPV)`，`Tnew=390694−Δ+E`。`E`包含索引、Gather、DMA、同步、padding和新增固定成本。**这是可迁移比例假设下的条件上限，不是测得速度或严格硬件上界；目前无法保证E小于预算，净收益下界未建立。** 单靠NZ在这套旧循环条件下达不到1.5×；约6万cycles曾是执行完整实验的理由。现有两项完整结果均回退，这份条件账不能覆盖实测裁决，也不能把1.1487与1.1885相乘。

## 3. 发布方案的已测裁决与保留边界

| 方案 | 完整链的新增成本/风险 | 当前选择 |
| --- | --- | --- |
| H32保留旧lane16归属，K/V直接发布B-NZ；A/P仍ND | 每KV256的K+V payload仍512KiB；16次自然发布/32个burst变32次Gather+发布/1024个burst，增加992个burst、索引和MTE3_V等待；一次发布可供4成员消费 | **已完成，慢8.40%，退出组合**。P/score alias和A物理pitch未改；本版本的免费预pack上限未兑现为完整净收益 |
| AB-NZ | 除K/V外，每KV还要重排4成员约512KiB的P，拆独立P槽；Q和最终归一化acc也需NZ。上述敏感性账只比B多约14537cycles额度；PV24 matched AB还比B多40ticks | 暂缓，完整AB尚未实现。B已回退但E未逐相位测量；不能仅见AB的PV126微测倍率较大就扩展到完整AB |
| D列分工NZ | C4可减少publisher burst，但两lane重复packed/metadata读取，新增虚拟旧lane错误归属；decode H64本已128burst/K，没有同等改善 | 未实现、未入组合。两次净回退仍未定位burst的独立占比，不据它们认定列分工必然有益 |
| Vᵀ-NZ | H32 V发布burst512→16，保留原row ownership；但完整K128的Cube NZ→L1 B burst8→16，PV改走转置Load2D | **已完成，慢7.67%，退出组合**。减少V发布burst仍未产生完整净收益；不是数值失败，不再原样重跑 |
| K-major低层consumer | 新完整CAModel qk24 **13943→12557，1.11038×**；PV126 **18975→25853，0.73396×**。3例有限值位/oracle/K128顺序通过 | **本版本不入组合、不乘NZ倍率、不扩形状**。少搬A未转成PV更快，根因需原trace；[原始配对](../reports/kmajor_20261011/v3/reports/kmajor-v3-camodel-v2/comparison.json) |

上述发布次数为当前H32旧lane16映射的源码工作账，不是实测HBM事务或周期。H64 decode每KV只有一次QK/PV，不能套C4四成员摊销；q1 M6也没有本轮对应NZ微测资格。

## 4. 下一项仅审查 K/V 的 L1 复用

B-NZ、Vᵀ-NZ及当前PV K-major版本都不再作为下一次运行或本次组合。下一问题是：同一成熟C4簇的多个成员能否复用已经进入Cube L1的有界K/V，在保留每成员KV顺序、FP32、C1/precise与错误优先级的同时，减少实际GM→L1工作。这一方案的API支持、容量、跨成员生命周期和净收益仍在独立审查；目前既未验证，也未否决。

当前A2/CANN公开SDK不能提供所设想的**无GM的AIV UB→Cube L1直通**。`DataCopyUB2L1Impl`在123–168行明确为软件模拟，先分配GM workspace、执行`copy_ubuf_to_gm`，再由`ScmDataCopyMsg`请求GM→L1；ND2NZ版本172–213行同样经GM。[官方SDK源码](../artifacts/cann_sdk_cube_public_20261010/impl/basic_api/dav_c220/kernel_operator_data_copy_impl.h)。该事实约束的是这两种公开跨核拷贝接口，不能将GM发布成本直接从模型中删除。

仅K/V的L1复用需要单独核实实际可省读取与资源归属。布局回退、公开UB→L1经GM，都不足以判定这一独立方向无效。先完成源码/容量/依赖审查，再决定最小新实验；不重跑已有NZ或H32微测、不扩大超时，不索要AISBench或新的私有数据。

## D.4四问与整轮边界

- **相位**：fused history内反量化发布→QK/PV数据搬运；不重新优化已接入的pack+Init。
- **旧失败**：D.4全历史dequant6499.8–6655.1ms、store约209–216ms；#145/#154证明UB跨pipe误复用CPU可能看不出。旧95.4%不是现版份额。
- **结构规避**：仅当前KV256、沿现两槽流水，空间不随历史增长；位重排、不cast，不改变量化或错误域，保留精确source与事件归属。
- **成本边界**：B-NZ与Vᵀ-NZ的完整净回退已建立；producer E的独立相位归因、L1复用新增Δ及整轮调用/占比仍未知。目标仍native1342.4437秒的2–3倍；FULL容量比不能补速度。新组合完成后以冻结工作量模型重算剩余差距，未达到就继续定位其他OSCAR主成本。
