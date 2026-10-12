# C4 尾波拆分：已有源码计数与条件预算（2026-10-11）

**裁决：保留为组合中的结构候选，未实现、未测净收益。** 当前可确认特定任务形状存在双簇尾波；不能把核数比或历史条件倍率当成准备组合的 H32 + member-pipeline 的收益。本文只保存本轮已完成的只读推导；没有新增 VM、算子或矩阵实验。Roofline 与整轮组合机制由独立文档说明，本方向只能提供其中的 critical-path 项。

## 已核实的任务与所有权

依据当前 `csrc/include/oscar_cv_schedule.h`、`csrc/kernels/attention_tasks.cpp:129`、`csrc/experiments/c4_member_pipeline/c4_candidate.cpp:310` 与 `:405`。以下限定健康单请求、Hkv=1、GQA=6、S=1、sink=64、recent=256、连续有效 slot、context>64；不是任意 batch 的结论。

- query tile 为 `128/6=21` token，C4 bucket 为84 token。source0 的历史终点是 `min(context,max(sink,context+qbegin+qcount−recent))`。C4要求四成员全部21 token、同request/head/split/context/KV范围，且 `kvend=context`；因此第一个成熟 anchor 是252，不能把前3个 bucket 当成熟簇。
- 记 `G=ceil(Q/21)`、`B=floor(Q/84)`，在上述限定下成熟簇数为 `max(0,B−3)`；第一遍 source0 为12个前沿 C1，加末尾不足完整 bucket 的 query组。source0 C1组 g 的 owner 为 `g mod20`；source1组 g 为 `(G+g) mod20`；成熟簇 bucket b 为 `b mod20`。
- source1每组仍必须执行，候选 KV 数为 `64+max(0,255−21g)`，按256划分搬运单元；不能把它与空 current、followers、尾部、最终旋转等第一遍工作丢掉。source2是否压空必须按实际服务路由确认，本计数沿用已有报告的压空条件。

| 已完成计数的形状 | source0 query组 | 成熟C4簇 | 第一遍C1组 | source1账 | 当前热点及拆分边界 |
| --- | ---: | ---: | ---: | --- | --- |
| Q2048 | 98 | 21 | 12前沿+2尾组 | 每核4–5组；core3及拟选helpers12–15均5组/5个KV单元 | core3为1个C1+2簇；其他核最多1簇。可考虑只拆anchor1932的最后一簇。 |
| Q1924（2048−124） | 92 | 19 | 12前沿+4尾组 | 每核4–5组；cores12–14各有6个KV单元 | 无双簇尾波；cores8–11叠加两个C1。不能照搬Q2048方案。 |
| Q8192，条件为仍走C4 | 391 | 94 | 12前沿+3尾组 | 每核19–20组；cores11–13各有21个KV单元 | 每核4–5簇，cores8–10还有两个C1；全拆14个尾簇会增加太多工作。实际若路由C16，须另按C16核算。 |

Q2028/2016/2012尚未在本轮单独落盘执行 host 计数；上式给出可复核入口，本文不伪称已有对应运行报告。特别是原始预算2048不等于实际长分区Q；prefix与请求边界会改变成熟簇集合。

## 准备组合基底上的参数式

必须重新取得 **H32 + member-pipeline** 的成本。设同一历史范围内一个完整 C1 历史工作为 `A`，一个四成员 pipeline 簇为 `B`；它们包含实际KV循环与其同步，保留部分KV/部分query的单独成本。设 `F_i` 为第一遍source1、metadata/followers、空current、query/state初始化、末尾旋转/写回等不属于A/B的每核成本。拆分增加的 planner、元数据和固定成本记入 `P_i` 或 `F'_i`。

通用式是 `T_old=max_i(H_i+F_i)`、`T_new=max_i(H'_i+F'_i+P_i)`，不能先把旧owners理想均衡。Q2048的一种已推导任务安排是：从core3移去anchor1932的C4，将其四个原始C1 leader分别交给原无第一遍C1的cores12–15。第一遍source1及所有其他task保持原owner；移走/新增query、旋转等固定工作仍须计入 `F'_i`。

仅在统一固定成本F、所有C1不重于A、迁移后其他核不形成更大热点的理想模型中，才有：

`T_old=A+2B+F`，`T_new=A+B+F+P`。

达到1.5倍要求 **`B ≥ A+F+3P`**；达到自定的明显净倍率 `1+η` 要求 **`(1−η)B ≥ η(A+F)+(1+η)P`**。展开一簇使合计历史工作增加 `4A−B`，并使共享KV展开从1次变4次；因此必须证明 critical-path 收益覆盖总流量/竞争增长。若新pipeline令B接近A或剩余只是1–5%增量，继续暂缓。

## 历史预算怎样使用

已有 [完整分区条件账](../reports/c4_phase_profile_q336_20261009/full_partition_conditional_budget.json) 已覆盖原owner、第一遍C1/source1/尾部边界及**整簇LPT重排**，不能重复领取它的收益。它没有实现本次“一个尾簇变四个原C1”的计划。历史 `C1_unit=131472`、`proposed_pipeline_C4_unit=216547` 只能说明形状敏感性：后者本就是旧片段的流水假设，不是当前H32组合的实测单位成本；完整C337配对1849738→1610282也不能唯一反解A/B。

此前以这些旧系数算出的Q2048/C20K历史代理43471582→26797463、C65.1K的143268292→88396826，**不是当前组合已具备1.62倍**。它们仅提示值得检查该形状。反例同样重要：Q8192全拆14个尾簇后，剩余80簇仍每核4簇，已有15个C1加新56个C1共71个；按不可分C1工作，至少一核承担4个C1。因此历史单位代理由 `2A+5B` 变为至少 `4A+4B`。以旧系数为例1345679→至少1392076，已经变慢；不能制定“尾波一律拆开”的规则。

## 守恒与重开门槛

planner只能从真实设备task和现有 `IsClusterMember` 条件选完整簇，按实时估计代价选择原C4或四个原C1；不增加长度专用CLI，不改变global split、query/KV累加顺序或OSCAR数学。一个选中簇的四个leader恰好各执行一次；原C4 owner必须同时撤销该簇。followers仍由原路径写两lane status，非法/非成熟/尾组不丢弃，source1/2与原task IDs保持。输出/LSE/status、错误优先级、clusterStats和图地址均需按新owner守恒。

重开须先证明真实Q/成熟集合命中收益窗口，并以当前组合A/B、所有第一遍成本和planner费用满足上述条件，再做有限完整配对；不凭“21簇/20核”承诺两倍或整轮收益。D.4与#148/#154的影响：只重排有界工作，禁止全历史恢复；保留各query原数值与错误域，CPU结果不能替代异步设备同步和最终质量边界。

本轮无已保存的独立host分析脚本；计数来自已执行的只读内联Python和上述源码公式，未把临时推导声称为新工具或设备证据。
