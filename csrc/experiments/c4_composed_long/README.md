# C4 完整组合：生产 H16 → H32 + member pipeline

档案 #126/#129/#140/#145/#148/#154–156；启动 §0.7 与完整 D.4。只读取当前工程算子和既有独立实验，不改原生/GDN、服务参数或精度。**2026-10-13最新裁决：当前H16→完整H32/member流水同库CAModel为3540306→2671072 ticks，1.325425倍，通过1.15门，已进入两入口共用candidate。** 原始数学、全部1008任务、336query独立oracle及60actor均通过；CPU1341仍保留超时，CPU337完整证据与大形真实prepare单列。真实NPU/graph/整轮尚未验收，详见下方分后端资格段及`reports/c4_composed_long_20261013/selection.json`。基础设施审批超时期间未自行重复发起Lima操作，真实执行由主线程串行完成。

## 为什么重开

旧完整 `1.1487043884×` 是 **H32 → H32+pipeline、Q336/C337**，不是本轮要交付的生产 H16 分母。也不能把 H16→H32 的 C113 `1.005962×` 和旧 C337 倍率相乘。当前生产 H16 SHA为`14b28f…`；已实现 member-pipeline 算子 SHA为`0795a0…`。`generate.py` 从这两个完整生产文件只改符号/include，逆向恢复所有母源码，并逐方法校验，不重新拼接未验证数学。

本轮唯一形状 **Q336/C1341/R1/Hq6/Hkv1/D256/S1/20Cube**。保留真实 prepare 全部1008任务；只按生产 eager prefill 规则让合法 source2 的 end=begin，仍执行该source的空输出与状态。独立原空间oracle覆盖全部336query的source0+1，冻结atol/rtol=0.005；原生current FIA不在本次CV计时范围内，两侧相同。

## 结构与成本依据

按当前prepare、CvTaskSchedule与C4成员条件，16个querygroup中12个为C1前沿，最后4个合成1个成熟簇。成熟history为`[64,1341)`：4整KV256+253尾。每个C1也有4整KV256，尾为18/39/…/249行。20Cube中核心3同时承担qbegin63的C1及qbegin252的C4，因此不能只算成熟簇自身。

已有C769闭合子范围的H16周期为C1=131472、C4=390694；旧依赖模型的pipeline候选周期216547是**条件推演，未测**。只迁移整KV部分时，核心3预算为：

```text
A = 4 × (131472 + 390694) = 2088664
B = 4 × (131472 + 216547) = 1392076
A/B = 1.500395（只含整tile历史，不是完整CV结果）
```

完整核必须按`max_i(A_i+F_old_i+tail_old_i)`和`max_i(B_i+F_new_i+tail_new_i)`求比。固定/尾部包括真实task/follower扫描、Q加载、精确窗口、empty-current、state初始化、最后normalize/Rv/输出、起停及全部KV尾行。仅在同关键owner、共同固定成本F的假设下，1.15门要求`F+7.6667P <= 3251844 cycles`；P为模型没计入的候选额外成本，不能把atomic、state及两buffer说成零开销。来源及逐核账见`reports/c4_composed_long_20261013/predeclared_budget.json`。

H32与pipeline作为一个组合只计最终完整CV倍率。H32 helper2.346×不计入；本轮即使完整CV达到1.15，也还要扣除原生current与层外成本，不能直接宣称整轮2–4倍。

## 资源与精度

- baseline：真实生产H16；workspace1839104 B/core、显式UB157632 B/AIV。
- candidate：现有H32+member流水；workspace2494464 B/core、显式UB157760 B/AIV。相对C4专用旧arena新增640 KiB/core，20核12.5 MiB，固定容量不随history增长。**当前默认服务的`history_cluster_size=16`已预留4997120 B/core的公共CV arena并跨层复用，覆盖新C4需求；默认服务不会再次额外分配这12.5 MiB。** 实际活跃地址/搬运增加仍由完整周期计入，不能据预留容量不变推断运行开销为零。
- FP32/K128/softmax/每成员KV顺序、C1/precise数值、错误域、全部任务和归属保留原实现。没有新增MDL或NZ组合。
- CPU与CAModel都使用相同完整fixture，输出partial/LSE/status/clusterStats必须逐位一致、所有GM guards和输入不变、全部336query独立oracle、1cluster/4members/4skips。真实CPU prepare与host源码任务witness必须逐字相等；host生成不能充当CPUprepare通过。
- 新配对CPU每臂120秒、CAModel每臂900秒。超时保存真实rc/completion/cleanup，不加时、不自动重跑，不采用partial ticks。旧超时和失败报告不修改。

## 已编译产物能否复用

旧C4 MDL研究库中的`oscar_c4_mdl_baseline_kernel`源数学为当前H16；旧pipeline库中`oscar_c4_member_pipeline_candidate_kernel`源数学为H32+pipeline，完整生产库也已编译过这两个公开算子。它们提供已有编译/符号依据，但**旧CPU driver仅接受C113/C337，不能直接执行C1341**。分开旧库的测量也不是本轮预声明的同库配对。

因此新独立库同时编译两个仅改符号的原实现；可由构建工具重用合法缓存，但不复制旧编译结果冒充新源/新形状通过。源码、传递include依赖、driver与同库SHA均绑定；真实已编符号未经本轮VM重新查询，来源为现存已完成报告。

## 准备和后续执行

当前导出host fixture：`/private/tmp/oscar-c4-composed-c1341-20261013`。它只含独立输入/oracle与任务公式，未含设备结果。需要重建时选新目录，避免覆盖旧证据。

```bash
python3 csrc/experiments/c4_composed_long/generate.py --check
python3 csrc/experiments/c4_composed_long/fixture.py --output NEW_FIXTURE
```

恢复本机Lima可用性后，由主线程通过既有独占锁/有界supervisor顺序执行，不启动多套仿真：

1. 本目录CMake，`SOC_VERSION=ascend910b4`、实际`ASCEND_HOME_PATH`；CPU额外`OSCAR_C4_COMPOSED_CPU_DEBUG=ON`，二进制为`oscar_c4_composed_cpu`。两臂输入同一fixture，新输出目录。
2. CPU两臂全部完成且逐位/oracle通过后，CANN同库`liboscar_c4_composed_long.so`；`source_binding.json`由CMake配置时写在build根。
3. 复用`tools/experiments/run_camodel_arm.py`，每臂用`cv_replay.py --variant baseline|candidate --fixture FIXTURE --prepared MATCHING_CPU_OUTPUT --library LIB --output NEW_OUTPUT`。prepared必须来自对应臂，不能把仅prepare或另一臂的回执冒充完整CPU资格。
4. `cv_compare.py --old BASELINE --new CANDIDATE --old-control BASELINE_CONTROL --new-control CANDIDATE_CONTROL --output NEW_COMPARISON`。比较器验证同库、同输入/oracle、源绑定、20AIC/40AIV完整输出和真实deadline完成，再以完整task ticks判`old/new>=1.15`。低于门exit2，保留精度通过及实际倍率。

上面的入口保留初始预声明与失败证据。得到下述完整结果后，统一服务flag才由research升为candidate；该选择不改变原生预算或默认prefix使能。

## CPU1341 超时后的显式分后端资格（2026-10-13）

主线程已在同一VM完成本目录CPU/CANN同库构建（rc0）。首次完整CPU1341 baseline在120.0204秒达到预声明期限，child=-15、supervisor=124、cleanup=true；candidate未启动。原失败保存在`/tmp/oscar-c4-composed-20261013`，不覆盖、不改成数值通过、不延长时限。

真实prepare检查点已完成全部1008任务与336位置，明确`metadata_only`、`attention_execution=not_established`。stdout末尾仍为core3两个AIV的SoftmaxFlashV2 reuse警告，stderr为空；该CPU工作目录161文件约1.573GB，两个core3检查日志分别约172MB/167MB。这证明CPU检查器有大量输出，但**没有证据证明日志是唯一超时原因，也不构成OOM或算子数值失败判定**。

继续方法明确分开证据，保留上面的原严格入口不动：

- `make_small_cpu_evidence.py` 重新核验既有真实CPU337小形：当前H16母源对应的旧C4-MDL baseline和当前pipeline母源的旧candidate，完整partial/LSE/status/clusterStats/tasks/positions原始字节一致；两侧控制均在120秒内完成且cleanup通过，独立全部query oracle通过。H16源绑定与pipeline逆向打包身份同样核对。生成报告绑定归档及全部输出/回执hash。它仍然只证明**CPU337**。
- 新`backend_split_contract.py` 要求原CPU1341的真实失败/清理回执、原执行计划/可执行文件/同库/fixture哈希、已完成prepare元数据与全1008任务公式逐字一致、336位置逐字一致。它不会读取未完成的attention输出作gold，不启动CPU1341重跑。
- 新`cv_replay_backend_split.py`只替换CPU前置合同；同库CAModel执行、全部输出、input/GM guards、独立336query原空间oracle、原lane状态、clusterStats以及20AIC/40AIV完整完成门与原严格replay正文逐字一致。两臂各900秒；完整task ticks达到1.15方能组合。
- `qualify_backend_split.py`逐臂执行上述独立入口，所有工作与日志在新`/tmp/oscar-c4-composed-backend-split-20261013`。报告分别保留CPU337已完成、CPU1341未完成、CAModel1341实际结果；不把不同形状后端证据合并成“CPU1341通过”。2项focused host合同测试通过，CAModel新结果以真实回传为准。

此途径的收益判据与完整大形数值门没有降低，变更的是各后端证明的范围；CPU检查器的超时本身不允许阻止使用独立完整硬件模拟取得大形数值/周期证据，也不允许冒充它已经通过。

### 完整裁决及实际接线

两臂CAModel已真实完成：当前H16 **3540306**、H32/member pipeline **2671072** ticks，速度 **1.325425倍**（耗时降低约24.55%）。全部partial/LSE/status/clusterStats逐位一致、336query独立原空间oracle、全部1008任务及20AIC/40AIV完成门均通过。原始VM结果为`/tmp/oscar-c4-composed-backend-split-20261013/comparison.json`。小形CPU337完整、大形CPU1341仅prepare完成、其numeric超时仍保留；没有把它合并为CPU1341通过。

`tools/serving_variants.py`已把C4 flag从RESEARCH移入FEATURE；两个固定candidate命令实际选择已存在的`attention_cv_member_pipeline_cluster4_out`，不只是编译算子。默认C16共享arena无需增大。`tools/probe_c4_member_pipeline.py`改为1341及同地址1373：成熟簇必须执行5轮→6轮，保留全部query/source/错误域、图回放、2warmup/5交替median及1.15完整门，任何失败仍阻断observe服务。39项最小host接线/门契约通过；未改CANN生产源、GDN、status_guard或精度。

这是C4局部完整路径收益，不能相乘q4/q1局部倍率或当作整轮2–4倍；真实NPU/graph和全模型证据仍待各自验收。

## D.4 四问

对应fused history dequant+FIA；旧全历史恢复6499.8–6655.1ms、store209–215ms的失败禁止重演；两臂仅有界KV256通信、FP32与既有固定状态，未引入全历史恢复、逐请求host循环或GDN改动；成本仍包含原12个C1、所有窗口/尾部与新增640KiB/core。本轮只预声明至少1.15完整CV门，不把tick代理换算为NPU毫秒或模型分钟。
