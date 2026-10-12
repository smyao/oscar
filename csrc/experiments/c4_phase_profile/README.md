# C4 同源相位诊断（未执行设备验证）

## Q1344失败后新增的有界诊断点

Q1344 normal 实际达到原定900秒死线，监管器记录`timed_out=true`、TERM清理期间child rc=-11、`cleanup_complete=true`，没有completion或数值输出。不得采用任何ticks，也未证明算子数学错误；旧记录保留，不重跑、不增加900秒。

现只允许两种明确声明的Q：1344（保留记录）和336（下一次有限相位定位）。`plan_q336.json`先于新oracle生成。Q336保持R1/C769/S1/20core和全部1008任务、原CANN算子/FP32数学/库；保留12个历史前沿C1组、一个成熟C4簇及完整window/current空任务。预计source0 member工作由189降到45个KV256，须以真实prepare账本确认。**Q336低占用，不能代表K32整步、Q2048饱和或64K整体目标；目标未改变。**

生成新fixture命令：`python3 csrc/experiments/c4_phase_profile/fixture.py --tokens 336 --export --output NEW_FIXTURE`。只需重编metadata导出driver，其输入字节数从声明Q派生；CANN attention源码与SO不变。replay/marker/compare从经过校验的fixture/actual tasks读取Q，仍为同形两arm和同900秒门，不允许Q336与Q1344配对。

档案依据：#126/#132–135/#140/#145/#148/#154/#155；已亲读启动文档0.7及D.4完整原始打点。D.4四问：本实验归因fused FIA；旧6.5秒全历史恢复不能复现；仅增加可逆scalar计数/时钟/MOV marker，不加完成屏障或数据路径；任何收益须来自同形正常arm，profile只在扰动门通过后归因。

## 固定范围

- 唯一源：`csrc/kernels/attention_cv_window_range_cluster4.cpp`，SHA见manifest。
- Q1344、R1、Hq6/Hkv1、D256、S1、20 Cube、context769、sink64/recent256/spec3。
- 这是生产mixed分区中的一个prefill CV代表点，不是K32整步或端到端。
- 成熟簇历史区间[64,769)，705行、三个KV256；C1025会是四块，不采用decode工作量公式。
- 实际prepare的4032个任务全部保留。source2按生产`_empty_current_source_ranges`仅把健康任务end设为begin；前沿非成熟组、精确窗口、所有follower/status均保留。
- oracle逐个检查全部1344查询的source0+1合并结果。**不包含native current FIA及完整attention wrapper的时间。**

## 身份、空间和测量

`generate.py --check`校验全部依赖固定SHA；normal/profile的每一项替换均可逆，完整恢复唯一源全部字节，原数学/FP32/HF32/fence/setflag/wait数量不变。没有迁移64-row unpack、没有改C4生命周期。

独立profile ABI schema3：`[20,3,4,32] int64`，每Cube3072B。字段0–23复用原phase helper；24/25为state_spill/state_restore，26 clusters，27 member_kv_units，28 state_bytes，29 c1_tasks，30 cluster_members，31为finite_reduce。`tasks`计执行的C1/cluster函数次数，cluster一次含4成员；不能直接作为逻辑query任务数。`kv_units`是一次共享tile，`member_kv_units`另列每组工作。state_bytes为本actor所有spill/restore字节，包含初始化和结束阶段。

`finite_reduce`只包住Softmax现有`ReduceSum(scores行,256)`、`V_S`等待、`Finite`检查三条语句；原来的前置`PIPE_V`、循环、所有原行顺序与位置关系保持。它嵌套在softmax_p内，用exclusive时间区分V2与逐行归约，不能重复加总。marker次数按实际task/member/每KV256/lane有效query-head，重新执行原`any`区间判定推导；整行不可见或padding不计。首次CANN编译成功属于尚无field31的旧profile，须保留旧成功记录，并重编此修订后的normal/profile同库再计时，不能复用旧库结果。normal源码不变。

正常ABI16个GM指针；profile17个，profile位于clusterStats之后（index16）。原17个int64属性和一个float不变。每Cube固定workspace1,839,104B，20核36,782,080B；原AIV显式UB157,632B。profile scalar局部计数的真实资源影响尚待CANN编译/模拟，不以显式UB账冒充编译结果。

每arm计算硬截止**900秒**，prepare CPU硬截止120秒；失败保留、不加时、不重试kernel。normal/profile须同库、输入、SIM配置、核数、死线；partial/LSE/status/clusterStats逐位且全行独立oracle通过；绝对扰动不得超过5%。SYS_CNT零表示不可用，需完整60 actor MOV完成日志，不能当成相位耗时为零。嵌套相位不相加，各actor最大值不相加成wall。

## 当前交付与后续执行

已提供同源CANN构建目标和独立真实prepare CPU导出工具，尚未编译或运行。

1. `python3 csrc/experiments/c4_phase_profile/generate.py --check`
2. `python3 csrc/experiments/c4_phase_profile/fixture.py --export --output <新fixture目录>`（仅host oracle）。
3. 独占VM slot后，以`OSCAR_C4_PREPARE_CPU_DEBUG=ON`构建`oscar_c4_prepare_cpu`；有界执行`oscar_c4_prepare_cpu FIXTURE NEW_OUTPUT`。
4. `python3 csrc/experiments/c4_phase_profile/fixture.py --output FIXTURE --prepared NEW_OUTPUT`消费实际tasks导出、执行生产source2抑制并产生逐owner账本。预估13成熟簇、72历史tile、67窗口tile必须以这份真实输出复核；host结构测试的合成表不作设备输入。
5. 默认CANN构建产出`liboscar_c4_phase_profile.so`；`OSCAR_CAMODEL_MARKERS=ON`同时保留normal/profile。

## 独立执行适配

`replay.py --arm normal|profile --fixture FIXTURE --prepared PREPARED --output NEW_ARM --library SO --simulator-config CONFIG`只支持本固定shape。需由VM本地`run_camodel_arm.py --timeout 900`监管，SDK preload只在child env中，所有路径使用绝对路径。normal/profile使用同一SO和SIM配置。profile首次可不保留指令日志；通过配对数值与扰动门后，显式`--completion-markers`保留每actor有界completion普通文件、丢弃popped日志、顺序压缩并验证完整60actor。原4.5GiB可用tmpfs和64MiB/actor上限保留，达到上限即标不完整。

`markers.py`独立schema3解析、实际task账本推导的两遍LoadTask/cluster/C1归属、各phase计数及真实profile整数counter对齐；复用已验证普通文件collector时只复制其函数globals并替换本schema parser，原decode模块不变。禁止把schema3送入decode的24字段门。

`compare.py --normal DIR --profile DIR --normal-control JSON --profile-control JSON --output JSON`检查真实durable supervisor完成/900秒/进程归属，全部落盘输出hash，固定来源/输入/配置/geometry及冻结oracle，partial/LSE/status/clusterStats逐位与±5%扰动。若采集marker，还重新检查全部actor闭合与实际工作计数。不得拿profile比normal的比值宣称优化收益。

以上执行适配仅经过host结构/门验证，尚待第一次真实SDK运行；本原型不注册任何生产符号、不启用candidate、不声称NPU/graph/速度达标。

## 已完成profile的逐actor分析

`analyze.py`接受与`compare.py`相同的五个路径参数，先通过完整arm及±5%扰动门，再要求完整60actor markers。它依据真实tasks两遍调度顺序分开source0的C1前沿与C4成熟簇；输出各actor的exclusive相位、固定/每KV/未归因成本，finite_reduce从softmax父相位中扣除。C4初始四次spill和末尾四次restore计固定，其余state搬运计变量；task-body残差不冒充某个算子。理想删除finite_reduce的比值只限同一actor/同一类，不能写成完整算子或端到端提速。尚无完整profile时不输出相位比例。
