# stable 基底上的 32K / 64K 优化

档案依据：#34/#36/#126/#129/#140/#145/#148/#151/#154–160，启动文档 §0.7 / D.4。以用户确认 APC 开/关均可工作的 `f96820f1` 为基底；已完整快进同步至 `codex/oscar-ascend`。本轮不连接真机，不请求重跑 AISBench。

## 2026-10-15 用户裁决

B1进入主分支candidate，作为后续叠加的比较基底。用户明确取消每个优化点必须达到1.15倍的规则；今后检查完整相关路径的净收益及资源代价，设备门保留不回退要求，冻结精度不变。历史1.15门的原始失败记录保留，不改写成当时通过。继续迭代后只推主分支，stable由用户真机确认后自行更新。

## 启动与验收分母

`configs/target.json`、全部 vLLM argv、默认物理设备0–3/端口8989、APC默认关闭及显式开启选项、16384 token预算、原生调度器均保持 stable。两个 candidate 入口共享 `tools/serving_variants.py`；独立保存的四份完整 argv 在 `reports/stable_32k_64k_20261015/frozen_launch.json`，不能重新生成它来掩盖漂移。FE0七个冻结核及 `configs/acceptance.json` 保持字节不变。

客户端完整结果、服务端片段与来源边界见 [原始证据分析](../reports/stable_32k_64k_20261015/analysis.md)。32K/64K分别为300请求、512输出：native为1895.8486731/3305.4846943秒，OSCAR为2216.6739733/6949.4660925秒。速度提升30%的边界为1458.345133/2542.680534秒；保守耗时下降30%的边界为1327.094071/2313.839286秒。局部倍率不能替代两场景整轮目标。

附件没有实际启动SHA、PERF_RUNTIME_CONFIG或cv_operator；以下是当前源码中确认的成本，不把它写成所测全程的相位占比。

## 本轮针对的工作

固定16384预算使大query走C16。此前的C4成员流水、q4/q1 MDL都不替换C16；旧C16每个成熟KV256块给16个成员分别做FP32 QK/PV，重复把同一K/V装入L1。条件32K→64K下，历史chunk工作量可从一份16K扩为16K+32K+48K，约6倍，不能据此推算实际秒数。

C16 B1方案保留16成员的INT2解包共享与逐输出FP32/K128顺序，将同一KV的K、V各装入单个256KiB B1一次，16个score/P槽保存成员结果。四组score/P通知轮转并确认消费，最后完整排空；覆盖B1前保留MTE1→MTE2及反向屏障。每成员的softmax、在线状态、错误域和最终Rv旋转保持。实际选择只替换原C16 reader，不改变请求、调度、GDN、prefix所有权或缓存格式。

q4 hybrid L1复用已有完整同库配对：MDL 2658960→2214530 ticks（1.200688倍，约8K历史机制形），短形约1.109倍。它只替换q4；q1继续MDL，C4继续既有已资格成员流水。Rv按FP32 NZ地址排列在模型加载时一次性生成，原始数值不变；旧路径继续使用natural Rv，禁止在forward临时打包或缺失时回退。新NPU门另含32K/64K短合成算子病例。

## 固定资源账

- C16 stable为4997120 B/core；普通流水5652480 B/core；B1为7487488 B/core。20核时B1较stable增加47.5 MiB/rank，在原生KV profiling前分配，不能藏在KV容量收益里。
- q4 GM仍为1146880 B/core；NZ Rv增加262144 B/层（16个target FULL层加draft约4.25 MiB/rank），同样在profiling前准备。
- 均不分配随历史长度增长的恢复张量。逻辑GM→L1字节减少不是实测HBM流量，更不是速度倍率。
- `reports/c16_metadata_source_cost_20261015.json`另记录固定任务扫描；N16384条件下78384次LoadTask不会由B1消除，未测其设备时间占比。

## D.4 四问

1. 新reader对应融合history dequant/FIA内部的浮点数据装载和成员等待；不是新增全历史dequant阶段。
2. 历史失败边界是约6.5秒全历史恢复、约215ms写路径及725ms主机prepare；这些旧数字不作本轮分母。
3. 本版仅有固定KV256、单个B1与固定score槽；FP32精度、错误域、真实物理页与metadata虚拟块边界保留，没有CPU oracle生产路由、全历史BF16恢复或静默原生回退。
4. 成本收益必须由完整算子配对与目标设备门决定。C16普通流水Q672/recent256 CPU120秒和CAModel基线900秒超时，保留失败，无倍率；该大形失败不代表后续小形未完成；普通流水作为B1依赖与研究基线，默认选择B1。prepare复用只有host等价证据，也保留research默认关闭。

## 验证边界与入口

本轮已完成冻结启动、配置传递、真实Python forward选核、RV布局、workspace和错误阻断等主机检查。普通C16流水及B1的小CPU验证包括完整16成员、两KV轮、非零qbegin/core、元数据错误lane与成员错误隔离；机制形recent1不代表目标recent256整轮验收。ENOSPC中断、日志边界与恢复后的新结果分别记录，不能覆盖失败。

44项能力的完整CANN/Torch fresh build已通过（157.04秒，rc0，cleanup=true），见 [构建报告](../reports/stable_b1_fullbuild_20261015/summary.json)。C16完整同库三臂已完成：stable 3888161 ticks / 587.763秒，pipeline 3302518 / 512.878秒，B1 3172246 / 487.095秒。B1对stable为1.22568倍，对pipeline为1.04107倍。范围为Q336/recent1/C65单KV完整CV核：1008任务、20 AIC＋40 AIV、partial/LSE/status/counters逐位与冻结oracle均通过。此处wall是模拟器耗时；不含设备merge/copy，不是32K/64K模型端到端收益。见[完整报告](../reports/c16_l1_reuse_20261015/summary.json)。尚未取得真实NPU、图或两场景30%端到端达标证据。

正式直启仍为 `git pull --ff-only && bash scripts/install_serve.sh --variant candidate`；它不提供测试证据。测试入口为 `git pull --ff-only && bash scripts/install_observe_serve.sh --variant candidate`：真实算子/图/完整reader配对任一失败即阻断服务，成功后按同一stable argv启动。无需新vLLM参数或AISBench重跑指令。
