# C4 跨成员流水预算：只基于闭合循环，不放行截断 trace

依据：档案 #132–135/#145/#148/#154/#155 与启动文档 D.4；当前源 `attention_cv_window_range_cluster4.cpp`。本轮只做宿主证据分析，没有修改 CANN、数学、dtype、精度门或原生预算。

## 证据与范围

Q336/R1/C769/S1/20core 的 normal 已完成，2,543,951 ticks；profile 已完成数值及全部输出，2,660,850 ticks，扰动 +4.595%。两个 core3 AIV 指令日志在最终写回达到64MiB上限，**完整 marker 门继续失败**。原 `checked_arm`、`compare`、`analyze` 的失败判据没有修改。

独立 `partial_loop_budget.py` 只读取 core3 第二个 source0 task，即实际任务表的唯一成熟C4簇（query起点252/273/294/315）。AIC的12次QK/PV及两条AIV三轮所有解包、softmax、PV消费、循环内state搬运均闭合；最终归一化/写回没有补值。输入、输出、同库、同配置、全行oracle及逐位结果再次核对；只允许已见的两条日志截断原因。

第一轮 `[64,320)`，第二轮 `[320,576)` 均256行；第三轮 `[576,769)` 是193行尾块。第三轮softmax更慢，不能按193/256缩放。

## 实际成本：第二轮full256

| 工作 | 实际闭合范围周期 |
|---|---:|
| 每成员QK | 20,665–21,657 |
| 每成员PV | 26,643–26,648 |
| AIV0每成员两个softmax块 | 23,717–24,195 |
| AIV1每成员两个softmax块 | 22,673–23,159 |
| QK/PV完成到AIV ready的可见间隔 | 约3,520 |
| AIV P发布到Cube可用间隔 | 773–777 |
| 最后state spill到Cube ack结束间隔 | 4,262–4,587 |
| 两AIV KV准备包络 | 53,896 |
| 上一轮Cube ack至本轮ack | 390,694 |

以上等待间隔作为DAG依赖边保留，**没有把Pwait或KVwait总时长直接视为可删除成本**。首轮407,342周期KVwait含query加载和此前actor不同步，不能作为稳态解包成本。

## 候选DAG（尚未实现）

Cube顺序：`QK0 → QK1 → PV0 → QK2 → PV1 → QK3 → PV2 → PV3`。

每条AIV顺序：`S0 → S1 → C0 → S2 → C1 → S3 → C2 → C3`。

- `S_i`：恢复旧state，原softmax/alpha缩放，再把scaled acc/state以FP32原位写到已有成员GM state。
- `C_i`：恢复该scaled state，按原顺序加PV，再写更新state。
- `QK_i → S_i → PV_i → C_i` 保持数值依赖，每个成员的KV累积顺序不变。
- 两套score/P槽按成员交替，只有对应PV读完P才能覆写。单PV槽仍受上一成员消费确认约束；预算保守沿用原spill后ack，没有假设更早ack免费可得。

每Cube额外score/P为128KiB，双KV再增加512KiB，20core合计12.5MiB。每C4/KV新增FP32 state搬运 **1,056,768 B**，模型已加入实测restore/spill周期，没有视作零成本。

双KV预取只把下个tile的八个完整`LoadPacked→K unpack/publish→V unpack/publish`包放入AIV空隙；不能与同AIV的softmax/PV消费同时占用UB。这里是可行时序预算，仍须实现固定的合法预取顺序、独立next-KV错误状态及完整缓冲所有权。

## 预算结果与决策

| 范围 | 预算结果 | 含义 |
|---|---:|---|
| full256，只有跨成员流水 | 270,443 cycles / **1.445×** | 单独未达到1.5× |
| 重复full256，再用空隙预取next-KV | 216,547 cycles / **1.804×** | 有条件的稳态预算，值得独立实现验证 |
| 三轮串接，首轮准备保留、tail没有下一轮 | 731,103–731,261 cycles / **约1.67×** | 仅选中C4循环 |
| Q336整个profile kernel，其他成本固定 | **约1.226×** | 不能声称整核1.5× |
| 同测时模型中，选中循环除现有Cube QK/PV外全免费 | **整核至多1.319×** | 此低占用形状中固定成本限制明显 |

这些是**维持现有相位时长、无新增GM争用、固定成本不变**的调度预算。并发改变缓存/搬运/SDK排队后，实测可能不同。没有使用未知finalize值，没有跨actor求和成wall，也没有把Q336结果换算成K32或64K端到端收益。

结论：safe-range单点继续淘汰；跨成员流水单独也不足门槛。**跨成员流水加有界next-KV预取有1.8×稳态预算依据，但完整算子/服务收益仍未建立**，后续应在保持原算子数值与错误语义的独立候选上验证，不能据此直接推送生产优化。

完整时间戳、DAG节点、预取包安排与来源hash：`reports/c4_phase_profile_q336_20261009/partial_c4_pipeline_budget.json`。复现工具：`csrc/experiments/c4_phase_profile/partial_loop_budget.py`。
