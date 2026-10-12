# 同core双task流水 v1：限定协议

母源为当前 `csrc/kernels/attention_cv_decode_bundle.cpp`，不是旧striped版本。档案#126/#140/#145/#148/#154–156与D.4全文已亲读；仅独立research，无生产、native、GDN或精度门修改。

## 已执行证据（2026-10-11）

- 同库 CANN 与官方 CPU-debug 编译通过；q4/q1 双请求 C641/701 的全部 source、partial/LSE/status 逐位、独立 oracle、输入不变和 guards 通过。
- q4 短形完整 CAModel：810475→775979 ticks，1.04445倍。增益过小，尚未加入组合；不据此声明长请求提速。
- q4 C8193/8225（每请求31个history单元）完整 CPU-debug 在120秒终止，cleanup完成，基线未完成；不记为CPU数值通过。原始620MB日志已完整校验归档。完整同库CAModel随后两臂通过全部输出逐位、任务、actor和独立oracle门：**3351197→2948283 ticks，1.13666倍**；显式标明无完整CPU长形参考。[裁决](../../../reports/decode_task_overlap_20261011/decision.json)
- 本次不扩写三任务park/resume：完整长形增益低于先前约1.566倍的条件预算，不能照搬该预算。短长两点相减得到每pair轮约87611/74907 ticks只是描述性差分；不同context改变工作集，不能当硬件常数，更不能写成整轮2–3倍。
- 已有真实prepare的K32、20–30K、S3合成表中，q4/q1均有80/96 history tasks、2600/3120 history units能被v1配对；是覆盖计数，不是吞吐比。奇数尾和不同轮数泛化仍未实现。
- 证据入口：[当前本地结果](../../../reports/decode_task_overlap_20261011/)、[实际任务覆盖](../../../reports/decode_task_overlap_20261011/v1_actual_pair_coverage.json)、[A2 flag契约](../../../reports/decode_task_overlap_20261011/flag_contract.json)。真实NPU/graph/整轮性能未执行。

## 范围与任务守恒

- 第一pass按原owner处理所有padding、invalid、source1/2和不合格history，逐task保持原完整数学；第二pass只收集合法source0、至少2个KV256的task。
- 两个task必须同原owner、不同request、KV块数相同；尾长可不同。不能配对及最后单task走原OSCAR bundle，均保留全部task/status，不是native回退。
- 一对全部完成后才开始下一对；v1不实现不同块数的共同段/尾段、odd-triad或跨core重分配，因此不声明K32整核≥1.5倍。

## 每KV协议

Cube顺序 `QA → QB → PVA → PVB`。两AIV各自顺序 `K_A_next → S_A → S_B → V_A_next → K_B_next → C_A → V_B_next → C_B`。无next时仅保留当前softmax/消费；各请求仍依原KV256顺序更新。

| 数据 | 所有权与释放 |
| --- | --- |
| A/B acc、max/sum、softmax输入stats | 两份原尺寸UB缓冲，任务生命周期内常驻；每KV无GM spill |
| shared packed/packedNext | A next K后一直留到A next V完成，才给B next K；两个softmax和PV消费不写packed |
| pendingLive/keyError/valueError | 每ctx独立标量副本；Activate/Save只换ctx，未来错误在本ctx上一PV消费后按旧K0/V0/K1/V1次序提交 |
| Q、双KV、score/P、PV | 每ctx独立原1146880B工作区；总2293760B/core。下一QK可覆盖本ctx已消费P，下一PV仍等本ctx下一PReady，故不能越过上一PV消费 |

每ctx每轮KVReady/ScoreReady/PReady/PvReady各一收一发；末尾NormReady/RotReady/TaskDone各一次。两lane都发送/等待，AIC沿原mode2聚合。bank为A的8–14和B的0–6，**不使用24–30**：A2 SDK将flagId掩为4bit。避开7/15；当前A2 GM输入、仅AIC调用的MatmulImpl不使用KFC/SyncAll，禁止向此版本加入占用11–14的SyncAll。

UB158656→175296B/AIV（仅加acc16384B、stats256B），原dequant/Softmax scratch及其GetSize未缩。原FP32、K128、HF32off、finite/status及同请求归约次序保持；生成器可完整恢复母源。原serial方法仅将acc/stats访问改为同尺寸对应ctx view。

## 编译接口与裁决边界

- `python3 csrc/experiments/decode_task_overlap/generate.py --check`；CMake记录`source_binding.json`。
- CANN库：`liboscar_decode_task_overlap.so`；symbols：`oscar_task_overlap_{baseline,candidate}_{q4,q1}_kernel`。
- CPU选项：`OSCAR_DECODE_TASK_OVERLAP_CPU_DEBUG=ON`；target：`oscar_decode_task_overlap_cpu`。本地probe仅改共享模板的基线符号、当前bundle workspace与标签；原模板不动。
- root负责fixture/runner/独占VM。先同owner两长task全source位/oracle/error门，再看完整配对；不同tail、future bad scale与另一ctx QR NaN必须隔离。未运行的CPU/CANN/CAModel/NPU/graph均不得称通过。

D.4四问：相位是fused history反量化与QK/PV依赖；旧全历史恢复6499.8–6655.1ms禁止重现；本方案仅两个固定ctx与UB状态、不增加历史长度相关缓存；目标是隐藏现存串行等待，**不减少INT2元素总量、scale次数或GM总流量**。旧条件预算不是本实现速度，初始化、二pass扫描、ctx切换、flags和serial余项全部计入完整核，不能重复领取已有1.698倍。
