# B1 + Rv 基底后续组合（研究准备）

唯一生成链固定Rv候选SHA `9603817d…`，然后依次应用本目录`apply_finite`和
`c16_anchor_skip_20261015/generate.py::apply_skip`。只改变Softmax与Process；
其他方法、CubeCluster、跨核flags、GM/UB/L1声明逐字不变。本组合现已完成独立
CANN、CPU特殊值及完整CV、同库CAModel配对，生产迁移由主线程单独处理。
生成manifest中的not_run是生成时记录；实际资格以文末及reports完成回执为准。

有限性检查仍对每个active row执行原FP32 `ReduceSum(..., tmp[16], 256)`，顺序不变，
不引入sentinel，也不删除溢出/NaN/Inf检查。仅当kind0、jEnd256、start≥sink且tileEnd
不超过最早query的context/cut/position+1时批量回读结果。位置与cut随row非递减，
因此每个原始visible interval都等于整个tile，原any必为true且原mask为空；其他精确源、
尾块、partial或空行走原for-loop字节。activeRows之外的padding不加入检查。

每行结果放在已死亡的naturalBuf的`8*r` FP32偏移，32B对齐，最多1024B；原tmp[16]
仍作ReduceSum工作区。每次PIPE_V保证工作区复用顺序，最后一次V_S等待全部结果，
按原行序判finite，再S_V保护标量读取完成后后续alpha Brcb覆写。没有新buffer，
没有把V2输出alpha换成finite结果，也没有提前覆写score/P。

预声明后续门：原完整C320/recent1/Q336同库两臂，history[64,320)恰好一个满256tile，
1008tasks/20AIC40AIV/独立oracle/完整completion，600s每臂。C65全部fallback，不能
作为finite收益测量。额外CPU须覆盖有限值sum overflow、有限取消/中间溢出、NaN、
正负Inf、被mask的非有限值及空行；每个原ReduceSum和原状态域均保留。未通过前不
领取任何增量；失败保留、不扩时。只计这一组合完整新增净收益，不乘两项局部假设。

D.4的全历史恢复6.5秒与store约209–215ms是旧结构失败红线；本方案只减少有界
同步/元数据访问，不把旧毫秒当当前相位占比。#126/#145约束UB读写所有权，#134/#140
约束原mask及pre-mask finite语义，#148/#154要求CPU与设备精度边界分别记录。

完成记录：CANN/CPU构建10.933s；实际Softmax十二特殊值病例2.842s、完整C320两臂
CPU34.201/33.875s全部通过，均rc0/owned清理完成。新同库CAModel为Rv2884171→
组合2640555 ticks，1.0922593924倍，wall455.509/416.679s；所有1008任务、20AIC/40AIV、
独立oracle和partial/LSE/status/counters逐位门通过。主线程接受本项9.23%零资源增量
并处理同ABI生产打包；不拼乘之前C65/旧库倍率，NPU/图/native端到端30%仍未建立。

后续tail8仅为只读想法：最后波C16簇拆成两C8共享簇需要真实prepare与新增展开/owner
成本证明，未实现、未测、未启动实验。本轮优先交付现有合格组合。
