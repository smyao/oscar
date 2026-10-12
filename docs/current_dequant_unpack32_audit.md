# 当前反量化审计与 HalfKv32 候选

2026-10-10；档案 #13–22/#70–73/#126/#144–145/#148/#151/#154–156，启动文档0.7与D.4。这里只分析当前本工程生产源与同源CAModel证据，没有读取失败工程代码、恢复全历史、降低精度或跳过非法metadata。

## 当前确实存在的差异

| 路径 | 当前helper | 批次/自然行stride | metadata检查 |
|---|---|---|---|
| decode_bundle q4/q1 | `attention_striped_unpack_padded.h` | HalfKv64 / 544B | SIMD原始FP16位检查 |
| window_range C4 | `attention_striped_unpack.h` | HalfKv16 / 512B | 逐live行FP32 scalar检查 |
| request-window decode | 同当前decode helper | 64 / 544B | 同上 |
| request-window C4 | 同当前C4 helper | 16 / 512B | 同上 |

C4源码中的`oscar_ascend_fast::Unpack`通过文件头macro实际指向`oscar_ascend_striped`，不能误当旧fast的双大Gather。当前两个helper都已直接Shift/And写自然INT2码；**只剩metadata Gather，不再有code大Gather**。C4仍分配旧plane/word/laneIndex空间，但striped helper不消费这些大数组。

准确入口：`attention_cv_decode_bundle.cpp:179–181,302–305,622–626`；`attention_cv_window_range_cluster4.cpp:427–429,533–558`；对应request-window文件仅改变地址/epoch协议，不能把其未测速度当作已测结果。

PR明确：FP16 metadata转FP32，INT2 level转FP32后`q*scale+zero`；decode QK/PV及归约保持FP32。候选不改INT2位序、metadata舍入、Mul→Add次序或HF32状态。

## 现版已有计时，而非旧热点

- 当前C4 core3/AIV0闭合history范围：96次Unpack、212598 cycles，即约2214.56/16行；包含前沿C1和成熟C4，未拿截断finalize补总量。
- 当前decode q4/R32/N128/S3/C1025、core0/AIV0：20次Unpack、46527 cycles，即2326.35/64行；10次packed DMA共14310，20次KV publish共2994。其整体60actor日志完整，但每split仅一个history tile，不是长历史稳态。
- 两种helper每次耗时相近却批量相差4倍，支持批量化研究；它们不是同形A/B。旧#151的bits/meta比例属于旧实现，不套给现在。
- 当前profile只圈定完整Unpack，不能从这些总量伪造当前bits、Gather、half转换或Mul/Add的独立周期占比。

## 唯一已推进方向：C4 source0 HalfKv32

64-row直接扩容会超UB；32-row配合删除legacy大缓冲，C4显式UB **157760B**（旧157632B），低于184KiB，余30656B。C1 acc64KiB仍常驻，score32KiB独立。

每AIV、每完整KV256的静态变化：

| 操作 | 当前16 | 候选32 |
|---|---:|---:|
| Unpack K/V调用 | 16 | 8 |
| ShiftRight / And调用，各 | 256 | 128 |
| metadata Gather / half→float，各 | 16 | 8 |
| Brcb | 32 | 16 |
| FP32 Mul / Add，各 | 64 | 32 |
| code Cast调用 | 16 | 32（padded四片Cast） |
| Scalar metadata逐行检查 | 256 | SIMD位mask检查 |
| packed GM字节 | 17408 | 不变 |
| FP32 K/V发布字节 | 262144 | 不变 |

元素运算量没有因API调用减半而减半。新helper用compareRows64、scale低32bit、zero高32bit、liveMask0–32、scratch1024B，保持完整非法数据检查。默认C4的metadata逐行同步与512B自然行stride被当前已验证decode机制替代。

**旧lane归属必须保持**：候选32行来自本lane原来的两个16行片段，`logical=start+(j/16)*32+lane*16+j%16`。不能用`lane*32`，否则非法scale0的错误status会换lane。Publish两段16行恢复原GM行位置。因此Physical/GM读取段数没有宣称减半。

精确window/current、PV消费仍16行；全部非Unpack scratch view保持原512B，Softmax临时view保持原16KiB，避免扩大buffer顺带改变V2 tiling。QK/PV/归约顺序、状态后续覆盖和跨核flags保持。源恢复检查及旧lane集合/尾行主机契约已通过。

## 已取得的局部验证

`reports/quant_unpack32_20261010/comparison.json`记录：

- 193例官方CPU：old/new输出/status逐位、独立oracle、guards通过。
- 包括全部65536 packed-word模式、33个live边界、dead NaN/live负零scale及K/V scale/zero边界；另有65536种FP16编码guard的host穷举。
- 同库、同8个32-row块、相同SIM配置CAModel：旧 **98478**、新 **41974 ticks**，局部 **2.34617×**，独立oracle与保存输出逐位通过。

该短点含helper Init/load/output；不是纯反量化流水周期，更不是完整C4/NPU/图/服务收益。当前没有第二个已证明同量级的反量化方案，因此不为凑数推进K/V metadata合并等小改动。

完整C4候选正在独立验证Q336/C113、1成熟簇、所有1008任务及bad80/bad96原lane错误归属；源码与入口位于`csrc/experiments/c4_unpack32/`。生产路由未启用。本报告不把2.346×乘到整核或全服务。

## root 后续完整C4结果（2026-10-10）

Q336/C113的旧新完整CAModel均完成，全部partial/LSE/status/counters逐位和独立oracle通过，但周期1355015→1346984，仅1.005962倍。三组CPU配对valid/bad80/bad96均通过。此短形不代表64K，因此不将H32单点独立接入生产或声称整体加速；证据为`reports/quant_unpack32_20261010/full_c4_camodel_comparison.json`。
