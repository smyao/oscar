# C16 B1 最终 Rv 驻留

基底为用户接受的完整 C16 B1 组合，母源SHA固定于`pins.json`。档案#126/#145的
读写所有权、#148/#154的CPU与设备精度边界及完整D.4已重新亲读。生产B1与包装器
保持原样；本目录双臂只用于本地Lima官方CPU/CAModel，不访问真机。

成熟簇最后16个成员仍依次做原FP32/K128逆旋转。唯一修改是把相同256KiB Rv的
`LoadResidentB(0,true)`提到final loop前一次；循环调用原`MatmulResidentB`。
首次装载保留MTE1→MTE2保护上次PV的V读，MTE2→MTE1发布Rv；每成员的
MatmulImpl::End不拥有externalB1。下个簇/任务覆写仍经原装载fence。

生成器要求完整逆向恢复固定母源；除CubeCluster外全部方法（含Init、所有Vector、
Matmul、错误/发布/扫描）逐字一致，CubeCluster跨核flags逐字同序。
GM7487488B/core、UB157760B/AIV、B1=256KiB全部不变；不新增资源。
每成熟簇减少15次Rv载入=3.75MiB逻辑GM→B1，不是测得HBM或整个kernel倍率。

预声明门：先C337/recent1/Q336两KV CPU两臂，再末成员q315/head0 QR NaN隔离
两臂，全部1008tasks、原lane/status/独立oracle/guards/输入不变和逐位比较；各120s。
通过后只运行C65/recent1/Q336同库双臂CAModel，各600s，全部1008tasks、20AIC/40AIV，
真实completion、冻结原空间oracle与输出逐位。只用官方LOG mask0，完全没有FD轮询
线程；parsim0、L2、硬件拓扑均与已完成基底一致。任何CPU或超时失败停止，不自动加时。

计时仅CV核，实际partial离线LSE合并用于精度；不把它写成设备merge/BF16-copy全链。
单KV机制形不代表recent256、多KV性能、32K/64K、NPU、图或整模型native30%目标。
用户已放宽小正增量门，按同库完整净结果决定去留，不拼乘旧不同分母的倍率。

完成结果：独立CANN/CPU编译11.17s；C337两KV和末成员QR错误隔离四个CPU臂全部逐位/oracle通过。
新同库C65 CAModel：B1 3168844→Rv 3045844 ticks，1.0403828955倍；wall486.941/461.580s，
两臂rc0/清理完成，1008tasks、20AIC/40AIV全部profile、输出逐位和独立oracle通过。
用户/主线程接受该4.04%零资源净增量并单独处理生产打包；不与旧库stable/B1倍率相乘。
完整小型证据见`reports/c16_rv_reuse_20261015/summary.json`。
