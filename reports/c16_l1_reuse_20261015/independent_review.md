# C16 B1 独立只读审查

当前所审源码的哈希见 `independent_review.json`。重读完整D.4及G3/G7/G8/#126/#145/#148/#154/#157后，未发现需要修改当前草稿的flag轮转、B1覆盖、score16生命周期或错误顺序缺陷；这不是编译、设备或性能验收。

- **四组flag**：Cube在发Score(i)前（i≥4）等待P(i−4)，避免覆写尚未被两个AIV消费的通知；成员15结束后另排空P12–P15。AIV依次等待Score再发布P，实际score/P数据放在16个独立槽内，flag复用不复用未消费的数据槽。
- **B1所有权**：`LoadResidentB` 的MTE1→MTE2等待约束旧L1读完后才覆盖；ND2NZ结束后的MTE2→MTE1等待约束新L1发布。K一次装载服务16次QK，随后V装载服务16次PV。C1提前加载V仍发生在已有KVReady之后，V的GM数据此前已发布。
- **工作区**：一个原始score/P槽加15个附加槽，尾界为7487488 B/core；附加score与K/V双buffer、各成员acc/stat状态无重叠。各成员状态的原Spill/Restore屏障保留。
- **错误和数学顺序**：原TaskValid、IsClusterMember、INT2/precise publisher、Softmax、SpillState、RestoreState、PublishRows/Status按生成器逐字节校验。每个成员仍按原KV顺序进行FP32在线更新；不同成员的QK/PV重新排程不作为数值已通过的证据。
- **SDK边界**：审阅了当前保存的官方B1输入与cross-core实现；直接MatmulImpl的外部B1不同于KFC客户端协议。是否实际生成预期L1装载和屏障，仍须完整编译及指令/设备证据。

执行命令：`.venv/bin/python reports/c16_l1_reuse_20261015/independent_review.py`。结果为1809个可达host交错状态、1个完成状态、0个flag重复通知、0个死锁、完成后无残留token。脚本是一个Cube、两个独立AIV的mode-2通知/确认抽象，不模拟DMA、Matmul或硬件流水；不能据此覆盖#126/#145/#154的异步内存竞争验收。

没有连接真机、占用VM或修改内核。短形基线超时或未完成配对不能改写为达到1.15倍，独立NPU入口仅准备后续验证。
