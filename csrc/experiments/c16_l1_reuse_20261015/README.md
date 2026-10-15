# C16 单块 B1 驻留

独立候选，分母是本轮 C16 H32/member pipeline，不能重复领取旧 Normal→MDL 或
C4 pipeline 的倍率。档案 #126/#129/#145/#148/#154–157 和完整 D.4 已亲读。
最初源码生成时尚未编译或执行。现已完成独立CANN/CPU编译、owner0两形、
owner1两请求、metadata三错误域及末成员QR隔离的CPU验证；官方日志设施修复后，
单KV三臂完整CAModel门通过。多KV、目标recent256、NPU及整服务性能仍未建立。
B1保留独立研究，默认选择由主线程结合增量收益与工作区成本裁决。

每成熟 C16 簇的16次 QK重复装同一 K、16次 PV重复装同一 V。新核用一个256KiB
外部 B1，先装 K计算16组 QK，再装 V计算16组 PV；每组数学和K128累加顺序不变。
原 AIV INT2解包/发布、softmax、Spill/Restore、错误覆盖顺序均逐函数字节保留。

16组各自拥有 score/P 槽。只有4对跨核 flags：Score9..12 与 P0..3；Cube在提交
QK(i+4)前消费 P(i)作为score flag已被AIV消费的确认，末尾再drain最后4个P。
因此不使用16+ flag，也不在尚未消费的同一flag上重复set。P矩阵不因ack而复用，
一直保留至该轮 PV全部完成；RoundDone仍保护下一轮RestoreState。下一KV仍写
另一个固定GM槽。覆写B1前 MTE1→MTE2，装载后 MTE2→MTE1。

单B1占262144B，SDK A/reserve与B1总和编译断言不超过512KiB；只使用一个
MatmulImpl。workspace为7487488B/core，比pipeline多1835008B/core、比stable
多2490368B/core，20核时较stable新增47.5MiB/rank，必须在KV profiling之前计入。
逻辑B载入从8MiB变512KiB/成熟KV不是实测HBM量，也不代表16倍提速。P全员发布后
才开始PV会损失部分交错机会，增加score活跃集合；完整净结果必须计入这些代价。

完整门复用本轮C16：小CPU Q336/recent1/C65与C337两臂独立oracle/逐位/全部状态，
bad80/bad96/bad336仍须真实运行；全Q672/C337/recent256保2016任务和20Cube，
使用原真实prepare检查点，最初完整CAModel计划每臂900秒。stable Q672基线实际
900秒超时，未获数值完成；没有增时或原样重跑。改用已完成CPU的Q336/recent1/C337
机制形做同库stable/pipeline/B1三臂，每臂600秒，保全部1008任务。最终选择依据
stable→B1完整组合1.15门，pipeline→B1增量单列，不拼乘旧倍率。
不能把小CPU当大形CPU通过，不能把本地模拟当NPU/graph/32K或64K最终性能验收。

原日志设施下有界结果：Q336/recent1/C337的stable首臂600秒超时；进一步声明的C65
单KV完整机制形stable首臂亦600秒超时，两次均无completion产物，后续pipeline/B1
臂未运行。两次小形的profile文件为0字节，不能把缺失记录写成actor完成数为0。
只读host线程快照表明模拟进程仍活跃，不能据此区分计算、设施调度/缓冲或协议自旋。
这些失败完整保留，不用后来的成功回写退出码或宣称唯一根因。

日志设施修复只改`LOG.core_enable_mask=["0x0"]`并完整移除旧轮询FD的quiet线程；
原线程存在readlink后dup2的FD复用竞态。parsim0、L2、model_top、FFTS、核、输入和
数值/profile门不变。已知短CV控制完成1 AIC/2 AIV与原oracle后，以新ID将原C65三臂
重试一次，仍每臂600秒。三臂均完整返回，所有1008任务、20 AIC/40 AIV profile全正，
partial/LSE/status/counters逐位一致，冻结独立oracle无错误。

| 臂 | CV task ticks | 实际模拟wall秒 |
| --- | ---: | ---: |
| stable | 3888161 | 587.763 |
| pipeline | 3302518 | 512.878 |
| B1 | 3172246 | 487.095 |

同库stable/pipeline=1.177332、stable/B1=1.225681、pipeline/B1=1.041066。
只计CV核；精度通过的是实际partial离线LSE合并，未测设备merge/BF16-copy全链时间。
scope为Q336/recent1/C65、一个16成员成熟簇、单history KV轮，不代表目标recent256、
多KV或32K/64K。B1增量4.1%尚不足以支持比pipeline多35MiB/rank的默认资源成本，
建议默认采用pipeline，B1保留研究。stable单KV已经587.763秒，因此不启动接近截止的
C337双KV重试，不扩时。原始失败不能证明该FD竞态是唯一根因，修复后的完成只建立
这一配置和这一完整机制形的新证据。

非零owner门为合法两请求：qlens336/336、contexts0/337、recent1；第一请求的CV
空源逐位检查0/-inf，第二请求qbegin336..651在core1有完整16成员、两KV轮。
stable/pipeline/B1三个CPU臂均已通过完整2016任务、冻结原空间oracle和逐位门。
末成员QR门仅将qr[315,0,0]注入NaN，精确要求leader315/lane0错误2，其64条history
输出行fail-closed，其他1952行原空间oracle通过；pipeline/B1逐位相同。

环境失败另存：CPU checker每臂约1.3–2.6GB scratch累积后发生ENOSPC；失败bad80
的最终control无法落盘，child rc未知，不补造退出码、不当数值失败。已归档校验的
本轮成功scratch回收后，仅以新report ID重试该环境失败一次；三错误域随后通过。

用户后续裁决（2026-10-15）：接受stable→B1完整组合1.225681作为下一轮主分支基底，
不因相对pipeline仅4.1%剔除；上文pipeline推荐保留为裁决前历史判断。继续在B1基底
迭代，不机械要求每个增量1.15倍；原实测范围、47.5MiB/rank资源账和32K/64K目标
尚未达成的边界不变。下一独立实验为`../c16_rv_reuse_20261015/`。
