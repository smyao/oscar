# Kt/Vt-NZ v2：仅去除重复索引转换

独立research；v1及其完整回退证据不修改。基线仍原ND H32＋C4成员流水，不能以较慢v1为分母。

- `BuildVt16Indices`直接构造永久4096个uint32字节偏移，完整范围0–16380，索引公式/位布局不变。
- 删除Init末尾narrow Cast及每次`PublishNz16`的两次4096元素Cast；完整KV256 K/V删除 **64×4096=262144次索引元素转换**。KV payload、32次Gather、32个连续16KiB发布块、FP32/K128/flags/error保持。
- 共享索引8→16KiB，显式UB165952→**174144B/AIV**，仍低于188416；未缩任何原缓冲或Softmax view。GM仍2494464B/core。
- 新库`liboscar_c4_kvt_nz_u32.so`；CPU选项`OSCAR_C4_KVT_NZ_U32_CPU_DEBUG`，target `oscar_c4_kvt_nz_u32_publisher_cpu`；两臂symbols `oscar_c4_kvt_nz_u32_{baseline,candidate}_kernel`。
- 三个生成器均支持`--check`。`publisher_probe.py --verify OUTPUT`保留两个完整K/V位布局、所有payload、输入不变和guards；回放保留独立oracle、1008tasks、全部60actors和同库完整输出门。
- 仍为Q336/C337/R1/20core/全部1008tasks，合法source2按生产prefill抑制；native current FIA不在此核时间内。CAModel **600s/arm**、publisher CPU **60s**。本v2完整CPU未跑；原B-only CPU超时不改标。
- 不预称收益：必须和ND基线完整配对。若本次完整核仍无明显净收益，停止此布局路线，保存结果，不继续用小改动延长试验。
