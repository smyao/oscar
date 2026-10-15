# C16 H32 成员流水

档案 #126/#129/#145/#148/#154–157 与启动文档 D.4 原始日志已亲读。
固定分母为当前 `attention_cv_window_range_cluster16.cpp`，不是 C4 或旧 decode。
此次不修改 vLLM 启动参数、调度器、GDN、prefix 协议或 FE0 冻结集。

`max_query_len >= 8192` 原路由进入 C16；此前仅 C4 接入的成员流水不能覆盖该路径。
新核保留 C16 `Process` 与 `IsClusterMember` 原文，包括 scheduleTile4、两遍任务
归属、16 成员精确历史域和全部错误状态。将已资格 C4 的 H32 解包、两 score/P 槽
流水、complete-K atomic 累积与下一 KV 预取推广到16成员；每成员 FP32、K128
归约、softmax、归一化和最终旋转顺序不变。两个槽按 member%2 回用，跨核 flag
数量不增加。其余 C1/window/current 路径保留原资格数学。

生成器逐函数字节验证 C4 数学方法不变、C16 任务域不变；每次文本编辑可逆恢复
两份固定母源之一。`pins.json` 固定母源，`manifest.json` 记录生成核和资源。

| 项目 | stable C16 | 新候选 |
| --- | ---: | ---: |
| workspace/core D256 | 4997120 B | 5652480 B |
| 显式 UB/AIV | 157632 B | 157760 B |
| 每成熟簇成员 | 16 | 16 |
| H32解包错误lane | 原16行交错 | 原16行交错 |

新增655360 B/core为第二 KV tile 和第二 score/P tile；20 Cube 时为12.5 MiB/rank。
这必须计入预算，不将其藏入 KV 容量收益。没有增加随历史长度增长的空间。

独立 ABI 为 `attention_cv_member_pipeline_cluster16_launch`，16指针、17个 int64、
float scale 与原 C16 一致；Torch 能力 `attention_cv_member_pipeline_cluster16_out`。
生产源/header由主线程注册；源码出现不代表已启用或验收。

验证预声明：原全形 Q672/C337/R1/GQA6/D256/S1、sink64/recent256，20 Cube、
全部2016任务，健康 current 源按生产规则为空。其成熟簇位于 qbegin336..651，
必须实测 cluster=1、members=16、skip=16。冻结原空间 oracle 为 atol/rtol=.005，
全部 partial/LSE/status/counters 要求逐位。bad336 位于第二 KV 尾，bad96/80
覆盖两侧原16行错误域。图门需要固定容量/地址后改 context337→593→337。

2026-10-15 本地结果：独立 CANN/官方 CPU-debug 编译通过；Q672/C337 stable CPU
在120秒截止超时，rc124/child-15、owned清理完成，原日志保留，未增时重跑。
补充的 Q336/recent1/C65 与 C337 小 CPU 门分别覆盖完整16成员的一/两 KV 轮；
它们不等于 Q672 CPU 完成。完整 Q672 CAModel 两臂必须使用真实 prepare 检查点，
独立 oracle、所有任务、源、60 actor 与真实 completion 才能判断局部1.15门。
每臂 CAModel 截止900秒，不扩大超时或使用部分 ticks。

D.4 对照：对应 history dequant/FIA；旧全历史恢复6499.8–6655.1ms、host725ms、
store209–215ms是禁止的历史结构。新核只保固定 KV256 双缓冲，累计工作仍须核算；
既有 C4 倍率和 D.4 毫秒均不能冒充 C16、64K 整轮或真 NPU 性能。

后续完成：小CPU C65/C337四臂全输出/状态/任务逐位与冻结oracle通过，bad80/96/336
六臂原lane错误域亦通过。Q672/C337 stable CAModel在900秒截止超时、清理完成，
没有completion数值产物，candidate未运行，未采部分ticks。这些旧设施失败保留。

后续在单B1研究的同库三臂中，使用官方LOG零mask并完整删除FD轮询quiet线程，
Q336/recent1/C65单KV机制形完成全部1008任务、20 AIC/40 AIV、原空间oracle与
partial/LSE/status/counters逐位门。stable CV task ticks3888161，pipeline3302518，
局部倍率1.177332；模拟wall587.763/512.878秒。这只计CV核，离线LSE合并检查精度，
未测设备merge/BF16copy全链。单KV接近600秒边界，不再启动C337双KV重试。
pipeline建议进入候选选择，B1额外4.1%不足以支持其新增35MiB/rank默认资源成本。
完整证据见`reports/c16_l1_reuse_20261015/completed/b1-camodel-single-kv-safe/`；
NPU、图、recent256完整形、32K/64K和native30%目标均未验收。
