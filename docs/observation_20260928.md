# 835fc22 真实负载片段：q1 调度偏载与剩余证据

依据：档案 #129/#143/#148–#150、启动文档 D.4、当前源码、用户三份复制片段。原文保存在 `reports/target_observation_20260928/`；`analysis.json`只提取完整 sample 对象，并保留片段与损伤边界。用户已完整运行一次 AISBench，后续不再要求重跑。

## 已完成与缺失

- C4 本轮真NPU逐位9/9、有效oracle6例、独立图捕获/回放通过；成熟20K CV为75.6423→55.3756ms（耗时下降26.79%）。
- 候选服务已ready且收到32并发负载。完整AISBench新总耗时与质量结果尚未在提供文本中出现，不能写成已追平8分钟或完整模型质量已验收。
- 复制文本并非完整JSON：首片段末尾截断且多了一条无phase的`duration_ms=6.9`记录；中间段首尾均截断。全程各阶段次数/权重、完整32并发分布不可从这些片段补造。

## 同一步的确定分解

rank2、`step_id=12371-11`，32请求、128 scheduled decode tokens：

| 区间 | 时间 | 解释 |
|---|---:|---|
| 整步 | 428.101ms | 同rank/step/stream事件包络 |
| 目标模型图 | 293.687ms | 与target_forward同一包络；图内OSCAR/GDN/MLP不可再拆 |
| MTP proposal | 122.770ms | 包含下列三次CV，不能重复相加 |
| 三次MTP CV | 14.769 / 48.480 / 48.666ms | draft0每请求q4；draft1/2每请求q1 |
| 其余包络外时间 | 11.644ms | 由同一步区间算得；不是全程调度总耗时 |

后两次q1 CV共97.146ms，占这一decode步22.69%；即便完全消失，该步也最多约1.29倍加速。若仅作为假设降至前次q4的14.769ms/次，则整步省约15.8%，不是设备预测，更不是AISBench收益预测。最大的未知项仍是目标模型图的293.687ms。

## 在当前源码中确认的调度偏载

`attention_tasks.cpp`按请求生成leader；原生MTP后两轮将query starts更新为每请求1token，实际32行，但保持128行模型缓冲。当前CV的全局query tile固定为`128/GQA6=21`；N128共7个tile，S3。

source0的q1 leader仅在前两个tile，三个split的有效work id为`{0,1,7,8,14,15}`。20个Cube中至多6个承担历史；若32请求均有有效历史，每split的两核分别连续处理21/11个独立请求。q4的leader起点为0,4,...,124，有效tile为前6个，每split共6个work item，合计18个核承担历史。此前把q4说成20个活跃核不准确：最后tile126–127只有follower。

因此q1历史工作从18核降到6核，与实测q1 CV反而慢约3.3倍吻合。它是可证的分配问题，但尚无逐核NPU时间把48ms全部归到它。

新独立`attention_cv_q1_out`仅把遍历任务的tile改为1，使独立q1请求分布到各核。保留全部padded task、原source/split、FP32计算、mask、状态、workspace、输出和图地址。只在显式候选且`is_draft && draft_index>0 && max_query_len==1`时使用；不改主模型图路由，不截hidden states，不修改已认可的fe0/C4内核。

## 没有优先改的项目

首16K prefill步为2655.294ms，主模型attention相位约399.198ms，MTP attention约305.211ms；后两次MTP store仍按16384行量化后才过滤负slot，各35.22ms。源码确有padding浪费，但这两次总共只占该步2.65%，不是追回约14分钟的主要方案。本轮不同时改变store数学、有效行范围或GDN。

`residual`包含未覆盖计算、通信、流等待与launch间隙，不能叫作GDN耗时。headline中的step/attention/residual是独立中位数，不能相加做全程分解。KV峰值0.61909表示61.9%；running32与waiting30是独立峰值，不能相加成同一时刻62请求。preemptions=None原因为采集器未识别Counter的`_total`后缀；本轮修正读取，旧片段仍记未知。

## 下一次只用短probe

```bash
git pull --ff-only && bash scripts/install_observe_serve.sh --variant candidate --probe-only
```

只安装/编译和执行真实NPU门，之后退出；不启动模型、不发推理HTTP请求、不运行AISBench。新q1探针采用32独立历史、真实1token/请求，分别保留N128/S3和N16384/S1的padding；对同一task表做fe0逐位、冻结oracle、独立图同地址改输入回放以及2次预热/5次交替A/B。任何精度/图/速度门失败均停止，不放宽阈值。旧decode32算子probe固定S1，本轮已对齐生产S3并打印split来源；不能把旧40ms直接乘模型层数。

这些短probe只能决定该调度候选是否有价值。目标图内部热点及完整服务8分钟目标仍须通过更小的合成诊断逐步定位，不再要求用户重跑AISBench。

## 本轮本地验证

CANN ascend910b4编译通过；官方CPU-debug5/5通过，包含与fe0逐字节partial/LSE/status、独立oracle、padding/S1/S3/空历史/错误元数据/D256。N128/S3病例的source0任务96个，旧6核、新20核；这是CPU任务归属计数。相关主机回归119 passed、28真NPU skipped、4 subtests passed。编译源码指纹与边界见`reports/q1_schedule_local_validation.json`。新q1真NPU精度、图和速度均待上述短probe验证。

## a13af95 真机短probe回传

`observe-20260928T031553.257802Z`已执行完成，原文`reports/target_q1_schedule_pass_20260928.txt`。下表均为同输入、相同source split、2次预热/5次交替Event中位数；不是全服务时长。

| q1实际形状 | fe0 | 新q1调度 | 耗时下降 | 数值/图 |
|---|---:|---:|---:|---|
| 32请求、N128、S3 | 45.112ms | 13.807ms | 69.39%（3.27倍速度） | 逐位、oracle、独立图捕获/回放通过 |
| 32请求、N16384、S1 | 137.883ms | 20.544ms | 85.10%（6.71倍速度） | 逐位、oracle通过；该形状未运行图门 |

C4成熟20K仍为76.350→55.855ms；q4生产S3的同fe0算子自比15.278/15.302ms，不是候选退化。`OBSERVE_PROBE_DONE service_started=false`为本轮预期完成，不是漏启动。q1调度收益获得设备证据，应保留；用户AISBench不重跑，端到端总收益仍不外推。

下一步只诊断剩余q4：原生实际路径是BF16 paged-cache FIA TND，而不是current-only FIA。独立合成对照需从同一逻辑Q/K/V构造两种cache，分别验证native BF16 oracle和OSCAR INT2 oracle；两者不要求bitwise。缓存构造/编码/任务准备在测速外，报告native FIA和OSCAR CV+merge的明确范围。另以复制fe0数学的独立profile算子采每核AIC/AIV、每source原始时钟；profile新增局部完成栅栏及计数会改变时序，必须与原fe0逐位对照并单列Event开销，不将它的时间替代正常速度，也不跨核相加为wall。三个已验证内核fe0/C4/q1均保持不动。
