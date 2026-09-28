# 从19分钟继续压缩：三条CV路径的精确解包候选

依据：启动文档D.4、档案#126/#129/#140–#151、q4真机分段证据及用户本轮最新19分钟反馈。用户确认的fe0精度基准保持有效；本轮没有要求或执行新的AISBench。

## 目标预算与已确认的代码问题

19→8分钟约需减少660秒、57.9%总耗时，整体加速2.375倍。已测q4原生FIA1.4509ms，OSCAR CV+merge15.4011ms；critical history AIV解包占其task span64.71%，AIC KV等待81.25%。解包是当前有真实证据的优先项，不能把这个单actor比例当作整轮占比。

当前三个已验证CV算子均用相同INT2展开：8个uint16位平面在D256下间隔1024B，然后Gather重排并Cast；每16行的scale/zero逐个half标量读取、转换、检查。新的实现只改变这些步骤的数据组织和精确转换，保留原FP32 Mul→Add、QK/PV、Softmax、mask前检查、KV遍历次序、slot/页表、三源/split和状态语义。

## 实现

1. 八个位平面间距从`kWords`改为`kWords+16`个uint16；D256为1024→1056B。索引同时改写，32B对齐和所有2bit码的位置对应不变。减少bank冲突是待实测假设，不据地址算式宣称提速。
2. 一次Gather取得16行scale和zero，放入独立64B half scratch，一次Cast成32个FP32数。每个live行的有限性/正scale检查照旧，随后仍用原Brcb、Mul、Add序列。没有FMA/HF32或浮点归约改序。
3. 同一个共享helper供三个独立新算子使用：`attention_cv_fast_out`、`attention_cv_fast_q1_out`、`attention_cv_fast_cluster4_out`。分别保留原q4、q1均衡调度和C4历史复用。
4. fe0/C4/q1/profile四个已有kernel源文件逐字节不变，始终可用于独立同输入对照；生产路由没有原生BF16全历史替代。

显式UB增加448B，D256为157632B，小于A2可用188416B；GM工作区、缓存和图地址均不改变。整数码映射、半精度转换以及生产图分别验证，不把CPU结果代替NPU。

## 已完成的验证

- CANN ascend910b4编译通过。
- 官方CPU-debug真实执行全部65536种uint16 packed word × 8个码值，524288个FP32结果全部精确。
- 21项整CV用例通过：D64/128/256、合法FP16 subnormal/负零、非法live值、dead tail、q1 padding/S1/S3、C4成熟/非成熟/poison与共享tile特殊值；旧→新partial/LSE/status和C4统计逐位，独立oracle通过。
- 证据与编译源码SHA：`reports/fast_unpack_local_validation.json`。这些均为本地CANN/官方CPU证据，目标NPU速度与half特殊位型尚待测。
- 关键主机回归130 passed、28真NPU skipped、4 subtests passed，覆盖三路径实际分流、安装/设备配置、NPU门失败阻断、ABI及无回退约束。跳过项不记为设备通过。

## 已排除的捷径与收益边界

A2 SDK的AIV→TSCM/L1写入实际通过UB→GM→AIC L1的软件通道，不会直接消除GM往返。本轮没有把它当作提速方案。q4已有18/20核参与历史工作，也不复制q1的调度结论。简单双buffer只能利用当前约8%的Cube计算窗口，还会增加工作区与同步复杂度，未据此改生产。

即使64.71%这一部分加速4倍，该actor也最多约1.94倍，不能单独承诺整体2.375倍。达到8分钟的剩余预算仍取决于新解包实测、图内FULL路径与其他模型阶段，不能用单算子×层数伪造整图实测。

## 一键真机门

```bash
git pull --ff-only && bash scripts/install_observe_serve.sh --variant candidate --probe-only
```

原数值/C4/q1门保留，再执行新fast门：18个数值病例、5个2次预热/5次交替A/B的速度病例、q4/q1 N128同地址修改输入图回放。原q4/q1/C4与新算子分别实际调用；全部partial/LSE/status/合并输出逐位，另过独立冻结oracle；非法metadata以fe0最终状态为准，允许中间error3被后续error2覆盖，但不允许新旧最终状态不同。任何失败都停止，不放宽阈值。

终端只新增5行`PERF_FAST_UNPACK`与一行最终结果。输出为`fast-unpack-report.json`，与阶段状态`fast-unpack.json`分离。旧q4原生/profile诊断已经提供了本轮所需数据，默认不重跑；必要时才显式加`--diagnose-q4`。

`--probe-only`不启动模型、不发推理请求、不跑AISBench。新门通过后，已有直启命令仍是`install_serve.sh --variant candidate`，本轮有效配置包含`experimental_fast_unpack=true`，终端显示`FAST_UNPACK=on`。后四卡仍通过`--rear-cards`显式选择，默认卡和端口不变。基准模式及无参数的原配置仍不启用实验路径。

## 0677bcb 真机复验

用户回传`observe-20260928T064725.021030Z`，签名前缀`a1304fb6f1dd`；所有fast精度、独立图及性能门passed，正常`OBSERVE_PROBE_DONE service_started=false`结束。原文`reports/target_fast_unpack_pass_20260928.txt`。

| 场景 | 原路径ms | fast ms | 耗时下降 |
|---|---:|---:|---:|
| q4 decode32 N128/S3 | 15.298 | 11.282 | 26.25% |
| q1 decode32 N128/S3 | 13.855 | 9.842 | 28.96% |
| q1 mixed32 N16384/S1 | 20.468 | 15.766 | 22.97% |
| C4 mature20K | 55.010 | 48.772 | 11.34% |
| mixed q1/q4/long | 19.301 | 15.853 | 17.87% |

这些是同输入算子测量；不推出19分钟已变成某个新总时长。q4/q1指定图门通过，C4行graph=not_run，不伪称所有路径都完成整模型图验收。直启命令在0677bcb已经设置两个候选开关；随后将两入口的开关配置集中到`tools/serving_variants.py`，避免未来只更新probe。不会为这次配置整理要求重跑真机。
