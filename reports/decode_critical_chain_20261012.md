# 当前多KV decode：完成链已闭合

档案 #129/#132–135/#140/#145/#148/#154–156 与 D.4；从属总量化优化SOP。
输入：`reports/decode_completion_20261012/completed/reports/analysis.json`。
汇总脚本：`tools/experiments/summarize_decode_critical_chain.py`；[机器表](decode_critical_chain_20261012.json)。本次只处理已有完整trace，没有新VM运行。

## 完整范围

d817当前母源；R2/q4/GQA6/D256/C1341/S1、1 Cube＋2 AIV，每history task四个full KV256加一个tail。全部source/task、partial/LSE/status位一致及独立原空间oracle通过；normal1072361、profile1093991 ticks，扰动2.017%。六个完整popped/completion gzip有真实结束、hash与marker闭合。

## 每轮完成链（ticks）

`A=前轮PvReady→本轮ScoreReady`，`B=ScoreReady→PReady`，`C=PReady→PvReady`。后续各轮严格有 `A+B+C=PV周期`，不相加不同actor，不把wrapper提交时长当完整计算。

| task/轮 | KV生产者尚未ready | A | B | C | PV周期 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0/0 首轮 | 30602 | — | 9260 | 20973 | — |
| 0/1 稳态full | 0 | 17064 | 5919 | 20970 | 43953 |
| 0/2 稳态full | 0 | 17070 | 5988 | 20975 | 44033 |
| 0/3 full后接tail | 0 | 14090 | 7425 | 20974 | 42489 |
| 0/4 tail | 0 | 14093 | 5208 | 20974 | 40275 |
| 1/0 首轮 | 34580 | — | 9630 | 20970 | — |
| 1/1 稳态full | 0 | 17075 | 5913 | 20974 | 43962 |
| 1/2 稳态full | 0 | 17083 | 5910 | 20969 | 43962 |
| 1/3 full后接tail | 0 | 14088 | 7416 | 20972 | 42476 |
| 1/4 tail | 0 | 14094 | 5162 | 20970 | 40226 |

两个中间full轮的闭合周期中位43962；QK/PV交付链合计约86.54%，softmax＋P交接约13.46%。**这是依赖交付归属，不是Cube纯忙率或HBM roofline。**

中间full轮AIV0 Softmax完整scope5352–5431 ticks，AIV1为3601–3660；准确落在对应轮ScoreReady/PReady之间。q4已是16/8行分担，进一步12/12最多只动这约13.5%周期的一部分，且V2补齐增加。[lane审查](decode_lane_balance_audit_20261012.md)

首KV暴露的30.6K/34.6K还包括query装载与初始化，不能全部叫dequant。之后四轮KVReady均无生产者暴露等待，说明现有next-KV预取在这份完整工作集已隐藏供给。继续只把解包helper提速，不能直接削减这份稳态43.96K主周期；它可能改变争用，需另测才能领取。

## PV交付的具体瓶颈线索

task0/轮1：PReady112679；PV开始112796；8个B的MTE2 `MOV_OUT_TO_L1_MULTI_ND2NZ` 完成于116248、118933、121195、123462、125730、127990、130254、132520。后六块几乎每2260–2268 ticks推进。每个逻辑B块是FP32 K128×N64，即32KiB。

最后B完成132520→最后MMAD完成133167→最后FIX133644→PvReady133649。此链显示B装载推进直接约束结果交付。八个MMAD各自pop→complete均277 ticks，不能把约21K整个交付当纯乘法；也不能把这些延迟求和冒充硬件busy，pipeline等待和服务仍须区分。

QK/PV方法结束可早于其实际交付；当前下一QK会与上一PV尾部重叠，因此应使用上述ready链，不能把两者delivery span直接相加。L2在当前CAModel配置关闭，该装载现象不证明真NPU HBM带宽数值。

## 下一实现主线：AIC侧整块/大块ND→L1装载

**保持AIV现有FP32 GM publisher和原FP32/K128数学顺序，只改变AIC将这些块搬到L1的形状/驻留策略。** 目标是消除目前8个跨行32KiB ND2NZ载入的串行暴露；它不是重新尝试失败的AIV NZ重排，也不改变scale/zero或矩阵归约精度。

官方已存SDK `matmul_config_utils.h::IsBFullLoad` 要求 `stepN*baseN>=N` 且 `stepKb*baseK>=K`；`data_copy_wrapper_nd.h::CopyND2NZ`由真实tile height/width生成Nd2NzParams；`scheduler_norm.h`通过CopyCubeInB加载。必须继续核合法GetMatmulApiTiling配置是否能让完整256×256 B一次/少次载入，再由原K128/MMAD消费，不能手捏tiling字段或仅看源名称宣称全载入。

FP32 B整块256KiB，真实M24的A为24KiB，二者可在512KiB L1内共存；这是单B账，不能额外塞两个完整B。代价包括等全B到达后才能开第一MMAD、L1/L0双缓冲变化及FIX尾部；应在同一完整算子比对，不以减少DMA条数代替速度。

### 避免重做历史失败

- `decode_static_matmul`仅移除动态shape控制，仍basic64/64/128，局部1.6%已排除；不是当前整B载入方案。
- `decode_mmad_barrier`旧失败C为实际M32/N128/**K64**，去8个特定小M barrier却整体变慢约34.4%；改变了K累加分组，不能拿它直接否定或放行K128方案。
- `fp32_kmajor_matmul`保持N64/K128，但输入在host已完整NZ，M24 QK或M126 PV；K-major累积所有输出后统一FIX，PV尾部增加8112 ticks，整体退化。新方案不重排为K-major，不把FIX推到全部计算后，且不使用免费host预pack。
- 若研究basicM32/N128/K128：实际`SetSingleShape(m)`仍是M24或M6，不会因basicM32自动免barrier；`floor(M/16)*(N/16)<10`在M24/N128仍成立。若显式pad实际M32，q4多33%MAC、q1多5.33倍，且每轮新增P行必须初始化，不能假借旧Q padding。N128/K128使L0B单块64KiB，丢失双缓冲可能抵消四次MMAD/四次B装载收益。

当前优先查整B/多块L1装载的合法SDK路径；N128/K128是待成本核对的次选。尚未实施内核，尚未证明完整加速，更没有64K/K32或整轮2–3倍结论。

### 已到本地的官方tiling接口核查

来源绑定：`artifacts/cann_sdk_matmul_tiling_20261012/provenance.json`。

- 公共`GetMatmulApiTiling(mmCFG,l1Size)`先调用`GetL1Factor`；`GetL1StatusBothFullLoad`在容量满足时选择A/B各单buffer，因此完整B的256KiB并非必然需要512KiB。
- 但**大L1缓存不等于大DMA指令**：当前Norm scheduler的`CopyCubeInB::LoadData`仍按当前K/N子块调用；实际trace的8条32KiB ND2NZ是决定性事实。新方向需核`doMultiDataLoad/GetMDLConfig`而不能只声称step条件满足即可。
- 带`SingleShape/L1Shape/BaseShape`模板的公共重载固定`depthA1=stepM*stepKa*2`、`depthB1=stepN*stepKb*2`。将该重载的B L1Shape直接设256×256，会仅B就占512KiB，加A后超限，不能采用。
- 若自动MDL完整B单buffer不成立，可审查合法的“B K128×N256”半K全N载入：base64/64/128、A L1Shape64×K256，双bufferA128KiB＋B256KiB=384KiB；B从8个32KiB改为2个128KiB。它保持K128计算，区别于旧K64失败；但是否减少实际暴露的ND2NZ成本、是否能保持首块流水，仍需完整配对。
- `MatmulConfig`原生注释明确`doMultiDataLoad`启用另一模板，指定native GMM中有`GetMDLConfig`先例；其量化dtype不能替代当前FP32及动态transpose资格。还需读MDL loader/scheduler和配置默认值才可生成最小候选。
