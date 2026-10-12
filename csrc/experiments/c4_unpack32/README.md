# C4 HalfKv32 反量化原型（未启用生产）

档案：#13–22（metadata/反量化/精度）、#70–73与完整D.4（6.5秒历史恢复禁区）、#126/#144–145/#148/#151/#154–156。只使用当前生产C4及decode padded helper，未读旧失败工程。

## 现有证据与假设

当前C4确实还是16-row非padded striped helper；decode_bundle和对应request-window decode是64-row padded、SIMD metadata guard。当前C4 core3/AIV0已闭合history unpack共96次/212598 cycles，约2214.56/16行；现版decode短形core0/AIV0为20次/46527 cycles，约2326.35/64行。这是不同整核中的helper现象，支持摊薄固定调用成本，**不是同形A/B，更不能套旧#151的64.71%占比**。

本实验只选32行：去掉legacy大plane/index分配，保留四个32B guard mask；natural行stride544B；compareRows64，scale取低32bit、zero取高32bit，liveMask覆盖0–32，scratch1024B。全C4显式UB157760B（旧157632B），acc64KiB不spill、score32KiB仍独立。

## 精度与生命周期

- INT2位序、FP16 scale/zero、先FP32 Mul再Add不改，无FMA/BF16舍入或全历史恢复。
- 完整C4 source0的32行由本lane原来两段16行组成：`logical=start+(j/16)*32+lane*16+j%16`。因此非法metadata不会换到另一条AIV发布错误。
- GM Publish按两段16行写回原natural布局。Physical/DMA段数不宣称减半；Unpack和metadata调用数减半。
- source1/current以及PV消费仍为16行，保留精确窗口错误顺序。
- dequant buffer扩至32KiB，但Softmax临时LocalTensor view固定原16KiB，scratch view固定512B；不夹带V2 tiling变化。原QK/PV/Softmax/acc更新/跨核flags保留。

## 最小helper入口

`generate.py --check`校验当前源pin及可逆helper变更。默认CMake编译`liboscar_unpack32_helper.so`；`-DOSCAR_UNPACK32_CPU_DEBUG=ON`编译`oscar_unpack32_helper_cpu`。

生成全套：`python3 fixture.py --output NEW_DIR`。193例覆盖所有65536 packed word、33个live边界、dead NaN、live负零scale以及K/V scale/zero非法模式。`--check-guard`另外穷举全部65536 FP16编码及live bitmask的host语义。

CPU：`oscar_unpack32_helper_cpu FIXTURE NEW_OUTPUT`，检查旧16/new32全部输出与status逐位、独立有限值/零位模式、非有限类别、输入不变和GM边界。它不提供NPU/图证据。

CAModel：通过有界durable supervisor串行调用`replay.py --variant 16|32 --library SO --fixture FIXTURE --output NEW_OUTPUT`。可用`fixture.py --case all_words_00 --repeat 8`生成短周期数据，摊薄一次性Init；仍包含load/output，不叫纯unpack或完整C4耗时。`compare.py`要求两arm完成、同库/输入/SIM配置/死线、输出/status逐位。

## 完整C4原型

`generate_c4.py --check`校验`c4_baseline.cpp`/`c4_candidate.cpp`。当前仅源码生成与UB/数学身份/旧lane集合主机检查；尚未编译、运行或注册生产。原16行oracle与新32行所有数值/尾/非法metadata helper通过后，再进入完整C4 source0+1/图/真NPU门。

最小完整C4入口：CMake加`-DOSCAR_UNPACK32_CV=ON`；再加`-DOSCAR_UNPACK32_CPU_DEBUG=ON`得到`oscar_c4_unpack32_cpu FIXTURE NEW_OUTPUT baseline|candidate`（每臂独立120秒）。不加CPU_DEBUG则生成`liboscar_c4_unpack32.so`。`cv_fixture.py`只生成Q336/C113/R1/S1、完整1008任务，49行成熟history横跨旧16行与新32行的合并边界；`--bad-position 80`与96分别验证原AIV1/0的metadata错误归属。source1/current仍16行。

`cv_replay.py --variant baseline|candidate --fixture FIXTURE --prepared CPU_BASELINE_OUTPUT --library SO --output NEW_OUTPUT`要求真实CPU完成导出的tasks；使用原16GM+17int64+float ABI、20core，核对完整oracle/全部status lane/guard/输入不变/C4计数。`cv_compare.py`通过durable control核对两臂真实完成、同输入/库/配置、partial/LSE/status/clusterStats逐位；CPU还要求tasks逐位。无Q1344重跑或预算增加。

错误优先级：旧/新LoadPacked都先清零本lane packed UB；任何live Physical失败留下scale0，后续同lane K/V metadata guard都置error3覆盖address1。无地址/meta错误则保留先前error，后面的QK/softmax覆盖顺序仍原样。此分析限source0；source1未改。全部非Unpack scratch view保持512B，Softmax dequant view保持16KiB。

预计降低的是反量化固定调用、metadata同步及原512B自然行stride开销；完整算子收益必须另测，不能把helper改善比例直接当整核或服务收益。
