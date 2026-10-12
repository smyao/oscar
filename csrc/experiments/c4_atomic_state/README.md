# C4 FP32 atomic state：最小可否证原型

**research，未注册生产。** root 已完成 revision1 CPU/device 编译（rc0）、五组官方 CPU-debug 和五组 CAModel 逐位验证，包括 anti-K128、random、ties、subnormal/±0、nonfinite payload；原始证据在 VM `reports/c4_atomic_state_cpu_20261009` 与 `reports/c4_atomic_state_camodel_20261009`。paired 总计74698 ticks不当作单路径性能。**目标NPU/图/完整 C4 仍未验收；revision2 尚未编译或运行。** 本目录只判断完整 PV 写回后的 FP32 atomic-add 能否逐位替代当前 Vector Add；不改变 HF32、dtype、C4 预设或验收阈值。

档案 #126/#140/#145/#154 与完整启动文档 D.4：真实异步 UB/GM 所有权必须明确；CPU 顺序模拟不等于真 NPU 等价；D.4 的全历史 dequant 6.5 秒不在本原型中。此处对应 fused FIA 的 PV/acc 更新，不生成历史尺寸缓冲，也不宣称速度收益。

## 固定算子与顺序

同一 kernel、1 Cube + 2 AIV，M126、物理 M128、N256、K256。
静态 MatmulConfig 从当前 C4 字节一致复制：basicM64/basicN64/basicK128、singleCoreM128/N256/K256、GetNormalConfig、HF32-off。`fixture.py --check-source`检查当前 C4/common 固定 SHA 和完整配置片段；CMake 配置阶段也执行此检查。

两个路径使用同一组 P/V，按以下顺序执行：

1. 两 AIV 从 alpha 输入按原始字节初始化 reference/atomic GM，保留 alpha 到各自 64-row UB。
2. Cube 普通完整 K256 PV 写入独立 PV GM。
3. AIV 使用原 C4 的 `DataCopy → MTE2_V → Add → PIPE_V → V_MTE2`，产生 reference；写回后通知 Cube。
4. 同一个 Matmul 对同一 P/V，以 `IterateAll<true>(atomic,1)` 写入预置 alpha 的 atomic GM。
5. 两 AIV 完成通知与独立状态写回；Cube 确认两者完成后发布自己的状态。

阶段事件使用不同的 8/9/10/11/12 ID，各只 set/wait 一次；两个 AIV 保留原 MIX_AIC_1_2 模式配对，不用同 ID 连续双 set。初始化 DMA 使用已有原生 `MTE2_MTE3` 交接，状态按三个独立 64 B 区域保存。

SDK 源码审查支持当前 normal scheduler 在完整 KLoop 后 GetTensorC，再启用 F32 atomic Fixpipe；本原型用 anti-K128 病例进一步否证分 K 写回。SDK 原生 A2 有 F32 atomic Fixpipe 使用先例，但没有证明 atomic 舍入/denorm 与 Vector Add 逐位等价。

## 病例及判据

| 病例 | 目的 |
|---|---|
| anti_k128 | P全1；V前128行+32768、后128行−32768；alpha=0.25。完整PV=0，reference必须0.25；若K128 partial分别atomic，结果无法保持0.25 |
| random | 固定种子的有限 FP32 输入，成对完整输出逐位比较 |
| ties | ±1及相邻FP32 alpha，加±半ULP/三半ULP，发现舍入方式差异 |
| subnormal_zero | 最小subnormal、最大subnormal、最小normal和±0组合；不预设FTZ行为，用实际Vector路径作对照 |
| nonfinite | 保留qNaN/sNaN payload与±Inf；逐位差、类别差、NaN payload差分别计数 |

物理 padding 行126/127使用±0 alpha，单列两路 padding 位差。所有病例有64 B首尾 guard。CPU driver在任何比较之前保存全部输入回读、PV、reference、atomic、三个actor状态，以及各自包含guard的完整字节文件。

**任何位差、输入变更、guard损坏、actor未完成、anti-K128数学见证失败都返回 rc2。** nonfinite 即使类别相同但payload不同，也照实记录位差并返回rc2，不默默忽略。该结果是否涉及完整 C4 的非法输入契约需后续判断，不能由这个独立原型宣布完整 C4 精度通过。

## 已准备的入口

Host仅生成输入，不运行算子：

```bash
python3 csrc/experiments/c4_atomic_state/fixture.py --check-source
python3 csrc/experiments/c4_atomic_state/fixture.py --output NEW_FIXTURE_DIRECTORY
```

以下由 root 在排他 VM slot 中执行；本目录不会自行启动 VM 或重试：

```bash
cmake -S csrc/experiments/c4_atomic_state -B NEW_CPU_BUILD \
  -DSOC_VERSION=ascend910b4 -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \
  -DOSCAR_C4_ATOMIC_CPU_DEBUG=ON
cmake --build NEW_CPU_BUILD -j1
NEW_CPU_BUILD/oscar_c4_atomic_state_cpu FIXTURE_CASE NEW_OUTPUT_CASE

cmake -S csrc/experiments/c4_atomic_state -B NEW_CANN_BUILD \
  -DSOC_VERSION=ascend910b4 -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
cmake --build NEW_CANN_BUILD -j1
```

独立库 `liboscar_c4_atomic_state.so`，设备 symbol `oscar_c4_atomic_state_kernel`。
ABI为7个GM指针，顺序 `P,V,alpha,pv,reference,atomic,status`，无shape属性；launch blocks固定1。
P/alpha/PV/reference/atomic各128×256×4 B，V为256×256×4 B，status为3×64 B。
alpha是已乘过在线softmax系数的输入状态；本原型不额外执行乘法。

后续必须给单次运行预定有限墙钟截止，保存真实退出码和输出。CPU/CANN通过仅建立对应范围证据，不能代替目标NPU的atomic舍入、图、完整C4错误状态或性能门。整次paired-kernel事件时间也不能被拆成“atomic比普通Fixpipe快”的结论。

## revision1 独立 CAModel replay（已完成，旧证据保留）

`replay.py` 由 root 的既有 durable supervisor 启动，并在启动前明确每次不超过900秒的截止。SDK preload和SIM配置必须先通过child环境传入，runner会核对同一配置拓扑/cache、固定fixture/source身份；只执行真实7指针kernel，不调用CPU参考代替算子。

```bash
python3 csrc/experiments/c4_atomic_state/replay.py \
  --fixture FIXTURE_CASE --output NEW_OUTPUT_CASE \
  --library ABSOLUTE_CANN_LIBRARY --simulator-config ABSOLUTE_SIM_CONFIG
```

先执行 anti_k128、subnormal_zero、ties；这些通过后再执行 random、nonfinite。每例必须是新目录，不自动重试。全部输出和输入回读及guards先落盘；然后逐位比较、单列finite/NaN/±Inf及payload差异、检查anti数学见证。任何失败均保留真实失败报告和非零退出，SDK原始错误码单独保存。

不采集完整phase trace，dump输出仅定向到本轮/dev/null。仍要求SDK的task=1/AIC=1/AIV=2完整正时长记录、相同task/stream身份及正确core集合。task ticks只表示一次含两条路径的完整诊断成本，不提供atomic相对普通Fixpipe的独立速度比。CPU逐位结果不升级为CAModel或目标NPU通过。

## revision2：AtomicDone 后立即 MTE2 回读（CPU random与device编译已完成）

revision1 在 AIV 收到 AtomicDone 后仅写状态；host 同步后下载的正确结果不能证明 next-KV 的立即 MTE2 Restore 可见性。revision2 保留旧 `kernel.cpp/probe.cpp/replay.py` 和7指针 ABI，新增独立生成的 `kernel_readback.cpp/probe_readback.cpp/replay_readback.py`。

新 symbol 是 `oscar_c4_atomic_state_readback_kernel`。**第8个GM指针为 atomic_readback，排在原status之后**，大小128×256×4 B。两个AIV执行：

```text
WaitAtomicDone(11)
DataCopy(atomic GM → 原accBuf，每lane64行)
MTE2_MTE3
DataCopy(accBuf → atomic_readback GM)
写status → VectorsDone(12, PIPE_MTE3)
```

原 reference 写回在 ReferenceDone→Cube atomic→AtomicDone 的依赖链之前已完成，所以不会仍占用被回读复用的 accBuf。没有改变数学、shape、flag ID 或 Matmul；原 flag12 覆盖新增回读的最终写回完成。

所有128行的 readback 都必须逐位等于 atomic 和 reference，分别报告 active/padding 对两者的差异。任一不等返回rc2。新增 readback及其guard同样先落盘；旧 NaN类别、输入、guard、状态、SDK profile门全部保留。

```bash
python3 csrc/experiments/c4_atomic_state/generate_readback_v2.py --check
python3 csrc/experiments/c4_atomic_state/fixture_readback.py --output NEW_V2_FIXTURES
```

v2输入字节与v1相同，但manifest记录独立revision/kernel SHA和v1 baseline SHA，不能用v1运行器误接v2。构建时使用**新目录**并追加 `-DOSCAR_C4_ATOMIC_READBACK_V2=ON`；CPU目标为 `oscar_c4_atomic_state_readback_cpu`，库为 `liboscar_c4_atomic_state_readback.so`。CPU选项 `-DOSCAR_C4_ATOMIC_CPU_DEBUG=ON` 其余不变。

CAModel入口为 `replay_readback.py`，四个路径参数保持不变。2026-10-10 root回传v2 random官方CPU readback逐位通过、device库编译通过；尚未跑v2 CAModel。后续改用revision3一次同时检查即时回读和atomic模式恢复，不重复v2 CAModel或全部五例。目标NPU与图仍未通过。

## revision3：normal → atomic → normal 模式恢复（独立准备）

保留所有v1/v2文件和证据。`generate_recovery_v3.py`固定v2 kernel SHA `bcd752fbf0448fa61c18aab9e1bd620e5de5fe202fbd81a26f6834eea590ce93`与driver SHA；仅生成`kernel_recovery.cpp/probe_recovery.cpp`。fixture/replay为独立`fixture_recovery.py/replay_recovery.py`。新入口`oscar_c4_atomic_state_recovery_kernel`、库`liboscar_c4_atomic_state_recovery.so`。

ABI为10个GM指针：原8个后追加`normal_before`、`normal_after`，各128×256×4 B。前者全字节初始0x85，后者每FP32初始1.0。保持同一个mm对象、HF32-off和原完整K256配置：

```text
WaitInitialized
normal P@V.T → normal_before（IterateAll省略atomic参数）
原 normal P@V → PV → 原Vector Add参考
原 atomic P@V → atomic state
原AtomicDone → 两AIV立即MTE2回读 → VectorsDone
WaitVectorsDone
normal P@V.T → normal_after（同mm，IterateAll再次省略atomic参数）
写AIC完成状态
```

原readback、reference/atomic全128行逐位、输入只读、guard、actor状态、profile门全部保留。新增：live126行before/after逐位相同且有限；两者padding各保留自己的初值。after预填1.0，能够使残留atomic-add产生可见差异。仅计划random一例，并由fixture记录PV非零、普通转置结果有限且加1可分辨的见证，不用anti-PV0代替。

### 已亲核的SDK恢复位置

只读当前本机Lima CANN9.1，未改SDK、未编译/模拟：

- `asc/impl/adv_api/detail/matmul/matmul_impl.h:118–136`的v220分支，`IterateAll(gm,enAtomic=0)`每次将本次参数传到`GetTensorC`。
- `matmul_impl_base.h:692–696`将该参数传入Scheduler。
- `scheduler/base/scheduler_base.h:459–507`是A2所走的else分支：`SetAtomic(enAtomic)`→`CopyCubeOut`→`ClearAtomic(enAtomic)`；不是旧1001/2002分支的误用。
- 同文件`:512–515`在`enAtomic!=0`时调用`SetAtomicNone()`。因此普通默认0依赖前次atomic结束正确恢复；revision3实际验证这一依赖，没有用理论替代见证。

### root运行入口

```bash
python3 csrc/experiments/c4_atomic_state/generate_recovery_v3.py --check
python3 csrc/experiments/c4_atomic_state/fixture_recovery.py --output NEW_V3_FIXTURES
```

使用新构建目录，CMake追加`-DOSCAR_C4_ATOMIC_RECOVERY_V3=ON`；与V2选项互斥。CPU继续追加`-DOSCAR_C4_ATOMIC_CPU_DEBUG=ON`，目标`oscar_c4_atomic_state_recovery_cpu`；device库如上。仅运行`NEW_V3_FIXTURES/random`。构建目录写`atomic_source_binding.json`；replay要求其与当前fixture source严格一致，不接收来源不匹配的库。

后续CAModel用`replay_recovery.py`，沿原四个路径参数及root排他、预定截止/真实退出码监管。全部字节先保存，任何原门或模式恢复门失败仍返回非零，不放宽阈值。v3当前仅host源码/fixture准备；不声明CPU/CANN/CAModel、NPU或图通过，更不从四次Matmul的总周期推导单路径加速。
