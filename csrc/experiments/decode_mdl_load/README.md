# 当前 decode 的 AIC MDL 载入研究

档案 G3–G13/#21/#81–85/#126/#129/#140/#145/#148/#154–156、startup D.4全文已亲读。
独立research；无生产注册、native/GDN修改、AIV NZ publisher或精度放宽。

## 2026-10-12 已完成结果

当前CANN编译通过，资源静态断言已过；C1341/R2/q4/S1完整官方CPU-debug两臂partial/LSE/status逐位、独立oracle、guards及输入不变通过。首次CPU编译仅因stage漏传`oscar_launch.h`失败，原失败由root归档，补齐同工程头文件后通过；不是本算子数值失败。

完整同库CAModel **1079788→931042 ticks，1.1597629倍**（时间减少13.78%）。两臂全部输出位一致、原空间oracle、所有source/task、3 actors、6个完整gz及真实rc/清理门通过。[root执行证据](../../../reports/decode_mdl_load_20261012/completed/reports/)、[载入与依赖提取](../../../reports/decode_mdl_load_20261012/load-analysis.json)

实际B ND2NZ条数232→34；完整34次Matmul每次只装入一次B。history每QK/PV由8×32KiB变1×256KiB，GM逻辑载荷没有减少；每方法F32F32 MMAD参数XT序列与旧版完全相同。稳态full轮PV周期中位43831.5→33165.5，主要来自PReady→PVReady的20901.5→12307.5，Softmax交接5859.5→5866基本不变。证据支持减少当前AIC分块搬运这一原因，不把交付延迟称作纯忙率。

按用户最新指定的局部净收益≥1.15门，此完整q4病例具备进入候选组合评估的性能证据；q1、其他形状、真实NPU、graph和整服务仍须分别记录，不能从此宣布整轮目标达成。

## 问题及唯一差分

当前完整多KV证据：稳态full轮43962 ticks中约86.54%属于QK/PV交付链；首KV后KVReady生产者等待为0。PV的8条32KiB ND2NZ按约2264ticks推进，最后B完成直接先于末MMAD/FIX。[报告](../../../reports/decode_critical_chain_20261012.md)

母源固定`csrc/kernels/attention_cv_decode_bundle.cpp` SHA `d8171f0403cf176087e1c76c64e454e6fb1e74fd79252aab3b59e848cca19401`。
候选调用公共`GetMDLConfig(false,false,0,false,false,false,true)`并恢复原`batchMode`，最终与`GetNormalConfig()`只有`doNorm/doMultiDataLoad`两字段区别。没有加入M/N预取、AIV格式转换、HF32、近似仿射计算或新跨核flags。

保持basic64/64/128、真实M6/M24、原动态`SetOrgShape/SetSingleShape`、QK/PV/Rv顺序、全部source、任务/输出/status、K128累加顺序、原workspace。生成器完整逆向恢复母源并逐函数核AIV/math/error/flags哈希。

## 官方实现已核实

来源绑定在`artifacts/cann_sdk_matmul_tiling_20261012/{provenance,mdl-provenance,mdl-implementation-provenance}.json`；参考树只读，代码只使用公共API。

1. `copy_cube_in_utils.h::GetCopyCubeInType/IsFullLoad`：GM的MDL配置进入MDL loader；静态step满足A/B全载入条件。
2. `scheduler_mdl_fullload.h::ScheduleOnce`：首迭代`CopyCubeInB.LoadData(0,0,GetSingleCoreK(),NLoop.GetTileShape())`，不是Norm的baseK/baseN逐块调用。
3. `copy_cube_in_mdl_fullload.h`：`SetInput`重置`isFirstIter`；第一次CopyTileToCube后EnQue/DeQue，后续GetBuffer，不把上一Matmul的B混作本次B。
4. `n_loop_mdl_base.h::SetSingleShapeFromTiling`保留动态tail：tile宽为`baseN*stepN`，实际末tile用`singleShape % mainTileShape`；source1/2的实际TailColumns仍限制消费。
5. `scheduler_mdl_common.h::ReduceKOneIter`：L1整K不等于一次K256乘法；内部KLoop逐baseK递增，调用ComputeKDB，保留原K128数值顺序。SDK内部实现差异仍必须通过逐位与原空间oracle验证。

## 资源与需要实证的事项

两臂均由官方`GetMatmulApiTiling`生成，静态断言：

| 项 | 预期值 |
| --- | ---: |
| stepM/N/Ka/Kb | 2 / 4 / 2 / 2 |
| depthA1/B1 | 4 / 8 |
| A L1 / B L1 | 128KiB / 256KiB |
| 量化向量API预留 | 2KiB（维持原开关） |
| shareL1Size | 395264 B，小于512KiB |
| L0A/B | 各两份32KiB |
| L0C | 16KiB |
| 原D256 GM workspace/core | 1146880 B |
| 原D256显式UB/AIV | 158656 B |

不直接填写tiling结构。显式L1Shape的另一公共重载强制双buffer，完整B会单项512KiB，因此未使用该重载。

**不能凭此宣称提速。** 全B先载入增加首个MMAD启动延迟；MDL可改变L0装载、内部flags、FIX overlap。必须查看完整指令里B载入形状/次数是否确实变大、原K128/FP32/实际M是否保持，再看同库完整算子周期。此前M32/N128/K64失败、K-major末尾集中FIX回退均保留，不复用其倍率。

## 执行接缝（由root独占VM）

生成检查：

```bash
python3 csrc/experiments/decode_mdl_load/generate.py --check
```

新建独立CMake构建目录，SOC=ascend910b4、ASCEND_HOME_PATH指当前CANN；设备目标`oscar_decode_mdl_load`生成`liboscar_decode_mdl_load.so`。CPU选项`-DOSCAR_DECODE_MDL_CPU_DEBUG=ON`生成`oscar_decode_mdl_cpu`，保留共享probe的独立oracle/bytes/status/guards/input检查；prepare-only不当作CPU数值通过。

复用已完整准备的`q4_r2_n8_c1341_s1` fixture/export，不另造数据集。每臂由`tools/experiments/run_camodel_arm.py`独占锁和600秒deadline包装：

```text
python3 csrc/experiments/decode_mdl_load/run.py run
  --root STAGE --arm baseline|candidate
  --fixture-root FIXTURE_ROOT --export-root EXPORT_ROOT
  --library DEVICE_LIBRARY --simulator-config-dir LOGGING_CONFIG
  --output NEW_ARM_DIR --cpu-reference required|optional
```

CPU reference默认required；若root明确使用既有prepare-only状态，optional会原样写入计划及报告，不能称CPU full pass。两臂均完整保存1 Cube/2 AIV共6个popped/completion gzip；每文件64MiB有界上限、RLIMIT及原所有数学/guard/actor门，不截断冒充完整。

```text
python3 csrc/experiments/decode_mdl_load/run.py compare
  --baseline BASELINE_DIR --candidate CANDIDATE_DIR
  --baseline-control BASELINE_CONTROL --candidate-control CANDIDATE_CONTROL
  --output NEW_COMPARISON_JSON
```

compare核真实rc/cleanup/deadline/owned输出、同库/输入/配置、全partial/LSE/status位、独立oracle及完整trace hash。加载变大但完整周期回退仍不合格；小收益不自动入组合。当前完成范围见开头，不将本地通过视为NPU、图或整服务证据。

无marker指令提取入口为`extract_trace.py`，使用相同两臂/control并加`--tasks`指定实际prepare的tasks.bin。工具重核全六gz、同库输入和输出位一致，再按CvTaskSchedule、GM双KV对象及原ready flag序列对应每次矩阵；原始XM/XT保留，不猜ISA位域、不以pop→complete求busy。
