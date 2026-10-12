# Decode FP32 MMAD barrier：独立三臂否证

仅研究，未注册生产、未改变两个服务入口。档案G3–G13/#21/#81–85/#126/#140/#145/#154与启动文档完整D.4。源码/输入已准备；CANN、官方CPU、CAModel、NPU、graph均未在本目录执行。

## 问题与边界

SDK9.1 `asc/impl/adv_api/detail/matmul/stage/compute/mmad_compute.h:79–85` 在 `(actualM/16)*(actualN/16)<10` 时增加硬件要求的`PIPE_M`。本原型不改SDK或删除barrier；测试另一合法形状能否由SDK自行避开该分支。`SetHF32Mode(false)`始终保留。

| arm | basic M/N/K | singleCore M/N/K | 实际M | D256名义MMAD块数 |
|---|---|---|---|---|
| A | 64/64/128 | 128/256/256 | 18或24 | 4N×2K=8 |
| B | 32/64/128 | 32/256/256 | 18或24 | 4N×2K=8 |
| C | 32/128/64 | 32/256/256 | 32（多余输入行为+0） | 2N×4K=8 |

A复制当前decode配置。B只缩M容量，仍保留K128；C改变累加分段，**不能先验认定逐位等价**。M24→32增加1/3逻辑乘加，真实物理成本未知。旧static1.01586×和wide5120.82794×不是这一配置的测量。

每个kernel只有一个TPipe/Matmul；三臂独立launch，不在同一TPipe申请两份L0。GM均为A[32,256]、B[256,256]、C[32,256]，FP32，grid1、纯AIC（原生`batch_matmul_transpose_kernel.cpp:661`先例）。QK/Rv用B转置，PV不转置；保持当前Matmul输入类型及组织stride。

`status.bin`16个int32保存：完成标记、arm、liveM、computeM、layout、basicM/N/K、singleCoreM、SDK shareL1Size、shareL0CSize、depthA1、depthB1、dbL0A、dbL0B、HF32=0。它记录SDK实际编译tiling，不用手算tile容量冒充整个实例分配。

## 输入与第一道否决门

`fixture.py`固定8例：M18/M24各正常QK/PV/Rv，以及两例K64/128/192边界抵消。正常链使用BF16输入→独立PR Hadamard旋转→原INT2量化器及FP16 metadata重建；保存codes、packed、metadata和中间P/PV。它是合成输入，不是用户数据集。全部A尾部为+0。

CPU driver逐臂保存输入回读、输出、status及前后guard。A/B未计算尾行必须保留poison，C补零尾行必须数值为0。**live输出只要相对A出现一个bit不同，即打印真实差异、保存结果并返回2；不以5e-3把这个快速否决改为通过。** 独立FP32 oracle仍按冻结5e-3检查；oracle失败会保留并使最终失败，但允许完成第一道old/new比较。CPU通过不能替代设备舍入、完整attention错误域或图验证。

## root最小命令

Host生成（只生成，不运行设备）：

```bash
.venv/bin/python csrc/experiments/decode_mmad_barrier/fixture.py --output artifacts/decode_mmad_barrier_20261010
```

仅由root在已授权CANN VM排他执行，沿用其监管、日志与清理，单次CPU/后续CAModel至多180秒，不自动重试：

```bash
cmake -S csrc/experiments/decode_mmad_barrier -B NEW_BUILD/cpu -DSOC_VERSION=ascend910b4 -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DOSCAR_MMAD_CPU_DEBUG=ON
cmake --build NEW_BUILD/cpu -j1
NEW_BUILD/cpu/oscar_decode_mmad_cpu FIXTURES/normal_qk_m24 NEW_RESULT

cmake -S csrc/experiments/decode_mmad_barrier -B NEW_BUILD/device -DSOC_VERSION=ascend910b4 -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DOSCAR_MMAD_MARKERS=ON
cmake --build NEW_BUILD/device -j1
```

建议先normal_qk_m24、normal_pv_m24和cancel_qk_m24；遇bit差立即报告。M18和Rv由首批结果决定，不自动启动8例长矩阵。设备库为`liboscar_decode_mmad_barrier.so`；入口`oscar_decode_mmad_{a,b,c}_kernel`，ABI为4个GM指针A/B/C/status和2个int64 liveM/layout。CPU/CANN构建目录均写`source_binding.json`，需要与fixture源指纹核对。

## 真实指令采集计划

后续CAModel每次只运行一个入口，保存4个guarded GM回读及SDK task/core/profile。仅1个AIC，完整保留该actor的completion与popped记录后gzip；仍由root指定既有有界trace cap，触顶或marker不闭合即无指令占比结论，不修改SDK/cache model/global trace policy。

可选MOV标记值=`0x4d4d426100000000 | (arm<<8) | end`，begin/end分别0/1。两标记包住shape设置、IterateAll和End，没有额外完成同步；标记区间不可冒充裸Cube耗时。核对真实MMAD类型F32F32、M/N/K/初始化累加序列、实际PIPE_M及其完成cycle，不只看文本出现次数。源码预计A/B的8次小M MMAD命中barrier，C的8次M32/N128 MMAD不命中该特定分支；**必须由实际trace验证**，其他SDK barrier另计。

插桩库用于诊断；若进入性能比较，normal无marker库与marker库需分别标识，不能混作同一计时分母。任何primitive倍数不外推完整decode，更不外推整体相对native2–3倍。

## 后续集成未解决项

默认两个普通Matmul在同TPipe会各申请64KiB L0A/B；L1默认也约288KiB/实例。双实例方案暂停。当前只读原生检索发现`get_masked_input_and_mask_kernel.cpp:18`在Vector算子析构中调用`pipe.Reset()`；未找到A2 task边界End→Reset→不同Matmul重建的现成实例，合法性、事件资源和重建成本均未知。

完整decode若采用M32，不能只依赖现有Q尾清零：每轮PV额外P行也必须清零，且不破坏MTE3→MTE2协议。K64的full-attention数值、overflow/status契约以及M24→M12→M24图复用全部仍需独立资格。
