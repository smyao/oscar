# A2 FP32低层Matmul：只读可行性审计

当前未写CANN、未编译或执行。依据当前native A2源码、实际FP32 trace及root只读导出的CANN9.1 SDK（`artifacts/cann_sdk_matmul_move_20261010/provenance.json`记录来源与哈希）。档案#13–22/#70–73/#81–85/#126/#140/#145/#148/#154–156和startup D.4；不取arch35能力，不改精度/原生代码。

## 结论

可构造独立低层数值原型，三个操作方向均有当前A2依据。目标是有限固定几何下去除通用scheduler/资源队列及可能重复的L1加载，不改变basic M64/N64/K128。**应全替换该原型的MatmulImpl，不在同一TPipe同时初始化两套实现。** 同一个独立库可以有分别启动的old/low-level kernel，但不能让一个kernel实例同时占两套L1。

收益尚未成立。small-M的PIPE_M、GM↔L1↔L0的必要搬运和真实数据依赖都保留；此前静态SetShape约1.016×和M32/N128/K64慢34.4%的结论仍有效，不重复算成收益。

## 1. 固定合法几何与字节账

沿当前 `attention_cv_decode_bundle.cpp:183–191` 的物理矩阵合同：A/B/C行跨度均为D256；逻辑m∈[1,128]，n/k为16倍数且≤256。QK物理B=[N,K]、转置右乘；PV物理B=[K,N]、不转置；Rv物理B=[N,K]、转置右乘。m是真实24/6/126等，不擅自扩大MMAD m到对齐值。L1/L0搬运的M尾允许对齐，Fixpipe仅写真实m/n。

| 独占资源 | 预算 | 说明 |
| --- | ---: | --- |
| L1完整A | alignM16×alignK8×4，最大128KiB | GM实际行跨度256，不按短K紧密GM解释 |
| L1完整B | 最大256×256×4=256KiB | 方向不同只改NZ维度解释 |
| L1合计 | 最大384KiB | **不能再双缓冲完整A/B：768KiB不成立** |
| L0A | 2×64×128×4=64KiB | K128 ping/pong，每块最大32KiB |
| L0B | 2×128×64×4=64KiB | 同上 |
| L0C | 2×64×64×4=32KiB | 两个独立输出tile；各自完整K归约后写回 |

资源依据：native A2 `moe/hc_pre/op_kernel/hc_pre.cpp:19–27`明确区分310和非310；其`hc_pre_cube_compute.h:146–164`实际分配L1 A/B各256KiB、L0A/B各64KiB、L0C64KiB。通用`common/include/kernel/hardware.h:26–35`列512/64/64/128KiB，但这是模板，不冒充910b4设备查询。上述原型L0C只要32KiB，不依赖未实测的128KiB极限。

## 2. GM→L1：FP32 ND2NZ

native A2 `hc_pre_cube_compute.h:178–207`对`GlobalTensor<float>`与L1 float使用公开`DataCopy(..., Nd2NzParams)`。本方案ndNum=1、srcDValue=256、dstNzNStride=1；逻辑nValue/dValue与目标C0 stride如下：

| 输入 | nValue（GM行数） | dValue（每行有效列） | dstNzC0Stride |
| --- | ---: | ---: | ---: |
| A | m | k | align(m,16) |
| B，QK/Rv转置 | n | k | n |
| B，PV非转置 | k | n | k |

FP32的C0为8个float，NZ逻辑地址是`(col//8)*alignedRows*8 + row*8 + col%8`。srcDValue始终256，尤其PV尾K不能错读紧密stride=k。SDK具体不对齐M的ND2NZ padding仍需对照其copy-in wrapper及最小poison/guard输入，不以“L1容量够”替代尾部证明。

## 3. L1→L0：实际SDK分支

只读SDK的`stage/split/load_to_l0a/load_to_l0a_loadInstr.h:118–130`是当前A不转置路径：

- FMatrixA为H=1、W=align(fullM,16)。
- `LoadData3DParamsV2Pro.channelSize=align(fullK,8)`。
- M轴start=当前M块偏移、step=align(当前真实m,16)；K轴start=当前K块偏移、step=align(当前k,8)。
- dtype=float，目标L0A，不转置。原型可以使用公开`LoadData3DParamsV2<float>`表达同参数；不能直接包含SDK内部Matmul实现头作为自定义核依赖。

### QK / Rv：物理B=[N,K]

`load_to_l0b_loadInstr.h:91–145`的isBTranspose=true路径使用`LoadData2dParams`，不是再将float转FP16：C0=8、N0=16。fullN=256、madN=64、madK=128时：

- 4次N块加载；每次repeat16、srcStride16、dstGap3、ifTranspose=false。
- 初始L1 offset=`nOffset*8+kOffset*fullN`；每次L1/L0 offset均加128个float。
- madK尾≤64时要用该SDK的另一个循环分支，不能硬套上面repeat16。

native A2 `hc_pre_cube_compute.h:440–453`也给出同方向float LoadData先例；但其`MmadAB:462–476`打开HF32且K32，**只用于搬运API参考，不可照搬计算**。

### PV：物理B=[K,N]

`load_to_l0b_loadInstr.h:74–86,177–196`对FP32实际走LoadData3DV2：

- FMatrixB为H=1、W=align(fullK,16)。
- channelSize=`align(fullN,16)`，不是K。
- M轴start=K块偏移、step=align(madK,16)。
- K轴start=N块偏移、step=align(madN,16)。
- `fMatrixCtrl=true`选择B FMatrix；输出L0B。不是引用int8 LoadDataWithTranspose或arch35特殊通路。

实际证据：`reports/fp32_bf16_mmad_ratio_20261010/reports/mmad-ratio-camodel/pv_m126/fp32/log_ca/`的MTE1记录为`LOAD_L1_TO_DST_3DV2 / Dst:L0B / Dtype:B32`，并配置FMATRIX_B，与这条SDK分支一致。

## 4. MMAD与输出，不删除必要屏障

每个M≤64/N≤64输出tile，按K=0..127、128..255顺序（末尾不足128保持真实尾K）执行FP32 MMAD；首K初始化C、后K累加同C。HF32关闭；不更换K64，也不把两个K128分别输出再相加。当前trace包含M64和M62，短形M24/M6同理保留。

SDK `stage/compute/mmad_compute.h:46–80`显示m/n/k及cmatrixInit/Source由参数直接给出，并在`(m/16)*(n/16)<10`时强制PIPE_M。**低层版也保留该条件**。它不能靠去掉必需BAR取得虚假加速。

float→float Fixpipe保持NoQuant、真实mSize/nSize、srcStride=align(realMtile,16)、dstStride=256、ndNum=1；依据SDK `stage/copy_cube_out/copy_cube_out_utils.h:50–74,140–166`及native arch22 sparse attention `:822–835`。只写当前m/n有效矩形，其他GM poison/padding仍由原CV拥有者处理。

## 5. 事件与生命周期方案

1. 每次Matmul开始前，上一轮L1的最后MTE1读取完成后才允许MTE2覆盖A/B；GM输入仍由原CV的两lane→Cube ready协议发布。
2. ND2NZ A/B完成后，MTE2_MTE1发布/等待一次，才发L0加载。
3. L0A/B两个slot用M_MTE1消费完成信号保护覆盖；每次A/B LoadData都完成后用MTE1_M发布/等待，再发MMAD。下一slot可在上一MMAD执行时加载，不改K归约顺序。
4. L0C槽必须在前次Fixpipe消费完成后才复用；首K初始化、后K累加；最后K完成后才Fixpipe。初版选择一套明确协议：可沿当前A2的unitFlag2/3自动同步，或用公开M_FIX/FIX_M显式同步，**不能同时混用两套而将重复等待当算法收益**。实现前需把选定协议与SDK当前MatmulUnitFlag参数核到一致；当前本文件尚未决定删除哪一种同步。
5. 每个槽的最后反向事件在结束时drain。最终GM结果ready仍在PIPE_FIX之后；不在Scalar函数返回时提前唤醒Vector。下一次QK/PV/Rv重置几何和slot索引，不继承上一调用的pending状态。

先保持非atomic写回；C4研究atomic路径另需复用已独立验证的完整K atomic与恢复普通模式协议，不在第一低层原型同时扩大范围。

## 6. 未闭合项与最小后续资格

- FP32不对齐M的ND2NZ零填充/poison与公开LoadDataV2参数需由对应SDK wrapper闭合，再编最小搬运/identity witness；不是全部数学无误的结论。
- 即使MMAD维度相同，手动L0布局、initC、累计次序和尾部写回仍要与同库MatmulImpl逐位比全部输出、NaN/±0/非有限、输入及双guard；覆盖QK转置、PV非转置、Rv与N/K16尾。
- 普通数据量一次L1完整加载可行，不保证比当前MatmulImpl少读；SDK本身有L1/L0 cache。要用完整trace数实际copy/ready/compute依赖，不能把调度C++行数折成周期收益。
- 这是有限独立Matmul研究，不改生产、不宣称≥1.5或整轮2–3×；如果去通用调度后净收益不足就停止，不再做另一轮只有SetShape差异的长实验。

D.4对应fused FIA QK/PV/rotation；不增加历史长度相关的物化，只在当前KV256矩阵范围使用固定L1/L0。原失败全历史解包约6.5秒不能被这个局部研究掩盖。
