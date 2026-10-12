# 紧凑 Batch8 Hadamard，独立研究

档案 #13–22/#70–73/#126/#145/#148/#154–156 与 startup D.4。未改生产；本目录不注册算子或服务开关，VM由root排他安排。

## 当前真实路径与差异

`impl.py:161–163` 的 Q 旋转仍调用canonical `rotate_out`；`rotate_clip_store.cpp:44–54` 每核重复4096个scalar索引/sign写入，`76–116` 仍逐row执行Hadamard。striped writer已合入向量Init和SIMD pack，但 `rotate_clip_store_striped.cpp:100–110` 的Transform仍逐row。不是重复计算此前writer1.796×收益。

三臂writer：此前冻结 `writer_baseline.cpp`（old）、此前冻结 `writer_pack_init.cpp`（current，和当前生产逐字对应）、本目录 `writer_batch.cpp`（在current上仅改紧凑Transform及work容量）。三臂Q rotate：当前canonical原文件（old）、只向量Init的 `rotate_init.cpp`、向量Init＋紧凑Transform的 `rotate_batch.cpp`。

`generate.py --check`验证旧writer固定哈希、current与生产映射、新源逆向恢复；canonical与全部helper哈希写入构建绑定。不改旧A/B/C源码、fixture或报告。

## 算法、空间、精度

每stage执行所有row的原Gather到各自`work[row*D]`，一道PIPE_V；再逐row原Mul(sign)，一道PIPE_V；最后对`rows*D`一次Add，一道PIPE_V。每row的Gather→Mul→Add仍以distance1…D/2顺序执行，Gather只从该row输入读取；无跨row归约。最后统一Muls原FP32 normalization。阶段间所有数据依赖保留，MTE2/V/MTE3跨管道屏障与状态检查不变。

每stage索引/sign仍各D个（D256总16KiB）。work从1KiB扩大为8KiB，索引Init数据量不变；未采用需要128KiB索引/sign的全展开方案。显式D256 Hadamard UB：engine50272B，完整writer61440B，RotateOnly51392B。非Hadamard仍分配原1KiB work并执行原generic Transform；本次不重做已测只有约0.3%整步收益的identity单点。

Full8/D256按源码发射数：

| 项 | old Transform | compact Transform |
| --- | ---: | ---: |
| Gather | 64 | 64 |
| Mul(sign) | 64 | 64 |
| Add | 64 | 8 |
| Muls(normalization) | 8 | 1 |
| 显式PIPE_V屏障 | 200 | 25 |

每个元素执行的FP32算术未少。row=1时调用/屏障数不增加；partial rowCount仍只处理有效count。Q原有negative-slot清零、all-padding跳过Transform、finite/status路径保留；writer原有无效slot/error/raw-window/量化舍入路径保留。NaN payload、±0、subnormal与非有限状态仍须跨臂按位验证，不凭数学独立性宣称设备通过。

## ROI：条件，不是性能声明

向量调用数理论上仅200/137=1.460×，要得到完整writer≥1.5必须有足够的屏障/发射成本可消除。若以current完整writer96566ticks为示例，需净省≥32189ticks；即使Transform部分理想8倍，该部分原占比也至少38.1%，其余时间不得增长。尚无当前完整writer内部H比例；canonical Q另有独立Init收益，但不能把7.11×元数据与Batch8比例相乘当整算子收益。

判定使用同库完整old/current/batch8 writer与完整old/init/batch rotate，不用指令结束标记或单个向量phase替代task。NPU/graph/整轮2–3×仍未知。

## 最小入口与期限

```bash
python3 csrc/experiments/hadamard_batch8/generate.py --check
python3 csrc/experiments/hadamard_batch8/entrypoints.py --check
```

CMake设置`ASCEND_HOME_PATH`、`SOC_VERSION=ascend910b4`。CPU加`OSCAR_BATCH8_CPU_DEBUG=ON`，writer为`oscar_batch8_store_cpu FIXTURE NEW_OUTPUT [0|1|2|all]`，默认只执行current(1)→batch8(2)，逐臂立即保存全部回读并按字节oracle/guards比较；显式1/2可分别运行，旧writer不重跑。`OSCAR_BATCH8_STORE_ONLY=ON`可只构建writer入口。CPU固定120s。完整库`liboscar_hadamard_batch8.so`含全部六臂；CAModel单臂固定180s，通过现有有界supervisor执行，不自动重试。

Writer fixture已导出：`/private/tmp/oscar-hadamard-batch8-writer-fixtures-20261010/`，N128/H1/D256 suffix-padding与N16/D64非零clip；独立oracle内容沿用已冻结writer fixture，只新增源绑定。

Query rotate fixture在`artifacts/hadamard_batch8_rotate_20261010/`：主N128/H6/D256，另7个紧凑例覆盖1/7/8/9rows、D64/128/256、BF16/FP16、partial/allpadding、±0与非有限值。CPU入口`oscar_batch8_rotate_cpu FIXTURE NEW_OUTPUT [ARM [BASELINE_DIR]]`；默认0old/1Init/2Batch8依次执行。单臂1/2必须带已经通过的0臂目录，并核输入/shape及完整output/status字节；独立finite2e-3沿冻结store_dequant，仅独立host oracle的NaN按类别核实，跨臂NaN payload/±0仍逐字节。

```bash
python3 csrc/experiments/hadamard_batch8/replay_writer.py --arm current --fixture FIXTURE --output NEW_OUTPUT --library LIBRARY --simulator-config CONFIG
python3 csrc/experiments/hadamard_batch8/compare_writer.py --directory THREE_ARM_REPORT_DIRECTORY --output NEW_COMPARISON
```

Compare保留源/同库/输入/初始字节/独立oracle/全部guard/全部actor/真实rc/180s门；current/batch8为本轮必需配对。old仅在显式同库完成时额外报告，默认保留旧证据而不重跑、不跨库拼出新比值，不重算或抹掉旧失败。

## D.4四问

对应rotate与phase1_stores；旧失败full-history恢复6499.8–6655.1ms不能重现。本方案只增加固定7KiB work，不读历史、不造完整BF16 KV、不改旋转或量化语义。源码发射削减不是NPU毫秒，完整算子与整轮结果仍以实际完成证据为准。
