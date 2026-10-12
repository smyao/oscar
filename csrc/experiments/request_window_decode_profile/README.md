# 已交付 decode bundle 的同源分相位原型

档案：#132–135（enqueue背压/异步归因/诊断不作性能验收）、#145（MTE3→MTE2所有权）、#150（真实q4/q1形状、actor时间不能相加），以及完整启动文档D.4。本目录是研究诊断，未注册生产算子、未改candidate入口或任何原生文件。

**目录名含 request_window，但唯一被测源是已交付 `csrc/kernels/attention_cv_decode_bundle.cpp`，即旧 per-page `striped_v1`。这不是新 dual-region/request-stable window_map reader 的测量。** source0共同数学可用于定位候选方向，source1地址与生命周期不同，不能直接移用占比。生成时的manifest/plan保留初始not_run快照；实际后续执行记录见本节下方，不能把初始状态当最新结果。

2026-10-09后续：CANN编译/链接rc0；C1025/R32/N128/S3/20core的q4/q1 normal/profile四臂完整数值通过，逐位partial/LSE/status相同，插桩扰动分别+1.0490%/+3.1657%，均过预设5%门。SYS_CNT为0且未收marker，phase份额仍未知；C1025每history split一块，没有覆盖hasNext稳态。NPU/graph/整轮未跑。详细结果及C2049最小下一点的静态资源账见 `reports/request_window_decode_profile_20261009/summary.md`；未修改本目录原计划或运行第二context。

## D.4四问与源码身份

1. 观测fused FIA内部history packed DMA/unpack/FP32 publish/QK/PV/softmax/flags，以及window/current和finalize。
2. D.4的全历史dequant约6.5秒与prepare背压提醒：只测当前有界tile，不增加history-sized物化；异步等待不能归错phase。
3. `generate.py` 对当前bundle、全部直接依赖和成熟profile helper固定SHA。生成四个隔离symbol；每一次修改计数、逆序恢复所有原始字节，并检查completion fences/barriers/flag发布/HF32调用数不变。旧数学、错误顺序、buffer alias、全部任务遍历均保留。无host metadata循环进入服务路径。
4. 不预言毫秒或倍数；正常/插桩同形完整A/B先验证数值与扰动，再允许归因。若原型导致额外spills或明显扰动，必须公开并拒绝把占比套回正常kernel。

`normal.cpp`只改include路径和symbol。`profile.cpp`额外加Scalar `GetSystemCycle`、固定actor counters和可选MOV markers；不增加完成同步。它们同一个library同一次编译，不能拿历史SIMD二进制作normal分母。profile保留所有15个原GM指针，在workspace后加第16个profile指针，再接原17个int64和float scale；normal ABI仍15指针。两者workspace均 **1,146,880 B/core**。

| arm | q4 | q1 |
|---|---|---|
| normal | `oscar_bundle_normal_q4_kernel` | `oscar_bundle_normal_q1_kernel` |
| profile | `oscar_bundle_profile_q4_kernel` | `oscar_bundle_profile_q1_kernel` |

`manifest.json`记录source/helper/header/hash、每个数学method的hash、逆向恢复结果、symbols和待验证项。生成/一致性检查是host源码契约，绝非CANN编译通过。

原型host检查：`generate.py --check`、Python AST语法检查、`python3 -m unittest csrc.experiments.request_window_decode_profile.test_host_contract -v`（3项）通过。这些测试只验证证据门拒绝R1伪装、status/library/symbol不匹配及过大扰动，本身没有kernel执行；生成manifest保留初始not_run快照，后续CANN/CAModel实际结果以本页开头指向的报告为准。

## 相位定义、覆盖与重叠

兼容成熟schema2：profile为int64 `[cube_cores,3,4,24]`，每actor768 B，每Cube2304 B。actor依次AIC/AIV0/AIV1；source依次history/window/current/total。每actor只写自己的12条64 B cacheline，先重置全部96计数槽，Process结束后完整flush；flush在field23之外，但仍影响整体task tick，应由normal/profile扰动门捕获。

| 字段 | 定义 |
|---|---|
| 0–2 | 合法task数、KV256单元数、单元有效候选行数；每actor独立记录 |
| 3 | 完整合法task调用，包含其全部嵌套阶段与等待 |
| 4 | LoadQueries |
| 5 | LoadPackedInto：包括清零、Physical查询、DataCopyPad及既有MTE2完成等待；初始LoadPacked wrapper不另计，避免双计 |
| 6 | Unpack / PipelineUnpack，共用同字段；两者不会相互调用 |
| 7 | LoadPrecise，包含其LoadWindowRuns子函数；后者不再单独套相同scope |
| 8 | PublishKv / PipelinePublishKv（既有发布与fence）；不等于全部GM后端延迟 |
| 9/11 | AIV原QK/PV等待；history预取仍在原来等待之前 |
| 10 | 原Softmax全过程与P发布，包含原mask、finite检查、V2、alpha乘法与MTE3→MTE2 |
| 12 | 原PV读取/累加；history对应PipelineConsumePv |
| 13/15/17 | AIC等KV/P/AIV累加；history pipeline原来没有17，保持0 |
| 14/16/18 | 原Matmul helper的QK/PV/Rv阶段；含SDK搬运、提交和已有同步，不能称裸Cube busy |
| 19 | AIV归一化开始到函数结束，包含22旋转等待和21最终写回 |
| 20 | 所有LoadTask，包括padding/follower；marker统一归total，SYS_CNT计数在元数据读完后也记入读到的source |
| 21 | PublishRows，包括padding/错误行路径；保留所有status写入 |
| 22 | 原旋转/规范化/任务完成等待；按actor分别计 |
| 23 | actor Init+Process，不含诊断flush；只在total有值 |

同actor是严格嵌套区间树：23包含3，3包含4–22，19再包含21/22。`analyze_markers.py`减去直接子scope后得exclusive，并要求exclusive总和恰等于actor外层span。相同field的多次调用可求同actor不重叠并集；不能再加其父scope。不同actor并行：报告每actor独立区间和wait；若展示跨actor在公共cycle轴上的interval union/overlap，它只表示同时覆盖的时间，**不是可删除成本，绝不把actor总量或最大值相加为wall**。完整task tick来自同一次CAModel task record，不由phase拼出来。

GetSystemCycle在已有CAModel中为0。计数仍可用，但SYS_CNT计时必须标 `unavailable_zero_SYS_CNT`；MOV magic沿用 `0x5a77c01d` 与原解析器，使用实际instruction completion cycle。不得将0解释成phase无开销。profile markers会扰动Scalar指令调度，不能替代normal计时。

## 固定几何与有限斜率

`fixture.py`默认只用stdlib写 `plan.json`，不生成tensor也不运行算子。真实比较固定：R32，N128，Hq6/Hkv1/D256，S3，20 Cube；q4有128个有效token，q1有32个有效token及96个padding。全部1152 task槽和两lane status均保留。

先C1025，normal/profile数值与扰动门完成后才考虑C4097；也允许root在测前选择C4097/C8193。每个context的q4/q1各有32条独立历史，不能换成R1的大N。当前shape的理论KV单位（仅工作计数）为：

| context | history q4/q1 | window q4/q1 | current q4/q1 |
|---:|---:|---:|---:|
| 1025 | 96 /96 | 96 /96 | 64 /32 |
| 4097 | 480 /480 | 96 /96 | 64 /32 |
| 8193 | 1056 /1056 | 96 /96 | 64 /32 |

比较counter时，分别在AIC各core、AIV0各core、AIV1各core求和，各自应等于以上工作计数；不能将三个engine相加。两个context只能给固定shape/固定实现/固定simulator下的有限差分，如`Δpacked阶段/Δhistory单元`；不能认定全程线性、外推64K或推模型倍数。必须同时报告三source、正常task tick变化、profile扰动变化与核间分布，防止把window固定开销当history斜率。

## 复现入口与后续执行边界

先检查与生成源码（host only）：

```bash
python3 csrc/experiments/request_window_decode_profile/generate.py --check
python3 csrc/experiments/request_window_decode_profile/fixture.py
```

可复现的CANN配置，与成熟diagnostic工程相同；实际首轮路径/rc见报告，后续仍须排他执行：

```bash
cmake -S csrc/experiments/request_window_decode_profile -B build/request-window-decode-profile -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 -DSOC_VERSION=ascend910b4 -DOSCAR_CAMODEL_MARKERS=ON
cmake --build build/request-window-decode-profile -j1
```

输出库 `liboscar_request_window_decode_profile.so`。禁止修改现有SDK/cache model，保留当前exclusive `/tmp/oscar-cann-validation.lock`，不得与root已有CPU-debug/CAModel并跑。编译产物须保存构建日志、库SHA和本目录manifest。

后续显式离线golden导出可用：

```bash
python3 csrc/experiments/request_window_decode_profile/fixture.py --export --contexts 1025 --output artifacts/request_window_decode_profile
```

它复用 `window_batch_decode/fixture.py` 的真实metadata输入、独立PR oracle、冻结0.005容差和per-page原始布局。用已有bundle CPU harness的 `q4|q1 GOLDEN_DIR EXPORT_DIR prepare-only` 生成真实tasks/attrs；**prepare-only不算attention数值通过**。若已有完整CPU参考，replay保持required；若仅prepare，两个arm都必须明确`--cpu-reference optional`，仍强制两臂完整独立oracle和相互逐位partial/LSE/status检查，不能称CPU attention通过。

同一 `window_batch_decode/host_camodel.py` 可复用：两臂都指定新library、同fixture/export root、同host deadline、同core、同simulator配置、两个workspace参数1146880、同run-label；baseline symbols用normal，candidate symbols用profile。**仅candidate加 `--profile-counters`**。第一阶段`--instruction-budget-bytes 0`，不要求marker完整性，取得完整数值和总tick扰动；SYS_CNT为0照实记录，不输出phase份额。

现有 `window_batch_decode/compare_camodel.py` 要求两臂profile flag相同，不能直接用来比较正常/插桩。这里的 `compare.py` 只允许该诊断ABI差异，复用其`checked_arm`：完成记录、全部输出哈希、实际task tick、冻结oracle、input/library/config/deadline/core一致性均保留，还强制R32/N128/S3/20core与这四个symbol。逐位partial/LSE/status有任何差异直接失败。预先固定研究扰动门为 `abs(profile/normal−1)≤5%`；超过则返回rc2，保留数值通过与原始tick，不能事后放宽门槛来宣布低侵入。

```bash
python3 csrc/experiments/request_window_decode_profile/compare.py --directory reports/CHOSEN_RUN_LABEL --case q4_r32_n128_c1025_s3
```

这只是插桩扰动比，不能命名为优化收益；q1同样完成后才能推进第二context。

## 完整marker的资源门

不改现有全局trace caps。当前replayer在>2core时上限2 MiB/actor文件，正常数值A/B可先不采trace。完整marker研究要覆盖 **20 AIC+40 AIV=60个actor**；若后续root选择仅本次64 MiB/actor completion-log cap，上限为3.75 GiB；若还收同量popped logs则7.5 GiB，加runtime/metadata/输出/gzip副本另算。该提议没有在代码启用，不保证C1025/C4097能装下。

root可以后续选有界流式marker聚合，保留全部actor起止/嵌套/数量/源hash/截断标记，或按本次资源预算调整独立collector；本轮未实现collector，也未扩大旧cap。达到cap即明确 `incomplete`，只保留诊断原文，不求完整占比。第二context必须等第一个context实际完整与资源消耗已知后决定，不能预设它可装下。

若完整marker条件满足，可原样复用 `decode_phase_profile/analyze_markers.py --directory ... --case ... --variant candidate`；它要求60个actor全部完整、event count匹配、区间树闭合、exclusive守恒。须同时附这里的normal/profile扰动报告和新源manifest；不能把旧SIMD marker JSON混入新bundle报告。没有这些证据前，不新增DMA/Gather或其他优化。
