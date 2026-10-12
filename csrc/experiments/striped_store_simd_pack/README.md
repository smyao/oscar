# Striped store：精确 SIMD pack 与 Hadamard Init 三臂研究

**本目录保留固定的已测研究三臂；候选writer源码已整合，尚未推送。** 2026-10-10，root已完成两个完整writer用例的CPU-debug三臂，以及同库、同输入N128/grid16完整writer的CAModel三臂；逐位、独立oracle、guards均通过。组合方案的完整writer CAModel为原来的0.5568倍耗时，即**1.79586倍局部加速**。目标NPU、图与完整服务收益尚未验收；不能把这一结果换算成整服务提速或原生基线达标。

档案 G25–34/#13–22/#70–73/#126/#144–145/#151/#154–156与完整启动文档D.4。原始PR量化规则来自 `references/oscar-vllm-pr46774/vllm/v1/attention/ops/triton_oscar_store.py`；所有参考树保持只读。

## 已核实的实际生产路径

`tools/serving_variants.py`统一候选配置；`cache_storage_ops.store_current`按实际cache_format选择：`striped_v1`调用`rotate_clip_store_striped_out`，`striped_request_window_v1`调用`rotate_clip_store_request_window_out`，`canonical_v1`调用`rotate_clip_store_out`。`active_identity_store`未注册生产CMake/contracts/selector，其2026-10-06局部证据不代表已经部署。

writer在校验slot前仍对K/V做旋转和量化；负slot最终覆盖status为0。`ClipQuantize`先做原FP32 ReduceSum检查、可选clip、min/max；scale/zero先FP16舍入再参与除法/加0.5/clamp/int32截断。冻结旧D256 writer每侧末尾pack有256次Scalar GetValue和64次byte SetValue，K/V合计512读+128写/token/head；本轮替换这一段。默认config的clip_ratio为0，不执行Sort32/MrgSort；不能把默认写延迟归于percentile排序。

request-window writer仍使用同一个量化/pack数学前缀；它有packed与window两个同stream launch，后者只发布至多323个精确窗口行，没有再量化。本轮仅将精确pack/Init接入两条striped writer，不改slot/epoch/MTP生命周期。

历史 `active_identity_store_result_20261006.md`：同grid16下q1 identity局部8.56倍，但已复制的旧decode单step中两次q1 store仅1.138ms/428.101ms，不能靠其宣称整轮追平。本轮完整writer A/B可以归因到pack替换，但尚无真实服务中该writer的完整占比。D.4的208.6–216ms/16K store与6.5秒历史恢复均只作失败背景，不套给本轮。

## 已完成证据与采用边界

| 层级/用例 | 实际结果 | 结论范围 |
|---|---|---|
| CPU-debug，以下两个完整writer用例 | old/pack/pack+init全部逐位、独立PR oracle、inputs/guards通过 | 两种shape的本地数值与边界契约 |
| CPU-debug，D256 compact-domain | 66112 words/side通过：全65536-word域加每个物理位置的边界模式 | 不等于每位置完整65536域穷举 |
| CPU-debug，D256 full-per-position | 120秒rc124，无完整数值结论；原始日志已归档 | 失败保留，未改成通过 |
| CANN，同一个三臂库 | 构建完成，随后三臂均真实运行 | 不能替代目标NPU完成 |
| CANN，整合后的生产全库 | `tools.build_ops`干净重编通过；四个正式源SHA与host一致 | source signature `0687dfd3983614a05cb08498fcc6519dc9ccd8718c21230d2d1448b3bbbd3f4e`，不是NPU完成 |
| 官方CPU-debug，整合后的request-packed接缝 | `clipped_hadamard_d256`的正式striped、request-packed及window全部bytes identical通过 | 独立PR oracle、raw/tag/status/guard与新共享头整合 |
| CAModel，N128/Hkv1/D256/BF16，grid16，112 live/16 padding，Hadamard，clip0 | 下表三臂均通过bytes/oracle/guards/inputs、真实control、完整task/AIV profile及严格compare | 完整writer局部周期，可进入生产整合准备 |
| 目标NPU、NPU图、完整服务 | 本研究未执行 | 均待验收 |

| CAModel arm | 完整writer ticks | 相对old加速 |
|---|---:|---:|
| old | 173419 | 1.00000× |
| vector_pack | 136213 | 1.27315× |
| vector_pack_and_init | 96566 | 1.79586× |

这次比较由root统一串行执行，同一library、fixture、SIM配置与grid，三臂真实退出码均为0、未超时且清理完成；严格比较器要求完整profile和保存字节一致。数值及计时来自这次完整writer运行，独立Init的33393→4695 ticks只用于交叉说明，未相加或拼接成上述结果。原始三臂证据由root的VM运行目录保留；本文件不伪造尚未回传的宿主机报告路径。

组合节省44.32%的该writer周期。若真实端到端中该writer占比为f，且其余工作不变，可支持的理想整轮加速仅为`1 / (1 - f + f / 1.79586)`；这里没有完整服务f，不能据此声称整轮1.5倍、4–5倍或达到时间目标。

## A/B/C 保持哪些字节与算术

| arm | kernel | 唯一变化 |
|---|---|---|
| old | oscar_rotate_clip_store_striped_kernel | 冻结的旧生产原源码，见下方SHA |
| vector_pack | oscar_rotate_clip_store_striped_simd_pack_kernel | 只替换末尾整数pack |
| vector_pack_and_init | oscar_rotate_clip_store_striped_simd_pack_init_kernel | 上项加独立精确Hadamard索引/sign构造 |

旧生产SHA固定为 `0515fd2a527812dc6e7c09e46ebae50fd19be62e7645705c8caffbb132a31d29`。`writer_baseline.cpp`只调整两条相对include，文件SHA为 `674b6115f93b1c8387bdc8ba8440218c7b1e78e3563cb4f76e37624e0eae62eb`；逆向include重定位后必须恢复旧生产SHA。`frozen_sources.json`同时锁定已测B/C、pack helper、CPU harness及共享依赖的SHA。B/C不再从变化中的生产writer生成，现有实测报告不改写。

`generate.py --check`检验固定三臂并可逆校验新生产writer等于已测C臂；`--check --frozen-only`用于独立复现旧实验，不要求未来生产仍停留在同一版本。CMake的CPU和CANN目标均编译`writer_baseline.cpp`，并在构建目录写`frozen_source_binding.json`。新fixture记录该来源；新replay要求库的构建清单相同，并报告实际编译源SHA。compare仅允许完整旧报告（原生产SHA与已测helper SHA均一致）或完整新冻结报告，拒绝混配，不能把更新后的生产源码重新标成old。

B保留原RotationEngine::Init；C只在Load前借用死input/work/red生成完全相同indices和±1 sign，Transform算术不变。本轮冻结仅通过host生成/哈希/可逆映射检查；没有重新编译或模拟，不新增性能结果。

两候选都不改归约树、Div、量化bin、clip次序、half metadata、错误优先级、WriteRaw、tags或发布边界。保持原来“先两侧成功，后cache发布”，保留负slot/非法slot/负position的覆盖顺序。

## pack 的精确实现与 D64 对齐

安装SDK已确认 `Cast<int16_t,int32_t>(CAST_NONE)`、uint16 ShiftLeft、uint16 Or可用。codes已为0..3，整数窄化无损；最高`3<<14`仍在uint16内。保持当前striped物理排列：`word[i]=Σ code[b*(D/8)+i]<<(2*b)`，LE存储。此排列是PR连续四码格式的既有可逆置换，不能误改回另一套字节布局。

- D128/256：复用已死work，8 planes对齐；7 ShiftLeft、7 Or，最后直接写精确D/4字节。
- D64：compact planes仅16B，V目标偏移也仅16B。先用16次uint64读/16次写搬到32B对齐临时planes；最后仅2次uint64写回16B，绝不扩写32B覆盖邻域。
- 原Cast读work结束、plane依赖、末尾Vector→Scalar与后续MTE3所有权均显式保护；metadata写法保持原样。CPU equality仍不能证明NPU异步正确。

## 两个完整 writer 用例

已生成 `artifacts/striped_store_simd_pack_20261010`：

1. `n128_h1_d256_bf16_b128_suffix_padding`：Hadamard、112活跃/16负slot padding、grid16、clip0。
2. `n16_h1_d64_bf16_b128_clip`：Hadamard、grid2、K clip0.75/V clip0.95，并覆盖D64两侧不对齐pack边界。

继续使用既有独立PR rotation/quantizer oracle，只有这两个固定fixture的Hadamard选择在私有函数副本中显式设置，不修改旧oracle模块。比较完整packed、FP16 metadata、raw BF16 NaN poison、tags、status、输入和前后guards，任何字节差均失败。

pack-only另备D64/128/256各65536个word模式，在**每个word位置及K/V两侧**遍历完整16-bit域。不同位置使用不同可逆偏移，避免同值广播掩盖错位；metadata与邻接区为poison。该完整输入约D256 128MiB，输出10MiB；只准备入口，不自动运行长模拟，截止不超过180秒，失败不扩限。

root原D256全位置CPU枚举实际达到预定120秒、rc124；CPU-debug npuchk/cceprint写满约4.6GB tmpfs，未完成数值结果。进程已退出，root完整gzip归档该轮原始日志至宿主机并校验后，仅回收本轮已归档的VM日志，失败保留，**不得宣传完整每位置穷举通过**。本轮另增加明确的`compact-domain`模式，不覆盖旧模式：主域`rows=65536/width`，`word=r*width+((i+r)%width)`，V侧加12345再mod65536。每侧整体覆盖完整65536-word域且物理位置轮换；额外18行让每个位置均覆盖0/ffff/16个单bit，邻位内容不同且metadata/邻区poison照查。报告分别标`domain_words_per_side`、`boundary_words_per_side`和`total_words_per_side`，**不称每位置覆盖65536**。root实际D256 compact完成并通过66112 words/side；D64/128 compact未在本轮追加运行。完整writer oracle门不变；compact每个D仍预定120秒，不加时。

## root 一键构建与有限运行

```bash
# 仅在已授权CANN VM中执行，串行取得现有排他锁；不自动启动模拟。
bash csrc/experiments/striped_store_simd_pack/build.sh NEW_BUILD_DIRECTORY

# Host生成，仅输入/oracle，不是设备测试。
python3 csrc/experiments/striped_store_simd_pack/fixture.py --output NEW_FIXTURES

# CPU完整writer三臂；保存每臂全部guarded字节，然后独立oracle与逐位比较。
NEW_BUILD_DIRECTORY/cpu/oscar_simd_pack_store_cpu CASE_DIRECTORY NEW_CPU_OUTPUT
# 独立pack完整模式；每次一个D。
NEW_BUILD_DIRECTORY/cpu/oscar_simd_pack_exhaustive_cpu 64 NEW_PACK_OUTPUT_BIN
# 本轮短枚举模式；完整模式仍是默认且不改变旧结果。
NEW_BUILD_DIRECTORY/cpu/oscar_simd_pack_exhaustive_cpu 256 NEW_COMPACT_OUTPUT_BIN compact-domain
```

完整writer CPU/CAModel由已有 `tools/experiments/run_camodel_arm.py --timeout 180`监管；pack-only compact使用`--timeout 120`。独立control/stdout目录，不覆盖旧结果，不重试或加时。CPU三臂在同一调用内完成，采用180秒总上限，更严格而非扩充预算。

CAModel每臂的child命令：

```bash
python3 csrc/experiments/striped_store_simd_pack/replay.py \
  --arm old --fixture CASE_DIRECTORY --output CASE_RESULT/old \
  --library ABSOLUTE_BUILD/device/lib/liboscar_striped_store_simd_pack.so \
  --simulator-config ABSOLUTE_SIM_CONFIG
```

依次把arm改为`vector_pack`、`vector_pack_and_init`，同库/同fixture/SIM配置/grid；各自durable control路径为`CASE_RESULT/<arm>-control.json`，限180秒。child环境由既有`--env-json`传入CANN LD_PRELOAD/LD_LIBRARY_PATH/CAMODEL_CONFIG_PATH，supervisor本身不preload。根负责唯一VM执行。

```bash
python3 csrc/experiments/striped_store_simd_pack/compare.py \
  --directory CASE_RESULT --output NEW_COMPARISON.json
```

比较器要求完整task与所有AIV profile、真实rc0/未timeout/清理完成、全部保存文件hash/guard/oracle相同。它只报告完整writer ticks；pack访问次数和Init局部7.11倍都不能替代该比较。已测完整writer局部1.79586倍；目标NPU、graph及整轮收益仍另验，不扩大这一结论或放宽任何精度门。

## 最小生产整合（CANN/官方CPU完成，NPU门待验收）

1. 已将资格版本`pack.h`和`hadamard_indices/vector_indices.h`映射到`csrc/kernels/striped_store_pack.h`与`hadamard_indices.h`。`tools/package_request_window_kernels.py`逐字校验仅include/来源注释改变且可逆；保持原Cast/ShiftLeft/Or、D64窄写回、Cast→Not依赖与全部barrier，不新增ABI、op或候选flag。既有`tools.build_ops`对全部`csrc/`做文件指纹，新头会触发重新构建。
2. `csrc/kernels/rotate_clip_store_striped.cpp`已复用两个helper。`csrc/experiments/request_windows_store/generate.py`继承相同helper，再通过既有packager生成`csrc/kernels/rotate_clip_store_request_packed.cpp`；生成器新增`--check`校验生成体和共享头SHA，packager保持非include正文逐字一致。`rotate_clip_store_request_window.cpp`只负责窗口发布，本轮未改其正文。
3. 真实主`rotate_out`位于`csrc/kernels/rotate_clip_store.cpp`（由`rotation_bindings.cpp`调用`rotate_launch`），本轮保持原样；canonical/fe0也未改。将来若共享Init，**须单独做旋转资格，不套用本轮writer证据**。它的canonical pack排列不同，不能使用striped pack helper。
4. 保持`tools/serving_variants.py`与`cache_storage_ops.store_current`原有dispatch，无新增用户参数。完成后再确认`install_serve.sh --variant candidate`与`install_observe_serve.sh --variant candidate`均构建并实际装载相同新实现；后四卡只改变显式设备/端口。本研究当前没有修改任何服务入口。

## 现有真NPU门只需补齐的边界

- **striped store的Hadamard分支（探针已补，设备未跑）**：旧`tools/probe_striped_cache.py::store_gate`及`_store_changed_input_graph`虽传Hadamard矩阵，调用参数实际为`hadamard=False`。本轮保留原dense分支，追加一次`True`的完整bytes/metadata/raw/tag/status/guard及同地址K/V改变图回放；不能把旧dense gate视为Init验收。host测试确认两种参数均传入op且Hadamard失败不会被吞掉。
- **D64的16B边界及D128对齐**：复用现有原语golden准备，给实际striped writer补D64夹值病例，逐位检查K/V相邻16B、metadata及poison；D128补短字节病例确认aligned分支。N128带padding的D256病例同时验证scratch复用，无需真机运行65536词长枚举。
- **request-window生命周期**：`tools/probe_request_windows.py::case_plan`已有`segment_boundary`、`reordered_hkv2`、`clipped_hadamard`的`hadamard=True`，`run_store_graph`也已有N128/32 live的True路径、全padding捕获→同地址live/lease/epoch/position变更。复用这些门与既有独立PR/poison检查，确认生成writer实际包含新helper即可；不得删掉非法状态、rollback、stale epoch等现有门，也不需要另造大型probe。
- **仅在主rotate_out也改Init时**：保留`tests/test_rotation_npu.py`已有D64/128/256×Hadamard开关、masked/error与两种clip的全部goldens；该文件当前不提供图验收，额外补一个真实主rotate的同地址输入/slot改变图回放与eager对拍。未改主rotate时不为此追加无关验证。

本轮17项相关host测试及7项既有固定入口/候选选路回归通过；冻结研究生成器、request-packed生成器、packager的`--check`均通过。root随后完成生产全库干净CANN构建及`clipped_hadamard_d256`官方CPU接缝，实际记录位于`reports/quant_production_integration_20261010/integration-summary.json`与`request-packed-cpu-seam.log`；先前失败的增量编译日志也保留。当前未推送；上述证据不能替代NPU、图或整服务性能验收。
