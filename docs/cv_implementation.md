# Cube / Vector attention 实现与证据边界

本轮新增 `attention_cv.cpp`、`attention_tasks.cpp` 和独立 torch registration。实现是 Ascend A2 的 `MIX_AIC_1_2`：一个 Cube 配两个 Vector。QK、PV、历史输出的 `Rv.T` 均由 `matmul::MatmulImpl` 执行；向量核处理 INT2 解包、精确 BF16 搬入、因果 mask、稳定 softmax 与分块累计。没有标量 dot 或调用 Python oracle 的生产路线。

源码、真实 CANN 编译、官方 CPU-debug、设备执行、图捕获、图重放、精度与性能是独立状态。此文件的资源/流量数字是源码推导，不能视为实测性能。

## 档案、PR 与 SDK 对照

修改前重新读取启动文档 §0.7 与 D.4 原始 JSON（短 decode、两次16K prefill、32K continuation）。按标题检索症状表、G1–G13、G25–G34、#3–22、#53–73、#79–93、#98–122 的同族故障。未访问失败项目代码。

| 证据 | 实现决定 |
|---|---|
| G3–G13、#3/#21/#81–83/#87–92 | 原生 API + CANN 9.1 SDK 核实；单个 primitive-only launch 声明；`ascend910b4`；静态资源边界；不做 float/unsigned 数值强转 |
| G26–G34、#13–20 | PR136B(D256) slot；metadata 原样 FP16→FP32；解包是 4-per-byte LSB-first，SIMD bit-plane→自然顺序；没有重新量化或整向量 BF16 降精度 |
| G30–G34、#12 | padding/empty/source split 也必须写 output 与 `-inf` LSE；初始输出 poison 不可能当有效值复用 |
| #34/#36 | graph 固定 N/R/task/workspace 容量；qstarts、seq_lens、slot_mapping 每次从设备读取；Cube scalar metadata 在入口清 DCache，Vector 使用 DMA |
| #37–49 | 窗口从物理页＋页内位置查找；tag 必须匹配页内位置；verify 每个 query 使用独立绝对位置与 recent 边界 |
| #53–69 | 不依赖 FIA 可选 LSE，不猜测 reshape；Q/K/V 同设备，causal 在每个 score row 显式处理 |
| #71/#140–141 / D.4 | 整历史恢复曾为6499.8–6655.1ms、原生FIA仅18.5–18.9ms：四份32-token子块组成一份128-token工作单元，供同一query tile复用；禁止先物化全长历史。K4实测由182.5s降到约131.7s，仍比原生慢约9倍；后续结构修订必须真机复测。 |
| #98–122 | direct-launch 单路径；不复制原生文件，不替换 OPP 环境，不把 config/符号存在当执行通过 |

只读原生先例：`references/vllm-ascend/csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h` 的 mode-2 CV flag 与固定 GM 通信；`hc_pre_cube_compute.h` 的 FP32 Cube 数据通路；`moe_grouped_matmul.h` 的 MatmulImpl；`add_rms_norm_bias_multi_n.h` 的 Gather。指定 PR：`triton_oscar_decode.py` 的 INT2 解包、FP32 score/PV 与自然对数 LSE；`oscar_attn.py` 的 window BF16 与历史输出逆旋转后合并。

CANN VM 中只读核实了 9.1 SDK 的 `matmul/constant_tiling.h`、`MatmulImplBase::SetSingleShape`、`SetHF32`、Gather/ShiftRight/cache 接口。`SetSingleShape` 调用 SetTail；normal config 的 `enableSetTail=true` 允许同一已初始化 matmul 使用 QK / PV / rotation 的实际形状。FP32 Cube 明确 `SetHF32Mode(false)`，不把 HF32 精度冒充 FP32。仍必须执行冻结容差数值门。

## 可执行 ABI

`prepare_attention_tasks_out(query_start_loc, seq_lens, slot_mapping, tasks, positions, query_heads, kv_heads, sink_tokens, recent_tokens, splits)`：

- starts / lengths 是 int32 `[R+1]` / `[R]`；slots 是 int32/int64 `[N]`。
- tasks 是预分配 int64 `[N*Hkv*3*splits,16]`，128B/row；positions 是 int64 `[N]`，padding 写 -1。
- 不依赖 host token 数值；二分定位 request。R≤4096。`qtile=floor(64/GQA)`；GQA≤16，所以 q_len4 一定共享历史 tile。
- tasks 前11列：`q_begin,q_count,kv_head,kv_begin,kv_end,request,output_split,source_kind,context,request_query_begin,metadata_status`。后5列置0。
- source0 为压缩历史；source1 为旧 BF16 sink/recent；source2 为当前 forward 的 BF16 K/V。`output_split=source_kind*splits+split`。
- q_count=0 为同querytile的非leader task；q_count=-1 为padding/非法metadata，输出行由该token自己的task初始化。带负slot的请求不因dummy seq_len=0触发有效请求错误；querytile遇内部负slot截断，后续valid run单独成为leader，避免两个task覆盖同一输出行。

`attention_cv_out(query, query_rot, current_key, current_value, rotation_v, raw, block_table, window_key, window_value, window_tags, tasks, partial, lse, status, workspace, block_tokens, physical_blocks, raw_ssm_offset, physical_page_stride, sink_tokens, recent_tokens, speculative_tokens, splits, scale, cube_cores)`：

- query BF16 `[N,Hq,D]`、query_rot FP32同形；current K/V BF16 `[N,Hkv,D]`；Rv FP32 `[D,D]`。D64/128/256。
- raw 为 uint8 flat native SoA allocation；block_table int32 `[R,columns]`，存原生virtual128页号。`virtual_block/(block_tokens/128)` 得物理页，余数和 `position%128` 得页内位置。
- window K/V 为 BF16 `[physical_blocks,S+R+spec,Hkv,D]`；后三维连续，允许页stride大于有效数据字节；tags int64 `[physical_blocks,S+R+spec]` 同理。binding 读取真实tensor stride，不伪设整张窗口连续。
- partial FP32 `[N,Hq,3*splits,D]`；LSE FP32 `[N,Hq,3*splits]`；status int32 `[task_count,2]`，每AIV独立写状态，无错误位丢失竞态。非0状态必须由 runtime 的设备异步断言/探针检查拒绝，不能忽略。
- workspace uint8 flat，至少 `cube_cores*(384*D+8192)*4` bytes；cube_cores∈[1,32]。同stream不做动态分配、主机值拷贝或同步。

所有partial均为原始 V 空间。source0在整段online-softmax完成后，只对64query行的结果作Cube `@Rv.T`；不对历史K/V逆旋转。随后可把partial展平为 `[N*Hq,3*splits,D]`，交给已有 `merge_lse_out`。

## 因果窗口与 query 复用

令旧context长度 C，当前query绝对位置 p，sink S、recent R。历史mask：`S <= key_pos < max(S,p+1-R)` 且 `key_pos<C`。旧精确mask：`key_pos<C` 且 `(key_pos<S or key_pos>=max(S,p+1-R))`。当前forward K/V按 `C<=key_pos<=p` 直接用BF16；它们尚未经过INT2往返。

在同一个querytile里，source0读取上述历史集合的并集，source1读取旧精确集合的并集，然后按每个query再mask。并集内部没有按query重新读取INT2；每份128-token工作单元供最多64个query/head行共用。目标TP4 Hq6/Hkv1、MTP4共有24行，完全位于单个querytile。prefill超过 `floor(64/GQA)` queries需要后续querytile重新扫描，这是精确attention的计算/带宽事实。

每次仅对128个KV tokens构建FP32工作单元，不随全历史长度分配。两个AIV各负责32个query row；在每个32-token子块里，各AIV读取并解包16个KV row，四个子块按行写入K `[128,D]`，按列写入转置V `[D,128]`，每个INT2 slot只读取一次。SIMD Gather提取16bit编码字，8次Shift/And生成bit planes，再Gather还原维度顺序；scale/zero只在每向量应用，数据映射与PR保持一致。四个子块全部发布后，Cube只做一次QK和一次PV，Vector只做一次128列FP32在线softmax更新，跨核握手由每32列一轮变成每128列一轮。末尾不足128列仍写入K/V和P的零尾、每query精确因果mask与原始LSE/输出owner。

GM scratch的活跃区间明确互斥：Q `[64,D]`，K `[128,D]`，V `[D,128]`，score/P `[64,128]`，PV/最终`Rv^T`结果 `[64,D]`。QK的CubeReady在Vector覆盖score为P之前；PV读P只在VectorReady之后；PV被Vector累加后才用于最终旋转。K与V、score/P与PV区间不重叠。GraphWorkspace及C++ ABI用相同字节公式，不能让单独算子调用方传旧尺寸。

Softmax统计全程FP32：每row有running maximum与denominator，旧acc按max变化重标度，Cube完成 `P@V` 后累加。每个空tile概率置0；整段全空输出0/LSE-inf。Log采用不同输入/输出buffer，避免官方CPU-debug已发现的默认非原地API约束。

## D.4 四问与资源上界

1. **相位**：tasks对应prepare；source0替代dequant+FIA；source1/2对应精确FIA部分；最终已有merge只处理有界partial。
2. **历史失败**：D.4主因是全历史恢复独立路径导致6.5s；prepare达到725ms；phase1_stores209–216ms与本读kernel分开验收。
3. **结构规避**：无 `[L,D]` FP32/BF16历史tensor；固定128-token工作单元、内部四个32-token子块，GQA/MTP复用；QK/PV是真Cube，SIMD解包是真Vector；当前chunk直接BF16；host只传固定属性和指针。旧全历史读/恢复路径没有出现。source0的INT2字节数不变，K/V bounded GM通信仍为Θ(LD)。
4. **量级**：16K首次prefill的current源每FULL层静态Q×KV工作单元从420761个32列变为105805个128列，Cube QK/PV调用和跨核flag轮次约减四分之三；每query的点积与历史字节数没有减四分之三。目标是short decode对照0.6–1.1ms、32K attention对照18.5–18.9ms，必须同硬件实测。这些数字不是本kernel已达到的性能；#140的12.4倍端到端差距也不能由静态计数推断已收敛。

D256每Cube workspace=425984B，20Cube共8519680B（8.125MiB）、32Cube共13631488B（13MiB），和L无关。单AIV显式UB分配约173312B（原161024B加score buffer 12288B；含两个4096-entry Gather索引），小于192KiB；编译器内部资源和实际容量必须由CANN编译确认。`TCubeTiling`由SDK编译期算法推导，不手填未经验证的字段。Matmul内部Cube L1/L0分配另由SDK静态tiling检查。

A2 Vector→Cube通信使用原生precedent的固定GM buffer，**没有把UB→GM→L1称作直接片上通路**。压缩源每token/head读136B(D256)；每tile仍写/读FP32 K/V通信，总通信量是Θ(LD)，只容量有界。严格H18“历史读取对L亚线性”与任意精确full attention冲突，本实现不声称满足o(L)，不以此停止其它可实现工作。

当前通信是单buffer、显式阶段flag协议。它优先建立真实数值与无竞态基础；没有把源码中的CV分工当作性能证明。后续若profiler显示通信/解包/softmax吞吐不足，必须在相同冻结精度下优化流水，不可增大全历史workspace或降低精度。

## 验证入口

- 本地 `tests/test_cv_contracts.py` 有3个源码/ABI硬闸；6个真实NPU用例默认skip，设置 `OSCAR_RUN_NPU_TESTS=1` 仍要求本任务devices已选择且环境完全一致；不以CPU替代。
- `tests/test_cv_contracts.py <golden_dir>` 生成独立dense PR oracle golden：D64/Q1、D64/Q4 history/window/current、D64/Q4首次prefill、D128/Q4、D256/Q4物理页边界。包含非identity Rk/Rv与Hq6的实际GQA。
- `csrc/cpu/cv_probe.cpp` 使用官方 `ICPU_RUN_KF` 执行生产同一kernel body：prepare AIV→CV MIX→merge AIV，验证所有status与partial写入，对照固定5e-3输出/LSE。官方CPU-debug仍然不能替代NPU/graph/performance。

编译和CPU-debug的当前执行结果由主流程写入本轮reports/checklist；未执行的设备完成、图捕获、图重放、32K性能必须继续为unverified。

## 后续 MTP draft 的真实位置

再次核实原生 `llm_base_proposer.py:1893–1979`：`prepare_inputs_padded`保留拒绝token，明确不修seq_lens；后续draft的`attn_update`仅对该长度+1，但slot_mapping根据接受后的实际位置从virtual128 block table重新生成（`:1608–1664`）。因此draft_index≥1不能继续把`seq_lens-1`当真实context，也不能用mRoPE某轴替代逻辑token位置。

prepare ABI追加可选`block_table=None,use_slot_context=False`。默认主模型/首个draft不变；后续draft传真实BT并设true，要求该请求qlen=1。设备端匹配 `slot/128`，得到唯一logical column后计算`position=column*128+slot%128`作为context与positions输出。扫描范围限制为`min(table_columns,ceil(seq_lens/128))`；原生过估计长度仍提供位置上界，未用capacity列的0/陈旧页号不参与匹配。匹配缺失、重复、位置越过seq上界或qlen不为1均写metadata/status5并禁止发布位置。没有CPU位置回传。

官方CPU-debug同一kernel body已验证：两个请求native长度134/260对应slots导出真实位置130/257；重复页与缺失页均写status5，且先前dummy请求与slot孔洞测试继续通过。日志`logs/cv-cpu-debug/slot-context.log`；它仍只是CPU-debug证据。

## 本轮 CPU-debug 发现与修正

扩展错误注入发现CPU Cube对NaN输入的行为不足以保证从score检查捕获非有限值。现在Q、精确K/V在进入Cube之前先做Vector FP32有限性检查；坏query/current value写status2。FP16 metadata只有真实有效tile row可判scale>0，padding lane的0 metadata不误报；真实scale0写status3。缺失window tag写status4。所有错误分支继续完整CV flag协议后发布错误，不留下Cube等待。该问题来自官方CPU-debug，不冒充真机错误写入故障档案。

最终官方CPU-debug报告为 `reports/cv_cpu_debug_final.json`：12个kernel端到端病例通过（8个正确输入对拍，包括Q17跨querytile与Hkv2/GQA6；4个非法输入状态检查）。每个病例还执行上述5个metadata契约。报告中的`npu_acceptance=not_run`保留；没有借CPU-debug宣称NPU、图或性能验收。
