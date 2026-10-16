# OSCAR 历史 attention BF16 候选执行方案

初稿日期：2026年10月15日。方案基准：`oscar-new-3`，提交 `f96820f154f62686c319edba5ac31e52a20bad4e`。2026年10月16日开始执行，代码母源为 `8cd23d2840e1b6a535fa315bccf27e42592b5b12`；两者之间仅新增本方案，算子母源未变。

**2026-10-16 最新执行状态：按用户“一次改完、不分阶段”的指令，一次补齐六路研究算子、typed workspace、C++ 绑定、可选构建、Python 路由、统一开关、独立 NPU/图探针和 observe 接线。** 当前分支为 `codex/bf16-history-implementation`，本轮母源 `6029f2a8c6277ab8136e5a5eeaa2c2c16313ca08`。下文原阶段编号仅作为依赖/验收检查项，不再作为分批开发或交付边界。

**代码已提供不等于数值合格或生产放行。** 已保存的 [CPU 数值报告](../reports/bf16_history_stage1_20261016.json)仍是 11 个旧 FP32 对照通过、BF16 组合 7 例失败/4 例通过，其中 2 例合并输出也失败。没有修改 oracle、容差或失败样本。上一轮 `6029f2a` 仅有 CPU screen/拒绝配置的历史状态保存在 [原交付回执](../reports/bf16_history_implementation_20261016.json)，不能把该旧回执当成本轮文件清单。

失败的解析反例仍成立：identity 旋转、Q=0、context=1023 时，第 0 维历史 INT2 重建的 352 对 `2048`、`-2048.0625` 总和为 -22；转 BF16 后正负相消，精确窗口/current 同样相消，最终参考为 `-22/1024=-0.021484375`、候选为 0，超过约 0.00510742 的冻结容差。FP32 累加无法恢复输入 Cast 丢失的信息。该结论不是 NPU 或模型质量实测。

新实现以独立符号提供；`tools.build_ops --bf16-history` 才编入六个新能力，默认 CMake 选项关闭。`research_variant_config` 校验研究依赖，生产 `variant_config`、CLI、deploy、runtime 仍拒绝已知未合格的 True，没有环境变量绕过，也不会静默调用旧核。两个固定服务入口保持原合格组合；本轮不能宣传成已启用的 BF16 优化。

当前主机没有 CANN、Lima 或 torch_npu。[工具链尝试](../reports/bf16_history_toolchain_20261016.json)实际 rc=1；[真实 probe 的本机尝试](../reports/bf16_history_no_npu_delivery_20261016.json)实际 rc=2，明确禁止 CPU 替代。新内核未经过设备编译、官方 CPU-debug、NPU、graph 或服务验证。尤其 Matmul `End → PIPE_ALL → TPipe.Reset → Init` 的重初始化契约仍需当前 SDK/设备验证，不能仅以原生存在 Reset API 作保证。

相关主机回归按测试项去重并采用最近复验，结果为 **468 通过、8 跳过**；其中最终完整路径/独立 probe 的复验为 [76 项通过](../reports/bf16_history_full_pipeline_host_20261016_v2.xml)。原 AST 依赖注入错误和新增测试夹具共享可变状态的失败均保留原始 XML，修复未放宽任何数值断言。另执行全仓静态检查，结果为 2 通过/3 失败；三个首报触发文件均与本轮母源逐字节相同，详见[静态检查边界](../reports/bf16_history_static_baseline_20261016.json)。本轮新增源的档案头及无静默吞错检查已过；不能据此宣称全仓静态检查通过。

本轮文件与验证入口：

| 部件 | 本轮实现/证据 |
|---|---|
| 六条数学路径 | `attention_bf16_history_mdl.cpp` 提供 q4/q1；另有 C4 member、C16、range、balanced 四份独立源 |
| 派生可审计性 | [生成器](../tools/experiments/generate_bf16_history_kernels.py)固定 5 个母源哈希；每个替换逆向还原母源；[资源及身份回执](../reports/bf16_history_kernel_sources_20261016.json) |
| 工作区/绑定 | [launch header](../csrc/include/oscar_bf16_history_launch.h)、[绑定](../csrc/bf16_history_bindings.cpp)；旧 ABI/验证模板保留，全部容量显式放大 |
| 构建/运行接线 | 可选编译清单、实际 capability、Meta、runtime、六项显式 selector 映射、clusterStats 参数同步 |
| 新 probe | [probe_bf16_history.py](../tools/probe_bf16_history.py)：30 个固定算子诊断病例、完整原空间 partial/output/LSE/最终 BF16、source1/2 逐位、错误 lane、同址改输入图回放 |
| 完整调用链门 | [bf16_history_pipeline.py](../tools/bf16_history_pipeline.py)：另设六个与真实 selector 一致的形状，按实际 Cube 核数/工作区推导 split；执行 prepare、适用的 source2 suppress、rotate、CV、current slot guard/native FIA/partial 写入、merge/copy、striped store、冻结 status guard |
| 不扩大结论 | CV、CV+merge/copy、完整 eager 调用链分别计时。后者使用合成物理页、实际目标 head 几何和独立输入/任务绑定，不冒充模型真实物理布局、完整图路径、服务性能或模型质量 |
| 回归/验收边界 | [本轮总回执](../reports/bf16_history_all_code_20261016.json)；保留首次回归失败和修复复验，42 个 checkpoint 不重排 |

研究构建与设备诊断入口（仅在已有 CANN/torch_npu/NPU 的环境执行；不是真机默认服务的新前置步骤）：

```bash
python3 -m tools.build_ops --soc ascend910b4 --bf16-history
python3 -m tools.probe_bf16_history --config /path/to/research-target.json --output /path/to/bf16-report.json
```

研究配置在当前 target 的副本中显式增加 `"experimental_bf16_history": true`；probe 用统一 candidate 依赖规范化，仍核对本任务设备和真实模型几何。`configs/target.json` 本身不改。该配置不能启动生产服务；实际执行/数值失败保留在报告，不放宽门或改默认以促成启动。

计时有全局前置条件：完整 CPU 数值集、30 例算子数值/协议/图检查及六例完整调用链数值/状态/写后字节检查全部通过，才可运行任何局部或完整计时。完整调用链的 C16 使用真实路由阈值所需的 q8192，balanced 使用能实际选中该路由的 padding；不把小算子诊断的形状/split 冒充生产选择。完整路径两臂使用各自最大 C16 arena，原空间 oracle 按 query 分块覆盖全部实际行；每次在计时外同址恢复初始缓存，计时内保持先读后写。当前 CPU 门已失败，所以这些计时代码存在但不能产生通过证据。

目标是保持 INT2 缓存及现有服务协议，研究将历史 attention 的矩阵操作数改为 BF16，累加、softmax、逆旋转和合并保留 FP32。先隔离数值影响，再实现独立算子，最后接入统一服务入口。BF16 会增加舍入，功能协议保持与数值误差合格必须分别证明；不能承诺任意输入逐位不变，也不能用短探针证明完整模型质量不下降。

档案约束：G4、#81–83 对应 SOC/BF16 编译；#126/#145/#154 对应数值、异步缓冲区所有权及 CPU/NPU 证据边界；#148 对应独立候选与模型质量；#153 对应 capability 登记；#157–163 对应冻结 guard、MTP discard/位置和 prefill 状态协议。条目按标题定位，禁止沿用旧条数和行号。启动文档 D.4 全文在本轮分析中已读；实施人员涉及 AscendC 时仍须亲读并在新核文件头回答四问。技术改法依据下列可执行源码，档案仅提供约束和回归来源。

## 1 基线与适用范围

用户确认的无 OSCAR 基线是以下命令。仅将粘贴文本中 `method` 值缺少的双引号补齐，不改变其余参数：

```bash
vllm serve /softwarePlatform/models_directory/Qwen3.8-27B \
  --served-model-name qwen3.8 --host 0.0.0.0 --port 8011 \
  --data-parallel-size 1 --tensor-parallel-size 4 \
  --max-model-len 262144 --max-num-seqs 128 --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_capture_sizes":[1,4,8,12,16,24,32,48,56,64,72,84,96,108,112,128,160,172,196,200,212,232,272,288,312,328,344,360,384,400,416,432,448,480,512],"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative_config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  --additional-config '{"enable_cpu_binding":true}' \
  --hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}' \
  --trust-remote-code --async-scheduling --allowed-local-media-path / \
  --mm-processor-cache-gb 0 \
  --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16
```

两个 Mamba 参数控制 conv/SSM 状态缓存，不决定 FULL attention 的乘法和累加类型。FULL 模型 dtype 仍为 `auto`，KV dtype 还需经过模型配置解析。原生 MTP 场景在当前参考树进入 FIA，接口前没有将 Q/K/V 升为 FP32；FIA 内部实现不在当前参考树，不能声称基线全程 BF16。[原生 dtype][S1]、[Mamba 状态 dtype][S2]、[FIA 调用][S3]

原生 arch22 sparse attention 提供 BF16 A/B、FP32 C 的 Mmad，以及 FP32 softmax 后将 P 转为低精度的实现先例。这证明硬件路线可研究，不证明本项目 MatmulImpl 改型已经编译或设备验证通过。[原生矩阵实现][S4]、[原生 P 转换][S5]

当前工程配置仍指向 Qwen3.5，SOC ascend910b4、APC 关闭、预算 16384；本方案不自动更换模型。后续按用户 2026-10-16 启动配置要求，默认物理卡改为 4–7、端口 7878，既有 `--rear-cards` 保留兼容；这是设备分配变更，不改变 BF16 实现或验收门。性能配对必须使用同一模型/权重/旋转及一致的实际调度设置；不能把两个模型或不同预算、APC 状态的结果直接比较。[当前配置][S6]

第一实现目标为当前 striped D256 路径。实际 Qwen3.8 的架构、每 rank Hq/Hkv/D、dtype 和加载配置必须由既有短 probe/启动信息核实，不能由目录名或旧 Qwen3.5 的 Hq6/Hkv1/D256 推定。此为实验绑定，不新增默认部署环境审计或原生源码扫描。

## 2 必须保持的功能契约

| 对象 | 保持要求 | 验证方法 |
|---|---|---|
| 持久化 KV | INT2 位序、striped 布局、FP16 scale/zero、物理页与 metadata128 映射不变 | 相同输入 raw 字节比较，跨页与尾页验证 |
| 量化写入 | 旋转、裁剪、舍入、写入范围、错误码不变 | 使用现有 writer 与原语门，不修改期望 |
| source1 精确窗口 | BF16 原值、tags、sink/recent/speculative 范围和原计算保持 | 单独比较 source1 partial/LSE/status 与原核 |
| source2 current | 原生 current FIA、首 draft current-only、CV current 的选择及计算保持 | 路由回归与单独 source2 比较 |
| source0 历史 | 请求/head/split 范围、mask、历史读取及复用关系不变 | task 全表、计数、边界和独立 oracle |
| MTP | 3 个 speculative tokens；提交/拒绝/discard/逻辑位置不变 | later draft q1、混合、全丢弃/部分丢弃、后续步验证 |
| GDN | 原生算术、缓存 dtype、状态更新和现有外部接缝修复保留 | 原生接缝回归，尤其 warm qLen4 prefill 后的 decode |
| APC | 当前默认关闭；显式开启功能仍可用 | 两态、前缀分叉、释放复用、共享页快照 |
| 图 | 原捕获资格及固定地址契约保持 | 同地址修改输入/任务/物理页后回放 |
| 状态与异常 | 旧非法域的错误码、lane 归属和优先级保持；新数值异常不得吞掉 | 故障注入与非有限值分类，不接受仅“任意报错” |
| 冻结集 | FE0_KERNEL_SHA256 中全部文件字节不变 | 沿用现有冻结核检查，禁止更新哈希迁就改动 |
| 部署 | 两个 candidate 入口共用开关；原探针、日志、退出码、清理保持 | 入口配置到真实 op 的贯通测试 |

冻结集包括 `attention_cv.cpp`、`attention_tasks.cpp`、`store_int2.cpp`、`rotate_clip_store.cpp`、`merge_lse.cpp`、`status_guard.cpp` 和 `oscar_common.h`。不得改 guard、插入生产回读或通过放宽校验归因；既有 `OSCAR_GUARD_HOST_SYNC=1` 诊断方式保持。[冻结集][S7]

## 3 数值契约先于实现

### 3.1 区分重建与矩阵操作数舍入

本方案保留 FP32 反量化重建 `code * float(scale16) + float(zero16)`，随后在新 attention 算子内部生成 BF16 矩阵操作数；不提供替换现有接口的 BF16 反量化输出，不改变独立 oracle。原 FP32 反量化仍按 `store_dequant atol=0.002, rtol=0.002` 验证；新混合精度 attention 的原空间 output、LSE、merge 后 output 和最终 BF16 输出按冻结 `fused_attention atol=0.005, rtol=0.005` 验证。精确窗口相关门 `relative_l2 < 0.02` 继续执行。[冻结配置][S8]、[oracle][S9]

这两个门不能互相冒充：FP32 重建通过，不能把末次 Cast 后的 BF16 数值标称为已通过原反量化门；Cast 的新增误差必须单独报告，并由完整 attention 门验证其影响。如果要求“BF16 重建值本身也对全部合法输入满足原 store_dequant 门”，直接单 BF16 替换存在以下已知反例，应停止该定义下的生产替换，不自行放宽门槛。

合法 BF16 向量 `[16,16.5,17,18]` 重复组成 D256，现有量化得到 scale=`0.66650390625`、zero=`16`。FP32 重建值 `16.66650390625` 转成 BF16 为 `16.625`，误差 `0.04150390625` 大于冻结容差约 `0.035333`。该结果为本机 CPU 数值反例，不能写成 NPU 错误、完整 attention 失败或模型质量结论。

### 3.2 三类比较分别执行

1. 逐位保持：raw/cache 元数据、window/tags、准备后的 tasks、整数计数及未修改的 source1/2 输出。旧 FP32 算子的逐位门继续原样运行。
2. 允许浮点差异但须满足冻结门：新 source0 数学结果。比较对象必须是未改动的独立原空间 oracle；旧 FP32 同输入结果同时保留，用于定位新增误差，不能单独充当 oracle。
3. 同一新算子的 eager/graph/重复执行：沿用现有同输入逐位一致检查。若发现不确定性，先定位同步或原子归属，不擅自降为容差比较。

现有 MDL/C4 probe 明确要求旧候选与母源逐位一致。不得删除或改松这些断言，也不得把 BF16 新路径塞入原报告的 `bitwise_passed` 字段。新增独立 probe、报告 schema 和校验器，单独报告冻结 oracle 结果及误差。[旧 MDL 门][S10]、[旧 C4 门][S11]

### 3.3 模型质量边界

模型 logits、任务指标、MTP 接受率的阈值目前在 acceptance 中为 null。不得自行填写宽松阈值后宣布模型质量通过；不得把算子门、HTTP 成功或少量文本相同当作完整质量验收。复用既有日志和已保存输出，必要时仅做短合成服务对照；不要求重跑 AISBench。没有足够的质量证据时保留“质量未验收”，不能承诺性能优化不改变任何输出。

## 4 候选数学与数据流

仅在 source0 中执行以下变化，保持原 KV 顺序、缩放位置、mask、split 和归一化逻辑：

```text
Qrot32 = 既有 FP32 查询旋转结果
Krot32, Vrot32 = 既有 INT2 解包与 FP32 乘后加重建
Q16, K16, V16 = 各自一次经过验证的 BF16 最近舍入
score32 = BF16×BF16 的 QK，FP32 累加，再按原位置应用 scale
P32, max32, sum32, alpha32 = 既有 FP32 online softmax
P16 = P32 的一次 BF16 舍入
acc32 = alpha32 * acc32 + BF16×BF16 的 PV，FP32 累加
history_output32 = 既有 FP32 归一化及 V 逆旋转
output = 既有 FP32 LSE merge，再由既有出口转换模型 dtype
```

- scale/zero 保持 FP16 存储，重建保持原 Mul 后 Add 顺序，不改成 BF16 算术或未经验证的融合 FMA。
- Qrot 在一个 query task 内转换并复用；同一个共享 KV tile 只转换一次，保持 C4/C16 原有共享条件，不跨请求/层/target-draft 共享数值。
- max/sum/LSE、online alpha、累计 output 和逆旋转保持 FP32。softmax 分母用原 FP32 P 的和，不改为 BF16 P 求和。
- QK/PV 使用独立 BF16 A/B、FP32 C 的矩阵实例；V 逆旋转保留独立 FP32 实例。source1/2 仍使用原 FP32 算法。
- C4 对 FP32 accumulator 的原子累加、初值和每轮 rescale 顺序保持；BF16 不改变原子目标 dtype。
- Cast 必须核实目标 CANN 的合法向量重载及舍入语义；用中点、正负值和尾部用例验证，不能只凭 `CAST_RINT` 或 `CAST_ROUND` 名称猜测。不得用 reinterpret 替代数值转换。
- 不做全历史 BF16 恢复，不新增全历史张量、全局同步或生产 host 回读。

## 5 文件改动范围

下列新增文件名为计划名称，实施前不存在；修改既有文件仅限接线、容量和新门，不改变原实现的语义。

| 范围 | 计划动作 |
|---|---|
| 新增独立研究构建/实验入口 | CPU screen、派生源生成器和 `tools.build_ops --bf16-history`；不进入默认生产源清单 |
| 新增 `csrc/kernels/attention_bf16_history_mdl.cpp` | q4/q1 的新符号，保留原 source1/2；从当前母源建立可复核的语义差异 |
| 新增 C4/C16/普通及 balanced 候选文件 | 分别覆盖当前生产可选路径；每个核独立核对实际母源，禁止只复制算法说明 |
| 新增 `csrc/include/oscar_bf16_history_launch.h` | 明确新 ABI、各路径 typed 字节布局、容量计算和合法几何 |
| 新增 `csrc/bf16_history_bindings.cpp` | 独立 op 注册；沿用 dtype、shape、alias、geometry 检查，保留完整错误上下文 |
| [构建与 capability][S12]、[Python contracts][S13] | 同时登记编译源、绑定、导出符号与 capability，不伪置验证状态 |
| [runtime][S14] | 配置依赖、能力检查和预分配；原配置缺省值不变 |
| [selector][S15]、[impl][S16] | 在原资格选择结果上作显式映射，保持调用 ABI 和 clusterStats 参数选择 |
| [统一开关][S17] | `experimental_bf16_history` 初始默认关闭，先列研究项；通过发布门后才加入 candidate |
| 新增 `tools/probe_bf16_history.py` 及专用测试 | 新数值路径的独立 oracle/状态/图/性能门，报告不冒充旧逐位门 |
| [observe 入口][S18] | 在既有门外追加 BF16 门；失败实时输出并阻断服务，保留真实退出码 |
| 工程既有 checklist、优化台账 | 只回填真实证据，42 个 checkpoint 编号不重排；本机结果不新增真机档案 |

外部 ABI 保持：Q/current/window BF16；query_rot/rotation/partial/LSE FP32；raw/workspace uint8；tasks/tags int64；status int32。[当前 binding][S19]

## 6 一次性交付范围与验收依赖

### 阶段 0 固定母源与实验合同

记录当前提交、相关源文件/acceptance 哈希和实际选中路由。若母源已变化，只复核相关差异后重新绑定，不把旧版本实验套在新代码上。读取本方案引用的同类档案与 D.4 原始日志；新增 AscendC 文件回答 D.4 四问：对应 fused dequant+FIA；避开全历史物化；仅有界 tile 转换/复用；成本账含新增 Cast 与同步，历史 6.5 秒不能当作当前占比。

定义精度、状态、图、性能门与代表输入，再运行实验。旧候选、BF16 候选、无 OSCAR 原生是三个不同分母：增量收益相对当前候选，原生差距单独记录。正常部署的既有设备、模型、端口和参数不被实验改写。

完成条件：源与门可追溯，BF16 研究开关保持关闭。用户最新要求允许一次补齐受隔离保护的接线；这不授予生产启用资格。

### 阶段 1 先做 CPU 数值分解

新增独立实验脚本，复用现有独立 oracle 和 fixture，模拟 BF16 舍入后再升 FP32 做矩阵乘以分离输入舍入影响。该模拟不能代表 NPU 的真实累加顺序、性能或异步执行。

按固定输入分别测 K/V 舍入、Qrot 舍入、P 舍入以及完整组合；分项结果用于定位，只有完整组合参与最终数值判断。覆盖真实 writer 可产生的缓存、旋转 Hadamard 和已支持的普通正交旋转、attention 大小值与取消误差、长历史、GQA、多 split、窗口合并和边界。保留上节反例，不删失败样本。

所有实际 query/head 的 output/LSE 均比较原空间 oracle，不能抽样掩盖坏行。对诊断误差记录 max_abs、max_rel、超阈计数、首个坏位置和 source/split；相对误差分母为零时另报，不用平均误差替代逐元素门。

完成条件：完整组合通过冻结 attention 门，且原重建/缓存门保持通过。若失败，保留失败形状与误差来源，停止该候选的生产接入；允许继续独立研究，但不改阈值、不让线上按数据值切回旧核。

### 阶段 2 先完成工作区与内核资源设计

在编写设备读写和执行设备探针之前，为每个候选生成区域表：名称、dtype、字节偏移、长度、对齐、生产者、消费者、最后读取和复用事件。Python 分配公式与 C++ launch/binding 下限一致，按所有已启用路径的最大需求分配。

首版不主动缩小现有公共 arena；仅在证明新布局不超过容量后复用。若不够，明确增加固定容量并核算 KV 预算影响，禁止越界或偷偷降低可用并发。后续缩容是单独优化。

重点检查 score32/P16 不再共享同元素偏移，Q16 不能直接复用为 FP32 归一化结果；softmax/PV 临时空间仍须满足 FP32 使用，不能把原 dequant scratch 无条件减半。保留 MDL 双缓冲、C4 score 槽和 FP32 原子目标的存活期。

用实际 DMA 方向证明依赖：BF16 Cast 结束后才可 MTE3 发布；MTE3 读取 P16 的缓冲区未结束时不能被下一块 MTE2 或 Vector 覆写。参考 #145，不能以 MTE3_V 代替必需的 MTE3_MTE2。所有新矩阵实例的 L1/L0/事件资源及 End/复用关系也须单独验证。

完成条件：布局、同时存活资源、每次复用的必要依赖均有明确计算和审查；在此基础上才进入阶段 3。后续扩展 C4/C16 时先补齐其区域表，再编写该路径。

### 阶段 3 独立实现 q4 和 q1

先验证 A2 BF16 A/B→FP32 C 的 QK、PV 和 FP32 逆旋转可共存，随后实现完整 source0。保持原分工、owner、尾部和 source1/2 流程，不同时修改 tile 尺寸或调度算法。新增能力有独立符号，不覆盖现有 op。

必须重新计算 UB/L1/L0 的同时存活资源，两个 Matmul 实例可能各自占用资源；不因 BF16 元素变小就推断一定能装下。设置完整 shape/stride/transpose 和 K 尾部；确保 BF16 padding 为精确零，不遗留 poison。

先执行源码/ABI 检查、CANN ascend910b4 编译及必要官方 CPU-debug，再执行真实 NPU。编译或 CPU 通过仅解锁下一步，不等同设备通过。设备探针无 NPU 时必须明确失败。

完成条件：q4/q1 在所有三个 source、merge 和最终 BF16 copy 的完整路径通过冻结数值、错误域与设备完成门。

### 阶段 4 分别扩展 C4 C16 和剩余路径

当前路由必须逐项覆盖，不更改原选择资格：

| 原选择结果 | BF16 候选要求 |
|---|---|
| `attention_cv_mdl_decode_out` | target q4 及既有可支持 q1–q4 shape |
| `attention_cv_mdl_decode_q1_out` | later-MTP q1，保留 padding 和 slot-context |
| `attention_cv_member_pipeline_cluster4_out` | C4 共享历史、双 KV/score 流水、FP32 atomic accumulator |
| `attention_cv_window_range_cluster16_out` | 长 prefill C16，独立状态和完整共享条件 |
| `attention_cv_window_range_out` | 普通路径，包含不能使用 M32 的合法几何 |
| `attention_cv_window_range_balanced_out` | 混合/大 padding 的 balanced 路径 |

首期实验可以只有 q4/q1；此时只在独立 probe 直接调用，不能宣称生产全覆盖。发布到统一 candidate 之前，生产选择表中可能到达的分支均须有完整实现与测试，不能遇到缺失映射静默返回旧核。

六项映射只在完整母源组合下闭合。启用新开关必须同时满足 `experimental_history_reuse`、`experimental_fast_unpack`、`experimental_mixed_cv`、`experimental_striped_cache`、`experimental_decode_bundle`、`experimental_mdl_decode`、`experimental_c4_member_pipeline` 全部为严格布尔 True，且 request-window 研究格式关闭、实际 storage format 为本方案支持的物理页快照格式。variant=None 的显式研究配置也必须检查相同依赖，不能只依赖 candidate 自动置位。

保留原 selector 先计算资格，再使用包含全部六项的映射；新模式下原结果不在表中必须明确报错，禁止 `.get(old, old)` 静默返回旧算子。以逐项关闭依赖和不支持 storage format 的主机测试证明在启动/配置阶段拒绝非法组合。baseline/native 清除新开关，未开启新开关时原 selector 返回值逐项保持。

现有 D64/D128 和可选 request-window 研究格式保持原配置下可用。新 BF16 开关首期限定 D256/当前物理页快照格式；对未支持组合在启动能力检查处明确报错，不污染旧模式，也不在运行异常后自动回退。需要扩展这些域时另行完成相同验证。

完成条件：路由覆盖表与实测实际 op 对应，C4/C16 数值、所有权、错误 lane 及 clusterStats 保持，未覆盖域显式标记。

### 阶段 5 图协议与服务接入

图测试固定地址，依次更新 Q/QR、raw/scale/zero、block table/tasks、窗口和页映射，再与同一新核 eager 及独立 oracle 比较；连续两轮还要覆盖行数变小/变大、空历史和全 padding。捕获期间不得新增 host sync、临时全历史 Cast 或按设备数据选择 dtype。

图部分完成条件：容量和地址契约通过，真实 NPU graph capture/replay 分别通过，旧模式地址/容量行为不被意外改变。

新开关由统一 variants 模块维护，原配置缺省关闭。用新配置依赖检查、capability、selector 映射、CLUSTER_CV_OPS 等算子分类集合和实际绑定一起接入；不能只完成符号注册而漏传 clusterStats。

保持 current-only、首 draft FIA、mixed 分流和 later-MTP prepare 接缝。特别验证 none+MTP 的 warm qLen4 prefill 保留真实 prefill 分类并走既有 eager 选择，下一个普通 decode 能恢复既有图路径，不因新算子路由重引入 #163。

先通过下表必要主机/设备/短服务回归，再运行完整相位门。安装、编译、真实原语/CV/旋转 probe、旋转文件、服务 probe、资源清理的现有顺序与失败阻断规则保持；本轮不重新引入 readiness 前置审计或原生源码扫描。

### 阶段 6 配对收益与发布

增量配对以当前实际路由为母源，例如新 q4 对 MDL q4、新 C4 对 member pipeline；不能用更早且更慢的实现作为主要分母。分别报告历史 CV 算子与生产完整 attention 路径，不混用计时范围，也不能只测乘法或 helper。

在采用 native_current 的 prefill 路径中，CV 的 source2 区间被清空，随后在 CV 外执行原生 current FIA 并写入 partial；现有 C4 probe 也会清空 source2，因此仅计 CV 不能代表完整 attention。[生产调用顺序][S20]、[C4 probe 的 source2 处理][S21] 生产完整路径计时必须覆盖实际适用的 prepare、current source suppression、rotate、CV、current slot guard、原生 current FIA 及 partial 写入、merge、最终 dtype copy、store 和 status guard，以及新增 Cast/同步。保留 read-before-write 顺序；诊断用主机同步的开关和计时边界在母源、候选间保持一致。当前 source2 在 CV 内计算的路径则计入该计算，不能漏算或重复计算 current。

沿用 warmup 2、repeats 5、median 和逐 case 原始样本记录；单项完整相关路径 `t_old/t_new >= 1.15` 才获得性能组合资格，保留现有禁止退化的冻结要求。病例和形状在测量前固定，不删慢形状、不四舍五入放行。组合收益另测，不拼乘单项倍率。

对 D256/KV256，单份 K/V 展开载荷由 524288 B 降为 262144 B；它仅描述 K/V tile，不代表总 workspace、HBM 流量或延迟减半。FP32 状态、新 Cast、两个矩阵实例和同步均计入成本。

同模型原生短服务配对记录实际 dtype、预算、APC、capture、MTP 和 batch；历史完整 AISBench 只作原有证据，不重跑、不把短服务结果冒充完整质量或 8 分钟性能验收。

默认推广必须同时满足：冻结数值门、功能协议、真实设备、图、完整相关路径收益，以及预先确定的模型质量判据和对应证据。当前 model_quality 阈值为 null；在判据未确定或证据不足时，新开关保持默认关闭，只交付可测试的研究实现和分项证据，不提升到统一 candidate 默认组合。不得由执行者自行选择宽松质量标准，也不以 CPU 成功、短文本相同或局部速度替代质量资格；本方案不要求重跑 AISBench。具备推广资格后，再一次性完成两个固定入口的默认启用，不能只更新 probe。

## 7 最小必要回归矩阵

避免机械全笛卡尔积；从现有 fixture 选择能区分风险的病例，并保证每项至少有一个真实触发用例。

| 风险 | 必须执行的病例与判断 |
|---|---|
| BF16 舍入 | 中点/正负/零/极小/大有限值；原 FP32 有效输入不得因转换新增未解释的 Inf/NaN |
| 数值完整性 | identity 诊断旋转、真实 Hadamard、已支持普通旋转；完整原空间 oracle；合并前分段与合并后均检查 |
| q1–q4 | target 和 later draft，GQA6 及新核声明支持的其他比率；活跃少但 padding 多 |
| C4/C16 | 资格阈值两侧、共享域相同/不同、多 KV 轮次、多个请求/head/split |
| UB 复用 | 触发第二个 softmax 块、双 AIV、双 buffer 和 C4 原子轮次；重复 poison 后运行 |
| 页与尾部 | virtual128、KV256 边界及 ±1，非连续物理页、窗口跨页和物理页复用 |
| 全空与死行 | 空历史、all-padding、全 mask、合法死尾垃圾；输出和 LSE 哨兵/状态完全写全 |
| 错误域 | 活跃 scale=0/NaN/Inf、非法 zero、Q/QR 非有限、坏 tag/slot；精确错误码/lane/优先级 |
| 不变 source | 同输入 source1/source2 partial/LSE/status 逐位比较；原生 current 单独路由验证 |
| MTP | 首 draft、later draft、部分/全部 discard、位置修正；不能从 slot 失败反推 discard |
| GDN 接缝 | 16384/16380/4 的 prefill 边界与后续 decode，none/align 已支持模式 |
| APC | off/on、完整/部分命中、共享后分叉、释放复用、reset；快照与 tags 不变 |
| 图 | 固定地址换值/页/任务，replay 与同核 eager 逐位，且每态均过原空间 oracle |
| 历史长度/并发 | 短边界作数值诊断；按 acceptance 长度、q_len、并发范围取必要合成覆盖，区分算子覆盖与实际模型覆盖 |
| 服务回归 | TP4、MTP、混合请求短服务完成；实际 op 命中、返回结果、worker 退出和 NPU 资源释放 |

本地已有虚拟环境时可运行下列直接相关主机测试，之后按新增测试文件补充；这是实施时的验证命令，不表示本轮已经执行，也不是目标机启动前置条件：

```bash
.venv/bin/python -m pytest -q \
  tests/test_serving_variants.py \
  tests/test_mdl_decode_dispatch.py \
  tests/test_c4_member_pipeline_dispatch.py \
  tests/test_decode_bundle_routing.py \
  tests/test_integration_contracts.py \
  tests/test_compact_later_mtp.py \
  tests/test_mtp_small_shape_flow.py \
  tests/test_mtp_logical_slots.py \
  tests/test_prefill_state_contract_hooks.py \
  tests/test_prefix_caching.py
```

## 8 发布和失败处理

发布前回归固定入口实际启用新候选，服务启动摘要应能看到开关，probe 的实际调用记录应能证明 q4/q1/C4/C16 和剩余合法分支的 op 名；不能仅从编译清单推断启用。

```bash
git pull --ff-only && bash scripts/install_observe_serve.sh --variant candidate
```

observe 保留全部既有必要门并追加新门，任何失败都阻断正式服务、实时打印异常与真实退出码，清理本次创建的资源。真机已通过所需门后，直启仍为：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

直启只安装/编译并启动，不增加测试，也不产生验收证据。按用户 2026-10-16 最新要求，默认配置已为后四卡 4–7/7878，无需叠加 `--rear-cards`，该参数仍保留兼容。合法功能保持不依赖关闭 MTP/APC、减少上下文/并发、抬预算、加超时或更换调度器。

新门失败时保存母源/候选哈希、输入绑定、首错和真实退出码，停止候选推广。若需要恢复已知版本，在服务外显式撤销本次独立提交并重新安装；禁止运行中捕获错误后静默切回旧核或原生 attention。不得覆盖原生源码，不使用 cp 部署。

只有新真机错误才追加档案；本地失败、编译失败、CPU 反例和理论分析分别存实验报告。代码和报告最终按已有授权只推送本工程，保留真实验收边界。初稿交付仅新增方案；后续代码执行状态见本文开头，研究代码推送不等于候选默认推广。

## 9 交付证据与完成定义

每条路径交付一行状态，区分：源码接入、CPU 数值、CANN 编译、设备完成、图捕获、图回放、服务协议、模型质量、增量性能和原生差距。未运行填未运行，不把 null、skipped、health 或 HTTP200 写成通过。

报告至少包含：母源和候选提交/源码哈希、acceptance 哈希、模型与配置绑定、实际 op、完整形状/源区间/任务数、逐位保持项、独立 oracle output/LSE 误差、状态/lane 比较、图地址与变更输入哈希、workspace 字节、原始计时样本、失败码和资源清理结果。

本方案原定的全部验收完成条件是：独立实现覆盖所声明的生产域；不变契约证据齐全；新数值路径通过冻结门；获得生产资格后两入口能传递同一显式研究配置并选择实际新算子；冻结集、oracle、acceptance 与原生参考树均未被改写。用户最新要求下，以上各部件代码一次提供；已知数值失败和缺少设备工具链意味着全部验收条件尚未满足。模型质量与完整性能结论需要各自证据，不能由代码提供自动推出；candidate 默认启用另须满足阶段 6 的全部推广条件。

[S1]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/references/vllm/vllm/utils/torch_utils.py:375
[S2]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/references/vllm/vllm/model_executor/layers/mamba/mamba_utils.py:87
[S3]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/references/vllm-ascend/vllm_ascend/device/device_op.py:92
[S4]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/references/vllm-ascend/csrc/attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_cube_mla.h:792
[S5]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/references/vllm-ascend/csrc/attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_vector_mla.h:675
[S6]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/configs/target.json
[S7]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/tools/observe_serve.py:29
[S8]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/configs/acceptance.json
[S9]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/oscar_ascend/ops/reference.py:80
[S10]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/tools/probe_mdl_decode.py:163
[S11]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/tools/probe_c4_member_pipeline.py:84
[S12]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/csrc/CMakeLists.txt:45
[S13]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/oscar_ascend/ops/contracts.py:12
[S14]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/oscar_ascend/runtime.py:117
[S15]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/oscar_ascend/ops/cv_dispatch.py:49
[S16]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/oscar_ascend/integration/impl.py:175
[S17]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/tools/serving_variants.py:35
[S18]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/tools/observe_serve.py
[S19]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/csrc/striped_attention_bindings.cpp:43
[S20]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/oscar_ascend/integration/impl.py:158
[S21]: /Users/loki.yao/Documents/workgit/gpt_new_oscar_kimi_2/tools/probe_c4_member_pipeline.py:66
