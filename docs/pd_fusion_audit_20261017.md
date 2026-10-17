# PD 混合调度源码复查与本地修复（2026-10-17）

本轮母版本为 `b5d2255e29372ac3da599fc73cbccca29fd4068a`，分支 `oscar-new-3`。PD 指同一服务实例内 prefill/decode 混合调度，不是跨实例 PD 分离。本轮根据可执行源码判断，不以注释或 README 证明实现。档案关联：#34/#36 的验证必须执行真实路径，#129/#140/#142 的异步混合与 MTP 接缝，#148 的冻结精度，#153/#155 的原生预算和 current/partition 组合，#157–#163 的 discard/slot/guard，#164–#165 的 BF16 编译及未归因 Trap。本轮没有新真机事件，不新增档案编号。

**结论：修复了三个探针共用的 BF16 工作区分配和相关验证缺陷；714 项本地主机测试通过、1 项重复大形状跳过。没有新增已证实的生产 Attention/调度正确性缺陷，也没有证据宣称生产无 bug、32K Trap 已修复、BF16 精度合格或吞吐提高。** 正式 RuntimeProvider、CANN 核、原生代码、配置和冻结门均未修改。

## 1. 当前混合执行路径

源码入口：[原生请求重排](../references/vllm/vllm/v1/attention/backends/utils.py)、[OSCAR forward](../oscar_ascend/integration/impl.py)、[current-only 分流](../oscar_ascend/integration/current_only_dispatch.py)、[mixed 分流](../oscar_ascend/integration/mixed_decode_split.py)、[CV 选择](../oscar_ascend/ops/cv_dispatch.py)。当前配置关闭 APC、预算 16384、max-num-seqs 128；继续使用原生 scheduler。本轮不修改设备或端口配置。

| 场景 | 实际路径 | 必须计入的成本 |
|---|---|---|
| 没有历史的 fresh prefill | current-only suffix → 原生 current FIA → copy/validate → INT2 store | 无历史 CV，但任务准备、guard、拷贝校验、量化写缓存仍存在 |
| 有历史的 continued prefill | history/window BF16 CV，current 用原生 FIA partial，再 FP32 合并与 store | 历史读取/解包、Q 旋转、QK/PV、softmax、逆旋转、partial/LSE 合并 |
| 短前缀 ≤128 token，后缀满足长查询条件 | 按请求边界拆两次 FULL forward；短前缀 MDL，长后缀按形状选 C4/C16 | 增加 boundary guard/launch；两段依次执行，共用工作区，保留原 batch 的 split 选择 |
| 短前缀 >128 token ＋长 continued prefill | mixed planner 不拆分，整个 batch 按长查询形状选择 CV | 如 33 个 q4＋长 P 会共同走 C16；短请求不能单独选择 MDL |
| decode ＋fresh prefill | current-only 优先；有历史前缀独立 forward，零历史后缀用 current FIA | fresh 后缀不能吞入任何正 context 请求；前缀可能再满足 mixed 分流条件 |
| first/later MTP | first draft 保留 request 域；later draft 按既有 discard、logical-slot、compact 协议 | target 与 draft 的投影/缓存不同；不能按相同位置共用 K/V；compact 不改变 split 归约域 |

原生重排顺序为 decode → short extend → long extend → fresh。新增测试执行仓内原生重排函数本体，再执行 OSCAR metadata/planner；六类请求的 **720 种排列 × main/first-MTP 两种身份 = 1440 次**检查均保持请求归属、context/current 区间、virtual128 block table 和 slot 一致。Batch 的 swap callback 是明确声明的测试夹具，不代表完整 native InputBatch 或设备执行。

源码没有支持“first draft 必然携带过期累计长度”的反例。已有 q4 warm prefill 的 GDN 修复与 q1 decode 协议不同，本轮不扩大修改范围。后续 MTP 的显式 discard 与非法活行 guard 保持原协议，不能从 slot 失败反推 discard。

## 2. 修复的确定缺陷

| 缺陷 | 原因及影响 | 修改与回归 |
|---|---|---|
| 共用探针申请旧 FP32 大小的 CV arena，却创建 BF16 provider | `probe_current_only._runtime` 直接构造默认 WorkspaceGeometry；current-only、first-MTP、mixed 三个真实 forward 探针都会复用。D256/C16 每 Cube 旧值 4,997,120 B，BF16 需要 7,495,680 B，旧探针缺 1/3 必需空间 | 先归一化 provider 配置，再据其 BF16/C4/split 选择分配。CPU 适配器执行真实 allocator 的全部 shape/dtype/byte；2 Cube 实际分配 14,991,360 B。非法 false/不完整组合在分配前报错 |
| current-only 回执和三探针能力检查仍要求旧 reader | 正确 BF16 q4/q1 调用不能满足旧回执；能力检查没有覆盖实际使用的全部 BF16 CV/rotate-init 符号 | 回执通过生产 selector 选择 q4/q1；能力检查使用 BF16 符号集。负例拒绝旧 reader 回执；数值、逐位、后续缓存读写门不放宽 |
| first-MTP A/B 关闭 current FIA 时仍启用依赖它的 mixed split | provider 在进入数值用例之前正确拒绝配置，无法进行 A/B 对照 | 两臂都关闭 current-only 和 mixed split，仅 first-MTP current FIA 不同；两臂仍采用 BF16，未恢复旧 FP32 服务分支 |

正式服务的 [ensure_workspace](../oscar_ascend/runtime.py) 已根据 BF16 配置分配正确尺寸。**上述探针 arena 问题不能当作 `install_serve.sh` 正式服务 32K Trap 的根因。** 修改会进入正常 probe 安装链；直启仍安装当前 BF16 candidate，未增加新开关或伪装成服务性能优化。

另修正两类过期测试：native argv 现在显式使用 `native=True`，旧 FP32 baseline 应报退役错误；balanced clone 的文本同源检查只忽略精确三行说明注释，保留全部可执行代码、编译 guard 和 trap 的一致性检查。

## 3. 性能账：压缩存储不等于减少相同比例的运行时间

详细几何、七个实际 host planner 样本及源码 SHA 见 [成本账](../reports/pd_fusion_costs_20261017.json)。以下是源码推导的逻辑工作量，不是 NPU 时间、实测 HBM 事务或整轮占比。

1. **INT2 后仍有浮点工作区搬运。** Hq6/Hkv1/D256，一条历史 token 对一个 decode task：packed KV 136 B；解包后 BF16 K/V 写入 1024 B；Cube 逻辑读取 1024 B，合计 2184 B，尚未计 Q/P、partial、state 等。旧 FP32 对应小计 4232 B。原生 BF16 KV 一份读取的 payload 是 1024 B，不能假设原生只读一次，也不能把这些逻辑字节直接当作 HBM 事务。L2/重复装载/物理 DMA 需要设备证据。
2. **C16 共享解包，仍逐组计算和保存状态。** 对符合聚簇条件的同请求 16 组查询，满 KV256 tile 的 BF16 K/V 发布量为 262,144 B。GQA6 时每组 21 查询、有效 M126，QK＋PV 逻辑 FLOPs 为 `4×126×256×256=33,030,144`；16 组合计 528,482,304。每组 FP32 online state spill＋restore 为 `(128×256+2×128)×4×2=264,192 B`；16 组约 4.23 MB/tile。尚未计初始/最终状态、Q/P、softmax 和逆旋转；不能跨不同请求共享历史解包。
3. **切换 Matmul 类型有资源管理成本。** [SelectMatmul](../csrc/kernels/attention_bf16_history_operands.h) 在 BF16 QK/PV 与 FP32 inverse-Rv 切换时执行 End、PIPE_ALL barrier、TPipe Reset 和 Init。是否在目标 SDK 上完全正确复用资源、具体耗时，主机语法测试无法证明。本轮未改动该生命周期。
4. **空间容量与 operand 字节是两本账。** BF16 C16 arena 因新增 operand 区域是旧 FP32 的 1.5 倍。20 Cube 的示例为 149,913,600 B 对 99,942,400 B；N16384/H6/D256 的完整共享工作区为 729,711,872 B。该工作区每 worker 分配并由顺序 FULL 层复用，不按层数再次相乘。
5. **128-token 分流上限是性能策略边界，不是已证实的正确性 bug。** 当前 31q4＋长 P 拆分，33q4＋长 P 不拆分。MDL 核本身检查单 task 的 qcount/矩阵行数，没有 batch 总 N≤128 的约束；但扩大分流会改变 launch、负载均衡、缓存和不同场景成本，尚无完整配对证明净收益达到项目门槛，本轮未自动扩大。不能把 C16 的 M128 与 MDL 的 M32 容量之比当作 4 倍运算减少：两者 SetSingleShape 都传实际行数，q4/GQA6 为 M24，q1 为 M6。

因此，本轮未获得“比原生更快”或“比修改前更快”的证据；修复探针不是性能收益。需要设备验证的方向已明确，但不重跑用户已完成的 AISBench，也不拿短 probe 代表模型质量或整轮性能。

## 4. 本地结果与尚未通过的门

| 检查 | 结果及证据边界 |
|---|---|
| 最终 38 个文件的相关回归 | **714 passed / 1 skipped / 0 failed**；包含 PD/MTP、native seams、current-only/mixed、配置路由、探针、guard、BF16 解包/生成。主机 Clang 实际执行真实头的语言检查；不等于 CANN 编译 |
| 跳过项 | `test_runtime` 中重复 large FULL shape；同套参数已有 C16 覆盖。不是跳过数值失败项 |
| 初次回归 | 479 passed / 6 failed / 1 skipped；6 项为已修正的 native/baseline 测试 API 过期。保留 [原始 XML](../reports/pd_fusion_host_before_20261017.xml) |
| native reorder | 720×2 域检查包含于最终套件，不额外累加成更多独立 pytest 测试 |
| 冻结与派生 | FE0 7/7 哈希相同；BF16 六个派生源逆变换检查通过；42 个 checkpoint ID 不变；生产树/原生参考/精度门相对母版本无改动 |
| 已有完整数值筛查 | [原报告](../reports/bf16_all_sources_screen_20261016_v3.json)仍是 **7/11 失败，其中 2 例最终输出失败**；旧 FP32 comparator 全过。BF16 的旋转结果/P/解码舍入不保证原 oracle 容差，门槛未放宽。714 主机测试通过不表示该数值集通过 |
| 32K 请求 Trap | #165 仍缺当前 site/family/code，不能仅凭异步 GDN/event 栈定位；没有生产修复证据 |
| 本轮设备资格 | 本机无 CANN/NPU/torch_npu/Lima；CANN 构建、官方 CPU-debug、NPU 完成、graph capture/replay、服务、质量、性能均未运行，不提升其状态 |

最终 [JUnit](../reports/pd_fusion_host_final_20261017.xml)、[终端日志](../reports/pd_fusion_host_final_20261017.log)、[完整命令和退出码](../reports/pd_fusion_host_command_20261017.json)、[冻结检查](../reports/pd_fusion_invariants_20261017.json) 已保存。复跑同一主机套件：

```bash
.venv/bin/python - <<'PY'
import json, os, subprocess
with open('reports/pd_fusion_host_command_20261017.json') as f:
    command = json.load(f)
raise SystemExit(subprocess.run(command['argv'], env={**os.environ, **command['env']}).returncode)
PY
```

上述命令使用本机实际 Xcode clang 路径，只用于当前开发机。真机部署/运行不以这些 CPU 测试作为前置条件，也不会将 CPU 替代真实 NPU probe。
