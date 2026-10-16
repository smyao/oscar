# OSCAR 全源 BF16 精度契约改造

2026-10-16；母版本 `07a70d8`，五份派生核继续绑定 `6029f2a` 的固定母源哈希。用户确认的边界是“统一精度契约，保留 OSCAR 旋转与合并”。本文件覆盖旧 history-only 方案的默认关闭、source1/2 FP32 和生产拒绝配置规定；历史失败报告继续有效，不改写为通过。

## 实际实现

| 环节 | 本次实现 | 与原生的关系 |
| --- | --- | --- |
| Q/K/V 入口 | BF16 | 相同入口类型 |
| 普通历史存储 | INT2 striped 编码 + FP16 scale/zero | 仍有不可逆量化误差 |
| 历史解包 | 每 token 的四个码值分别按 FP32 `Mul → Add` 重建，RNE 一次舍入 BF16；整数索引直接 Gather 16 位结果 | 不展开 FP32 K/V tile；并非用 BF16 算术计算 scale/zero |
| 历史 Q | 原有 FP32 旋转，发布矩阵输入时 RNE 到 BF16 | OSCAR 自有旋转保留 |
| sink/recent/current 的 Q/K/V | 原始 BF16 直接发布；仅一行临时 FP32 用于原有限性检查 | 存储位和矩阵输入位保留，含正负零 |
| 全源 QK/PV | BF16 A/B，FP32 C；P 从 FP32 softmax 权重 RNE 到 BF16 | 统一矩阵输入与累加类型；不声称等同未公开的 dense FIA 内部舍入顺序 |
| scale/mask/softmax/max/sum | FP32；分母在 P 舍入前累计 | 保留稳定统计，未把全流程降为 BF16 |
| online 输出累加、partial/LSE/merge | FP32 | 原分源、多 split 与合并算法保留 |
| 历史输出逆旋转 | FP32 矩阵输入/输出 | 是唯一仍可达的 FP32 Matmul；不是 FP32 attention 备用路由 |
| 最终输出 | BF16 | 与入口/原生输出类型一致 |
| 已经调用原生 FIA 的 current-only/current 分支 | 继续原生 FIA | 不改写原生内核 |

`attention_bf16_history_*` 六个旧研究 ABI 名称兼容保留，但现在覆盖全部三源。q4、later-MTP q1、C4 member、C16、range、balanced 全部由共享 selector 映射到新核。生产配置缺省启用 `attention_compute_dtype=bfloat16` 和兼容别名 `experimental_bf16_history=true`。显式 FP32、False、缺依赖、非 D256 或非 `striped_v1` 均报错；生产 `baseline`（旧 FP32 OSCAR）已退役，`native` 继续表示禁用 OSCAR 的原生基线。冻结旧核只供独立对照，不被生产选中。

直接启动与带探针入口共用候选配置。默认构建必含六个 BF16 能力，旧产物不能凭编译缓存复用。带探针入口保留真实算子、图和完整服务门；新 BF16 数值门失败即停止，直接启动仍不运行探针、不提供验收证据。

## 所有权与其他功能

- INT2 外部布局、writer、精确物理页快照、MTP 位置/丢弃语义、原生 GDN、原生源码、冻结 `status_guard`、`ops/reference.py`、`acceptance.json` 不改。
- C4 每 lane 的两个原 H16 域仍按原位置发布；MDL 双缓冲、prefetch 错误提交顺序不变。query/KV 尾行、空 partial、tag→finite 错误顺序保留。
- sink/recent/current 直接 DMA 到 BF16 tile；逐行 FP32 `ReduceSum` 只保持旧检查，不进入矩阵输入。使用 MTE2→V、V→S、S→V 以及原发布/复用屏障保护 UB。
- H16 自然码缓冲从8192增为8704字节，防止 padded stride 越界。H32/H64 原缓冲已含 padding。新增索引 scratch 与 Q/P cast scratch 分时复用16KiB。
- 五个生成源的每个替换都可逆还原固定母源；生成器 `--check` 同时检查母源身份与生成字节。独立复审另核对 C4、MDL、half Gather 索引和缓冲地址。

## 工作量与新增开销

QK/PV 的乘加次数不变，只改变操作数精度。D256 的 K/V 仿射重建从每 token 两组256个元素降为两组4级表，即512次 Mul 和512次 Add变为8次 Mul和8次 Add；这只是仿射部分64倍减量，不是 attention 或模型64倍加速。

单个 KV256/D256 工作单元的 K/V 矩阵发布从524288降为262144字节；Q/P 发布也由4字节变为2字节。这里是源码 GM 地址流量，不等于真实 HBM 流量。位解包仍需要 Shift/And；新增 float→int32 的小批索引转换、整数 offset 运算和16位 Gather，故不能仅凭字节量断言净提速。

为隔离 FP32 score/state/Rv 与 BF16 操作数，本轮保留旧 FP32 arena并追加同索引BF16镜像，保留容量为旧核1.5倍：MDL每核1720320B，C4每核3741696B，C16每核7495680B，range/balanced每核1376256B。最大 C16 按20 Cube为149913600B，比旧核多49971200B，会占用KV预算。该容量不随历史长度增长。

显式UB为MDL175040B、C4174144B、H16三路174528B，均低于184KiB限制。BF16 QK/PV与FP32逆旋转之间仍需要 `End → PIPE_ALL → TPipe.Reset → Init`，避免同时预留两套L1/L0；这有切换成本，且该生命周期、新Gather/Cast组合仍需CANN和设备验证。不得把单个公开API存在当作完整执行通过。

## 已得证据与未通过项

2026-10-16新增真机回传#164：node93已尝试CANN9.1.0/ascend910b4编译，因共享解包头缺少dependent-template的`template`关键字两次失败，未进入正式服务。本次只修复该C++语法；主机Clang真实头+声明桩的旧错误对照与H16/32/64实例化通过，相关主机回归120项通过。修复后目标编译/设备执行尚未复验，以下数值失败与验收边界不变，见[修复回执](../reports/bf16_template_compile_fix_20261016.json)。

直接解码CPU模型通过65536个packed word×8码位、634872组FP16元数据组合及2539488个重建值比较；覆盖RNE中点、负零、有效/无效metadata和H16/32/64地址界限。它不是全部scale×zero笛卡尔积穷举，也不是设备执行。

[全源数值报告](../reports/bf16_all_sources_screen_20261016_v3.json)仍为11例旧FP32对照全部通过、统一BF16候选7例失败、4例通过；其中2例最终BF16合并输出也失败，真实rc2。新方案对INT2→BF16的实现正确不等于满足旧原空间oracle的精度门：例如已有identity抵消反例，BF16输入舍入丢失的信息无法由FP32累加恢复。没有提高容差、删样本或修改oracle；失败阻断带探针入口和所有性能资格，不妨碍用户明确授权的默认路由代码变更。

本机无CANN/Lima/torch_npu，官方CPU-debug、CANN构建、新NPU完成、图捕获/回放、服务、模型质量和性能均未验收。主机回归、冻结文件身份和启动配置证据汇总见[本轮交付回执](../reports/bf16_alignment_delivery_20261016.json)。不要求重跑AISBench，也不把合成测试写成模型质量结论。

本轮沿用既有 `configs/target.json`：后四卡4/5/6/7、端口7878，APC关闭、预算16384、原生调度器；没有更改模型路径、served name或MTP参数。
