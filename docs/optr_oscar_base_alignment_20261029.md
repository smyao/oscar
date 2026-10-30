# 按用户指定 oscar_base 恢复 OptR 拟合来源

## 2026-10-30：修改后的全面自查与修正

本轮以 `0e3cd5f` 为基底，按“检查 → 可复现问题修正 → 完整复查”执行。固定 candidate 升级为 `optr_oscar_shared_train_seed_ascend_v7`，继续实际执行共享 OptR。下节 v6 的共享数学仍适用；其 seed 直接复用和无界固定缓存描述由本节覆盖。

### 发现并修正

1. **seed 与 held-out 来源交叉。** 原 seed 按 GPQA 原始顺序取前 512 行，而 OptR 按记录 SHA 排序分 train/validation/test。真实 bundled 198 条记录中，原始前三条分别属于 test、test、validation；因此初始化也用到了 held-out 来源。现抽出同一分区函数，seed 只取 66 条 train 记录的有序前缀。统计仍取最多 512 个对齐 Q/K/V 行，FP32 qqt/sst、rank 合并、FP64 eigh+epsilon/UHP、共享和零均值均不变。该遗漏损害验证独立性，不能直接归为 LongBench 40.94 的根因。
2. **采够 seed 后仍校验无用 prompt。** 原流程先渲染完整 token budget，再截取覆盖 512 行的完整 prompt；后面的超长 prompt 可能阻断本无需使用它的准备。现完整 prompt 达到采样上限就停止，实际参与的 prompt 仍严格执行预算、长度和完整性检查。recipe2/3 行为保持。
3. **大采样配置的离线固定缓存无界。** 共享搜索同时驻留全头 train/validation，默认形状固定缓存约 3.60 GiB，但合法的四个约 64K 长度、每 forward 64 个 query、decode 64 配置约为 50.47 GiB。现先按 tensor 形状核算：不超过 8 GiB 保持原驻留方式；超过时按 query 分片，一次固定缓存不超过 512 MiB，保留完整 K/V、所有 query、原顺序和步数。chunk 按 query 数加权，保持原均值目标；单个完整历史加一条 query 仍超限时，在设备分配前明确失败。额外搬运和固定概率重算只发生于超预算离线配置；这两个阈值不是包含 autograd、工作张量和 allocator 的实际总峰值保证。
4. **正交检查对非有限结果显式拒绝。** 三个检查入口补上 Gram 误差的 finite 检查，复用已有标量，无新增矩阵乘法或放宽容差。负例模拟 backend NaN；本地有限极大值反例产生 Inf 已被旧门拒绝，不能把防御性修正写成已复现 NPU 故障。

新 seed 身份为 `oscar_base_shared_train_v2`（仍 recipe6）；最终 PT 仍 recipe5。旧 seed、最终 PT 和拟合断点不能充作当前结果，固定命令自动准备新身份；原生 trace 在 OSCAR 关闭的隔离进程采样，与 seed 数值无关，覆盖一致时仍按 v1–v6 原身份精确认证复用，不将旧 seed 换标签。历史六版 policy/profile 冻结 SHA 均匹配。

### 修正后的复查

| 检查面 | 结果与边界 |
| --- | --- |
| 本机 OSCAR 数学 | 只读指定源码 AST 对照、FP32 统计与 rank 合并、FP64 UHP、共享/零均值通过；本轮未更改这些公式 |
| OptR 核心 | 固定官方独立 GQA K/V loss、共享参数梯度、逐步 Adam、固定选中 K 后的 V 通过，含 D256/GQA6 |
| 分块等价与生命周期 | loss/gradient/Adam、完整 K→V 搜索和选择与驻留路径匹配预定 FP32 容差；weakref 检查捕获并修正评分 `enumerate` 暂存前一行导致重叠驻留的问题 |
| 默认准备开销 | 原逐头一次 stage、共享 W_O、teacher 和 V 固定概率缓存保持；选中 K 仍只搬运一次，默认不增加 NPU 重算 |
| 数据隔离与兼容 | 真实 GPQA seed train-only；trace 全计划逐 token SHA 与修改前一致；recipe2/3 保留；旧 seed/PT 拒绝、历史 raw trace 精确认证 |
| 接受/发布/加载 | 整层共享候选，validation 每请求/forward/terminal 非劣和整层严格改善保持；test 仍按已授权实验政策记录，不参与矩阵重选；原子发布/中断恢复/四进程清理通过 |
| 服务与安装包 | candidate 继续 OptR，摘要显示 train seed；前四卡、7878、完整 argv、线上 attention/codec/runtime/timer 保持；wheel 构建、字节核对及隔离导入通过 |

统一回归 **630 passed、0 failed、0 skipped**。证据：`reports/optr_self_review_20261030/verification.json`、`pytest.xml`、`source_package_checks.json`；seed 两项真实红例日志保留。分块开发期间的一项单 query 测试夹具错误和评分引用滞留在统一回归前修正，没有放宽产品质量门。

仍有明确适配差异：参考 OSCAR 用短中文 prompt，这里用固定 GPQA train 前缀；官方 OptR 独立优化各头，这里按用户要求约束为每层共享矩阵；官方 FP32 优化目标与当前 Ascend BF16/FP32 metadata 部署由独立输出验证连接，采样包含稀疏 query、真实 forward 窗口及 Qwen gate。它们没有被宣称为源码逐位等同，也没有证明对任意数据集都改善。独立 test 仍可产生带退化记录的实验 PT。

本轮未连接真机、未做 NPU/图/LongBench/速度测试；不能宣称准确率恢复或实测性能无退化。线上每 token 计算和 KV 格式未变，产物加载仅增加已有标量的有限性判断。只本地提交、暂不推送。原固定命令不变；远端未获得本地提交前，真机 `git pull` 不会取得本轮修复。

## 2026-10-30：保留 OSCAR 共享结构，只替换寻优过程

本轮基底为 `d08c3d7`，对应用户“尽量保持原有 OSCAR 逻辑，只更换核心 OptR 的拟合寻优”。固定 candidate 现在使用 `optr_oscar_shared_upstream_ascend_v6`，仍实际执行 OptR，不增加开关。下面的 10-29 记录保留为历史；其中逐头独立优化、头子集选择和 v5 身份已被本节覆盖。

已确认的差异：指定参考 `oscar_base@9cd2429` 的 `tools/gen_rotations.py:207–231` 始终为每层输出一对共享矩阵；d08c3d7 只让初始 seed 共享，随后独立优化每个 KV 头，并允许头子集混选，最终矩阵不再共享。这是源码上的偏离，尚不能归因为用户全部精度损失。

本轮改动限于离线搜索和相应产物认证：

- 每层只学习一个 `A_K` 和一个 `A_V`，`R = R_base @ exp(A−Aᵀ)`，所有 KV 头使用同一 R。仍使用固定官方 FP32 独立 GQA 损失、原 clip/窗口、80 步、lr 0.02 和原验证频率。
- loss 按 `mean_KV_heads(mean_requests(phase_loss))` 归约。每一步固定同一个 R，逐请求累计 dL/dR，再一次反传到 A、一次 Adam 更新；不逐头轮流更新，不平均已独立拟合的矩阵，不二次除以头数。
- K 在全头 held-out 均值上选出唯一结果后，所有头固定该 K，再拟合共享 V。最终只允许整个共享候选与整个 OSCAR seed 二选一，保持每请求/forward/terminal query 非劣；不再混选部分头。独立 test 仍只报告实验结果，不重选矩阵。
- 保持现有 perhead_center 存储与 TP ABI，离线复制同一矩阵；发布、加载和完成层断点要求所有头的 Rk/Rv 严格相等、K mean 为零。共享拟合断点单独保存，不能用旧 head 断点代替。
- 已认证 recipe6 seed 直接复用；采样覆盖相同的 v1–v5 raw trace 按原身份认证复用。旧逐头 PT/fit 不再加载，固定命令自动准备新 v6 PT。

这称为“在 OSCAR 共享矩阵约束下使用 OptR 目标”，不是宣称官方 OptR 本身使用共享参数。D256、4 个 KV 头时，每层 K/V 的有效正交自由度从 261120 降至 65280；可减少针对少量校准请求的自由度，但不构成泛化或 LongBench 分数改善保证。新可行域仍包含原 OSCAR seed，验证未改善时保留该层 seed 是优化器的接受规则，不是关闭 OptR。

本轮保留的边界：采样仍为当前 GPQA/长序列 trace，没有改成参考短中文 prompt；seed 仍为参考数学的 FP32 共享统计/FP64 eigh+epsilon/UHP、零均值。INT2、K=.96/V=.92、sink/recent、BF16/FP32 metadata 数值链、MTP/GDN、CANN/runtime、timer、argv、前四卡和四 worker 均未变。不声称整套数据采集或线上实现与该参考逐位相同。

成本：线上每 token 没有新增操作、buffer 或 KV 字节。离线仍计算全部头的损失；CPU 正交矩阵/Adam 更新从每头一次变为每层一次。流式反向只保留一个请求的大图，V 阶段缓存选定 K 的概率后释放 Q/K/teacher/mask。固定 train/held 张量会同时驻留所有头，固定缓存约为旧逐头拟合的 H 倍；不能声称离线峰值显存不增或速度实测提升。

验证入口：`reports/optr_shared_oscar_20261030/verification.json`。覆盖固定官方函数在共享参数约束下的 loss/梯度、D256/GQA6、逐步 Adam、条件 V、严格共享、独立 test 报告、旧 PT/trace 身份、断点恢复、真实本机四进程和固定启动接缝。首次回归有一个旧负例同时破坏共享和正交性，因此先触发新共享门；已改为同时破坏各头相同元素以单独验证正交门，原失败记录保留。

无真机连接、NPU/图、模型准确率或性能测试；CPU 通过不能标为质量恢复。仅本地提交，暂不推送。固定命令仍为 `git pull --ff-only && bash scripts/install_serve.sh --variant candidate`，远端尚未包含本地提交时不能据此声称真机已生效。

## 以下为 2026-10-29 历史交付记录

本轮将固定 candidate 改为：**用户指定 OSCAR 的共享、零均值基底 + 官方 OptR 的 FP32 分阶段优化 + 现有部署输出验证**。它继续实际运行 OptR，不增加关闭或绕过开关。修改只涉及离线准备、身份认证和启动摘要，线上 attention、codec、缓存布局、MTP/GDN、timer、服务 argv 均保持。

本轮基底为 `478fe18c850edfdba4505c37c38da44fd25afa58`。用户指定参考 `/Users/loki.yao/Documents/workgit/oscar_base`，明确授权只读该目录；未读取其他被禁止的实现，未修改参考目录。按用户已有指令只做本地提交，不推送，不连接真机、不跑性能实验。

## 源码核对结论

参考仓库为 `smyao/oscar`，commit `9cd2429fb4fb934889fa6f51da5d54396012b5df`。它是用户指定的本机实现，不能称为官方 FutureMLS-Lab OSCAR 仓库。

| 环节 | 用户指定 OSCAR | 修改前 OptR v4 | 本轮 |
| --- | --- | --- | --- |
| K 统计 | FP32 `Q_g.T @ Q_g / (N·G)`，同层跨头、跨 rank 合并 | recipe3 逐 KV 头 FP64 统计、trace 归一 | 按指定源码共享统计，不做 trace 归一 |
| V 统计 | FP32 `w=kᵀΣqk`，归一后平方根加权 V 的协方差 | recipe3 逐头 FP64 energy moment | 按指定源码 FP32 sqrt-weighted 顺序 |
| 特征分解 | 合并后 CPU FP64 `eigh(sym(cov)+1e-6 I)`，`U·H·Pbr` | 逐头 eigh，无这项 epsilon | 按指定源码；epsilon 先 FP32 创建再转 FP64 |
| 基底矩阵/均值 | 每层共享一对 R，零 K mean | 每头不同 R，非零 K mean | 共享 R 展开到现有逐头存储，均值严格为零 |
| OptR 优化目标 | 该仓库没有 OptR 优化器 | BF16 STE、native teacher、gate、joint GQA 自定义目标 | 固定官方 OptR 源码的 FP32 原空间 K/V 损失 |
| 梯度归约 | OptR 官方对所有 KV 头再取平均 | 分别训练各头时漏掉该系数 | backward 除以全局 KV 头数，lr 保持 0.02 |

源码哈希写入 seed profile：`calib.py` 为 `a4db413470e123102b1423bfb933e83a789b417590ebde2ddfd9eb2fc1b5b9b7`，`gen_rotations.py` 为 `0dfbd105e9ff63018bbaacd93a84e6a8dea1efe3c99a07fa7e2d410e38ab405e`。

以上是确定的实现差异。现有 40.94 分数没有现场 commit/PT 身份与同条件配对结果，不能把某一个差异认定为全部掉点原因，也不能把数学对齐写成质量已恢复。

## 实际拟合路径

1. `tools.prepare_optr_seed` 在独立原生采样进程中收集 post-RoPE Q/K/V。按参考默认上限，只保留前 512 个对齐 token；不足 512 时使用本次实际完整样本数。取得上限后立即计算协方差并释放收集缓冲。
2. 继续使用当前固定 GPQA/chat-template 数据来源，保留完整 prompt，取得 512 行的那个完整 prompt 后停止后续前向。没有复制参考的短中文 synthetic prompt；这一采样差异已写入 provenance。原配置 30000 是选取完整 prompt 的预算上限，不表示此 seed 拟合用了 30000 行。
3. CPU 合并 rank 的 FP32 协方差，按指定源码进行 FP64 eigensolver 和 U·H·Pbr。生成独立 recipe6 seed，矩阵展开成现有 `[global_kv_heads,D,D]`，K mean 填零。`perhead_center` 字段在此仅是既有存储 ABI；`fitted_grouping=layer`、`key_centering=none` 与启动摘要说明真实拟合含义。
4. 在基底上实际运行 OptR K→V 优化，来源固定为 `daniel-eai/Output-Aware-INT2-KV-Cache-Quantization@66d46561b9bdf5d715f591e0f5845a4cbd927dee`。对给定 μ，K 使用 `Keff=quant((K−μ)Rk)Rkᵀ`；本轮 μ 按所选 OSCAR 基底为零。K loss 为每个 GQA 头的 `KL(Pt||Pq) + mean((((Pq−Pt)V)WOjᵀ)²)`；V 固定选中的 K，使用 `mean(((Pq(Veff−V))WOjᵀ)²)`。逐 GQA 头和序列分别平均。
5. 保持 80 步、lr=0.02、上游零基 0/20/40/60/79 验证，以及 `R_base @ exp(A−Aᵀ)`。各头独立 Adam 的反传 loss 除以全局 KV 头数，复现上游 all-head mean；不通过缩小学习率替代。
6. 现有 BF16/native teacher/gate/全 TP 联合输出只用于最终部署验证，不再定义训练损失。每请求、每 forward、terminal query 和双分片的非劣验证不放宽。独立 test 仍完整执行并按已授权实验政策报告；不用于重选矩阵，退化 PT 明确标为实验结果。

保留两项必要部署适配：INT2 分组仍为现有 head_dim=256 和 K=.96/V=.92；一个采样轨迹包含多个真实 forward，精确窗口按各 forward 的 context/cut 计算，不能直接使用轨迹最终长度的 recent 窗口，否则重引 #175。稀疏 query、GPQA 拼接与 gate-aware 部署验证也是现有适配。因此只声称指定纯数学函数与官方优化公式对照成立，不声称复刻官方完整模型、数据、CUDA 执行或论文结果。

## 缓存、启动和成本

- 新拟合 policy 为 `optr_oscar_base_upstream_ascend_v5`，最终仍为 recipe5 OptR PT；中间 source seed 独立为 recipe6。旧 PT/拟合断点自动隔离，不能混入新均值/矩阵。
- 原 recipe2/3 拟合实现及其 standalone prepare 保持。旧 raw payload 只有 FP64 QGram，不能拿它冒充参考 FP32 `Q.T@Q`，所以首次新 seed 会自动短采样。
- 相同采样覆盖的 v1–v4 原始 OptR 长序列 trace 按历史完整 profile、rank、请求、张量哈希认证复用。历史 profile 仍记录原 recipe3；不会重新贴成 recipe6。默认仍为 train/validation/test 各 4K、32K、64K，共 9 条独立请求，每条 8 个 decode token。
- 完整层继续由四个独立 spawn worker 调度，使用本次配置的前四张物理卡 `0,1,2,3`。seed 文件和完成标记原子发布，损坏或不完整产物不进入拟合，CPU 测试产物不能进入生产。
- 推理每 token 的算子、矩阵尺寸、KV 字节、设备 buffer 与服务参数无新增；同形状源码工作量保持。首次新 seed 采样和重新拟合有离线成本，设备拟合耗时及模型端到端性能未测，不承诺具体分钟数或速度倍率。

固定入口无需增加参数：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

本轮按用户要求未推送；远端取得本地提交后，这条命令会自动启用新拟合路径，无需手工删 PT。启动摘要可见 `optr_fitting_policy=optr_oscar_base_upstream_ascend_v5`、`optr_seed=oscar_base_shared_zero_mean`、`key_centering=false`。显式指定的旧 PT/旧 pair 仍因身份不符报错，不能静默替换用户明确指定的文件。

## 验证与未完成边界

测试包含指定参考函数的 portable AST 快照，并与实际只读参考的 SHA/AST 核对；包括 FP32 协方差、跨 rank 合并、FP64 eigensolver/U·H·Pbr 的 CPU 逐位配对，以及 D256/GQA6、官方 OptR loss/梯度、多头 Adam 每一步参数和最终选矩阵对照。另覆盖 seed 上限/释放、旧 Gram 拒绝、原子发布/恢复、旧 trace 身份、CPU 产物隔离、真实主机进程并行一致性、默认入口和原 recipe2/3 回归。

最终 **543 项通过，0 失败、0 跳过**；Python wheel 构建、隔离导入及 12 个生产变更模块的包内字节校验通过。机器可读结果与 JUnit 位于 `reports/optr_oscar_base_alignment_20261029/`，早期测试 fixture 的失败及修正保留记录。源码核对确认 `rotations.py` 仅改离线身份函数，候选配置与 argv 对比基底完全相同。未修改 CANN，因此本轮未重新编译 CANN；未运行目标 NPU、图、LongBenchv2 或吞吐验证。已解决本轮确定的拟合来源偏离，**模型精度是否改善仍未验收**。
