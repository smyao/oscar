# 按用户指定 oscar_base 恢复 OptR 拟合来源

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
