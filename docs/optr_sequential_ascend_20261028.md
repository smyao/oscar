# OptR 官方分阶段拟合与昇腾离线加速（2026-10-28）

本轮按用户“参考官方拟合逻辑、加快昇腾准备”的要求，将固定 candidate 的候选拟合从完整 BF16 输出重放优化改为官方 sequential K/V 目标。仅离线准备改变；线上算子、runtime、PT 矩阵尺寸、clip、窗口、MTP/GDN、timer 和服务 argv 不变。独立 test 继续按已授权规则记录并允许实验发布，validation 的最终逐请求非劣门不变。

## 官方一致性与保留差异

参考固定上游 [66d46561 的 calibrate_optr.py](https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization/blob/66d46561b9bdf5d715f591e0f5845a4cbd927dee/optr/calibrate_optr.py)，不声称完整复刻其 CUDA/SGLang 环境。

| 项目 | 本轮之前 | 本轮 |
|---|---|---|
| K 训练目标 | K/V 同时量化的 BF16 输出归一化误差 | 官方 KL(P_ref‖P_K) + post-W_O 输出 MSE，lambda=1，V保持全精度 |
| V 训练目标 | 同上，每步完整重放 | 固定选中K的P_K，优化 P_K(V_hat−V) 的 post-W_O MSE |
| 头间训练聚合 | GQA头求和后平方 | 官方每个query头损失取平均；整层最终验证仍保留所有GQA/TP交叉项 |
| 参数化/优化器 | R0 exp(A−Aᵀ)，Adam | 相同，R0不随中间候选重置 |
| 默认steps/lr | 每阶段16 / 0.002 | 官方每阶段80 / 0.02 |
| held-out选择时机 | 第4/8/12/16步 | 官方零基step%20==0或最后一步，即1/21/41/61/80 |
| 选择指标 | 头级BF16逐请求门 | 官方阶段proxy held-out均值严格改善、margin=0；之后原整层BF16逐请求门 |
| 训练计算 | CPU完整tile重放及大投影 | NPU FP32 dense proxy；CPU小矩阵指数/Adam |

必要适配：Qwen真实sigmoid gate进入逐头输出误差；量化组沿用生产D256而非上游默认128；使用本项目已采集的稀疏query及完整K/V，不伪造上游末尾64个query。proxy沿用官方FP32及每条完整轨迹prefix/recent窗口；最终严格验证仍使用生产BF16、FP32 metadata、真实forward cut和两种分片。proxy不冒充NPU逐位参考。

原生采样来源、7条请求、train/validation/test题目划分均保持。当前仅一条4K训练请求、长请求题目拼接/可能重复、稀疏query、基础recipe3与test同源的局限没有在本轮解决。对齐训练目标不意味着自动获得官方论文的质量收益。

## 减少的离线计算

1. 每头一次性把完整历史和稀疏query搬到本次配置第一张物理卡对应的逻辑npu:0；仅一张卡承担拟合，未宣称四卡并行。模型采样子进程先退出，不同时驻留27B模型和优化器。
2. NPU使用FP32，显式设置ACL_PRECISION_MODE=must_keep_origin_dtype、关闭matmul HF32，并使用原生工程已有的jit_compile=False模式。接口依据为[Ascend PyTorch v2.7.1 npu_config](https://github.com/Ascend/pytorch/blob/v2.7.1/torch_npu/npu/npu_config.py)。若NPU不可用或设备可见性冲突则明确报错，无静默CPU拟合。
3. 官方matrix_exp和Adam保留CPU，小型R及其梯度通过可微设备复制传递。未依赖NPU matrix_exp的实现或其反向，不将其称为全NPU优化器。
4. K阶段缓存teacher概率/对数；V阶段只在阶段开始算一次选中K的概率，之后每步不再做K量化、QK、softmax。
5. 使用每query头G=W_OᵀW_O/hidden，计算zGzᵀ代替每轮5120维输出投影；按GQA头批量矩阵乘法。该等价仅用于官方逐头proxy，不能替换含跨头交叉项的整层验证。
6. 每个训练step只构造一次matrix_exp，不再每个query microbatch重建小矩阵指数图。
7. 最终CPU部署验证保持原算术次序和门限，但每个request/head仅旋转和量化历史一次。原先M个query、batch size B需2×ceil(M/B)次全历史K/V量化；现在为2次。回归检查完整输出逐位相同。tile attention仍在CPU，不能承诺首次准备变成分钟级。

所有缓存仅活在离线准备调用内；优化器和Gram矩阵不进入服务或PT。完整PT匹配时直接认证复用，不初始化拟合NPU。

## 身份与恢复

policy_version升级optr_sequential_ascend_v2，包含新目标、80步和执行设备身份，自动产生新产物目录。不能把旧joint-output目标的PT、头/层断点当成新算法结果。旧文件保留；新版本仍按头/完整层落盘并支持中断恢复。

只为旧原生trace提供明确兼容：从当前profile精确推导前版profile和目录，所有采样、模型、来源、TP、分片、token/teacher和文件哈希检查仍执行。不扫描或随意选择相似目录，不复制/篡改manifest和大trace，不提升失败JSON为已完成层。显式指定了不匹配旧PT仍拒绝，不覆盖用户指定文件。

固定入口仍为 `bash scripts/install_serve.sh --variant candidate`，无需新增开关。fit_plan显示fitting_objective、fitting_device、matrix_exp_device和deployment_validation_device；阶段日志显示proxy_loss，不能把其数值与旧normalized_error直接比较。

## 验证边界

258项本地回归通过，离线wheel构建/隔离导入通过，690个受保护线上文件哈希不变。证据在 `reports/optr_sequential_ascend_20261028/`。对经过SHA256核验的官方源码仅提取数学函数执行，使用相同query、无gate（全1）、相同量化组及FP32比较K/V损失、梯度和完整Adam选择，明确限定为CPU小样本。另验证真实gate、D256/GQA6、NPU设备映射/精度设置、NPU缺失失败、trace复用/篡改拒绝、旧PT和断点隔离、固定candidate真实发布加载和最终BF16重放逐位一致。

未连接真机、执行NPU编译/拟合/图或模型质量与性能实验。NPU执行路径的本机检查包括调用契约，不等于设备运行验收。未给出未经测量的加速倍率、ETA、精度提升或性能无回退保证。
