# v4后精度仍低：定向复核与下一条可区分证据

用户反馈eb1f8be5后精度仍低，尚未确认实际启动摘要中的v4标识。本轮不重跑大批回归、不连接真机、不改timer或生产数学，也不把此前CPU通过写成模型质量恢复。

## 最高优先级差异是PT配方

官方README的结果表注使用逐(layer, KV head)旋转、key centering与逐头K clip，并在variant表明确把`center`标为结果表配方。Qwen3.8脚本默认加载既有RotationZoo文件；它不运行本地这套共享层V1拟合。启动脚本没有绑定具体PT内容，单凭同名qqt/sst文件不能证明配方相同。

本地rotation_calibration.py先对全部KV头/TP的协方差求平均，再每层拟合一个[D,D]矩阵；产物没有k_mean。official_math.py的K路径直接K@R，没有减每头均值。v4修了跨来源旋转域和合并次序，未改变该PT配方。因此“v4算术修复已部署”与“复现官方发布结果”必须分开。

官方固定源码38680852的fit_oscar2_variants.py使用每头M_q，经trace/D归一化后独立eigh、E·H·Pbr；center在所有key上减该头校准均值。官方逐头clip另从配置表加载，不由PT文件名自动启用。本轮只读诊断比较共享V1/perhead/center的K误差，固定同一个标量clip，隔离旋转/中心化因素；不宣称已经比较完整逐头裁剪配方。

之前“无需重新生成PT”仅对eb1f8be5那次数学顺序修复成立。若切换官方perhead/center配方，PT内容和版本必须变化，不能继续复用旧V1 PT。但现在直接删除重生仍只会得到V1，无法补上该差异。

## 已排除的具体猜测

- native与官方都在一次Q/K norm、RoPE之后，attention之前采样；没有发现重复RoPE或gate混入校准输入。
- FULL gate均在attention输出之后；scale均D^-0.5，TP4的Q6/KV1分组一致，校准没有MTP输入。
- 实际CPU helper组合核验连续/非连续QKV和current-only两态，finalize没有覆盖待写入Krot/Vrot；混合current确实进入旋转域CV。
- group256、sink64/recent256、clip .96/.92、FP32 metadata对应官方标量配置，没有发现W8A8专用补偿分支。

这些检查不能排除真机算子/内部ND-NZ格式问题，也不能证明LongBench输入和采样参数一致；当前没有证据据此修改生产。官方表为GPQA和PG19 KL，不与用户LongBench分数直接等同。

## 最小现场取证

`tools.audit_precision_identity`只读当前服务进程环境中指向的配置与PT；多个不一致候选时拒绝猜测。它区分checkout版本、配置选择和实际算子执行证明：读取/proc配置不等于在worker内部验证已加载的Python/kernel。

可选`--compare-calibration`读取一键启动已经保存的rank*.pt，不加载模型、不调用NPU、不发请求、不改PT。比较共享V1、逐头旋转、逐头旋转加中心化在同一批真实校准K上的误差，并分别列出prefill/decode反量化。结果是校准分布上的量化误差与q_cov加权score误差代理，不是attention/LongBench分数；中心化引起的合法公共logit偏移不计为误差。

服务保持运行，在本工程根目录执行：

```bash
git pull --ff-only && python3 -m tools.audit_precision_identity --compare-calibration --output reports/precision_identity.json
```

终端仅输出身份和误差摘要，详细逐层结果落盘。若无可识别的运行服务或存在多个不同配置，明确要求`--config`，不猜测最新日志。校准样本缺失/身份不同会拒绝比较，并保留已读取的配置/PT身份信息。主机5项针对性检查通过；另用固定官方函数小样本对拍basis最大差为0。这不是新的真机精度证据。

下一步由这条证据决定：若没有实际启用v4，先修启动使用；若确认v4且perhead/center显著降低真实样本误差，优先补完整PT配方与各来源相同的中心化；若无改善，不凭配方名字盲改，继续定位目标算子或评测输入。不存在由本轮源码阅读直接宣称五点根因已经确定的依据。

官方来源：

- https://github.com/FutureMLS-Lab/OSCAR/blob/38680852e52c10bd6b9e9a64ed9bc815e742859b/README.md
- https://github.com/FutureMLS-Lab/OSCAR/blob/38680852e52c10bd6b9e9a64ed9bc815e742859b/rotation/run/qwen3.8-27b.sh
- https://github.com/FutureMLS-Lab/OSCAR/blob/38680852e52c10bd6b9e9a64ed9bc815e742859b/rotation/tools/fit_oscar2_variants.py

源SHA及公开RotationZoo刷新结果见reports/official_semantics_followup_20261021/。原始首次审计与数值修复见docs/oscar_official_numerical_alignment_20261021.md和docs/official_math_v4_20261021.md。
