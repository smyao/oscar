# OPTR：完整精度链审计、CPU 准备提速与离线 PT

本轮以 `oscar-new-6@7720e3d` 为基底，按用户最新指令只处理 OPTR。改动限离线拟合、候选选择、数据搬运与认证；在线 CANN、codec、attention、KV 布局、PC/GDN/MTP、clip、窗口、服务 argv、timer 和后四卡/7878 不变。档案依据 #97/#166/#170/#172/#175–177；没有新的真机事件，不追加真机档案。

**结论边界：已改进候选搜索能力和离线执行路径，尚不能证明模型 accuracy 已改善或追平不量化。** 全链路源码检查不等于全模型质量评测。当前机器没有生产采样 PT、已拟合 PT 或配置所指的模型权重；不能用提示文本、CPU 合成 fixture 或修改 `test_only` 制造可部署 PT。

## 精度链逐项结果

| 阶段 | 已检查/保持 | 剩余边界 |
|---|---|---|
| 输入来源与拆分 | 固定 prompt/source hash，排除基线拟合用过的完整记录，train/validation/test 按完整 source record 隔离 | 同一来源语料内部的相关性及与用户任务的分布差异仍存在 |
| 原生采样 | native post-RoPE Q/K/V、pre-gate BF16 attention teacher；捕获 gate 与原生 sigmoid 对齐；每 rank W_O 分片和 TP token/frame 身份一致 | 当前 BS1/eager/PC-off/no-MTP 采样不代表所有线上调度；默认每 split 共80个 sampled query，不是全量 token |
| 重放 | BF16 中心化/旋转、INT2 clip/舍入、FP32 metadata、history/window/current、split/merge 与既有部署参考对齐 | CPU 参考不是 NPU 逐位复现；原生 FIA 内部归约不完全可见 |
| 训练目标 | 真正捕获的 teacher/gate，全头 W_O 投影后求联合残差；FP32、最终 BF16、rank BF16 三视图；两种 partition | STE 只是提议方向，不保证量化前向误差下降；单层 teacher-forcing 未包含多层量化误差传播 |
| 候选空间 | 原单头单侧候选保留；新增固定的全头 K/V 联合候选，复用同次 train 已算方向 | 能找到更多候选不等于必然存在可行改善点，也不代表泛化改善 |
| 选择与 test | 原逐 query/frame/terminal/request、双 partition、三视图非劣门保留；validation 严格改善；test 前持久锁定 | 有限样本 SSE 门不能替代 logits、生成轨迹或 LongBench accuracy |
| 发布与使用 | test 退化不重选、不发布；全层零改善不发布；源模型/base/trace/hash/实现摘要、全层覆盖、均值和正交性认证 | 最终模型质量与实际推理性能仍未测 |

采样链未发现新的确定性语义 bug，不能将旧日志的低分归因于未证实的问题。具体 CPU/source 审计见 `reports/optr_cpu_20261105/capture_audit.json`。

## v13 的精度改进是什么

v12 已经用全头联合输出计算 loss，但其候选每次只改变一个 KV head 的 Rk 或 Rv；**联合目标并不自动产生联合候选**。这样可能遗漏依赖跨头或 K/V 协同的变化。

v13 在原8个 train 梯度方向、最多16个单侧 train 候选和8个 validation 候选之外，增加：

- 同时应用所有 train 方向的两个固定半径候选，仍为 `.01` 和 `.0025`；不额外求梯度。
- 只由 train 选出其中最多一个，送入原 validation 门。每层总预算最多18个 train、9个 validation。
- 与原候选一起按预先固定的 validation 规则确定唯一结果，再进入独立 test。失败不回选第二名，不以缩半径或扩大样本搜索重试 test。

对同一组数值输入，原候选和指标保留，所以扩大后的候选集合所选 validation 总 SSE 不会高于原集合的最优值；这只是一项有限样本、同前向算术下的性质，不保证所有指标相对旧赢家更好，更不保证独立 test 或模型 accuracy 更好。独立 test 失败时仍停止，不能拿该集合性质放行。

新 policy 身份为 v13；`joint_native_residual_v1` 配置开关保留，默认 candidate 自动使用新 policy。v12/11/10 符合来源要求的原始 capture 可以复用，旧拟合 checkpoint/旧最终 PT 不能冒充新策略证据；实现摘要变化也要求重拟合，不能通过跳过摘要换取表面提速。

固定 CPU 反例采用真实基线 fitter、非零 gate、同层固定 W_O、独立随机输入和 dense attention teacher：原单侧候选全部被 train 门拒绝，联合候选通过原 train/validation 全部门，SSE 分别 `1.64271145→1.60993935`、`0.32933790→0.31569689`。但独立 test 为 `0.27251898→0.36121172`，所以它仍必须拒绝发布。这个反例证明搜索范围缺口及 test 门的必要性，**不是已获得精度收益的例子**，不根据 test 再寻找能过的策略。

## 准备速度改进与成本

同一个候选、head、request 的两套 partition 原来分别对完整历史做中心化、旋转、clip、量化/反量化。现在只准备一次历史，分别执行原两套 attention/mask/split/merge；完整指标按原顺序聚合。只缓存当前 request 的展开历史，不跨 request 累积。对象、版本、request、clip 绑定防止串候选复用；梯度调用保留独立计算图。

多头联合候选只重新计算 BF16 操作数变化的头；其他头复用原投影，但最终仍按完整 TP 头顺序相加再求误差，不能把各头 SSE 相加替代联合输出。选择、独立 test 的完整标量证据和常量哈希需与禁用共享的路径一致。

合成 CPU 候选评估曾测得 `0.12227s→0.08641s`，仅代表该 D256/GQA6/4头/4096历史/8query 的评估范围。进一步对同 v13、D256/GQA6/4头/541历史/3query 的完整单层拟合及 test 做3次配对：独立随机 split 的中位耗时 `1.82838s→1.54018s`（降低15.76%），选中联合候选的控制例 `1.85298s→1.58003s`（降低14.73%）。后者不同 split 使用同数值，仅检验变化赢家和 test 路径等价，不能当泛化证据。所有候选指标、最终 test、赢家与常量哈希一致。

原始时间、环境和复现脚本在 `reports/optr_cpu_20261105/history_reuse.json`。它比较**同 v13 下是否共享历史**，不是旧 v12 与新 v13 的整份 PT 对比；新增联合候选有额外回放成本，跨机器搬运与哈希也有 I/O 成本，真实完整准备耗时尚未知。

已有持久 spawn worker、完成层续跑、投影缓存、线程上限、单 split 等价复用不是本次新增收益。没有采用 `exp(0)` 梯度线性化：虽然数学一阶导数相同，FP32 实测有差异，不以它冒充位级等价优化。

## 无 NPU 的本地生成

拟合本来就在 CPU；以前的障碍是源机器绝对路径和模型身份认证耦合。新增 `python -m tools.optr_offline` 提供 export/fit/import，三个步骤均不采样、不加载模型、不初始化 NPU。首次获得真实 Q/K/V、teacher/gate/W_O 仍需要原生模型采样；缺少它们会明确失败。

使用同一份代码版本完成三步。在原来有完整采样与认证基线 PT 的机器、工程根目录运行：

```bash
python3 -m tools.optr_offline export \
  --config configs/target.json --variant candidate \
  --output /tmp/optr-inputs
```

默认发现当前或兼容旧策略的 `native-traces/capture-manifest.json`。非默认缓存可显式给 `--trace-manifest /绝对路径/capture-manifest.json`。export 不会因缺文件启动采样；目标目录必须尚不存在。它复制必要的真实采样、source、模型 config 与原基线配对文件，保留原 PT/manifest 字节，输出文件数和总大小；不用完整模型权重。

将整个 `/tmp/optr-inputs` 目录传到本机（目录可改名、改位置），在同版本工程的 CPU PyTorch 环境运行：

```bash
.venv/bin/python -m tools.optr_offline fit \
  --bundle /本机绝对路径/optr-inputs \
  --output /本机绝对路径/optr-result \
  --workers 1 --threads 4
```

默认1 worker/1 thread；显式 `--workers 4 --threads 4` 可并行处理4层，但每 worker 都持有一层全部 train/validation/test 及拟合临时数据，需足够 RAM。输入 bundle 与输出必须分离。采样、PT 和 manifest 字节不会被拟合重写；原配对认证会创建/打开 `base/.oscar.lock`，该目录须可写，不宣称支持整个目录 `chmod` 只读。中断后重跑同一命令会认证并续用当前实现的完成层与锁定选择，不省略独立 test。

只有所有门通过时才生成 `optr-result/refined.pt` 和 `.optr.complete.json`。将**整个结果目录**传回原机器，运行：

```bash
python3 -m tools.optr_offline import \
  --config configs/target.json --variant candidate \
  --result /绝对路径/optr-result
bash scripts/install_serve.sh --variant candidate
```

import 重新读取真实模型身份、认证原始基线与 trace、验证当前实现摘要和完整证书，才原子发布到正常 candidate 缓存。模型指纹仍绑定原机器路径/文件身份；不允许用“架构相同”替代。结果可带回目录中的检查点和诊断；服务不读取这些工作文件。标准服务仍执行原 NPU 装载校验。

若 export 使用了显式外部 `--trace-manifest`，import 同样传入原机器上该 manifest 的绝对路径；其内容 SHA 必须与结果证书一致。不要对旧拟合配置中的 `steps=80` 等参数作静默转换；当前 CPU 策略使用固定预算，以上命令直接由当前 `configs/target.json` 派生 candidate。

不要删除原始 raw/base，也不要把 test-only fixture 当作导出来源。跨 CPU 架构的 BLAS/舍入可能影响拟合选择；每份结果仍按实际本地前向严格选择，并明确标为 CPU 部署参考，不能宣称与 NPU 逐位一致。

## 验证记录

源码保护门以本轮基底逐字节检查在线源码、生成器、配置/argv、timer 和原基线 fitter；`rotations.py` 只允许 profile 中显式离线路径解析接缝变化，其余函数逐字保持。正常 profile 输出另与原函数对照。

新回归覆盖共享/独立回放的完整指标与常量、联合候选预算/来源/锁定/test失败、真实 CPU spawn、移走原模型/raw/base/source 后拟合、缓存续跑、文件篡改/路径逃逸/模型身份/test-only 拒绝。合成测试始终保留 test-only；CPU 成功不记作 NPU、图、模型质量或推理性能验收。

最终相关回归 **1100 passed / 1 skipped**；跳过的是可选本地源树审计，常规 CI 使用既有 pinned AST。初次运行的2个失败为旧测试仍预期 v11 optimizer 字段和前四卡，已按当前已授权策略修正测试，未修改生产门。命令、完整日志、保护范围和未测项存于 `reports/optr_cpu_20261105/delivery_validation.json`；其中 `uncommitted_local_workspace` 和未推送描述记录的是验证当时的状态。没有连接真机、运行模型或要求重跑 AISBench。

## 生产 PT 交付前检查（2026-11-05）

用户随后要求本地生成生产 PT 并通过 Git 交付。重新检查工程（包含被 Git 忽略的 artifacts）、已知临时输入目录及用户附件后，未找到真实 `bundle.json`、原生 trace 或配套基线 PT；配置模型目录在本机不存在。日志及合成 CPU 反例不能替代这些输入。本次源码交付包含离线工具、OPTR 改进及验证记录，**不包含生产 PT**，也不表示真机已可直接复用新 PT。

必须先从保有原数据的机器执行上述 `export`，将完整输入包带到本机；此步骤只导出现有文件，不启动采样或 NPU。本地 `fit` 通过全部门后，服务所需文件应放入源模型身份对应的目录：

```text
artifacts/rotations/<源机器模型指纹>/
  qqt_sst_<recipe3配置摘要>/
    calibrated.pt
    k_rotation_qqt_r_h_pbr.pt
    v_rotation_sst_r_h_pbr.pt
    .oscar.complete.json
  optr_<v13配置摘要>/
    refined.pt
    .optr.complete.json
```

若源机器已保留完全相同的基线四件，只需交付新 OPTR 两件；否则六件一起保留原字节交付。`artifacts/` 被 Git 忽略，应精确 `git add -f` 这些经认证文件，不提交锁、pending、raw trace 或拟合检查点。完整目录摘要只能在拿到真实 bundle 后确定，不能用本机路径或同名模型重算源身份。
