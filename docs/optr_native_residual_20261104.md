# OPTR v12：对齐原生输出残差，保留 recipe3 严格发布门

本轮基底为 `oscar-new-6@a599e8a`，遵循档案 #170/#172/#174–177。固定 candidate 改用 `joint_native_residual_v1`，替代默认 v11 代理方向微步搜索。修改限定于离线旋转准备、证书和配置接线；不修改在线 attention、量化、clip、窗口、KV、PC/GDN/MTP、timer、设备或服务参数。

## 最新错误证明了什么

附件原字节见 [user_log.txt](../reports/optr_native_residual_20261104/user_log.txt)，解析见 [user_log_analysis.json](../reports/optr_native_residual_20261104/user_log_analysis.json)。仍是 `require_publishable` 拒绝发布零改善 PT；本片段没有新的 RESULT/耗时，不能再算作一次 113 分钟重跑。开头残行不归属到任何层；完整小步摘要只有 layer59、63。

| 项目 | layer59 | layer63 |
|---|---:|---:|
| 新小步候选 | 64 | 64 |
| 总 SSE 改善的候选 | 26 | 26 |
| 改善者最少退化 query 检查数 | 180 | 180 |
| 最佳小步总 SSE 相对变化 | −1.139% | −1.549% |
| 该小步最差项相对变化 | +8.050% | +6.737% |
| BF16 操作数完全不变 | 0 | 0 |

这两层的失败不支持“只是步子太小、被 BF16 全抹掉”的解释。降低平均误差仍损害部分位置是明确现象。query 计数横跨 partition 和投影视图，不是独立 token 数。日志汇总仍为 16 层、1216 候选、未发布，不能用两层详情补造其他层的分布。

## 与原 OSCAR 和旧 OPTR 的差异

原 recipe3 OSCAR 对逐头中心化样本统计 FP64 协方差，经 `eigh` 和 U·H·Pbr 构造固定旋转。它不做旧 OPTR 的 80 步 K + 80 步 V 代理优化，也不要求搜索结果逐 query 严格优于自身；所以准备 PT 的工作量不同。OSCAR 本身仍有原生采样和统计成本，不能解释为零校准成本。

源码审查与可复现 CPU 反例见 [source_audit.json](../reports/optr_native_residual_20261104/source_audit.json) 和 `tests/test_optr_native_objective_alignment.py`：

1. 旧 `optr_sequential.prepare_samples/phase_loss` 从 Q/K 重建 FP32 teacher 概率，不使用捕获的原生 teacher 和 gate。改 gate 后旧 loss/梯度不变，而真实输出误差改变。
2. 旧代理分开计算头贡献后平方；部署评估先联合投影求和，再计算误差。它缺少头间交叉项，也没有完整 K/V 联合量化误差。
3. v11 的 V 方向来自已选 K 下的训练，但单侧 V 微步配回原 recipe3 K，丢失了训练时的条件。

固定合成反例中，旧代理 loss 从 0.0265003 降至 0.0264896，部署输出 SSE 反而从 0.3814196 升至 0.3814622。该反例证明目标不一致，不能量化它对用户 42.52 分数的贡献。原严格 gate 正确拒绝退化，不能通过放宽 gate 修复训练目标。

## 新候选如何产生和发布

`oscar_ascend/optr_native_residual.py` 从认证的完整 recipe3 `Rk/Rv/k_mean` 出发，按实际 replay 使用 BF16/INT2、原生捕获 teacher、gate、W_O 和全部 KV 头。训练目标是双 partition、三个投影视图下的逐 query teacher-energy 归一化联合残差均值，包含首段 prefill、带历史帧及 decode。

每个头每一侧只计算一次分批反向梯度，投影到反对称切空间并归一化。以 `R0 @ exp(radius * tangent)` 生成固定半径 0.01、0.0025 的候选；每个候选只修改 recipe3 的一个头和一侧，不把前一候选累计到下一候选。分批前向与原完整 replay 必须逐值相等，否则明确失败。仅反向使用 STE，实际候选评估继续执行真实离散前向。

对当前四 KV 头，每层最多 8 次梯度计算、16 个 train 候选。只有 train 上原全部严格门通过且总 SSE 下降者，才能成为该方向唯一的 validation 候选；validation 最多 8 个。validation 仍要求两个 partition、FP32/BF16 output/BF16 rank output 三视图、每 request/frame/query/terminal 非劣且总 SSE 严格下降。

选择结果在独立 test 前持久化。test 退化就终止，续跑不重选；零改善层保留原 recipe3，全层零改善仍拒绝发布。新的训练筛选没有取代 validation 或 test。报告单列 train 和 validation，空 validation 也能说明原因，不把 train 通过写成部署验证通过。

旧 v10/v11 原始 capture 经过原身份/哈希/来源排除校验后可复用，原 recipe3 PT 保留。新实现摘要与 profile 独立，旧 fit/PT 不能冒充新策略证据。没有清空已有文件，也不把已有失败方向当成新训练结果。

## 理论收益的准确范围

设原 recipe3 的联合输出残差为 `e = y_recipe3 − y_native`，某个旋转变化导致输出变化 `δ`，则：

```text
E_new − E_recipe3 = 2〈e, δ〉 + ||δ||²
```

旧代理不直接优化这个 `e`，其下降不控制交叉项。新方向针对相同的联合残差，在理想可实现的 `δ = −ηe` 情形，`0 < η < 2` 时新误差为 `(1−η)²E_recipe3`。但旋转自由度有限、BF16/INT2 非光滑，STE 不能保证产生这种 δ，也不能保证存在满足所有逐 query 约束的严格改善点。**真正可认证的收益来自候选实际前向通过冻结的全部门，而不是梯度符号或“小步”标签。**

使用真实 recipe3 fitter 生成基线的固定合成 CPU 用例，直接残差方向可使 train/validation 全部位置和三视图非劣，其中 validation 总 SSE 从 0.2578071 降至 0.1379147；独立 test 却退化，已保留并断言拒绝发布。这同时验证方向有机会改善以及 test 门不可省略，不能把该 46.5% 当作模型精度收益。另有完整准备、发布、重新加载及身份认证的合成功能用例；指定方向用例明确标识，不冒充真实模型训练成功。

因此本次修复解决已复现的目标错位，并保证只认证实际测得的有限样本收益。该证据针对各层固定原生采样激活上的局部输出，不能保证多个量化层串联后 logits 或任务准确率的变化；**不能在没有新目标模型结果时保证 LongBench 超过 recipe3 OSCAR 的 45+，也不能保证这批现场样本一定产生新 PT**。未获满足约束的候选时仍明确失败，不会发布不变或退化的矩阵冒充成功。

## 成本、入口和验收

v12 删除默认流程中的旧 80+80 代理迭代及 64 个微步 validation 搜索，但新增真实 BF16 回放的 CPU 反向与 train 筛选。16 个 train、最多 8 个 validation 不是端到端提速比例；首次新拟合仍需计算，未测准备耗时。线上仅加载相同形状的旋转常量，没有新增算子、分支、buffer 或每步检查；源码保持不等同于重新测得性能零回归。

固定入口不需要新增参数：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

该命令只适用于包含本次改动的版本已同步到部署仓库后；本地启动摘要应显示 `optr_recipe3_native_residual_ascend_v12`。设备 0–3、端口 7878、TP4、预算 16384 和既有开关保持。原 v10/v11 仅保留显式历史策略能力，重新指定 candidate 会清除旧 repair 标记并启用 v12。

本地主机回归、打包和逐字源码边界见 [validation.json](../reports/optr_native_residual_20261104/validation.json)。相关回归 971 通过、1 个既有可选检查跳过；另新增真实 spawn 首次拟合检查 1 项通过，累计 972 项。该子进程使用真实非零梯度，train/validation/test 指标与串行逐值相同，并禁止进入旧 NPU fitting_device。4 项旧 research-window 配置测试在 a599e8a 的原配置模块上也失败，单独保留证据，不改无关生产策略来绕过。

887 个既有生产文件逐字保持，10 个既有文件限定为离线接缝；新鲜临时 build 生成的 wheel 中 223 个 Python 模块与源文件逐字相等，隔离导入策略/digest 一致。首次默认本地打包包含旧 build/lib 残留，验证已拒绝该包并用全新临时 build 重建，未删除用户原 build。

没有连接真机，没有运行 NPU/图/性能实验或 AISBench，没有新模型分数；42 项 checklist 的设备、质量、性能状态不升级。本轮修改留在本地 oscar-new-6 工作区，未提交或推送。
