# BF16 attention 分支交付（2026-10-16）

本轮从主分支 `codex/oscar-ascend` 的 `7047ce36a19e001046a07a7d85de2a4092cd7941` 创建 `codex/oscar-bf16-attention`。主分支与 `codex/aisbench-stable` 不移动。用户反馈主分支32K约34分钟，仅为用户报告，不推断本轮BF16速度。用户最新要求取消测速，由用户进行真机验证；本轮不以性能门阻止交付，不声称达到30%或变为访存瓶颈。

## 计算精度与原生依据

原生调用依据和文件指纹见 `reports/bf16_attention_20261016/native_dtype_contract.json`：Qwen3.5 FULL 使用 Qwen3NextAttention，Q/K/V沿模型activation dtype进入原生FIA；公开A2 sparse attention实现也支持BF16乘数、FP32 L0C累加。闭源dense FIA的内部概率舍入和归约顺序不可见，因此只声明对齐BF16计算类型，不声明原生逐位相同。

| 环节 | 当前BF16分支 |
|---|---|
| INT2编码、FP16 scale/zero、物理页、BF16精确窗口 | 保持原协议 |
| Q旋转、INT2 tile重建 | 原FP32计算，进入Cube前RNE到BF16 |
| QK、PV、最终inverse Rv | BF16乘数，硬件FP32累加，单个CV kernel |
| online max/sum/alpha、归一化、LSE、partial、merge | FP32 |
| Rv缓存 | 初始化时生成natural ND BF16；原FP32 writer矩阵保留 |
| 最终输出 | BF16 |

BF16×BF16→FP32使用BF16乘法路径，不等于旧的FP32×FP32路径。FP32累加与online softmax状态用于稳定长序列归约，未强行将所有中间状态降为BF16。每tile未归一化P先RNE到BF16再PV；分母继续使用原FP32 exp。新合同 `configs/acceptance_bf16.json` 为 `oscar_bf16_attention_v2`，独立oracle在 `tools/bf16_reference.py`，容差在数值验证前固定为atol/rtol各0.005。原 `configs/acceptance.json` 和 `oscar_ascend/ops/reference.py` 未改；有意改变的BF16舍入与旧FP32输出差异记录为诊断，不再用FP32逐位门否决BF16。

## 系统连接

四个新能力为 `attention_cv_bf16_decode_out`、`attention_cv_bf16_decode_q1_out`、`attention_cv_bf16_cluster4_out`、`attention_cv_bf16_cluster16_out`。candidate由共享 `tools/serving_variants.py` 开启 `experimental_bf16_attention`。短prefill/通用布局使用C4内的完整C1处理；q4、q1及长prefill使用对应BF16核。缺能力、非D256或缺BF16 Rv均明确失败。没有原生BF16历史回退或完整历史恢复。

BF16 Rv在KV profiling之前分配并缓存，forward直接传入，不添加每层cast/矩阵回读。q4/q1不再使用旧FP32 direct-NZ入口；各核采用匹配BF16的布局。Q与P分开存放，避免half-width P跨AIV覆盖另一行FP32 score；原finite检查、错误域、padding、同步与FE0冻结算子保留。

| 资源 | q4/q1 | C4 | C16 |
|---|---:|---:|---:|
| 每core GM workspace上限(B) | 1,146,880 | 2,494,464 | 7,487,488 |
| 每AIV UB(B) | 158,656 | 157,760 | 157,760 |

上述是容量审计，不是实际流量或性能测量。原MTP、GDN、APC开关、默认0–3卡/8989、APC-off/显式16384及原生调度器保持现有语义；显式rear模式仍为4–7卡/7878。

## 验证与边界

- 新鲜完整CANN9.1/ascend910b4和Torch扩展构建通过，48个能力一致，126个host/stage/VM源码哈希一致，实际两份.so与manifest匹配。见 `reports/bf16_fullbuild_20261016/summary.json` 和 `artifact_verification.json`。该构建不访问目标NPU。
- 官方CPU-debug q4/q1正常及错误路径4例通过；完整prepare→BF16 CV→merge、status/padding/input/GM guards通过。见 `reports/bf16_decode_20261016/initial_validation.json`、`fault_correctness/summary.json`。
- C4/C16六个正常/故障用例通过，覆盖成熟簇、多个KV头、S3/C1、坏scale和最后成员NaN；完整输出和LSE通过独立BF16 oracle。见 `reports/bf16_clusters_20261016/validation_summary.json`。
- source/split task ABI按真实prepare核为field6=`kind*S+split`、field7=`kind`；oracle修正及原S1凭证不受影响的逐位证明见 `task_abi_correction.json`。不将手写test误差伪写成设备故障。
- 宿主接缝总套件577通过/140跳过（132为未启用真实NPU，8为重复大形状），后续父门模式绑定修订的117项定向测试通过；两组有重叠，不相加。启动plan确认默认/后四卡均实际启用BF16。见 `reports/bf16_attention_20261016/delivery_validation.json`。
- 实际NPU执行、NPU图捕获/回放、服务启动、模型质量及速度均未在本轮验收。CPU-debug和宿主接缝测试不能替代它们。42个历史checkpoint不因此升级。
- 用户取消测速之前，曾有一次q4模拟候选因本地VM OOM被SIGKILL，未生成完成性能结果。停止后没有重试/测速；原记录保留，不将它解读为数值失败或性能结论。

probe入口已适配：BF16核心精度/错误/图门替代旧FP32 reader性能门，生命周期门继续检查自身数学oracle、状态和物理页一致性；跨数学路径差异单独诊断，同一BF16路径的逐位约束保留。metadata/striped/native-current及四个生命周期门在BF16配置下不计时，记录 `not_run_user_requested`。若用户运行probe，无NPU或任一正确性/清理失败仍必须停止。

## 2026-10-16 APC-off故障修复

用户回传VEC UB越界后，已定位并消除新增大count FP32→BF16转换在实际指令模拟中的确定性失败，改为统一normal-mask、每次最多64元素的RNE转换。BF16计算、精度合同、MTP、APC-off和启动参数不变。相同q4完整功能case由异常终止变为全部数值/状态/guards通过；详见档案#164及 `reports/bf16_ub_fix_20261016/`。未测速，不将本地功能模拟描述为真机完成。

## 用户启动

首次切换到独立分支：

```bash
git fetch origin && git switch codex/oscar-bf16-attention && git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

此后固定直启为 `git pull --ff-only && bash scripts/install_serve.sh --variant candidate`。该入口编译安装并启动服务，不跑测试/probe，也不构成验收证据。显式后四卡追加 `--rear-cards`。省略 `--variant candidate` 会遵循现有配置，不能作为已启用BF16的证明；plan中的 `attention_compute.operands=bfloat16` 和 `optimizations.bf16_attention=true` 为选择依据。
