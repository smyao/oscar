# Qwen3.8/Qwen3.5 模型、旋转PT与BF16对齐（2026-10-18）

本轮按用户最新要求，默认改为 **Qwen3.8-27B 原版**，默认命令不再传 `--quantization ascend`。Qwen3.8/Qwen3.5 的原版与 Ascend W8A8 均有显式模型选择流程。仅在 `codex/oscar-bf16-attention` 交付，main/stable不变；性能测试暂停，没有连接真机。

## 对用户对照文档的核实

[对照文档](https://github.com/ava131/vllm-learning/blob/main/oscar/rotation-pt-comparison.md)的四个核心结论成立：旧自动入口只生成固定Hadamard；官方默认先找K/V两个PT，启用启动校准才学习矩阵；文件schema不同；默认裁剪参数不同。官方 [Qwen3.8脚本](https://github.com/FutureMLS-Lab/OSCAR/blob/339368e8fb440c53d5e157a5974036f25f3841a0/rotation/run/qwen3.8-27b.sh)本身是启动配方，不是独立矩阵拟合脚本。

实现对照固定到官方提交 `339368e8fb440c53d5e157a5974036f25f3841a0`。实际读取的源码URL和SHA见 [sources.json](../reports/official_alignment_20261018/sources.json)。用户文档针对旧stable f96820f；本轮结论还逐项核对了当前BF16分支，不把旧源码状态直接当作现状。

## 默认模型与启动

默认模型目录是 `/softwarePlatform/c00879303/Qwen3.8-27B`，API模型名 `qwen3.8`，activation dtype为BF16。官方HF配置确认仍使用 `Qwen3_5ForConditionalGeneration`、64层、16个FULL attention层、D256、24 Q/4 KV头，所以保留已有原生Qwen3.5类与MTP方法，不编造Qwen3.8专用后端。

移除了旧配置遗留的YaRN×4覆盖。Qwen3.8官方 `rope_type=default`、原生上下文262144；校准和服务均使用所选模型自身RoPE，显式用户HF覆盖仍由其配置决定。模型变更之外，MTP=3、APC默认关闭、16384 token预算、原生调度器、默认0–3卡/8989以及显式后四卡/7878保持。

所有入口共用模型元数据解析。原版不添加权重量化参数；显式选W8A8时从实际 `config.json` / ModelSlim描述识别，或由用户显式指定 `--quantization ascend`。不靠目录名推断权重格式，不让原版继承旧ascend开关。

```bash
# 默认原版，准备/复用校准PT后直接启动
git pull --ff-only && bash scripts/install_serve.sh --variant candidate

# W8A8：同一入口，自动读取该模型的量化元数据
bash scripts/install_serve.sh --variant candidate --model /softwarePlatform/c00879303/Qwen3.8-27B-w8a8 --quantization auto

# Qwen3.5 同样适用，不复用Qwen3.8的校准产物
bash scripts/install_serve.sh --variant candidate --model /path/to/Qwen3.5-27B --served-model-name qwen3.5 --quantization none
```

`install_observe_serve.sh` 和 `install_probe_serve.sh` 同样使用共享模型选择与旋转准备。BF16流程不启动native-synthetic吞吐负载、混合测速或后台NPU Event计时；保留真实正确性/错误/图门、短服务请求和资源清理。直启 `install_serve.sh` 仍无测试/probe，不能把直启成功当完整验收。

## 旋转PT如何生成

默认 `rotation_method=qqt_sst`。缺少匹配产物时，准备阶段启动独立的原生vLLM Ascend采样进程：模型在NPU推理，OSCAR缓存路由关闭；模型warmup完成后才安装采样hook，收集FULL层真实post-norm/post-RoPE Q/K/V。没有两次重放拼接Q与K/V，也没有拿随机数或固定H替代校准。

采用官方GPQA-Diamond数据源、seed0选项排列、模型chat template和30000 token上限；只选完整prompt。每个prompt单次prefill，长度/顺序/实际token范围严格核对，排除padding、decode帧、重复帧与缓存命中。采样进程只采一个prefill logits token，不产生后续decode前向；其MTP/图设置不影响随后正式服务。

Q统计按512 token块累计FP64 Gram，V统计按2048块计算官方attention-energy加权协方差；按TP全局KV头聚合，FP64特征分解后组成 `Rk=U_Q·H·Pbr`、`Rv=U_S·H·Pbr`，最终保存FP32矩阵。Ascend没有在此假定原生FP64 eigensolver：统计与拟合明确在离线CPU执行，模型采样和正式推理仍在NPU。这是显式准备阶段，不是推理回退。

输出官方层号格式的 `k_rotation_qqt_r_h_pbr.pt` / `v_rotation_sst_r_h_pbr.pt`，另有适配器使用的 `calibrated.pt`。模型身份、真实权重文件位置/size/mtime/inode、配置/index、tokenizer/chat template/ModelSlim描述、RoPE、prompt来源/预算和官方版本参与缓存绑定；该身份不是逐字节哈希全部权重。K/V/combined通过同一代际、pending/complete标记与SHA成对发布，运行时再次校验。旧Hadamard文件不会被当作本轮校准结果复用。

可在配置中设置 `rotation_calibration_prompts` 指向CSV/JSONL，`rotation_calibration_tokens` 指定预算；也可通过prepare入口显式导入官方K/V pair。原版与W8A8权重、不同模型/数据/有效RoPE设置各自校准。

## BF16与必要高精度

| 环节 | 本轮实现 |
|---|---|
| 模型activation、Q/K/V、精确sink/recent/current窗口 | BF16 |
| Q/K/V旋转 | 原生NPU BF16 matmul；BF16矩阵和结果，预分配当前chunk缓冲 |
| rotated Q的GM接口 | BF16，去除旧FP32中转载体 |
| QK/PV/最终Rv矩阵乘 | BF16操作数、硬件FP32累加 |
| clip、min/max、scale/normalized-zero和整数码决策 | FP32，与官方对应实现一致 |
| prefill反量化 | FP32 `(code-zero)*scale` 后舍入BF16 |
| decode反量化 | metadata先舍入BF16，减法结果和乘法结果各按BF16舍入；AIV必要FP32暂存不额外保留精度 |
| softmax max/sum/alpha/exp、归一化、LSE和merge | FP32稳定归约 |
| 离线校准统计/eigh | FP64，最终PT矩阵FP32，初始化时转BF16运行 |

新格式 `striped_fp32_v2` 保存四个FP32 scale/normalized-zero值。D256每KV头每token为144B；原136B/FP16 raw-min格式独立保留，不能混读。仅把metadata改成BF16虽然能匹配部分decode，却会改变官方prefill结果，因此没有这样压缩。

裁剪改为官方离散排序下标 `min(int(ratio*D),D-1)`，K=0.96、V=0.92；不再使用旧插值quantile。量化使用FP32 `scale=max(max-min,1e-8)/3`、`zero=-min/scale`，整数码决策前不再先舍入metadata到FP16。读者按原生prefill/decode类别显式选择精度，不用query长度猜；FULL_DECODE_ONLY图捕获固定decode模式。

新writer、q4/q1/C4/C16 reader和query检查均有独立能力注册，candidate统一选中新格式；旧格式不能静默进入新reader。保留#164的有界normal-mask BF16转换，避免重引入大count VCONV问题。无全历史BF16恢复，不修改原生vLLM/Ascend或GDN。

## 验证边界

独立合同在本轮设备功能验证前固定为 `configs/acceptance_aligned_bf16.json`（v3），数值容差仍为atol/rtol各0.005；旧v2/FP32 oracle不改。新oracle独立计算真实量化码、FP32参数、两个反量化模式、任务/padding/错误域和合并结果。

本地已完成54项能力的最终CANN/Torch整库编译，6个完整CPU算子用例通过（writer/Qcopy、两种反量化、C4/C16成熟簇及非padding尾部、错误域）。综合宿主回归732项通过、8个重复大形状跳过；后续112项定向检查通过，两组有重叠，不能相加。检查覆盖模型选择、启动参数、FP64统计公式与TP归并、PT原子发布/缓存拒绝、真实forward的新reader/writer路由、缓冲预分配。详见[本轮验证汇总](../reports/official_alignment_20261018/summary.json)。

额外整目录静态扫描仍有3项与本轮实现修改无关的既有问题（旧实验异常清理、已有metadata CPU镜像白名单、旧实验文件头），原错误已留存；没有宣称整个仓库所有测试全绿。真实NPU图、真实模型质量和性能不由这些本地结果代替；本轮没有声称已生成当前真机模型的最终PT或取得新的质量分数。

此外，Ascend适配保留原空间BF16精确窗口、各history partial的逆旋转与FP32分源LSE合并；官方SGLang使用旋转空间窗口及另一种最终旋转组织。校准公式、dtype与编码步骤逐项对照，不代表CUDA/SGLang和Ascend/vLLM跨硬件逐位相同。该差异已写进v3合同，不能用“100%对齐”掩盖未验证的模型级等价。
