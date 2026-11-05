# 2026-11-05：纯 OSCAR 精度审计与后四卡默认配置

本轮按用户澄清“当前只评估 OSCAR，未启用 OPTR”审查 recipe3；不把之前 OPTR 的42.52或v12实现归因于本次纯OSCAR分差。审计基底为 `oscar-new-6@04db2527e708dc127816249b06cbf9a58168c4d7`。后续用户要求默认后四卡，已单独落实。

结论：本轮没有确认新的在线精度实现bug，但确认了数值路径和静态校准代理的限制；不能据此声称已追平不量化版本。没有改动attention、INT2 codec、PT、clip、窗口、GDN/PC/MTP、timer或校准配方。生产变更只有设备默认配置，以及正常探针拒绝后四卡的设备校验接缝。

档案依据：#13–22数值/布局、#127–128物理页与元数据、#157–163 MTP/GDN、#170 recipe4失败、#172性能保护、#173–175窗口/current-only/PC处置、#176–177离线代理失败。未改CANN，不新增真机档案。指定官方OSCAR PR/paper完整快照在当前 `references/` 缺失；本轮使用本项目已有固定参考摘录与 `references/vllm`、`references/vllm-ascend`，不能声称重新全面核对了缺失的官方源树。

## 实现链检查

| 范围 | 证据与结论 | 限制 |
|---|---|---|
| Qwen外层算术 | 原生Q/K norm、RoPE在Attention之前；gate、o_proj在其后。新增CPU测试执行固定原生forward，未见重复RoPE/gate或遗漏projection | fused NPU算术使用边界替身，未执行完整模型 |
| TP4 | 执行原生QKV weight_loader四rank分片，与逐头Rk/Rv/mean对照正确 | 物理卡4–7不改变逻辑rank0–3；不是TP8验收 |
| 中心化/旋转 | current、精确窗口、压缩历史使用同一旋转域；K中心化一次，FP32 partial合并后一次最终逆旋转；MTP为identity/零均值 | BF16舍入不等于实数恒等变换 |
| INT2写读 | 检查clip、scale/normalized-zero、位平面pack/unpack、offset、mode0 FP32 metadata读端；未见新配对错误 | CPU参考和源码不能替代真实CANN指令执行 |
| attention | QK/PV BF16操作数、FP32累加；online max/sum/alpha与LSE merge为FP32；PV概率收窄到BF16 | native FIA内部没有完整可见证据，不能宣称逐位一致 |
| history/window/current | 可见token覆盖、因果边界、物理128页查表与FP32 LSE merge未确认新错误 | 精确窗口成员随forward分块变化 |
| PC/GDN/MTP | 复核已有checkpoint生命周期、later-draft discard/slot、none/align真prefill mask/count与图选择修复 | 默认PC关闭；CPU用例不等于用户实际cache/hit调度 |

不能笼统说“所有非attention逻辑都是未经适配的原生实现”：现有插件还包含已登记的PC引用保留、GDN prefill保护、MTP discard/slot与greedy rejection接缝。本轮保留这些修复，没有把旧问题重新当作当前新bug。

## 新增数值证据

以下均为构造的CPU反例，不是用户模型激活，也不是LongBench分差归因。

1. **BF16中心化可以先丢失信息。** 实际生产 `_center_key` 对可区分的BF16数 `1, 1.0078125` 减均值 `-1` 后，都得到 `2`。构造attention输出从 `0.0156237` 变为 `0`，没有INT2或非单位Rv。数学上 `softmax(q(K−μ)^T)=softmax(qK^T)` 成立，不代表BF16逐步舍入后仍精确成立。直接改FP32中心化会改变旋转输入与成本，当前证据不支持默认替换。

2. **小坐标量化误差也可能产生相关的attention偏差。** 默认K=.96/V=.92、mode0、sink64/recent256的16K构造中，K最大坐标误差 `0.06640625`，V误差为0；INT2引入的输出误差 `0.0644533`，额外KV256/PV-BF16/FP32 merge误差约 `0.000703365`。该例量化占主导，不能推广为真实模型误差占比，也不能把长context误差一律归因于merge累积。

3. **分块改变精度成员，不一定是token丢失。** context4096、最终query4099，q4固定cut=3841，连续四个q1最后cut=3844。两侧可见token完全相同且各一次，但位置3841的稀疏V在一侧仍精确，在另一侧被.92 clip量化掉，构造输出 `1→0`。这是已声明的fixed-per-forward窗口性质。旧query-relative/扩大窗口实验已有性能、精度或容量反例，不重新默认启用。

4. **recipe3的V协方差代理并不保持所有attention不变性。** 当前 `local_head_covariances` 使用未中心化K的 `kᵀQcov k` 加权V外积，而线上K减逐头均值。两个构造中，共同平移K后native attention、中心化K、Rk均不变，Rv却变化。真实BF16离线回放SSE分别为 `95.17065→107.50315` 和 `2.66503→2.35850`；第二例逐token K范数也不变。方向相反说明代理存在平移敏感性，不能据此推断改用中心化K能稳定提升模型精度。

5. **充分统计丢失时序信息。** 反转Q token顺序可保持Q Gram及recipe3 Rk/Rv/mean不变，而实际causal attention改变。旋转目标、K/V MSE或score SSE更好，不能单独认证attention、logits或accuracy更好；recipe4的既有失败也支持保留这一边界。

## PT来源与运行身份

完整K/V/combined生成配对、hash、模型身份、矩阵一致性和正交性检查仍存在，未发现TP4错取均值或矩阵。另有来源记录缺口：recipe3的 `calibration_cache_identity` 未纳入TP和部分执行配置 `additional_config`。真实测试中TP2拟合发布的全局逐头PT可在TP4同路径加载，各rank仍取得正确头。它是采样来源描述不完整，不足以解释当前固定TP4分差；本轮不因此批量删除或强制重生成PT。

校准默认累计30000 token，不代表包含一条30K长上下文。`tools/calibrate_model_rotations.py` 要求每条prompt加一个输出token放入显式预算，关闭chunked prefill/PC，`max_tokens=1` 不执行decode forward。短独立prompt的位置分布、KV注意力分布和原生层输入，不保证代表长上下文、多步生成及量化前层传播后的分布。

当前源码的 `install_serve --variant candidate` 会自动选择OPTR recipe5/v12；`--variant baseline` 则选择历史FP32路径并清除逐头中心化，它不是“迁移前BF16 recipe3 OSCAR”的同义词。用户实际部署版本/入口尚未核实，不能据本地HEAD判定用户启用了OPTR。本轮没有改变variant选择或替用户启用优化；已询问现有启动命令或 `PERF_RUNTIME_CONFIG`，不要求重新评测。

## 提升精度的优先方向

保留在线性能时，优先研究**离线校准代表性与目标对齐**：覆盖实际位置、context、decode/verify分布；以固定recipe3为对照，使用保留query/causal对应关系的样本，评价原生attention经gate/o_proj后的误差及下游logits。中心化K的V权重可以作为独立候选，但需要真正留出的attention验证，不能用相同协方差代理给自身通过。若仅更换同形状Rk/Rv/mean，在线算子调用形态可保持；离线采样与拟合仍有成本，目前没有模型证据证明收益。

若四级INT2的信息损失成为主要限制，可考虑分组、K4/V2或敏感层更高位宽。以下仅是D256每token/每KV head的**理论KV槽字节数**，不含精确窗口、GDN和页尾空余，不是实测压缩率或性能：

| 假设布局 | 槽字节 | 相对当前144B | 相对原生BF16 K/V 1024B |
|---|---:|---:|---:|
| 当前K2/V2，整头一个量化组 | 144 | 1.00× | 7.11× |
| K2/V2，每128维独立组 | 160 | 1.11× | 6.40× |
| K2/V2，每64维独立组 | 192 | 1.33× | 5.33× |
| K4/V2，整头组 | 208 | 1.44× | 4.92× |
| K4/V4，整头组 | 272 | 1.89× | 3.76× |

上表假定每组K/V共4个FP32 metadata。这些需要新codec/ABI/内核与性能验收，不是已实现方案，也不能承诺追平不量化。增加精确窗口、关闭PC/MTP、恢复已撤下native_exact、默认重新调clip，均不作为本轮捷径。softmax/merge已经FP32，不能把“升FP32”当作尚未实现的免费修复。

当前实际D256 **GM槽为144B**：K code64B、V code64B、FP32 metadata16B；**160B是UB对齐步长**，不是当前GM槽。物理页容量为 `A * floor(ssm_bytes / (A * Hkv_local * 144))`，none模式A=128，align/all为 `lcm(native_mamba_block_size,128)`；存在页尾空余。上表假设的160B新分组布局与当前UB步长无关。

## 后四卡交付及验证

- `configs/target.json` 默认物理卡改为 `4,5,6,7`；TP4、端口7878及其余配置保持。
- 默认直启与显式 `--rear-cards` 只读计划均选择4–7；配置生成的子进程环境覆盖继承的0–3。
- 正常observe probes共用的 `tools/probe_history_reuse._active_device` 原先硬编码前四卡，会拒绝新配置，现复用显式四卡校验；错误SoC、非法卡号、继承环境冲突仍失败。函数外所有字节不变，独立历史benchmark未改。
- 与 `04db252` 比较，服务argv完全相同；生产目录改动仅上述两个文件。q4/q1/C4/C16、生成器、运行时、timer和PT来源保持。
- 最后一组本地回归 **189 passed**，包含启动计划、配置传播、源码冻结、设备校验和15项新增精度审计用例。另有校准聚焦 **102 passed**、集成首组 **245 passed**。集合有重叠，不相加伪报总数。
- 集成第二组历史记录为244通过/2失败：后四卡环境失败已修复并进入55项相关通过；剩余legacy_v3测试同时设置 `official_math=False` 和 `fp32_metadata`，被正确校验拒绝。另一个旧附录端口测试预期8989而HEAD已7878；两者为基底已有的测试配置漂移，未伪称全库测试全绿。
- 未连接真机、未启动服务、未执行CANN/NPU/图/性能或模型质量评测，不升级42项验收状态。

原始明细：[数学/布局](../reports/precision_audit_20261105/kernel_audit.json)、[原生集成](../reports/precision_audit_20261105/integration_audit.json)、[校准](../reports/precision_audit_20261105/optimizer_audit.json)、[交付回执](../reports/precision_audit_20261105/delivery_validation.json)。
