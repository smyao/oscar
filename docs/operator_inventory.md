# 算子清单与验证状态

本清单区分数学oracle、候选AscendC源码和真实NPU验证。**尚无任何算子获得目标NPU数值/graph/performance通过状态。** `ops/reference.py` 只服务测试，生产loader没有oracle fallback。

| 能力 | 输入/输出和数据量 | 阶段/融合归属 | 入口与当前实现 | 同步/验证状态 |
|---|---|---|---|---|
| rotate_clip_quant_int2 | 原始K/V `[N,H,D]`；旋转 `[D,D]`；输出136B/head/token(D256) | store融合Cube旋转、Vector分位裁剪/量化/打包 | reference oracle；AscendC旋转/clip尚未实现，store只接FP32已处理输入 | C03未完成；不得用torch小算子拼接填补 |
| INT2 quant/pack/scatter | FP32 `[N,H,D]`→raw u8 SSM页，status i32 `[N,H]` | 增量History写入，一个launch，O(NHD) | `csrc/kernels/store_int2.cpp`; `store_int2_out`候选源码 | 设备事件；未编译/数值/性能实测；scale0明确拒绝 |
| INT2 unpack | u8压缩码+fp16元数据→FP32向量 | 应融合于C04 tile流水 | oracle存在；生产AscendC尚无 | 位序/metadata oracle可测，不等于device通过 |
| Sink/Recent store与迁移 | 原始BF16窗口 `[requests,W,H,D]`+position+commit事件 | window事务与History驱逐 | 需要device transaction/store kernel；未实现 | 不能发布未接受MTP token；不能双写全历史 |
| 有界解包/逆旋转 | 仅window/前缀所需的有界行与rotation | prefix restore/debug，禁止decode历史恢复 | oracle；device尚无 | 有界恢复测试待完成 |
| History Decode Stage1 | rotated Q `[R,Hq,D]`、INT2物理页、虚拟页表、GPU长度→partial FP32 | fused Cube QK/PV + Vector解包softmax | C04设计见ascendc_design；尚未实现 | 真实A2 UB→L1通路/GM流量约束待解，禁止假CV |
| Sink/Recent/window attention | Q+BF16窗口+positions→rotated-space partial/LSE | 与history统一causal/softmax | 尚无AscendC | FIA LSE/mask风险#53-69仍需实测 |
| LSE merge | partial FP32 `[R,S,D]`、LSE `[R,S]`→out `[R,D]`、LSE `[R]` | Stage2，单launch，O(RSD)，S≤128 | `csrc/kernels/merge_lse.cpp`; `merge_lse_out`候选源码 | 空段mask/全空写入已设计，未NPU验证 |
| MTP多query | query positions+接受长度+pending窗口 | C04query tile复用、窗口原子提交 | 只有集成契约；device算法未实现 | q_len1–4、全部/部分/零接受必须独立验证 |
| Prefill/chunked prefill | 原生QKV当前chunk，历史压缩tile | 当前chunk必要计算+增量store | native seam核实与oracle；缺device执行 | 禁止旧chunk全量dequant/重复量化 |
| 批量metadata | native qsl/seq_lens/block_table/slot、accepted→固定地址device描述 | prepare，消除Python逐请求热循环 | schema/CPU调试代数；device kernel未实现 | padding/query长度/事务错误状态待probe |
| 图与workspace | 输入/输出/status固定地址；small bounded partials | FULL_DECODE_ONLY，原生MTP scope | direct-launch出入口已写；没有捕获证据 | capture、replay、数值、性能全部独立pending |

所有数据同步仅允许明确的device事件和probe端设备同步；不以`.item()`把metadata拉回host。当前store的scalar-UB pack与merge的bounded split循环都有性能风险，已在kernel头和 `ascendc_design.md` 的D.4段登记，不能标成已满足预算。

唯一C++运行命名空间为 `oscar_ascend_ops`，两个out算子返回void，PyTorch schema和C++返回类型一致。构建manifest及extension.capabilities都只声明真实存在的两个component。`require_production_ops()`要求完整集合并明确抛出未实现列表；它不会自动回到原生BF16历史、Torch attention或Vector-only dot。
