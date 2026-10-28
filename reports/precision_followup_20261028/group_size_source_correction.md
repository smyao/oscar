# 官方量化 group size 的来源复核与搜索范围纠正

2026-10-28 只读核实。**官方 Qwen3.8-27B 专用脚本确实默认 `GROUP_SIZE=256`、`LLOYD_MAX=0`。本轮把旧 `references/oscar-paper` 搜索结果外推为“没有目标group256官方来源”的判断错误，现撤回。** 遗漏的是本工程此前已保存、带URL和SHA的较新官方快照。

已执行 `audit_official_group_size.py`。8份新版官方文件实际SHA均与已有manifest逐项匹配；完整URL、SHA与host路由证据在 `official_group_size_audit.json`。

| 本地原始来源 | 版本与实际内容 |
|---|---|
| `reports/official_alignment_20261018/official_qwen38_recipe.sh:12,15` | `sources.json`绑定官方339368e，目标Qwen3.8-27B默认group256、Lloyd-Max0 |
| `reports/official_semantics_audit_20261021/upstream/rotation/run/qwen3.8-27b.sh:12,15` | `upstream_manifest.json`绑定官方38680852，同样group256、Lloyd-Max0 |
| 两份目标recipe实际SHA256 | `ecd21aaf27fccbc44d5d4164d833dab623d0e1e49f2aff8c58a2f80e331e2375`，相同且均匹配各自旧manifest |
| `references/oscar-paper/rotation/eval_oscar_gpqa.sh:17,41` | 较旧通用GPQA脚本默认group128 |
| 旧 `rotation/qwen3-8B/eval_gpqa.sh:11`、`qwen3-32B/eval_gpqa.sh:11` | 这两个其他模型默认group128 |

新版目标脚本先导出256，再经 `_common.sh:50` 执行通用driver。driver的 `GROUP_SIZE="${GROUP_SIZE:-128}"` 保留已经导出的256；`:129,263`将它传给 `--kv-cache-quant-group-size`。Lloyd-Max0也由目标wrapper显式导出，并经新版driver `:325–327`传给SGLang。因此通用脚本的缺省128不覆盖该目标recipe。

## 显式 group128 在D256上是否被忽略

不会被忽略，但它属于不同于目标默认的额外数学配置。对38680852保存源执行未改的host AST（GPU kernel对象仅记录调用，不运行）：

- `unified_kv_pool.py:1101–1116` 对D256/group256解析为 `(256,1)`，对显式group128解析为 `(128,2)`。
- `oscar_rotation_clip_int2_kv.py:629–653` 按scale group数选择实际kernel；group256进入single核，`LLOYD_MAX=False`；group128的K/V均进入grouped核，实际constexpr为 `NUM_GROUPS=2`、`GROUP_SIZE=128`。
- grouped kernel `:293–306` 将整头clip后的数据reshape为 `[BLOCK_TOK,NUM_GROUPS,GROUP_SIZE]`，每组分别求min/max/scale/zero，所以不是只换标签。
- decode flush `_resolve_kv_quant_config` 在 `gpu_flush_int2.py:657–673` 返回group128的 `(BLOCK_QUARTER=64,NUM_GROUPS=2,GROUP_SIZE=128)`；其uniform路径 `:311–323`同样逐组归约并broadcast等级。reader `decode_attention.py:77–87,1364,1428`从metadata维度解析并传入GROUP_SIZE。

这不是GPU数值/速度测试，也没有运行完整模型。原始legacy `kv_quant_kernels.py` 的通用writer会拒绝num_groups2，但目标OSCAR mixed池调用的pretransformed-clip与flush专门核支持两组，不能只看legacy分支作相反结论。

本次不改变D256 packed ABI、kernel、PT或默认参数。不能把group128描述成“修复官方对齐错误”，也不能凭支持性推导四项模型质量或性能共同达标。
