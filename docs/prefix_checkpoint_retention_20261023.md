# 重复 prompt 的联合检查点保留

用户已确认 PC 开启，第 21 个请求与第 1 个相同。附件也包含
`PREFIX_CHECKPOINT_FIX joint_tokens=2304 gdn_tokens=768`，前一版边界修复已启用。
本次复现另一个缺口：FULL 页仍被请求持有时，GDN 检查点已被回收，联合命中降到零。

## 证据与原因

现场原文：[user_server.log](../reports/prefix_turnover_20261023/user_server.log)，SHA256
`084883647297945c16f522e575b75a05ed5ef8abe4831138c2b4c578c1edab7b`。
后续请求到来后仍显示 `Prefix cache hit rate: 0.0%`。这是近期 lookup 的 token 比例，并非每 10 秒清空。
日志没有逐请求 hash/驱逐记录，不能把合成重现的驱逐时刻当作现场记录。

原生调用链：

- `references/vllm/vllm/v1/core/single_type_kv_cache_manager.py` 的 Mamba `remove_skipped_blocks` 在状态推进时释放旧 checkpoint；FULL 页仍归活跃请求所有。
- `references/vllm/vllm/v1/core/block_pool.py` 的释放保留 hash，但页进入可回收队列，后续 `get_new_blocks` 移除旧 hash 再分配。当前占用未达 100% 不代表历史页没有被回收。
- `references/vllm-ascend/vllm_ascend/patch/platform/patch_mamba_manager.py` 要求同一逻辑位置的全部 GDN 组命中，且位置对齐 2304。只剩 FULL 页不能恢复状态。

前一版 ccc254b4 的单请求 prefill 后立即查询没有覆盖并发 decode 周转；旧 fixture 还用 object 替代 coordinator 基类，绕过真实继承的 cache_blocks 提交取整。本次修正 fixture，新增测试实际执行原生提交、分配、释放和查找方法。

## 配对重现

[测试源码](../tests/test_prefix_checkpoint_retention.py)、[结果](../reports/prefix_turnover_20261023/retention_results.json)：
20 个不同身份的 32768-token prompt，FULL2304/GDN768/MTP3，预算不超过 16384；decode 分配 q4、每步确认 1 token，持续 2048 步。池大小 795/800 页。
795 是依据附件 10.29 GiB 与本地页几何推得的邻近容量，不是现场 allocator 的精确读数。

| 条件 | 20 次 prefill 后 | decode 第 1022 步 | 第 1 个完成后重复 prompt |
|---|---:|---:|---:|
| 修改前，795/800 页 | 29952 tokens | 0 | 0 |
| 仅优先复用额外释放的未 hash 临时页 | 29952 | 0 | 0 |
| 本次保留检查点 | 29952 | 29952 | 29952 |

795 页对照在失去命中时仍有 194 页可分配。临时页回收调整没有解决反例，未进入生产。
以上使用真实原生管理方法与合成身份，不是模型推理或现场流量回放。

## 修复与成本

PrefixCheckpointLeases 仅在 OSCAR、APC、align 模式启用。原生 cache_blocks 返回后，核对 prompt 最后可读共同边界各 GDN 组的真实 request block、非 null、正确 hash/group id，使用原生 BlockPool.touch 各增加一个引用。
只持有这一组已有页，不复制状态、不产生额外 checkpoint、不追随 decode 增长。原生同一步 admission 屏障保持；hash 登记不等于 NPU 完成。

完成、取消、抢占及延迟释放统一经原生 coordinator.free；随后额外引用归还原生 LRU。插件卸载也释放全部引用。共享前缀按请求分别平衡引用，不恢复显式失效的 hash。
当前每活跃请求最多多持有 3 页，20 请求最多 60 页，即 795 页池的 7.55%。没有新增 NPU 分配，但减少可分配池容量，原生 admission/preemption 按真实引用计数处理，不能称为零显存成本。全体释放后非 null 页引用归零、free pool 恢复。

## 交付与边界

固定入口自动接入，保留用户 PC 配置，无需增加参数：

```bash
git pull --ff-only && bash scripts/install_serve.sh --variant candidate
```

原有启动摘要增加 `retention=active_prompt_checkpoint_v1`。PT recipe3、FP32 metadata 读端、旋转/clip、窗口、MTP、graph、timer、预算和设备保持，PT 不需重新生成。未修改原生/reference 源码或 CANN 内核。
真实 AsyncScheduler/KVCacheManager 接缝见 [生命周期测试](../tests/test_prefix_checkpoint_lifecycle.py)，主机验证见 [validation.json](../reports/prefix_turnover_20261023/validation.json)。
没有连接真机、运行性能或模型质量评测；不能据此证明 LongBench 差距小于 1 点。
档案关联：#26、#37–49 所有权；#127/#128 物理页与状态页区别；#157 MTP 原生边界；新增 #171。
